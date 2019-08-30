from __future__ import absolute_import, division, print_function

import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.util.nest import flatten

import utils.data as data_
import utils.label_data as label_data_
import utils.model as model
from hvae_model import decoder, encoder
from utils import parameters
from utils.beam_search import beam_search
from utils.ptb import reader

from tqdm import tqdm
import pickle
import logging
from logging.handlers import RotatingFileHandler


class PTBInput(object):
    """The input data."""
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name
        )


def kld(p_mu, p_logvar, q_mu, q_logvar):
    """
    compute D_KL(p || q) of two Gaussians
    """
    return -0.5 * (
        1 + p_logvar - q_logvar -
        (tf.square(p_mu - q_mu) + tf.exp(p_logvar)) / tf.exp(q_logvar)
    )


def zero_pad(sentences, max_len):
    return np.array([sent + [0] * (max_len - len(sent)) for sent in sentences])


def write_lists_to_file(filename, *lists):
    with open(filename, 'w') as f:
        for lst in lists:
            for x in lst:
                f.write(str(x))
                f.write(' ')
            f.write('\n')


def main(params):
    if params.input_ == 'PTB':
        # data_folder = './DATA/parallel_data_10k/'
        data_folder = './DATA/ptb/'
        # data in form [data, labels]
        train_data_raw, train_label_raw = data_.ptb_read(data_folder)
        word_data, encoder_word_data, word_labels_arr, word_embed_arr, word_data_dict = data_.prepare_data(
            train_data_raw, train_label_raw, params, data_folder
        )

        train_label_raw, valid_label_raw, test_label_raw = label_data_.ptb_read(
            data_folder
        )
        label_data, label_labels_arr, label_embed_arr, label_data_dict = label_data_.prepare_data(
            train_label_raw, params
        )

        max_sent_len = max(
            max(map(len, word_data)), max(map(len, encoder_word_data))
        )

    with tf.Graph().as_default() as graph:

        label_inputs = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name="lable_inputs"
        )
        word_inputs = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name="word_inputs"
        )

        d_word_labels = tf.placeholder(
            shape=[None, None], dtype=tf.int32, name="d_word_labels"
        )
        d_label_labels = tf.placeholder(
            shape=[None, None], dtype=tf.int32, name="d_label_labels"
        )

        with tf.device("/cpu:0"):
            if not params.pre_trained_embed:
                word_embedding = tf.get_variable(
                    "word_embedding", [data_dict.vocab_size, params.embed_size],
                    dtype=tf.float64
                )
                vect_inputs = tf.nn.embedding_lookup(
                    word_embedding, word_inputs
                )
            else:
                # [data_dict.vocab_size, params.embed_size]
                word_embedding = tf.Variable(
                    word_embed_arr,
                    trainable=params.fine_tune_embed,
                    name="word_embedding",
                    dtype=tf.float64
                )  # creates a variable that can be used as a tensor
                vect_inputs = tf.nn.embedding_lookup(
                    word_embedding, word_inputs, name="word_lookup"
                )

                label_embedding = tf.Variable(
                    label_embed_arr,
                    trainable=params.fine_tune_embed,
                    name="label_embedding",
                    dtype=tf.float64
                )  # creates a variable that can be used as a tensor

                label_inputs_1 = tf.nn.embedding_lookup(
                    label_embedding, label_inputs, name="label_lookup"
                )

        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        sizes = word_data_dict.sizes
        word_vocab_size = max(sizes[1], sizes[2], sizes[0])
        label_vocab_size = label_data_dict.vocab_size
        # seq_length = tf.placeholder_with_default([0.0], shape=[None])
        d_seq_length = tf.placeholder(shape=[None], dtype=tf.float64)
        # qz = q_net(word_inputs, seq_length, params.batch_size)

        Zsent_distribution, zsent_sample, Zglobal_distribition, zglobal_sample, zsent_state, zglobal_state = encoder(
            vect_inputs, label_inputs_1, params.batch_size, max_sent_len
        )
        word_logits, label_logits, Zsent_dec_distribution, Zglobal_dec_distribution, _, _, _ = decoder(
            zglobal_sample, params.batch_size, word_vocab_size,
            label_vocab_size, max_sent_len
        )

        ## keep the mu and sigma within (-50, +50)
        Zsent_distribution[0] = tf.clip_by_value(Zsent_distribution[0], -50, 50)
        Zsent_distribution[1] = tf.clip_by_value(Zsent_distribution[1], -50, 50)
        Zglobal_distribition[0] = tf.clip_by_value(
            Zglobal_distribition[0], -50, 50
        )
        Zglobal_distribition[1] = tf.clip_by_value(
            Zglobal_distribition[1], -50, 50
        )
        Zsent_dec_distribution[0] = tf.clip_by_value(
            Zsent_dec_distribution[0], -50, 50
        )
        Zsent_dec_distribution[1] = tf.clip_by_value(
            Zsent_dec_distribution[1], -50, 50
        )

        neg_kld_zsent = -1 * tf.reduce_mean(
            tf.reduce_sum(
                kld(
                    Zsent_distribution[0], Zsent_distribution[1],
                    Zsent_dec_distribution[0], Zsent_dec_distribution[1]
                ),
                axis=1
            )
        )
        # neg_kld_zsent = tf.placeholder(tf.float64)
        neg_kld_zsent = tf.clip_by_value(neg_kld_zsent, -20000, 20000)
        neg_kld_zglobal = -1 * tf.reduce_mean(
            tf.reduce_sum(
                kld(
                    Zglobal_distribition[0], Zglobal_distribition[1],
                    Zglobal_dec_distribution[0], Zglobal_dec_distribution[1]
                ),
                axis=1
            )
        )

        # label reconstruction loss
        d_label_labels_flat = tf.reshape(d_label_labels, [-1])
        l_cross_entr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=label_logits, labels=d_label_labels_flat
        )
        l_mask_labels = tf.sign(tf.cast(d_label_labels_flat, dtype=tf.float64))
        l_masked_losses = l_mask_labels * l_cross_entr
        # reshape again
        l_masked_losses = tf.reshape(l_masked_losses, tf.shape(d_label_labels))
        l_mean_loss_by_example = tf.reduce_sum(
            l_masked_losses, reduction_indices=1
        ) / d_seq_length
        label_rec_loss = tf.reduce_mean(l_mean_loss_by_example)
        # label_perplexity = tf.exp(label_rec_loss)

        # Word reconstruction loss
        # print(word_logits.shape)

        d_word_labels_flat = tf.reshape(d_word_labels, [-1])
        print(d_word_labels_flat.shape)
        w_cross_entr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=word_logits, labels=d_word_labels_flat
        )
        w_mask_labels = tf.sign(tf.cast(d_word_labels_flat, dtype=tf.float64))
        w_masked_losses_1 = w_mask_labels * w_cross_entr
        w_masked_losses = tf.reshape(w_masked_losses_1, tf.shape(d_word_labels))
        w_mean_loss_by_example = tf.reduce_sum(
            w_masked_losses, reduction_indices=1
        ) / d_seq_length
        word_rec_loss = tf.reduce_mean(w_mean_loss_by_example)
        # word_perplexity = tf.exp(word_rec_loss)

        rec_loss = word_rec_loss + label_rec_loss

        #anneal = tf.placeholder(tf.float64)
        # annealing=tf.to_float(anneal)
        #annealing = (tf.tanh((tf.to_float(anneal) - 5000)/1800) + 1)/2
        # overall loss reconstruction loss - kl_regularization
        #kld_loss = -1*(neg_kld_zglobal + neg_kld_zsent)
        #kl_term_weight = tf.multiply(
        #    tf.cast(annealing, dtype=tf.float64), tf.cast(kld_loss, dtype=tf.float64))

        alpha = tf.placeholder(tf.float64)
        # alpha_val = tf.to_float(alpha)
        beta = tf.placeholder(tf.float64)
        # beta_val = tf.to_float(beta)
        kl_term_weight = tf.multiply(tf.cast(alpha, dtype=tf.float64), tf.cast(neg_kld_zsent, dtype=tf.float64)) \
                         + tf.multiply(tf.cast(beta, dtype=tf.float64), tf.cast(neg_kld_zglobal, dtype=tf.float64))

        total_lower_bound = rec_loss - kl_term_weight

        gradients = tf.gradients(total_lower_bound, tf.trainable_variables())
        clipped_grad, _ = tf.clip_by_global_norm(gradients, 5)
        opt = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        # optimize = opt.apply_gradients(zip(gradients, tf.trainable_variables()))
        optimize = opt.apply_gradients(
            zip(clipped_grad, tf.trainable_variables())
        )

        saver = tf.train.Saver(max_to_keep=10)

        with tf.Session() as sess:
            print("*********")
            sess.run(
                [
                    tf.global_variables_initializer(),
                    tf.local_variables_initializer()
                ]
            )

            ## save the values weights are initialized by
            with tf.variable_scope("", reuse=True):
                vars_to_save = [
                    'encoder/zsent_enc_gauss/mu/weights',
                    'encoder/zsent_enc_gauss/logvar/weights',
                    'encoder/zglobal_enc_gauss/mu/weights',
                    'encoder/zglobal_enc_gauss/logvar/weights',
                    'zsent_dec_gauss/mu/weights',
                    'zsent_dec_gauss/logvar/weights'
                ]
                for vts in vars_to_save:
                    vts_val = tf.get_variable(vts, dtype=tf.float64).eval()
                    np.savetxt('values/' + vts.replace('/', '_'), vts_val)

            # ## Load saved state
            # try:
            #     path = os.path.join(params.MODEL_DIR, "vae_lstm_model-20600")
            #     print("Loading state from:",path)
            #     # chkp.print_tensors_in_checkpoint_file(path, tensor_name='', all_tensors=True)
            #     saver.restore(sess, path)
            # except:
            #     print("-----exception occurred--------")
            #     exit()
            #     # traceback.print_exc()
            # print("*******Model Restored*******")

            total_parameters = 0
            #print_vars("trainable variables")
            for i, variable in enumerate(tf.trainable_variables()):
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                print('{} {} {}'.format(i, variable.name, shape))
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            print("Total params:", total_parameters)
            print("#Operations:", len(graph.get_operations()))

            # exit()
            if params.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            summary_writer = tf.summary.FileWriter(params.LOG_DIR, sess.graph)
            summary_writer.add_graph(sess.graph)
            #ptb_data = PTBInput(params.batch_size, train_data)
            num_iters = len(word_data) // params.batch_size
            cur_it = 0
            iters, tlb_arr, klw_arr, kld_zg_arr, kld_zs_arr = [], [], [], [], []
            alpha_arr, beta_arr = [], []
            # wppl_arr = []

            ## for debugging
            # file_idx = -1
            # np.set_printoptions(linewidth=np.inf)
            # logger = logging.getLogger()
            # logger.setLevel(logging.DEBUG)
            # ch = RotatingFileHandler(
            #     'values/log', maxBytes=5 * 1024 * 1024 * 1024, backupCount=1
            # )
            # ch.setLevel(logging.DEBUG)
            # # create formatter
            # formatter = logging.Formatter('%(message)s')
            # ch.setFormatter(formatter)
            # logger.addHandler(ch)
            for e in range(params.num_epochs):
                epoch_start_time = datetime.datetime.now()
                print("Epoch: {} started at: {}".format(e, epoch_start_time))
                total_tlb = 0
                # total_wppl = 0
                total_klw = 0
                total_kld_zg = 0
                total_kld_zs = 0

                ## alpha, beta schedule
                alpha_v = beta_v = 0
                if e > 2000:
                    alpha_v = beta_v = min(1, float(e) / 5000 - 0.4)

                for it in tqdm(range(num_iters)):
                    params.is_training = True
                    start_idx = it * params.batch_size
                    end_idx = (it + 1) * params.batch_size

                    # zero padding
                    pad = max_sent_len

                    sent_batch = word_data[start_idx:end_idx]
                    label_batch = label_data[start_idx:end_idx]
                    sent_dec_l_batch = word_labels_arr[start_idx:end_idx]
                    sent_l_batch = encoder_word_data[start_idx:end_idx]
                    label_l_batch = label_labels_arr[start_idx:end_idx]

                    # not optimal!!
                    length_ = np.array([len(sent) for sent in sent_batch]
                                       ).reshape(params.batch_size)

                    # prepare encoder and decoder inputs to feed
                    sent_batch = zero_pad(sent_batch, pad)
                    label_batch = zero_pad(label_batch, pad)
                    sent_dec_l_batch = zero_pad(sent_dec_l_batch, pad)
                    sent_l_batch = zero_pad(sent_l_batch, pad)
                    label_l_batch = zero_pad(label_l_batch, pad)

                    feed = {
                        word_inputs: sent_l_batch,
                        label_inputs: label_l_batch,
                        d_word_labels: sent_dec_l_batch,
                        d_label_labels: label_l_batch,
                        d_seq_length: length_,
                        alpha: alpha_v,
                        beta: beta_v
                    }

                    ## for debugging
                    # z1a, z1b, z3a, z3b, kzg, kzs, tlb, klw, alpha_, beta_, grads_, lce, wce, l_logits, w_logits, zs_state, zg_state = sess.run(
                    #     [
                    #         Zsent_distribution[0], Zsent_distribution[1],
                    #         Zsent_dec_distribution[0],
                    #         Zsent_dec_distribution[1], neg_kld_zglobal,
                    #         neg_kld_zsent, total_lower_bound, kl_term_weight,
                    #         alpha, beta, gradients, l_cross_entr, w_cross_entr,
                    #         label_logits, word_logits, zsent_state,
                    #         zglobal_state
                    #     ],
                    #     feed_dict=feed
                    # )

                    # file_idx += 1
                    # file_idx %= 5
                    # file_idx_s = str(file_idx)

                    # for i, x in enumerate(grads_):
                    #     np.savetxt(
                    #         'values/{:02d}_grads_{}'.format(i, file_idx),
                    #         x,
                    #         delimiter=','
                    #     )
                    # np.savetxt(
                    #     'values/label_cross_entropy_' + file_idx_s,
                    #     lce,
                    #     delimiter=','
                    # )
                    # np.savetxt(
                    #     'values/word_cross_entropy_' + file_idx_s,
                    #     wce,
                    #     delimiter=','
                    # )
                    # with open('values/vals_' + file_idx_s, 'w') as oof:
                    #     oof.write('\n'.join(map(str, [kzg, kzs, tlb])))
                    # with open('values/sent_idx_' + file_idx_s, 'w') as oof:
                    #     oof.write(str(start_idx) + '\n' + str(end_idx))

                    # np.savetxt('values/zs_mu_' + file_idx_s, z1a)
                    # np.savetxt('values/zs_logvar_' + file_idx_s, z1b)
                    # np.savetxt('values/zs_dec_mu_' + file_idx_s, z3a)
                    # np.savetxt('values/zs_dec_logvar_' + file_idx_s, z3b)
                    # np.savetxt('values/zs_state_' + file_idx_s, zs_state)
                    # np.savetxt('values/zg_state_' + file_idx_s, zg_state)
                    # np.savetxt('values/word_logits_' + file_idx_s, w_logits)
                    # np.savetxt('values/label_logits_' + file_idx_s, l_logits)

                    # logger.info('zs_mu %s', z1a.tolist())
                    # logger.info('zs_logvar %s', z1b.tolist())
                    # logger.info('zs_dec_mu %s', z3a.tolist())
                    # logger.info('zs_dec_logvar %s', z3b.tolist())
                    # logger.info('zs_state %s', zs_state.tolist())
                    # logger.info('zg_state %s', zg_state.tolist())
                    # logger.info('w_logits %s', w_logits.tolist())
                    # logger.info('l_logits %s', l_logits.tolist())

                    z1a, z1b, z3a, z3b, kzg, kzs, tlb, klw, o, alpha_, beta_, clipped_grads_ = sess.run(
                        [
                            Zsent_distribution[0], Zsent_distribution[1],
                            Zsent_dec_distribution[0],
                            Zsent_dec_distribution[1], neg_kld_zglobal,
                            neg_kld_zsent, total_lower_bound, kl_term_weight,
                            optimize, alpha, beta, clipped_grad
                        ],
                        feed_dict=feed
                    )

                    # for i, x in enumerate(clipped_grads_):
                    #     np.savetxt(
                    #         'values/{:02d}_clipped'.format(i), x, delimiter=','
                    #     )

                    cur_it += 1
                    total_tlb += tlb
                    # total_wppl += wppl
                    total_klw += klw
                    total_kld_zg += -kzg
                    total_kld_zs += -kzs
                    if cur_it % 100 == 0 and cur_it != 0:
                        path_to_save = os.path.join(
                            params.MODEL_DIR, "vae_lstm_model"
                        )
                        # print(path_to_save)
                        model_path_name = saver.save(
                            sess, path_to_save, global_step=cur_it
                        )
                        # print(model_path_name)

                avg_tlb = total_tlb / num_iters
                # avg_wppl = total_wppl / num_iters
                avg_klw = total_klw / num_iters
                avg_kld_zg = total_kld_zg / num_iters
                avg_kld_zs = total_kld_zs / num_iters

                iters.append(e)
                tlb_arr.append(avg_tlb)
                # wppl_arr.append(avg_wppl)
                klw_arr.append(avg_klw)
                kld_zg_arr.append(avg_kld_zg)
                kld_zs_arr.append(avg_kld_zs)
                alpha_arr.append(alpha_)
                beta_arr.append(beta_)

                print("Time Taken:", datetime.datetime.now() - epoch_start_time)

                plot_filename = "./plot_values_{}.txt".format(params.num_epochs)
                write_lists_to_file(
                    plot_filename, iters, tlb_arr, klw_arr, kld_zg_arr,
                    kld_zs_arr, alpha_arr, beta_arr
                )


if __name__ == "__main__":
    params = parameters.Parameters()
    params.parse_args()
    main(params)
