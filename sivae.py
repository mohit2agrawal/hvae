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
from sivae_model import decoder, encoder
from utils import parameters
from utils.beam_search import beam_search
from utils.ptb import reader
from utils.schedules import scheduler

from tqdm import tqdm
import pickle
import traceback


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
        1 + p_logvar - q_logvar - (tf.square(p_mu - q_mu) + tf.exp(p_logvar)) /
        tf.exp(tf.cast(q_logvar, tf.float64))
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
    # data_folder = './DATA/parallel_data_10k/'
    data_folder = './DATA/' + params.name
    # data in form [data, labels]
    train_data_raw, train_label_raw, val_data_raw, val_label_raw = data_.read_data(
        data_folder
    )
    data, encoder_data, val_data, encoder_val_data, embed_arr, data_dict, labels, encoder_labels, val_labels, encoder_val_labels, label_embed_arr, label_dict = data_.prepare_data(
        train_data_raw, train_label_raw, val_data_raw, val_label_raw, params,
        data_folder
    )

    # max_len_word = max(max(map(len, data)), max(map(len, val_data)))
    # max_len_label = max(max(map(len, labels)), max(map(len, val_labels)))
    max_len_word = max(map(len, data))
    max_len_label = max(map(len, labels))
    print('max len word, label:', max_len_word, max_len_label)

    word_vocab_size = data_dict.vocab_size
    label_vocab_size = label_dict.vocab_size

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

        d_word_inputs = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name="d_word_inputs"
        )
        d_label_inputs = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name="d_label_inputs"
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
                    embed_arr,
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

        # seq_length = tf.placeholder_with_default([0.0], shape=[None])
        d_seq_length_label = tf.placeholder(shape=[None], dtype=tf.float64)
        d_seq_length_word = tf.placeholder(shape=[None], dtype=tf.float64)

        Zsent_distribution, zsent_sample, Zglobal_distribition, zglobal_sample, zsent_state, zglobal_state = encoder(
            vect_inputs, label_inputs_1, params.batch_size, d_seq_length_word,
            d_seq_length_label
        )
        word_logits, label_logits, Zsent_dec_distribution, Zglobal_dec_distribution, _, _, dec_word_states, dec_label_states = decoder(
            d_word_inputs,
            d_label_inputs,
            zglobal_sample,
            zsent_sample,
            params.batch_size,
            word_vocab_size,
            label_vocab_size,
            max_len_word,
            max_len_label,
            word_embedding,
            label_embedding,
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

        ## label reconstruction loss
        #d_label_labels_flat = tf.reshape(d_label_labels, [-1])
        #l_cross_entr = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #    logits=label_logits, labels=d_label_labels_flat
        #)
        #l_mask_labels = tf.sign(tf.cast(d_label_labels_flat, dtype=tf.float64))
        #l_masked_losses = l_mask_labels * l_cross_entr
        ## reshape again
        #l_masked_losses = tf.reshape(l_masked_losses, tf.shape(d_label_labels))
        #l_mean_loss_by_example = tf.reduce_sum(
        #    l_masked_losses, reduction_indices=1
        #) / d_seq_length_label
        #label_rec_loss = tf.reduce_mean(l_mean_loss_by_example)
        #label_perplexity = tf.exp(label_rec_loss)

        # Word reconstruction loss
        # print(word_logits.shape)

        #d_word_labels_flat = tf.reshape(d_word_labels, [-1])
        #print(d_word_labels_flat.shape)
        #w_cross_entr = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #    logits=word_logits, labels=d_word_labels_flat
        #)
        #w_mask_labels = tf.sign(tf.cast(d_word_labels_flat, dtype=tf.float64))
        #w_masked_losses_1 = w_mask_labels * w_cross_entr
        #w_masked_losses = tf.reshape(w_masked_losses_1, tf.shape(d_word_labels))
        #w_mean_loss_by_example = tf.reduce_sum(
        #    w_masked_losses, reduction_indices=1
        #) / d_seq_length_word
        #word_rec_loss = tf.reduce_mean(w_mean_loss_by_example)
        #word_perplexity = tf.exp(word_rec_loss)

        d_label_labels_flat = tf.reshape(d_label_labels, [-1])
        label_softmax = tf.nn.softmax(label_logits)
        l_mask_labels = tf.one_hot(
            d_label_labels_flat, label_vocab_size, dtype=tf.float64
        )
        l_masked_softmax = tf.multiply(label_softmax, l_mask_labels)
        l_softmax_single = tf.reduce_sum(l_masked_softmax, 1)

        ## reshape to get back per sent
        l_softmax_batch = tf.reshape(l_softmax_single, tf.shape(d_label_labels))
        ## mask to remove effect of padding
        l_sent_mask = tf.sign(tf.cast(d_label_labels_flat, dtype=tf.float64))
        l_sent_mask = tf.reshape(l_sent_mask, tf.shape(d_label_labels))
        l_softmax_batch_masked = tf.multiply(
            tf.math.log(l_softmax_batch), l_sent_mask
        )
        l_softmax_mean_per_sent = tf.reduce_sum(
            l_softmax_batch_masked, 1
        )  #/ d_seq_length
        l_softmax_mean_per_sent_seq = tf.reduce_sum(
            l_softmax_batch_masked, 1
        ) / d_seq_length_label#max_len_label

        label_rec_loss = -tf.reduce_mean(l_softmax_mean_per_sent)
        label_perplexity = tf.exp(-tf.reduce_mean(l_softmax_mean_per_sent_seq))

        # Word reconstruction loss
        # print(word_logits.shape)
        d_word_labels_flat = tf.reshape(d_word_labels, [-1])
        word_softmax = tf.nn.softmax(word_logits)
        w_mask_labels = tf.one_hot(
            d_word_labels_flat, word_vocab_size, dtype=tf.float64
        )
        w_masked_softmax = tf.multiply(word_softmax, w_mask_labels)
        w_softmax_single = tf.reduce_sum(w_masked_softmax, 1)

        ## reshape to get back per sent
        w_softmax_batch = tf.reshape(w_softmax_single, tf.shape(d_word_labels))
        ## mask to remove effect of padding
        w_sent_mask = tf.sign(tf.cast(d_word_labels_flat, dtype=tf.float64))
        w_sent_mask = tf.reshape(w_sent_mask, tf.shape(d_word_labels))
        w_softmax_batch_masked = tf.multiply(
            tf.math.log(w_softmax_batch), w_sent_mask
        )
        w_softmax_mean_per_sent = tf.reduce_sum(
            w_softmax_batch_masked, 1
        )  #/ d_seq_length
        w_softmax_mean_per_sent_seq = tf.reduce_sum(
            w_softmax_batch_masked, 1
        ) / d_seq_length_word#max_len_word

        word_rec_loss = -tf.reduce_mean(w_softmax_mean_per_sent)
        word_perplexity = tf.exp(-tf.reduce_mean(w_softmax_mean_per_sent_seq))

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
        kl_term_weight = - tf.multiply(tf.cast(alpha, dtype=tf.float64), tf.cast(neg_kld_zsent, dtype=tf.float64)) \
                         - tf.multiply(tf.cast(beta, dtype=tf.float64), tf.cast(neg_kld_zglobal, dtype=tf.float64))

        total_lower_bound = rec_loss + kl_term_weight

        gradients = tf.gradients(total_lower_bound, tf.trainable_variables())
        clipped_grad, _ = tf.clip_by_global_norm(gradients, 5)
        opt = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        # opt = tf.train.GradientDescentOptimizer(
        #     learning_rate=params.learning_rate
        # )
        # optimize = opt.apply_gradients(zip(gradients, tf.trainable_variables()))
        optimize = opt.apply_gradients(
            zip(clipped_grad, tf.trainable_variables())
        )

        saver = tf.train.Saver(max_to_keep=10)
        config = tf.ConfigProto(device_count={'GPU': 0})
        with tf.Session(config=config) as sess:
            print("*********")
            sess.run(
                [
                    tf.global_variables_initializer(),
                    tf.local_variables_initializer()
                ]
            )

            ## save the values weights are initialized by
            # with tf.variable_scope("", reuse=True):
            #     vars_to_save = [
            #         'encoder/zsent_enc_gauss/mu/weights',
            #         'encoder/zsent_enc_gauss/logvar/weights',
            #         'encoder/zglobal_enc_gauss/mu/weights',
            #         'encoder/zglobal_enc_gauss/logvar/weights',
            #         'zsent_dec_gauss/mu/weights',
            #         'zsent_dec_gauss/logvar/weights'
            #     ]
            #     for vts in vars_to_save:
            #         vts_val = tf.get_variable(vts, dtype=tf.float64).eval()
            #         np.savetxt('values/' + vts.replace('/', '_'), vts_val)

            ## Load saved state
            try:
                path = params.ckpt_path
                if path:
                    print("***Loading state from:", path)
                    # chkp.print_tensors_in_checkpoint_file(path, tensor_name='', all_tensors=True)
                    saver.restore(sess, path)
                    print("******* Model Restored ***********")
            except:
                print("-----exception occurred--------")
                traceback.print_exc()
                exit()

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
            num_iters = len(data) // params.batch_size
            cur_it = -1

            all_alpha, all_beta, all_tlb, all_kl, all_klzg, all_klzs = [], [], [], [], [], []
            all_rl, all_wrl, all_lrl = [], [], []
            all_l_ppl, all_w_ppl = [], []

            schedule = scheduler(
                params.fn, params.num_epochs * num_iters, params.cycles,
                params.cycle_proportion, params.beta_lag, params.zero_start
            )

            data = np.array(data)
            encoder_data = np.array(encoder_data)
            val_data = np.array(val_data)
            encoder_val_data = np.array(encoder_val_data)
            labels = np.array(labels)
            encoder_labels = np.array(encoder_labels)
            val_labels = np.array(val_labels)
            encoder_val_labels = np.array(encoder_val_labels)

            prev_epoch_elbo = cur_epoch_avg_elbo = 0

            alpha_v = beta_v = -(1 * 0.5) / num_iters

            current_epoch = 0
            # while True:
            for e in range(params.num_epochs):
                current_epoch += 1
                epoch_start_time = datetime.datetime.now()
                print(
                    "Epoch: {} started at: {}".format(
                        current_epoch, epoch_start_time
                    )
                )

                # prev_epoch_elbo = cur_epoch_avg_elbo
                # cur_epoch_elbo_sum = 0

                ## alpha, beta schedule
                # if cur_it >= 8000:
                #     break
                rand_ids = np.random.permutation(len(data))
                for it in tqdm(range(num_iters)):
                    # if cur_it >= 8000:
                    #     break
                    cur_it += 1
                    # alpha_v, beta_v = schedule(cur_it)
                    # beta_v, alpha_v = schedule(cur_it)
                    # beta_v = 1

                    alpha_v += (1 * 0.5) / num_iters
                    alpha_v = min(1, alpha_v)
                    if cur_it > 0 and (-kzs < 0.8 or -kzg < 0.8):
                        alpha_v = 0

                    beta_v = alpha_v

                    params.is_training = True
                    start_idx = it * params.batch_size
                    end_idx = (it + 1) * params.batch_size
                    indices = rand_ids[start_idx:end_idx]

                    sent_inp_batch = encoder_data[indices]
                    label_inp_batch = encoder_labels[indices]
                    dec_word_inp_batch = data[indices]
                    dec_label_inp_batch = labels[indices]

                    # not optimal!!
                    word_length_ = np.array(
                        [len(sent) for sent in sent_inp_batch]
                    ).reshape(params.batch_size)
                    label_length_ = np.array(
                        [len(sent) for sent in label_inp_batch]
                    ).reshape(params.batch_size)

                    # prepare encoder and decoder inputs to feed
                    sent_inp_batch = zero_pad(sent_inp_batch, max_len_word)
                    label_inp_batch = zero_pad(label_inp_batch, max_len_label)
                    dec_word_inp_batch = zero_pad(
                        dec_word_inp_batch, max_len_word
                    )
                    dec_label_inp_batch = zero_pad(
                        dec_label_inp_batch, max_len_label
                    )

                    feed = {
                        word_inputs: sent_inp_batch,
                        label_inputs: label_inp_batch,
                        d_word_labels: sent_inp_batch,
                        d_label_labels: label_inp_batch,
                        d_seq_length_word: word_length_,
                        d_seq_length_label: label_length_,
                        d_word_inputs: dec_word_inp_batch,
                        d_label_inputs: dec_label_inp_batch,
                        alpha: alpha_v,
                        beta: beta_v
                    }

                    kzg, kzs, tlb, klw, o, rl, lrl, wrl = sess.run(
                        [
                            neg_kld_zglobal, neg_kld_zsent, total_lower_bound,
                            kl_term_weight, optimize, rec_loss, label_rec_loss,
                            word_rec_loss
                        ],
                        feed_dict=feed
                    )

                    all_alpha.append(alpha_v)
                    all_beta.append(beta_v)
                    all_tlb.append(tlb)
                    all_kl.append(klw)
                    all_klzg.append(-kzg)
                    all_klzs.append(-kzs)
                    all_rl.append(rl)
                    all_lrl.append(lrl)
                    all_wrl.append(wrl)
                    write_lists_to_file(
                        'test_plot.txt', all_alpha, all_beta, all_tlb, all_kl,
                        all_klzg, all_klzs, all_rl, all_lrl, all_wrl
                    )

                    # cur_epoch_elbo_sum += tlb

                    if cur_it % 100 == 0 and cur_it != 0:
                        path_to_save = os.path.join(
                            params.MODEL_DIR, "vae_lstm_model"
                        )
                        # print(path_to_save)
                        model_path_name = saver.save(
                            sess, path_to_save, global_step=cur_it
                        )
                        # print(model_path_name)

                # w_ppl, l_ppl = 0, 0
                # batch_size = params.batch_size
                # num_examples = 0
                # for p_it in tqdm(range(len(encoder_val_data) // batch_size)):
                #     s_idx = p_it * batch_size
                #     e_idx = (p_it + 1) * batch_size

                #     sent_inp_batch = encoder_val_data[s_idx:e_idx]
                #     label_inp_batch = encoder_val_labels[s_idx:e_idx]
                #     dec_word_inp_batch = val_data[s_idx:e_idx]
                #     dec_label_inp_batch = val_labels[s_idx:e_idx]
                #     word_length_ = np.array([len(sent) for sent in sent_inp_batch]
                #                        ).reshape(params.batch_size)
                #     label_length_ = np.array([len(sent) for sent in label_inp_batch]
                #                        ).reshape(params.batch_size)
                #     sent_inp_batch = zero_pad(sent_inp_batch, max_len_word)
                #     label_inp_batch = zero_pad(label_inp_batch, max_len_label)
                #     dec_word_inp_batch = zero_pad(dec_word_inp_batch, max_len_word)
                #     dec_label_inp_batch = zero_pad(dec_label_inp_batch, max_len_label)

                #     num_examples += batch_size

                #     _, _, wp, lp = sess.run(
                #         [
                #             total_lower_bound, kl_term_weight, word_perplexity,
                #             label_perplexity
                #         ],
                #         feed_dict={
                #             word_inputs: sent_inp_batch,
                #             label_inputs: label_inp_batch,
                #             d_word_labels: sent_inp_batch,
                #             d_label_labels: label_inp_batch,
                #             d_seq_length_word: word_length_,
                #             d_seq_length_label: label_length_,
                #             d_word_inputs: dec_word_inp_batch,
                #             d_label_inputs: dec_label_inp_batch,
                #             alpha: 1,
                #             beta: 1
                #         }
                #     )

                #     l_ppl += lp * batch_size
                #     w_ppl += wp * batch_size

                # l_ppl /= num_examples
                # w_ppl /= num_examples

                # all_l_ppl.append(l_ppl)
                # all_w_ppl.append(w_ppl)
                # write_lists_to_file('test_plot_ppl.txt', all_l_ppl, all_w_ppl)

                # cur_epoch_avg_elbo = float(cur_epoch_elbo_sum) / (num_iters)
                # print('\navg elbo:', cur_epoch_avg_elbo)

                # if current_epoch > 1:
                #     print('prev elbo:', prev_epoch_elbo)
                #     perct_change = float(prev_epoch_elbo - cur_epoch_avg_elbo)/prev_epoch_elbo
                #     print('change%:', perct_change)
                #     ## stopping condition
                #     if perct_change <= 0.01:
                #         break

                print("Time Taken:", datetime.datetime.now() - epoch_start_time)

            ## save model at end of training
            model_path_name = saver.save(sess, path_to_save, global_step=cur_it)


if __name__ == "__main__":
    params = parameters.Parameters()
    params.parse_args()
    main(params)
