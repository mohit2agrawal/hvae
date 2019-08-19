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


class PTBInput(object):
    """The input data."""
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name
        )


def rnn_placeholders(state):
    """Convert RNN state tensors to placeholders with the zero state as default."""
    if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
        c, h = state
        c = tf.placeholder_with_default(c, c.shape, c.op.name)
        h = tf.placeholder_with_default(h, h.shape, h.op.name)
        return tf.contrib.rnn.LSTMStateTuple(c, h)
    elif isinstance(state, tf.Tensor):
        h = state
        h = tf.placeholder_with_default(h, h.shape, h.op.name)
        return h
    else:
        structure = [rnn_placeholders(x) for x in state]
        return tuple(structure)


def kld(p_mu, p_logvar, q_mu, q_logvar):
    """
    compute D_KL(p || q) of two Gaussians
    """
    return -0.5 * (
        1 + p_logvar - q_logvar -
        (tf.square(p_mu - q_mu) + tf.exp(p_logvar)) / tf.exp(q_logvar)
    )


def max_likelihood_loss(word_logits, d_word_labels):
    w_probs = tf.nn.softmax(word_logits)
    d_word_labels_flat = tf.reshape(d_word_labels, [-1])
    w_mask_labels = tf.sign(tf.cast(d_word_labels_flat, dtype=tf.float64))
    w_probs_flat = tf.reshape(w_probs, [-1])
    w_index = tf.range(tf.shape(d_word_labels_flat)[0]
                       ) * tf.shape(w_probs)[1] + d_word_labels_flat
    w_index_probs = tf.gather(w_probs_flat, w_index)
    w_log = -tf.log(w_index_probs + 1e-8)
    w_masked_cost = w_log * w_mask_labels
    w_cost_1 = tf.reshape(w_masked_cost, tf.shape(word_inputs))
    w_cost = tf.reduce_sum(w_cost_1, axis=1)
    return w_cost


def main(params):
    if params.input_ == 'PTB':
        # data in form [data, labels]
        train_data_raw, train_label_raw = data_.ptb_read(
            './DATA/parallel_data_10k/'
        )
        word_data, encoder_word_data, word_labels_arr, word_embed_arr, word_data_dict = data_.prepare_data(
            train_data_raw, train_label_raw, params, './DATA/parallel_data_10k/'
        )

        train_label_raw, valid_label_raw, test_label_raw = label_data_.ptb_read(
            './DATA/parallel_data_10k/'
        )
        label_data, label_labels_arr, label_embed_arr, label_data_dict = label_data_.prepare_data(
            train_label_raw, params
        )

    with tf.Graph().as_default() as graph:

        label_inputs = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name="lable_inputs"
        )
        word_inputs = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name="word_inputs"
        )

        d_word_inputs = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name="d_word_inputs"
        )
        d_label_inputs = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name="d_label_inputs"
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
        seq_length = tf.placeholder_with_default([0.0], shape=[None])
        d_seq_length = tf.placeholder(shape=[None], dtype=tf.float64)
        # qz = q_net(word_inputs, seq_length, params.batch_size)

        Zsent_distribution, zsent_sample, Zglobal_distribition, zglobal_sample = encoder(
            vect_inputs, label_inputs_1, seq_length, params.batch_size
        )
        word_logits, label_logits, Zsent_dec_distribution, Zglobal_dec_distribution, _, _, _ = decoder(
            zglobal_sample, d_word_inputs, d_label_inputs, seq_length,
            params.batch_size, label_embedding, word_embedding, word_vocab_size,
            label_vocab_size
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
        label_perplexity = tf.exp(label_rec_loss)

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
        word_perplexity = tf.exp(word_rec_loss)

        rec_loss = word_rec_loss + label_rec_loss

        #anneal = tf.placeholder(tf.float64)
        # annealing=tf.to_float(anneal)
        #annealing = (tf.tanh((tf.to_float(anneal) - 5000)/1800) + 1)/2
        # overall loss reconstruction loss - kl_regularization
        #kld_loss = -1*(neg_kld_zglobal + neg_kld_zsent)
        #kl_term_weight = tf.multiply(
        #    tf.cast(annealing, dtype=tf.float64), tf.cast(kld_loss, dtype=tf.float64))

        alpha = tf.placeholder(tf.float64)
        alpha_val = tf.to_float(alpha)
        beta = tf.placeholder(tf.float64)
        beta_val = tf.to_float(beta)
        kl_term_weight = tf.multiply(tf.cast(alpha_val, dtype=tf.float64), tf.cast(neg_kld_zsent, dtype=tf.float64)) \
                         + tf.multiply(tf.cast(beta_val, dtype=tf.float64), tf.cast(neg_kld_zglobal, dtype=tf.float64))

        total_lower_bound = rec_loss - kl_term_weight

        gradients = tf.gradients(total_lower_bound, tf.trainable_variables())
        opt = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        clipped_grad, _ = tf.clip_by_global_norm(gradients, 5)
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

            # try:

            #     path="./models_ckpts_"+params.name+"/vae_lstm_model-11900"
            #     # print(path)
            #     # chkp.print_tensors_in_checkpoint_file(path, tensor_name='', all_tensors=True)
            #     saver.restore(sess,path )
            # # saver.restore(sess, "./models_ckpts_1/vae_lstm_model-258600")
            # except:
            #     print("-----exception occurred--------")
            #     exit()
            #     # traceback.print_exc()

            # print("Model Restored")

            total_parameters = 0
            #print_vars("trainable variables")
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                print(shape, variable.name)
                # print(len(shape))
                variable_parameters = 1
                for dim in shape:
                    print(dim)
                    variable_parameters *= dim.value
                print(variable_parameters, total_parameters)
                total_parameters += variable_parameters
            print(total_parameters)

            # exit()
            if params.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            summary_writer = tf.summary.FileWriter(params.LOG_DIR, sess.graph)
            summary_writer.add_graph(sess.graph)
            #ptb_data = PTBInput(params.batch_size, train_data)
            num_iters = len(word_data) // params.batch_size
            cur_it = 0
            iters, tlb_arr, wppl_arr, klw_arr, kld_zg_arr, kld_zs_arr = [], [], [], [], [], []
            alpha_arr, beta_arr = [], []
            for e in range(params.num_epochs):
                epoch_start_time = datetime.datetime.now()
                print("Epoch: {} started at: {}".format(e, epoch_start_time))
                total_tlb = 0
                total_wppl = 0
                total_klw = 0
                total_kld_zg = 0
                total_kld_zs = 0
                for it in tqdm(range(num_iters)):
                    params.is_training = True

                    sent_batch = word_data[it * params.batch_size:(it + 1) *
                                           params.batch_size]
                    label_batch = label_data[it * params.batch_size:(it + 1) *
                                             params.batch_size]

                    sent_dec_l_batch = word_labels_arr[it * params.batch_size:
                                                       (it + 1) *
                                                       params.batch_size]

                    sent_l_batch = encoder_word_data[it *
                                                     params.batch_size:(it +
                                                                        1) *
                                                     params.batch_size]
                    label_l_batch = label_labels_arr[it *
                                                     params.batch_size:(it +
                                                                        1) *
                                                     params.batch_size]

                    # zero padding
                    pad = len(max(sent_batch, key=len))
                    # not optimal!!
                    length_ = np.array([len(sent) for sent in sent_batch]
                                       ).reshape(params.batch_size)
                    # prepare encoder and decoder inputs to feed
                    sent_batch = np.array(
                        [sent + [0] * (pad - len(sent)) for sent in sent_batch]
                    )
                    label_batch = np.array(
                        [
                            sent + [0] * (pad - len(sent))
                            for sent in label_batch
                        ]
                    )

                    sent_dec_l_batch = np.array(
                        [
                            (sent + [0] * (pad - len(sent)))
                            for sent in sent_dec_l_batch
                        ]
                    )

                    sent_l_batch = np.array(
                        [
                            (sent + [0] * (pad - len(sent)))
                            for sent in sent_l_batch
                        ]
                    )
                    label_l_batch = np.array(
                        [
                            (sent + [0] * (pad - len(sent)))
                            for sent in label_l_batch
                        ]
                    )

                    alpha_v = beta_v = 0
                    if e > 2000:
                        alpha_v = beta_v = float(e) / 5000 - 0.4
                    feed = {
                        word_inputs: sent_l_batch,
                        label_inputs: label_l_batch,
                        d_word_inputs: sent_batch,
                        d_label_inputs: label_batch,
                        d_word_labels: sent_dec_l_batch,
                        d_label_labels: label_l_batch,
                        seq_length: length_,
                        d_seq_length: length_,
                        alpha: alpha_v,
                        beta: beta_v
                    }

                    z1a, z1b, z3a, z3b, kzg, kzs, tlb, wppl, lppl, klw, o, alpha_, beta_ = sess.run(
                        [
                            Zsent_distribution[0], Zsent_distribution[1],
                            Zsent_dec_distribution[0],
                            Zsent_dec_distribution[1], neg_kld_zglobal,
                            neg_kld_zsent, total_lower_bound, word_perplexity,
                            label_perplexity, kl_term_weight, optimize,
                            alpha_val, beta_val
                        ],
                        feed_dict=feed
                    )

                    # print(c.shape)
                    # print(d.shape)
                    # print(c,d)
                    # print(e,f)
                    # print(d[69],d[119])
                    ##print('zsent', z1a, z1b)
                    # print('zglobal',z2a,z2b)
                    ##print('zsent dec', z3a, z3b)
                    # if cur_it % 100 == 0 and cur_it != 0:
                    #     print("TotalLB after {} ({}) iterations (epoch): {} Neg_KLD_Zglobal: "
                    #           "{} Neg_KLD_Zsent: {}".format(
                    #               cur_it, e, tlb, kzg, kzs))
                    #     print(
                    #         "Word Perplexity: {}, Label Perplexity: {}".format(wppl, lppl))

                    cur_it += 1
                    # iters.append(cur_it)
                    # tlb_arr.append(tlb)
                    # wppl_arr.append(wppl)
                    total_tlb += tlb
                    total_wppl += wppl
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
                avg_wppl = total_wppl / num_iters
                avg_klw = total_klw / num_iters
                avg_kld_zg = total_kld_zg / num_iters
                avg_kld_zs = total_kld_zs / num_iters

                iters.append(e)
                tlb_arr.append(avg_tlb)
                wppl_arr.append(avg_wppl)
                klw_arr.append(avg_klw)
                kld_zg_arr.append(avg_kld_zg)
                kld_zs_arr.append(avg_kld_zs)
                alpha_arr.append(alpha_)
                beta_arr.append(beta_)

                print("Time Taken:", datetime.datetime.now() - epoch_start_time)

            import matplotlib as mpl
            mpl.use('Agg')
            import matplotlib.pyplot as plt
            plot_filename = "./plot_values_" + str(params.num_epochs) + ".txt"
            with open(plot_filename, 'w') as wf:
                _ = [wf.write(str(s) + ' ') for s in iters]
                wf.write('\n')
                _ = [wf.write(str(s) + ' ') for s in tlb_arr]
                # wf.write('\n')
                # _ = [wf.write(str(s) + ' ') for s in wppl_arr]
                wf.write('\n')
                _ = [wf.write(str(s) + ' ') for s in klw_arr]
                wf.write('\n')
                _ = [wf.write(str(s) + ' ') for s in kld_zg_arr]
                wf.write('\n')
                _ = [wf.write(str(s) + ' ') for s in kld_zs_arr]
                wf.write('\n')
                _ = [wf.write(str(s) + ' ') for s in alpha_arr]
                wf.write('\n')
                _ = [wf.write(str(s) + ' ') for s in beta_arr]

            plt.plot(iters, tlb_arr, color='blue', label='Total lower bound')
            plt.title("Total Lower Bound vs Epochs")
            plt.xlabel('Epochs')
            plt.ylabel('Total Lower Bound')
            plt.savefig(
                './graph_elbo_epochs_' + str(params.num_epochs) + '.png'
            )
            plt.clf()

            # plt.plot(iters, wppl_arr, color='red', label='Word Perplexity')
            # plt.title("Word Perplexity vs Epochs")
            # plt.xlabel('Epochs')
            # plt.ylabel('Word Perplexity')
            # plt.savefig('./graph_ppl_epochs_'+str(params.num_epochs)+'.png')

            # plt.legend(bbox_to_anchor=(1.05, 1),
            #           loc=1, borderaxespad=0.)
            # plt.clf()

            plt.plot(iters, klw_arr, color='green', label='KL term Value')
            plt.title("KL Term Value vs Epochs")
            plt.xlabel('Epochs')
            plt.ylabel('KL term Value')
            plt.savefig('./graph_klw_epochs_' + str(params.num_epochs) + '.png')
            plt.clf()

            plt.plot(iters, kld_zg_arr, color='yellow', label='kld_zg')
            plt.title("KLD zg")
            plt.xlabel('Epochs')
            plt.ylabel('KLD zg term Value')
            plt.savefig(
                './graph_kld_zg_epochs_' + str(params.num_epochs) + '.png'
            )
            plt.clf()

            plt.plot(iters, kld_zs_arr, color='black', label='kld_zs')
            plt.title("KLD zs")
            plt.xlabel('Epochs')
            plt.ylabel('KLD zs term Value')
            plt.savefig(
                './graph_kld_zs_epochs_' + str(params.num_epochs) + '.png'
            )
            plt.clf()

            plt.plot(iters, alpha_arr, color='blue', label='alpha')
            plt.title("Alpha")
            plt.xlabel('Epochs')
            plt.ylabel('Alpha Value')
            plt.savefig(
                './graph_alpha_epochs_' + str(params.num_epochs) + '.png'
            )
            plt.clf()

            # figure_name = './graph_epochs_'+str(params.num_epochs)+'.png'
            # # plt.plot(iters, coeff, 'r--', label='annealing')
            # plt.savefig(figure_name)


if __name__ == "__main__":
    params = parameters.Parameters()
    params.parse_args()
    main(params)
