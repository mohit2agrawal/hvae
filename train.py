from __future__ import absolute_import, division, print_function

import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.util.nest import flatten

import utils.data as data_
import model
from utils import parameters
# from utils.schedules import scheduler

from tqdm import tqdm
import pickle
import traceback


def kld(p_mu, p_logvar, q_mu, q_logvar):
    """
    compute D_KL(p || q) of two Gaussians
    """
    return -0.5 * (
        1 + p_logvar - q_logvar - (tf.square(p_mu - q_mu) + tf.exp(p_logvar)) /
        tf.exp(tf.cast(q_logvar, tf.float64))
    )


def kl_simple(p, q):
    P = tf.distributions.Categorical(probs=p)
    Q = tf.distributions.Categorical(probs=q)
    return tf.distributions.kl_divergence(P, Q)


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
    data_folder = 'DATA/imdb_topic'
    encoder_sentences, decoder_sentences, documents, embed_arr, word2idx, idx2word = data_.get_data(
        data_folder, params.embed_size
    )

    max_len_word = max(map(len, encoder_sentences))
    print('max len word:', max_len_word)

    word_vocab_size = len(word2idx.keys())

    topic_beta_initial = (1 / word_vocab_size) * np.ones(
        [params.batch_size, params.num_topics, word_vocab_size]
    )

    with tf.Graph().as_default() as graph:

        enc_inputs = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name="enc_inputs"
        )
        dec_inputs = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name="dec_inputs"
        )
        doc_bow = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name="doc_bow"
        )
        topic_beta = tf.Variable(
            initial_value=topic_beta_initial,
            dtype=tf.float64,
            shape=[params.batch_size, params.num_topics, word_vocab_size],
            name="topic_beta"
        )

        with tf.device("/cpu:0"):
            # [data_dict.vocab_size, params.embed_size]
            word_embedding = tf.Variable(
                embed_arr,
                trainable=False,
                name="word_embedding",
                dtype=tf.float64
            )
            enc_vect_inputs = tf.nn.embedding_lookup(
                word_embedding, enc_inputs, name="enc_vect_inputs"
            )

        # seq_length = tf.placeholder_with_default([0.0], shape=[None])
        seq_length = tf.placeholder(shape=[None], dtype=tf.float64)

        enc_z_mu, enc_z_logvar, enc_z_sample, enc_topic_dist = model.encoder(
            enc_vect_inputs, params.batch_size, seq_length
        )

        dec_logits_out = model.word_decoder_model(
            dec_inputs, enc_z_sample, params.batch_size, word_vocab_size,
            seq_length, word_embedding
        )

        doc_mu, doc_logvar, doc_sample = model.doc_encoder(doc_bow)

        topic_beta_sm = tf.nn.softmax(topic_beta)
        topic_dist, decoded_doc = model.doc_decoder(doc_sample, topic_beta_sm)

        dec_z_mu, dec_z_logvar, dec_z_sample = model.get_word_priors(
            topic_beta_sm, topic_dist
        )

        doc_prior_mu = tf.cast(0, dtype=tf.float64)
        doc_prior_logvar = tf.cast(0, dtype=tf.float64)  ## var=1 => log(1) = 0

        #### LOSS COMPUTATION ####

        ## for word reconstruction loss
        word_cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=dec_logits_out, labels=enc_inputs
        )
        word_mask = tf.sign(enc_inputs)
        cross_entropy_loss_masked = word_cross_entropy_loss * word_mask
        word_loss_by_example = tf.reduce_sum(
            cross_entropy_loss_masked, axis=1
        ) / seq_length
        word_recons_loss = tf.reduce_mean(word_loss_by_example)

        ## KL for two GMMs
        kl_seq = 0
        for i in range(params.num_topics):
            kl_seq += tf.reduce_mean(
                enc_topic_dist[:, i] * tf.reduce_sum(
                    kld(
                        enc_z_mu[:, i, :], enc_z_logvar[:, i, :],
                        dec_z_mu[:, i, :], dec_z_logvar[:, i, :]
                    ),
                    axis=-1
                )
            )

        kl_seq += kl_simple(enc_topic_dist, topic_dist)  ## write the KL func

        loss_seq = word_recons_loss - kl_seq

        ## KL divergence loss on topic
        kl_topic = tf.reduce_mean(
            tf.reduce_sum(
                kld(doc_mu, doc_logvar, doc_prior_mu, doc_prior_logvar)
            )
        )

        ## TODO: doc reconstruction loss
        doc_recons_loss = 0

        loss_topic = doc_recons_loss - kl_topic

        total_lower_bound = loss_seq + loss_topic

        gradients = tf.gradients(total_lower_bound, tf.trainable_variables())
        clipped_grad, _ = tf.clip_by_global_norm(gradients, 5)
        opt = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        optimize = opt.apply_gradients(
            zip(clipped_grad, tf.trainable_variables())
        )

        saver = tf.train.Saver(max_to_keep=5)
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
            val_labels = np.array(val_labels)

            prev_epoch_elbo = cur_epoch_avg_elbo = 0

            alpha_v = beta_v = -(1 * 0.5) / num_iters
            kzs = -1  ## just to ignore "undef var" warning

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
                    alpha_v, beta_v = schedule(cur_it)
                    # beta_v, alpha_v = schedule(cur_it)
                    # beta_v = 1

                    # alpha_v += (1 * 0.5) / num_iters
                    # alpha_v = min(1, alpha_v)
                    # if cur_it > 0 and -kzs < 0.8:
                    #     alpha_v = 0

                    # beta_v = alpha_v

                    params.is_training = True
                    start_idx = it * params.batch_size
                    end_idx = (it + 1) * params.batch_size
                    indices = rand_ids[start_idx:end_idx]

                    sent_inp_batch = encoder_data[indices]
                    dec_word_inp_batch = data[indices]
                    label_inp_batch = labels[indices]

                    # not optimal!!
                    word_length_ = np.array(
                        [len(sent) for sent in sent_inp_batch]
                    ).reshape(params.batch_size)

                    # prepare encoder and decoder inputs to feed
                    sent_inp_batch = zero_pad(sent_inp_batch, max_len_word)
                    dec_word_inp_batch = zero_pad(
                        dec_word_inp_batch, max_len_word
                    )

                    feed = {
                        word_inputs: sent_inp_batch,
                        label_inputs: label_inp_batch,
                        d_seq_length_word: word_length_,
                        d_word_inputs: dec_word_inp_batch,
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

                    # if cur_it % 100 == 0 and cur_it != 0:
                    #     path_to_save = os.path.join(
                    #         params.MODEL_DIR, "vae_lstm_model"
                    #     )
                    #     # print(path_to_save)
                    #     model_path_name = saver.save(
                    #         sess, path_to_save, global_step=cur_it
                    #     )
                    #     # print(model_path_name)

                # w_ppl, l_ppl = 0, 0
                # batch_size = params.batch_size
                # num_examples = 0
                # for p_it in tqdm(range(len(encoder_val_data) // batch_size)):
                #     s_idx = p_it * batch_size
                #     e_idx = (p_it + 1) * batch_size

                #     sent_inp_batch = encoder_val_data[s_idx:e_idx]
                #     dec_word_inp_batch = val_data[s_idx:e_idx]
                #     label_inp_batch = val_labels[s_idx:e_idx]
                #     word_length_ = np.array(
                #         [len(sent) for sent in sent_inp_batch]
                #     ).reshape(params.batch_size)
                #     sent_inp_batch = zero_pad(sent_inp_batch, max_len_word)
                #     dec_word_inp_batch = zero_pad(
                #         dec_word_inp_batch, max_len_word
                #     )

                #     num_examples += batch_size

                #     wp, _ = sess.run(
                #         [word_perplexity, word_rec_loss],
                #         feed_dict={
                #             word_inputs: sent_inp_batch,
                #             label_inputs: label_inp_batch,
                #             d_seq_length_word: word_length_,
                #             d_word_inputs: dec_word_inp_batch,
                #             alpha: 1,
                #             beta: 1
                #         }
                #     )

                #     w_ppl += wp * batch_size

                # w_ppl /= num_examples

                # all_w_ppl.append(w_ppl)
                # write_lists_to_file('test_plot_ppl.txt', all_w_ppl)

                # cur_epoch_avg_elbo = float(cur_epoch_elbo_sum) / (num_iters)
                # print('\navg elbo:', cur_epoch_avg_elbo)

                # if current_epoch > 1:
                #     print('prev elbo:', prev_epoch_elbo)
                #     perct_change = float(prev_epoch_elbo - cur_epoch_avg_elbo)/prev_epoch_elbo
                #     print('change%:', perct_change)
                #     ## stopping condition
                #     if perct_change <= 0.01:
                #         break

                ## save model at end of epoch
                if current_epoch % 2 == 1:
                    path_to_save = os.path.join(
                        params.MODEL_DIR, "vae_lstm_model"
                    )
                    model_path_name = saver.save(
                        sess, path_to_save, global_step=cur_it
                    )
                print("Time Taken:", datetime.datetime.now() - epoch_start_time)

            ## save model at end of training
            model_path_name = saver.save(sess, path_to_save, global_step=cur_it)


if __name__ == "__main__":
    params = parameters.Parameters()
    params.parse_args()
    main(params)
