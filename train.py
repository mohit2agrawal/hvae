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
    encoder_sentences, decoder_sentences, documents, embed_arr, word2idx, idx2word, topic_word2idx = data_.get_data(
        data_folder, params.embed_size
    )

    max_len_word = max(map(len, encoder_sentences))
    print('max len word:', max_len_word)

    word_vocab_size = len(word2idx.keys())
    topic_vocab_size = len(topic_word2idx.keys())

    num_buckets = 41
    for i in range(len(documents)):
        documents[i] = np.clip(documents[i], 0, num_buckets - 1)

    topic_beta_initial = (1 / topic_vocab_size) * np.ones(
        [params.batch_size, params.num_topics, topic_vocab_size]
    )

    with tf.Graph().as_default() as graph:

        enc_inputs = tf.placeholder(
            dtype=tf.int32,
            shape=[params.batch_size, max_len_word],
            name="enc_inputs"
        )
        dec_inputs = tf.placeholder(
            dtype=tf.int32,
            shape=[params.batch_size, max_len_word],
            name="dec_inputs"
        )
        doc_bow = tf.placeholder(
            dtype=tf.int32,
            shape=[params.batch_size, topic_vocab_size],
            name="doc_bow"
        )
        topic_beta = tf.Variable(
            initial_value=topic_beta_initial,
            dtype=tf.float64,
            shape=[
                params.batch_size, params.num_topics, topic_vocab_size,
                num_buckets
            ],
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

        topic_dist = model.doc_decoder(doc_sample)

        ## sum over the buckets
        ## then softmax over the topic vocab
        topic_beta_sm = tf.nn.softmax(
            tf.reduce_sum(topic_beta, axis=-1), axis=-1
        )
        dec_z_mu, dec_z_logvar, dec_z_sample = model.get_word_priors(
            topic_beta_sm, topic_dist
        )

        doc_prior_mu = tf.cast(0, dtype=tf.float64)
        doc_prior_logvar = tf.cast(0, dtype=tf.float64)  ## var=1 => log(1) = 0

        #### LOSS COMPUTATION ####

        ## for seq reconstruction loss
        seq_cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=dec_logits_out, labels=enc_inputs
        )
        seq_mask = tf.cast(tf.sign(enc_inputs), dtype=tf.float64)
        cross_entropy_loss_masked = seq_cross_entropy_loss * seq_mask
        seq_loss_by_example = tf.reduce_sum(
            cross_entropy_loss_masked, axis=1
        ) / seq_length
        seq_recons_loss = tf.reduce_mean(seq_loss_by_example)

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

        loss_seq = seq_recons_loss - kl_seq

        ## KL divergence loss on topic
        kl_doc = tf.reduce_mean(
            tf.reduce_sum(
                kld(doc_mu, doc_logvar, doc_prior_mu, doc_prior_logvar)
            )
        )

        ## for doc reconstruction loss ##

        ## bs x 1 x T
        topic_dist_expanded = tf.expand_dims(topic_dist, axis=1)
        ## bs x T x (vocab_size x num_buckets)
        topic_word_probs_flat = tf.reshape(
            topic_beta, [params.batch_size, params.num_topics, -1]
        )
        ## weighted average over topics
        topic_word_probs_flat_avg = tf.squeeze(
            tf.matmul(topic_dist_expanded, topic_word_probs_flat)
        )
        ## batch_size x vocab_size x num_buckets
        topic_word_probs_avg = tf.reshape(
            topic_word_probs_flat_avg,
            [params.batch_size, topic_vocab_size, num_buckets]
        )
        ## batch_size x vocab_size x num_buckets
        doc_bow_onehot = tf.cast(
            tf.one_hot(doc_bow, depth=num_buckets), dtype=tf.float64
        )
        ## batch_size x vocab_size
        doc_recons_loss_by_word = tf.reduce_sum(
            topic_word_probs_avg * doc_bow_onehot, axis=-1
        )
        ## 0<x<1, log(x) < 0 , therefore, negative log
        doc_recons_loss_by_word_log = -tf.math.log(doc_recons_loss_by_word)
        doc_recons_loss = tf.reduce_mean(doc_recons_loss_by_word_log)

        loss_doc = doc_recons_loss - kl_doc

        total_loss = loss_seq + loss_doc

        gradients = tf.gradients(total_loss, tf.trainable_variables())
        clipped_grad, _ = tf.clip_by_global_norm(gradients, 5)
        opt = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        optimize = opt.apply_gradients(
            zip(clipped_grad, tf.trainable_variables())
        )

        saver = tf.train.Saver(max_to_keep=5)
        # config = tf.ConfigProto(device_count={'GPU': 0})
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(config=config) as sess:
            print("********** STARTING SESSION **********")

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

            sess.run(
                [
                    tf.global_variables_initializer(),
                    tf.local_variables_initializer()
                ]
            )
            print("********** VARIABLES INITIALIZED **********")

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

            # exit()
            if params.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            summary_writer = tf.summary.FileWriter(params.LOG_DIR, sess.graph)
            summary_writer.add_graph(sess.graph)
            #ptb_data = PTBInput(params.batch_size, train_data)
            num_iters = len(encoder_sentences) // params.batch_size
            cur_it = -1

            all_loss, all_kl, all_kl_seq, all_kl_doc = [], [], [], []
            all_rl, all_srl, all_drl = [], [], []
            # all_t_ppl, all_w_ppl = [], []

            # schedule = scheduler(
            #     params.fn, params.num_epochs * num_iters, params.cycles,
            #     params.cycle_proportion, params.beta_lag, params.zero_start
            # )

            encoder_sentences = np.array(encoder_sentences)
            decoder_sentences = np.array(decoder_sentences)
            documents = np.array(documents)

            prev_epoch_loss = cur_epoch_avg_loss = 0

            # alpha_v = beta_v = -(1 * 0.5) / num_iters
            # kzs = -1  ## just to ignore "undef var" warning

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

                # if cur_it >= 8000:
                #     break
                rand_ids = np.random.permutation(len(encoder_sentences))
                for it in tqdm(range(num_iters)):
                    # if cur_it >= 8000:
                    #     break
                    cur_it += 1
                    # alpha_v, beta_v = schedule(cur_it)

                    start_idx = it * params.batch_size
                    end_idx = (it + 1) * params.batch_size
                    indices = rand_ids[start_idx:end_idx]

                    enc_sent_batch = encoder_sentences[indices]
                    dec_sent_batch = decoder_sentences[indices]
                    doc_batch = documents[indices]

                    seq_length_ = np.array(
                        [len(sent) for sent in enc_sent_batch]
                    ).reshape(params.batch_size)

                    # prepare encoder and decoder inputs to feed
                    enc_sent_batch = zero_pad(enc_sent_batch, max_len_word)
                    dec_sent_batch = zero_pad(dec_sent_batch, max_len_word)

                    feed = {
                        enc_inputs: enc_sent_batch,
                        dec_inputs: dec_sent_batch,
                        doc_bow: doc_batch,
                        seq_length: seq_length_
                    }

                    loss, skl, dkl, srl, drl, _ = sess.run(
                        [
                            total_loss, kl_seq, kl_doc, seq_recons_loss,
                            doc_recons_loss, optimize
                        ],
                        feed_dict=feed
                    )

                    all_loss.append(loss)
                    all_kl.append(skl + dkl)
                    all_kl_seq.append(skl)
                    all_kl_doc.append(dkl)
                    all_rl.append(srl + drl)
                    all_srl.append(srl)
                    all_drl.append(drl)
                    # write_lists_to_file(
                    #     'test_plot.txt', all_loss, all_kl, all_kl_seq,
                    #     all_kl_doc, all_rl, all_srl, all_drl
                    # )

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

                write_lists_to_file(
                    'test_plot.txt', all_loss, all_kl, all_kl_seq, all_kl_doc,
                    all_rl, all_srl, all_drl
                )

                ## save model at end of epoch
                # if current_epoch % 2 == 1:
                path_to_save = os.path.join(params.MODEL_DIR, "topic_model")
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
