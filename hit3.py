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
from utils.schedules import scheduler

from tqdm import tqdm
import pickle
import traceback
from collections import Counter


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


def log_sum_exp(value, axis=None, keepdims=False):
    """Numerically stable implementation of the (torch) operation
    value.exp().sum(axis, keepdims).log()
    """
    if axis is not None:
        m = np.ndarray.max(value, axis=axis, keepdims=True)
        value0 = value - m
        if keepdims is False:
            m = np.squeeze(m, axis)
        return m + np.log(np.sum(np.exp(value0), axis=axis, keepdims=keepdims))
    else:
        m = np.ndarray.reduce_max(value)
        sum_exp = np.sum(np.exp(value - m))
        return m + np.log(sum_exp)


def calc_mi_q(mu, logvar, z_samples):
    mu_shape = mu.shape
    x_batch, nz = mu_shape[0], mu_shape[1]

    # [z_batch, 1, nz]
    z_samples = np.expand_dims(z_samples, 1)
    #print("z_samples:", z_samples.shape)
    #E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
    #(-0.5 * nz * math.log(2 * math.pi)- 0.5 * (1 + logvar).sum(-1)).mean()
    neg_entropy = np.mean(
        -0.5 * np.multiply(nz, np.log(2 * np.pi)) -
        0.5 * np.sum(1 + logvar, axis=-1)
    )

    # [1, x_batch, nz]
    mu, logvar = np.expand_dims(mu, 1), np.expand_dims(logvar, 1)
    var = np.exp(logvar)

    # (z_batch, x_batch, nz)
    dev = z_samples - mu
    # (z_batch, x_batch)
    log_density = -0.5 * np.sum(np.square(dev) / var, -1) - 0.5 * (
        np.multiply(nz, np.log(2 * np.pi), dtype=np.float64) +
        np.sum(logvar, -1)
    )

    # log q(z): aggregate posterior
    # [z_batch]
    log_qz = log_sum_exp(log_density, axis=1) - np.log(x_batch)
    return np.squeeze(neg_entropy - np.mean(log_qz, axis=-1))


def softmax(x, zero_probs=None):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:  ## for word logits passed in
        ## let zero logits be zero prob (softmax)
        x_nz = x[x != 0]
        sm = np.zeros_like(x)
        sm[x != 0] = np.exp(x_nz) / np.sum(np.exp(x_nz))
        return sm
    exp = np.exp(x)
    if zero_probs:
        for idx in zero_probs:
            exp[:, idx] = 0
    return exp / np.sum(exp, axis=1, keepdims=True)


def main(params):
    data_folder = 'DATA/' + params.name
    encoder_sentences, decoder_sentences, documents, embed_arr, word2idx, idx2word, topic_word2idx, topic_idx2word, topic_vocab = data_.get_data(
        data_folder, params.embed_size
    )

    max_len_word = max(map(len, encoder_sentences))
    print('max len word:', max_len_word)

    word_vocab_size = len(word2idx.keys())
    topic_vocab_size = len(topic_word2idx.keys())

    ## 80% - 20% train - valid split
    train_size = int(0.8 * len(encoder_sentences))
    val_encoder_sentences = encoder_sentences[train_size:]
    val_decoder_sentences = decoder_sentences[train_size:]
    val_documents = documents[train_size:]
    encoder_sentences = encoder_sentences[:train_size]
    decoder_sentences = decoder_sentences[:train_size]
    documents = documents[:train_size]

    # num_buckets = 41
    # for i in range(len(documents)):
    #     documents[i] = np.clip(documents[i], 0, num_buckets - 1)

    ## set batch_size as 1, will generate one sentence at a time
    batch_size = 1

    topic_beta_initial = (1 / topic_vocab_size) * np.ones(
        [params.batch_size, params.num_topics, topic_vocab_size]
    )

    with tf.Graph().as_default() as graph:

        # enc_inputs = tf.placeholder(
        #     dtype=tf.int32, shape=[batch_size, max_len_word], name="enc_inputs"
        # )
        dec_inputs = tf.placeholder(
            dtype=tf.int32, shape=[batch_size, None], name="dec_inputs"
        )
        doc_bow = tf.placeholder(
            dtype=tf.int32,
            shape=[batch_size, topic_vocab_size],
            name="doc_bow"
        )
        topic_beta = tf.Variable(
            initial_value=topic_beta_initial,
            dtype=tf.float64,
            shape=[params.batch_size, params.num_topics, topic_vocab_size],
            name="topic_beta"
        )
        topic_beta_input = tf.placeholder(
            dtype=tf.float64,
            shape=[batch_size, params.num_topics, topic_vocab_size],
            name="topic_beta_input"
        )
        alpha = tf.placeholder(dtype=tf.float64)
        beta = tf.placeholder(dtype=tf.float64)

        decoder_cell_state = tf.placeholder(
            dtype=tf.float64,
            shape=[batch_size, 2 * params.decoder_hidden],
            name="decoder_cell_state"
        )
        z_sample = tf.placeholder(
            dtype=tf.float64,
            shape=[batch_size, params.latent_size],
            name="z_sample"
        )
        topic_dist = tf.placeholder(
            dtype=tf.float64,
            shape=[batch_size, params.num_topics],
            name="topic_dist"
        )

        with tf.device("/cpu:0"):
            # [data_dict.vocab_size, params.embed_size]
            word_embedding = tf.Variable(
                embed_arr,
                trainable=False,
                name="word_embedding",
                dtype=tf.float64
            )
            # enc_vect_inputs = tf.nn.embedding_lookup(
            #     word_embedding, enc_inputs, name="enc_vect_inputs"
            # )

        # seq_length = tf.placeholder_with_default([0.0], shape=[None])
        seq_length = tf.placeholder(shape=[None], dtype=tf.float64)

        # enc_z_mu, enc_z_logvar, enc_z_sample, enc_topic_dist, enc_z_sample_all = model.encoder(
        #     enc_vect_inputs, batch_size, seq_length
        # )

        dec_logits_out, dec_cell_state = model.word_decoder_model(
            dec_inputs,
            z_sample,
            batch_size,
            word_vocab_size,
            seq_length,
            word_embedding,
            gen_mode=True,
            word_cell_state=decoder_cell_state
        )

        doc_mu, doc_logvar, doc_sample = model.doc_encoder(doc_bow)

        # topic_dist = model.doc_decoder(doc_sample)

        ## softmax over the topic vocab
        topic_beta_red = tf.reduce_mean(topic_beta, axis=0, keep_dims=True)
        topic_beta_sm = tf.nn.softmax(topic_beta_red, axis=-1)
        dec_z_mu, dec_z_logvar, dec_z_sample = model.get_word_priors(
            topic_beta_sm, topic_dist, batch_size
        )

        doc_prior_mu = tf.cast(0, dtype=tf.float64)
        doc_prior_logvar = tf.cast(0, dtype=tf.float64)  ## var=1 => log(1) = 0

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
            #ptb_data = PTBInput(batch_size, train_data)
            num_iters = len(encoder_sentences) // batch_size
            val_num_iters = len(val_encoder_sentences) // batch_size
            cur_it = -1

            encoder_sentences = np.array(encoder_sentences)
            decoder_sentences = np.array(decoder_sentences)
            documents = np.array(documents)

            bos_idx = word2idx['<BOS>']
            pad_idx = word2idx['<PAD>']
            eos_idx = word2idx['<EOS>']

            # topic_beta_ = sess.run(topic_beta)

            NUM_SENTS_PER_TOPIC = 1000
            TOP_N = int(0.01 * topic_vocab_size) + 1

            topic_scores = [0] * params.num_topics
            for topic_idx in range(params.num_topics):
                print('topic_idx: {}/{}'.format(topic_idx, params.num_topics))
                ## set topic distribution
                topic_dist_ = np.zeros((batch_size, params.num_topics))
                topic_dist_[0, topic_idx] = 1

                ## get mu and logvar for z, to be fed to decoder
                d_z_mu, d_z_logvar, topic_beta_ = sess.run(
                    [dec_z_mu, dec_z_logvar, topic_beta_red],
                    feed_dict={topic_dist: topic_dist_}
                )
                ## d_z_mu: batch_size (1) x num_topics x latent_size
                ## mu_: 1 x latent_size
                mu_ = np.matmul(topic_dist_, np.squeeze(d_z_mu))
                logvar_ = np.matmul(topic_dist_, np.squeeze(d_z_logvar))

                ## generate sentences
                generated_sentences = []
                with tqdm(total=NUM_SENTS_PER_TOPIC) as pbar:
                    while len(generated_sentences) < NUM_SENTS_PER_TOPIC:
                        eps = np.random.normal(size=np.shape(mu_))
                        z_ = mu_ + np.exp(0.5 * logvar_) * eps

                        dec_state = np.zeros(
                            (batch_size, 2 * params.decoder_hidden),
                            dtype=np.float
                        )
                        pred_word = '<BOS>'
                        pred_word_idx = bos_idx
                        pred_sentence = []

                        while True:
                            dec_logits, dec_state = sess.run(
                                [dec_logits_out, dec_cell_state],
                                feed_dict={
                                    z_sample: z_,
                                    decoder_cell_state: dec_state,
                                    dec_inputs: [[pred_word_idx]],
                                    seq_length: [1]
                                    # seq_length: [1 + len(pred_sentence)]
                                }
                            )

                            dec_logits = dec_logits[0][0]
                            dec_logits[bos_idx] = 0
                            dec_logits[pad_idx] = 0
                            if len(pred_sentence) == 0:
                                dec_logits[eos_idx] = 0
                            dec_sm = softmax(dec_logits)

                            pred_word_idx = np.argmax(dec_sm)
                            pred_word = idx2word[pred_word_idx]

                            if pred_word == '<EOS>':
                                break
                            pred_sentence.append(pred_word)
                            if (len(pred_sentence) > 2 * max_len_word):
                                break

                        ## end while True
                        # print(len(generated_sentences), ' '.join(pred_sentence))
                        if (len(pred_sentence) < 2 * max_len_word):
                            generated_sentences.append(pred_sentence)
                            pbar.update(1)
                    ## end while(len(generated_sentences)) < NUM_SENTS_PER_TOPIC

                ## topic_dist_: bs (1) x T
                ## beta: bs (1) x T x V
                ## => topic_word_dist: ( 1xT ) x ( TxV ) -> 1xV -> V
                topic_word_dist = np.squeeze(
                    np.matmul(topic_dist_, np.squeeze(topic_beta_))
                )
                top_topic_words_idx = np.argsort(topic_word_dist)[-TOP_N:]
                top_topic_words = set(
                    topic_idx2word[x] for x in top_topic_words_idx
                )

                generated_sentences = [set(x) for x in generated_sentences]
                ## HIT @ 3
                ## 1: if any of top 3 words from topic appear in sentence
                ## 0: else
                score = sum(
                    [
                        bool(top_topic_words.intersection(x))
                        for x in generated_sentences
                    ]
                )
                score = float(score) / len(generated_sentences)

                print('score:', score)
                topic_scores[topic_idx] = score
            print(topic_scores)
            print(float(sum(topic_scores)) / params.num_topics)


if __name__ == "__main__":
    params = parameters.Parameters()
    params.parse_args()
    main(params)
