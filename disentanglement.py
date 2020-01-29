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


def compute_disentanglement(z_list, y_list, y_dict, fl=1, M=1000):
    '''Metric introduced in Kim and Mnih (2018)'''
    N = len(z_list)  ## batch size OR num elements
    D = len(z_list[0])  ## latent size

    # Number of generic factors
    K = len(y_dict.keys())
    print('len(y_dict.keys()):', K)
    # N X F X D
    zs = np.array(z_list)
    zs_std = np.std(z_list, axis=0)
    #zs_std = np.reshape(np.std(z_list, axis = 1), [-1, 1])
    print('zs_std.shape:', zs_std.shape)

    zs_normalised = np.divide(zs, zs_std)
    #zs_variance =
    V = np.zeros(shape=[K, K])
    ks = np.random.randint(0, K, M)  # sample fixed-factor idxs ahead of time

    for m in range(M):
        k = ks[m]
        fk_vals = y_dict[k]  ##y_dict[k + 1]
        fk = fk_vals[np.random.choice(len(fk_vals))]
        #print(fk, k)

        # choose L random zs that have this fk at factor k
        indices = [
            i for i in range(N) if np.array_equal(np.array(y_list[i][k]), fk)
        ]

        if indices:
            zs_val = np.array([zs_normalised[i] for i in indices])
            # print("zs_val shape", zs_val.shape)
            zs_val_std = np.std(zs_val, axis=0)
            # print("zs_val_std:", zs_val_std.shape)

            temp_arr = []
            # for k_ in range(K):
            for k_ in range(K - 1):
                temp_arr.append(np.mean(zs_val_std[k_ * fl:(k_ + 1) * fl]))
                # temp_arr.append(np.mean(zs_val_std[k_ * fl:(k_ + 1) * fl]))
            d_star = np.argmin(temp_arr)
            V[d_star, k] += 1

    print(V)
    return (V.diagonal().sum() * 1.0) / V.sum()


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
            shape=[params.batch_size, params.num_topics, topic_vocab_size],
            name="topic_beta"
        )
        alpha = tf.placeholder(dtype=tf.float64)
        beta = tf.placeholder(dtype=tf.float64)

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

        enc_z_mu, enc_z_logvar, enc_z_sample, enc_topic_dist, enc_z_sample_all = model.encoder(
            enc_vect_inputs, params.batch_size, seq_length
        )

        dec_logits_out, _ = model.word_decoder_model(
            dec_inputs, enc_z_sample, params.batch_size, word_vocab_size,
            seq_length, word_embedding
        )

        doc_mu, doc_logvar, doc_sample = model.doc_encoder(doc_bow)

        topic_dist = model.doc_decoder(doc_sample)

        ## softmax over the topic vocab
        topic_beta_sm = tf.nn.softmax(topic_beta, axis=-1)
        dec_z_mu, dec_z_logvar, dec_z_sample = model.get_word_priors(
            topic_beta_sm, topic_dist
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
            #ptb_data = PTBInput(params.batch_size, train_data)
            num_iters = len(encoder_sentences) // params.batch_size
            val_num_iters = len(val_encoder_sentences) // params.batch_size
            cur_it = -1

            schedule = scheduler(
                params.fn, params.num_epochs * num_iters, params.cycles,
                params.cycle_proportion, params.beta_lag, params.zero_start
            )

            encoder_sentences = np.array(encoder_sentences)
            decoder_sentences = np.array(decoder_sentences)
            documents = np.array(documents)

            enc_zs, dec_zs = [], []
            num_sentences = 100
            for val_it in range(1 + (num_sentences // params.batch_size)):
                start_idx = val_it * params.batch_size
                end_idx = (val_it + 1) * params.batch_size

                enc_sent_batch = val_encoder_sentences[start_idx:end_idx]
                dec_sent_batch = val_decoder_sentences[start_idx:end_idx]
                doc_batch = val_documents[start_idx:end_idx]

                seq_length_ = np.array([len(sent) for sent in enc_sent_batch]
                                       ).reshape(params.batch_size)

                # prepare encoder and decoder inputs to feed
                enc_sent_batch = zero_pad(enc_sent_batch, max_len_word)
                dec_sent_batch = zero_pad(dec_sent_batch, max_len_word)

                feed = {
                    enc_inputs: enc_sent_batch,
                    dec_inputs: dec_sent_batch,
                    doc_bow: doc_batch,
                    seq_length: seq_length_,
                    alpha: 1,
                    beta: 1,
                }

                enc_z, dec_z, td = sess.run(
                    [enc_z_sample, dec_z_sample, topic_dist, opti],
                    feed_dict=feed
                )
                enc_zs.append(enc_z)
                dec_zs.append(dec_z)

            enc_zs = np.concatenate(enc_zs, axis=0)
            dec_zs = np.concatenate(dec_zs, axis=0)

            enc_zs = enc_zs[:num_sentences]
            dec_zs = dec_zs[:num_sentences]

            print('enc_zs:', enc_zs.shape)
            print('dec_zs:', dec_zs.shape)
            # exit()
            y_dict = {}
            for dim in range(dec_zs.shape[1]):
                y_dict[dim] = list(set(dec_zs[:, dim]))

            disentanle_score = compute_disentanglement(enc_zs, dec_zs, y_dict)
            print(disentanle_score)


if __name__ == "__main__":
    params = parameters.Parameters()
    params.parse_args()
    main(params)
