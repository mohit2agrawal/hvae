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
from hvae_model_ours import decoder, encoder
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
        self.input_data, self.targets = reader.ptb_producer(data,
                                                            batch_size,
                                                            num_steps,
                                                            name=name)

def kld(p_mu, p_logvar, q_mu, q_logvar):
    """
    compute D_KL(p || q) of two Gaussians
    """
    return -0.5 * (
        1 + p_logvar - q_logvar -
        (tf.square(p_mu - q_mu) + tf.exp(p_logvar)) / tf.exp(q_logvar))


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
        m = tf.reduce_max(value, axis=axis, keepdims=True)
        value0 = value - m
        if keepdims is False:
            m = tf.squeeze(m, axis)
        return m + tf.log(
            tf.reduce_sum(tf.exp(value0), axis=axis, keepdims=keepdims))
    else:
        m = tf.reduce_max(value)
        sum_exp = tf.reduce_sum(tf.exp(value - m))
        return m + tf.log(sum_exp)

def calc_mi_two(mu_x, logvar_x, mu_y, logvar_y, z_x, z_y):

    # MI(z_x, z_y) = H(z_y) - H(z_y | z_x)
    # H(z_y | z_x) = p(z_x) * p(z_y| z_x) * log(p(z_y| z_x)) = -E_{x,y} log(p(y|x))
    # H(z_y) = 0.5 * (1 + logvar + 2 * pi)
    mu_shape = tf.shape(mu_x)
    x_batch, nz = mu_shape[0], mu_shape[1]
    y_batch, nz = mu_shape[0], mu_shape[1]
    
    entropy_y = tf.reduce_mean(
        -0.5 *
        tf.cast(tf.multiply(tf.cast(nz, dtype=tf.float64),
                            tf.cast(tf.log(2 * np.pi), dtype=tf.float64)),
                dtype=tf.float64) -
        0.5 * tf.cast(tf.reduce_sum(1 + logvar_y, -1), dtype=tf.float64))


    mu_y, logvar_y = tf.expand_dims(mu_y, 0), tf.expand_dims(logvar_y, 0)
    mu_x, logvar_x = tf.expand_dims(mu_x, 0), tf.expand_dims(logvar_yx, 0)

    var_y = tf.exp(logvar_y)
    var_x = tf.exp(logvar_x)
    # (z_batch, x_batch, nz)
    dev = z_y - mu_y
    dev_x = z_x - mu_x

    density_x =  tf.exp(-0.5 * tf.reduce_sum(tf.square(dev_x) / var_x, -1) - 0.5 * (
        tf.multiply(tf.cast(nz, dtype=tf.float64),
                    tf.cast(tf.log(2 * np.pi), dtype=tf.float64))) + tf.cast(
                        tf.reduce_sum(logvar_x, -1), dtype=tf.float64))


    log_density = -0.5 * tf.reduce_sum(tf.square(dev) / var_y, -1) - 0.5 * (
        tf.multiply(tf.cast(nz, dtype=tf.float64),
                    tf.cast(tf.log(2 * np.pi), dtype=tf.float64))) + tf.cast(
                        tf.reduce_sum(logvar_y, -1), dtype=tf.float64)
    density_y = tf.exp(log_density)
    entropy_y_given_x = -log_density * dentisy_x * density_y

    return tf.squeeze(entropy_y - entropy_y_given_x)

def calc_mi_q(mu, logvar, z_samples):

    # mu, logvar = Zsent_distribution
    mu_shape = tf.shape(mu)
    x_batch, nz = mu_shape[0], mu_shape[1]

    # [z_batch, 1, nz]
    z_samples = tf.expand_dims(z_samples, 1)

    # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
    neg_entropy = tf.reduce_mean(
        -0.5 *
        tf.cast(tf.multiply(tf.cast(nz, dtype=tf.float64),
                            tf.cast(tf.log(2 * np.pi), dtype=tf.float64)),
                dtype=tf.float64) -
        0.5 * tf.cast(tf.reduce_sum(1 + logvar, -1), dtype=tf.float64))

    # [1, x_batch, nz]
    mu, logvar = tf.expand_dims(mu, 0), tf.expand_dims(logvar, 0)
    var = tf.exp(logvar)

    # (z_batch, x_batch, nz)
    dev = z_samples - mu

    # (z_batch, x_batch)
    log_density = -0.5 * tf.reduce_sum(tf.square(dev) / var, -1) - 0.5 * (
        tf.multiply(tf.cast(nz, dtype=tf.float64),
                    tf.cast(tf.log(2 * np.pi), dtype=tf.float64))) + tf.cast(
                        tf.reduce_sum(logvar, -1), dtype=tf.float64)

    # log q(z): aggregate posterior
    # [z_batch]
    log_qz = tf.cast(log_sum_exp(log_density, axis=1),
                     dtype=tf.float64) - tf.log(
                         tf.cast(x_batch, dtype=tf.float64))

    return tf.squeeze(neg_entropy - tf.reduce_mean(log_qz, -1))


def main(params):
    # data_folder = './DATA/parallel_data_10k/'
    data_folder = {
        'ptb': './DATA/ptb/',
        'ptb_ner': './DATA/ptb_ner',
        'ptb_pos': './DATA/ptb_pos'
    }.get(params.name)

    # data in form [data, labels]
    
    #train_data_raw, train_label_raw = data_.ptb_read(data_folder)
    train_data_raw, train_label_raw, val_data_raw, val_label_raw  = data_.ptb_read(data_folder)
    #word_data, encoder_word_data, word_labels_arr, word_embed_arr, data_dict, label_data, label_labels_arr, label_embed_arr, encoder_val_data, val_labels_arr = data_.prepare_data(
    #    train_data_raw, train_label_raw, val_data_raw, val_label_raw, params, data_folder
    #)
    word_data, encoder_word_data, word_labels_arr, word_embed_arr, word_data_dict, label_data, label_labels_arr, label_embed_arr, encoder_val_data, val_labels_arr, decoder_words, decoder_labels = \
    data_.prepare_data(train_data_raw, train_label_raw, val_data_raw, val_label_raw, params, data_folder)

    max_sent_len = max(max(map(len, word_data)), max(map(len, encoder_word_data)))
    max_sent_len = max(max_sent_len, max(map(len, encoder_val_data)))

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
                    "word_embedding",
                    [data_dict.vocab_size, params.embed_size],
                    dtype=tf.float64)
                vect_inputs = tf.nn.embedding_lookup(word_embedding,
                                                     word_inputs)
            else:
                # [data_dict.vocab_size, params.embed_size]
                word_embedding = tf.Variable(
                    word_embed_arr,
                    trainable=params.fine_tune_embed,
                    name="word_embedding",
                    dtype=tf.float64
                )  # creates a variable that can be used as a tensor
                vect_inputs = tf.nn.embedding_lookup(word_embedding,
                                                     word_inputs,
                                                     name="word_lookup")

                # val_vect_inputs = tf.nn.embedding_lookup(
                #     word_embedding, val_word_inputs, name="word_lookup"
                # )

                label_embedding = tf.Variable(
                    label_embed_arr,
                    trainable=params.fine_tune_embed,
                    name="label_embedding",
                    dtype=tf.float64
                )  # creates a variable that can be used as a tensor

                label_inputs_1 = tf.nn.embedding_lookup(label_embedding,
                                                        label_inputs,
                                                        name="label_lookup")
                # val_label_inputs_1 = tf.nn.embedding_lookup(
                #     label_embedding, val_label_inputs, name="label_lookup"
                # )

        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        sizes = word_data_dict.sizes
        word_vocab_size = max(sizes)
        label_vocab_size = word_data_dict.label_vocab_size
        #label_vocab_size = label_data_dict.label_vocab_size
        # seq_length = tf.placeholder_with_default([0.0], shape=[None])
        d_seq_length = tf.placeholder(shape=[None], dtype=tf.float64)
        # qz = q_net(word_inputs, seq_length, params.batch_size)

        Zsent_distribution, zsent_sample, Zglobal_distribition, zglobal_sample, zsent_state, zglobal_state = encoder(
            vect_inputs, label_inputs_1, params.batch_size, max_sent_len)
        word_logits, label_logits, Zsent_dec_distribution, Zglobal_dec_distribution, _, _, _, dec_word_states, dec_label_states = decoder(
            d_word_inputs, d_label_inputs, zglobal_sample, params.batch_size,
            word_vocab_size, label_vocab_size, max_sent_len, word_embedding,
            label_embedding)
        #decoder(zglobal_sample, params.batch_size, word_vocab_size,label_vocab_size, max_sent_len)

        neg_kld_zsent = -1 * tf.reduce_mean(
            tf.reduce_sum(
                kld(Zsent_distribution[0], Zsent_distribution[1],
                    Zsent_dec_distribution[0], Zsent_dec_distribution[1]),
                axis=1))
        # neg_kld_zsent = tf.placeholder(tf.float64)
        neg_kld_zsent = tf.clip_by_value(neg_kld_zsent, -20000, 20000)
        neg_kld_zglobal = -1 * tf.reduce_mean(
            tf.reduce_sum(
                kld(Zglobal_distribition[0], Zglobal_distribition[1],
                    Zglobal_dec_distribution[0], Zglobal_dec_distribution[1]),
                axis=1))

        # label reconstruction loss
        d_label_labels_flat = tf.reshape(d_label_labels, [-1])
        l_cross_entr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=label_logits, labels=d_label_labels_flat)
        l_mask_labels = tf.sign(tf.cast(d_label_labels_flat, dtype=tf.float64))
        l_masked_losses = l_mask_labels * l_cross_entr
        # reshape again
        l_masked_losses = tf.reshape(l_masked_losses, tf.shape(d_label_labels))
        l_mean_loss_by_example = tf.reduce_sum(
            l_masked_losses, reduction_indices=1) / d_seq_length
        label_rec_loss = tf.reduce_mean(l_mean_loss_by_example)
        # label_perplexity = tf.exp(label_rec_loss)

        # Word reconstruction loss
        # print(word_logits.shape)

        d_word_labels_flat = tf.reshape(d_word_labels, [-1])
        print(d_word_labels_flat.shape)
        w_cross_entr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=word_logits, labels=d_word_labels_flat)
        w_mask_labels = tf.sign(tf.cast(d_word_labels_flat, dtype=tf.float64))
        w_masked_losses_1 = w_mask_labels * w_cross_entr
        w_masked_losses = tf.reshape(w_masked_losses_1,
                                     tf.shape(d_word_labels))
        w_mean_loss_by_example = tf.reduce_sum(
            w_masked_losses, reduction_indices=1) / d_seq_length
        word_rec_loss = tf.reduce_mean(w_mean_loss_by_example)
        # word_perplexity = tf.exp(word_rec_loss)

        rec_loss = word_rec_loss + label_rec_loss

        alpha = tf.placeholder(tf.float64)
        # alpha_val = tf.to_float(alpha)
        beta = tf.placeholder(tf.float64)
        # beta_val = tf.to_float(beta)
        kl_term_weight = - tf.multiply(tf.cast(alpha, dtype=tf.float64), tf.cast(neg_kld_zsent, dtype=tf.float64)) \
                         - tf.multiply(tf.cast(beta, dtype=tf.float64), tf.cast(neg_kld_zglobal, dtype=tf.float64))

        total_lower_bound = rec_loss + kl_term_weight



        opt = tf.train.AdamOptimizer(learning_rate=params.learning_rate,
                                     name='adam')
        
        # Gradient all 
        gradients = tf.gradients(total_lower_bound, tf.trainable_variables(), name='gradients')
        clipped_grad, _ = tf.clip_by_global_norm(gradients, 5, name='clipped')
        optimize = opt.apply_gradients(
            zip(clipped_grad, tf.trainable_variables()))
        
        

        # Gradient decoder
        decoder_vars = tf.trainable_variables('decoder')
        gradients_decoder = tf.gradients(
             total_lower_bound, decoder_vars
         )
        clipped_grad_decoder, _ = tf.clip_by_global_norm(gradients_decoder, 5)
        optimize_decoder = opt.apply_gradients(
             zip(clipped_grad_decoder, decoder_vars)
        )


        # Gradient encoder word
        encoder_word_vars = tf.trainable_variables('encoder/word')
        gradients_word = tf.gradients(total_lower_bound, encoder_word_vars,
                                         name='gradients_encoder_word')
        clipped_grad_word, _ = tf.clip_by_global_norm(
            gradients_encoder, 5, name='clipped_encoder_word')
        optimize_encoder_word = opt.apply_gradients(
            zip(clipped_grad_word, encoder_word_vars))


        # Gradienr encoder label
        encoder_label_vars = tf.trainable_variables('encoder/label')
        gradients_label = tf.gradients(total_lower_bound, encoder_label_vars,
                                         name='gradients_encoder_word')
        clipped_grad_label, _ = tf.clip_by_global_norm(
            gradients_encoder, 5, name='clipped_encoder_word')
        optimize_encoder_label = opt.apply_gradients(
            zip(clipped_grad_label, encoder_label_vars))



        

        mi_mu = tf.placeholder(shape=[params.batch_size, params.latent_size],
                               dtype=tf.float64,
                               name="mi_mu")
        mi_logvar = tf.placeholder(
            shape=[params.batch_size, params.latent_size],
            dtype=tf.float64,
            name="mi_logvar")
        mi_samples = tf.placeholder(
            shape=[params.batch_size, params.latent_size],
            dtype=tf.float64,
            name="mi_samples")

        mutual_info = calc_mi_q(mi_mu, mi_logvar, mi_samples)
        #mutual_info_new = calc_mi_q(mi_samples_1)

       

        saver = tf.train.Saver(max_to_keep=10)
        #config = tf.ConfigProto(device_count={'GPU': 0})
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)

        with tf.Session(config=config) as sess:
            print("*********")
            sess.run([
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            ])


            total_parameters = 0
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

            if params.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            summary_writer = tf.summary.FileWriter(params.LOG_DIR, sess.graph)
            summary_writer.add_graph(sess.graph)
            #ptb_data = PTBInput(params.batch_size, train_data)
            num_iters = len(word_data) // params.batch_size
            cur_it = 0

            all_alpha, all_beta, all_tlb, all_kl, all_klzg, all_klzs = [], [], [], [], [], []

            smi, gmi, tmi = [],[],[]

            word_data = np.array(word_data)
            label_data = np.array(label_data)
            word_labels_arr = np.array(word_labels_arr)
            encoder_word_data = np.array(encoder_word_data)
            label_labels_arr = np.array(label_labels_arr)
            decoder_words = np.array(decoder_words)
            decoder_labels = np.array(decoder_labels)
            sub_iter = 0  ## for aggressive encoder optim
            aggressive = True
            pre_mi = 0

            # zero padding
            pad = max_sent_len

            alpha_v = beta_v = 1
            for e in range(params.num_epochs):
                epoch_start_time = datetime.datetime.now()
                params.is_training = True
                print("Epoch: {} started at: {}".format(e, epoch_start_time))

                for it in tqdm(range(num_iters)):
                    sub_iter = 0
                    burn_num_words = 0
                    burn_pre_loss = 1e4
                    burn_cur_loss = 0
                    rand_ids = np.random.permutation(len(word_data))
                    print('aggressive:', aggressive)
                    if aggressive:
                        while sub_iter < 1000:
                            sub_iter += 1
                            print('Sub iter:', sub_iter)
                            # print('sub_iter:', sub_iter)

                            start_idx = sub_iter * params.batch_size
                            end_idx = (sub_iter + 1) * params.batch_size
                            indices = rand_ids[start_idx:end_idx]

                            sent_batch = word_data[indices]
                            label_batch = label_data[indices]
                            sent_dec_l_batch = word_labels_arr[indices]
                            sent_l_batch = encoder_word_data[indices]
                            label_l_batch = label_labels_arr[indices]
                            dec_word_inp_batch = decoder_words[indices]
                            dec_label_inp_batch = decoder_labels[indices]
                            ## burn_batch_size, burn_sents_len = sent_l_batch.shape
                            ## TODO the burn_sents_len will be different for each idx?
                            burn_batch_size = len(sent_l_batch)
                            burn_sents_len = len(sent_l_batch[0])

                            # not optimal!!
                            length_ = np.array([len(sent) for sent in sent_batch
                                                ]).reshape(params.batch_size)

                            # prepare encoder and decoder inputs to feed
                            sent_batch = zero_pad(sent_batch, pad)
                            label_batch = zero_pad(label_batch, pad)
                            sent_dec_l_batch = zero_pad(sent_dec_l_batch, pad)
                            sent_l_batch = zero_pad(sent_l_batch, pad)
                            label_l_batch = zero_pad(label_l_batch, pad)
                            dec_word_inp_batch = zero_pad(dec_word_inp_batch, pad)
                            dec_label_inp_batch = zero_pad(dec_label_inp_batch, pad)

                            feed = {
                                word_inputs: sent_l_batch,
                                label_inputs: label_l_batch,
                                d_word_labels: sent_dec_l_batch,
                                d_label_labels: label_l_batch,
                                d_seq_length: length_,
                                d_word_inputs: dec_word_inp_batch,
                                d_label_inputs: dec_label_inp_batch,
                                alpha: alpha_v,
                                beta: beta_v
                            }

                            ## aggressively optimize encoder
                            loss, _ = sess.run(
                                [total_lower_bound, optimize_encoder],
                                feed_dict=feed)

                            if sub_iter % 15 == 0:
                                ## TODO why -1?
                                burn_num_words += (burn_sents_len -
                                                    1) * burn_batch_size
                                burn_cur_loss += loss
                                burn_cur_loss = burn_cur_loss / burn_num_words
                                ## stop encoder only updates
                                ## do one full VAE update
                            
                                if burn_pre_loss < burn_cur_loss:
                                    break
                                burn_pre_loss = burn_cur_loss
                                burn_cur_loss = burn_num_words = 0

                        # Decoder update
                        start_idx = it * params.batch_size
                        end_idx = (it + 1) * params.batch_size
                        indices = rand_ids[start_idx:end_idx]

                        sent_batch = word_data[indices]
                        label_batch = label_data[indices]
                        sent_dec_l_batch = word_labels_arr[indices]
                        sent_l_batch = encoder_word_data[indices]
                        label_l_batch = label_labels_arr[indices]
                        dec_word_inp_batch = decoder_words[indices]
                        dec_label_inp_batch = decoder_labels[indices]
                        length_ = np.array([len(sent) for sent in sent_batch
                                                ]).reshape(params.batch_size)
                        #print("Debug size", sent_l_batch.shape, sent_dec_l_batch.shape, length_.shape)
                        sent_batch = zero_pad(sent_batch, pad)
                        label_batch = zero_pad(label_batch, pad)
                        sent_dec_l_batch = zero_pad(sent_dec_l_batch, pad)
                        sent_l_batch = zero_pad(sent_l_batch, pad)
                        label_l_batch = zero_pad(label_l_batch, pad)
                        dec_word_inp_batch = zero_pad(dec_word_inp_batch, pad)
                        dec_label_inp_batch = zero_pad(dec_label_inp_batch, pad)
                        
                        feed = {
                            word_inputs: sent_l_batch,
                            label_inputs: label_l_batch,
                            d_word_labels: sent_dec_l_batch,
                            d_label_labels: label_l_batch,
                            d_seq_length: length_,
                            d_word_inputs: dec_word_inp_batch,
                            d_label_inputs: dec_label_inp_batch,
                            alpha: alpha_v,
                            beta: beta_v
                        }

                        ## aggressively optimize encoder
                        loss, _ = sess.run(
                            [total_lower_bound, optimize_decoder],
                            feed_dict=feed)
                    else:
                    ## standard VAE updates

                        start_idx = it * params.batch_size
                        end_idx = (it + 1) * params.batch_size

                        sent_batch = word_data[start_idx:end_idx]
                        label_batch = label_data[start_idx:end_idx]
                        sent_dec_l_batch = word_labels_arr[start_idx:end_idx]
                        sent_l_batch = encoder_word_data[start_idx:end_idx]
                        label_l_batch = label_labels_arr[start_idx:end_idx]
                        dec_word_inp_batch = decoder_words[indices]
                        dec_label_inp_batch = decoder_labels[indices]

                        # not optimal!!
                        length_ = np.array([len(sent) for sent in sent_batch
                                            ]).reshape(params.batch_size)

                        sent_batch = zero_pad(sent_batch, pad)
                        label_batch = zero_pad(label_batch, pad)
                        sent_dec_l_batch = zero_pad(sent_dec_l_batch, pad)
                        sent_l_batch = zero_pad(sent_l_batch, pad)
                        label_l_batch = zero_pad(label_l_batch, pad)
                        dec_word_inp_batch = zero_pad(dec_word_inp_batch, pad)
                        dec_label_inp_batch = zero_pad(dec_label_inp_batch, pad)
                        
                        feed = {
                            word_inputs: sent_l_batch,
                            label_inputs: label_l_batch,
                            d_word_labels: sent_dec_l_batch,
                            d_label_labels: label_l_batch,
                            d_seq_length: length_,
                            d_word_inputs: dec_word_inp_batch,
                            d_label_inputs: dec_label_inp_batch,
                            alpha: alpha_v,
                            beta: beta_v
                        }

                        ## both decoder and encoder updates
                        z1a, z1b, z3a, z3b, kzg, kzs, tlb, klw, _, alpha_, beta_ = sess.run(
                            [
                                Zsent_distribution[0], Zsent_distribution[1],
                                Zsent_dec_distribution[0],
                                Zsent_dec_distribution[1], neg_kld_zglobal,
                                neg_kld_zsent, total_lower_bound,
                                kl_term_weight, optimize, alpha, beta
                            ],
                            feed_dict=feed)

                        all_alpha.append(alpha_v)
                        all_beta.append(beta_v)
                        all_tlb.append(tlb)
                        all_kl.append(klw)
                        all_klzg.append(-kzg)
                        all_klzs.append(-kzs)
                        write_lists_to_file('test_plot.txt', all_alpha,
                                        all_beta, all_tlb, all_kl,
                                        all_klzg, all_klzs)

                    ## for MI
                    sent_mi = 0
                    global_mi = 0
                    num_examples = 0

                    ## weighted average on calc_mi_q
                    val_len = len(encoder_val_data)
                    for val_it in range(val_len // params.batch_size):
                        s_idx = val_it * params.batch_size
                        e_idx = (val_it + 1) * params.batch_size
                        word_input = encoder_val_data[s_idx:e_idx]
                        word_input = zero_pad(word_input, pad)
                        label_input = val_labels_arr[s_idx:e_idx]
                        label_input = zero_pad(label_input, pad)

                        ## batch_size = word_input.shape[0]
                        batch_size = len(word_input)
                        num_examples += batch_size

                        feed = {
                            word_inputs: word_input,
                            label_inputs: label_input,
                        }
                        zs_dist, zs_sample, zg_dist, zg_sample, _, _ = sess.run(
                            [
                                Zsent_distribution, zsent_sample,
                                Zglobal_distribition, zglobal_sample,
                                zsent_state, zglobal_state
                            ],
                            feed_dict=feed)

                        mi_s = calc_mi_q(zs_dist[0], zs_dist[1], zs_samples)
                        mi_g = calc_mi_q(zg_dist[0], zg_dist[1], zg_samples)
                        mi_zc_zl = calc_mi_two(zs_dist[0], zs_dist[1], zg_dist[0], zg_dist[1], zg_sample, zs_sample)
                        global_mi += mi_g * batch_size

                    sent_mi /= num_examples
                    global_mi /= num_examples
                    cur_mi = sent_mi + global_mi

                    smi.append(sent_mi)
                    gmi.append(global_mi)
                    tmi.append(cur_mi)
                    # smi_ext.append(sent_mi)
                    # gmi_ext.append(global_mi)
                    # tmi_ext.append(cur_mi)
                    # smi_ind.append(0)
                    # gmi_ind.append(0)
                    # tmi_ind.append(0)
                    write_lists_to_file('mi_values.txt', smi, gmi, tmi)
                    # write_lists_to_file('mi_ext_values.txt', smi_ext, gmi_ext, tmi_ext, smi_ind, gmi_ind, tmi_ind)

                    cur_it += 1
                    if aggressive and cur_it%num_iters==0:
                        print("pre mi:%.4f. cur mi:%.4f" %
                                (pre_mi, cur_mi))
                        if cur_mi < pre_mi:
                            aggressive = False
                            print("STOP BURNING")
                        pre_mi = cur_mi


                    if cur_it % 100 == 0 and cur_it != 0:
                        path_to_save = os.path.join(params.MODEL_DIR,
                                                    "vae_lstm_model")
                        # print(path_to_save)
                        model_path_name = saver.save(sess,
                                                     path_to_save,
                                                     global_step=cur_it)
                        # print(model_path_name)



if __name__ == "__main__":
    params = parameters.Parameters()
    params.parse_args()
    main(params)
