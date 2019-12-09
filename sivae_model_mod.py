from __future__ import print_function, division, absolute_import
import tensorflow as tf
import numpy as np
import os
from utils import parameters
import utils.model as model
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import xavier_initializer
params = parameters.Parameters()


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


##mu_nl and logvar_nl---- non linearity for mu and logvar
def gauss_layer(inp, dim, mu_nl=None, logvar_nl=None, scope=None):
    """
    Gaussian layer
    Args:
        inp(tf.Tensor): input to Gaussian layer
        dim(int): dimension of output latent variables
        mu_nl(callable): nonlinearity for Gaussian mean
        logvar_nl(callable): nonlinearity for Gaussian log variance
        scope(str/VariableScope): tensorflow variable scope
    """
    with tf.variable_scope(scope, "gauss") as sc:
        mu = fully_connected(
            inp,
            dim,
            activation_fn=mu_nl,
            weights_initializer=xavier_initializer(),
            biases_initializer=tf.zeros_initializer(),
            scope="mu"
        )
        logvar = fully_connected(
            inp,
            dim,
            activation_fn=logvar_nl,
            weights_initializer=xavier_initializer(),
            biases_initializer=tf.zeros_initializer(),
            scope="logvar"
        )

        mu = tf.clip_by_value(mu, -50, 50)
        logvar = tf.clip_by_value(logvar, -50, 50)

        eps = tf.random_normal(tf.shape(logvar), name='eps', dtype=tf.float64)
        sample = mu + tf.exp(0.5 * logvar) * eps
        return mu, logvar, sample


def encoder_model(word_input, label_input, batch_size, len_word, len_label):

    ## LSTM cells
    with tf.variable_scope("word"):
        word_cell_fw = tf.contrib.rnn.LSTMCell(
            params.encoder_hidden, dtype=tf.float64, state_is_tuple=False
        )
        word_cell_bw = tf.contrib.rnn.LSTMCell(
            params.encoder_hidden, dtype=tf.float64, state_is_tuple=False
        )

    with tf.variable_scope("label"):
        label_cell_fw = tf.contrib.rnn.LSTMCell(
            params.encoder_hidden, dtype=tf.float64, state_is_tuple=False
        )
        label_cell_bw = tf.contrib.rnn.LSTMCell(
            params.encoder_hidden, dtype=tf.float64, state_is_tuple=False
        )

    ## word_input.shape: [batch_size, time_steps, latent_dim]
    with tf.variable_scope("word"):
        _, word_output_states = tf.nn.bidirectional_dynamic_rnn(
            word_cell_fw,
            word_cell_bw,
            word_input,
            sequence_length=len_word,
            dtype=tf.float64
        )

    with tf.variable_scope("label"):
        _, label_output_states = tf.nn.bidirectional_dynamic_rnn(
            label_cell_fw,
            label_cell_bw,
            label_input,
            sequence_length=len_label,
            dtype=tf.float64
        )

    word_cell_state = tf.concat(word_output_states, -1)
    label_cell_state = tf.concat(label_output_states, -1)
    return word_cell_state, label_cell_state


def encoder(encoder_input, label_input, batch_size, len_word, len_label):
    with tf.variable_scope("encoder"):
        zsent_pre_out, zglobal_pre_out = encoder_model(
            encoder_input, label_input, batch_size, tf.cast(len_word, tf.int64),
            tf.cast(len_label, tf.int64)
        )

        zsent_mu, zsent_logvar, zsent_sample = gauss_layer(
            zsent_pre_out, params.latent_size, scope="word"
        )
        Zsent_distribution = [zsent_mu, zsent_logvar]

        gauss_input = tf.concat([zglobal_pre_out, zsent_sample], -1)
        zglobal_mu, zglobal_logvar, zglobal_sample = gauss_layer(
            gauss_input, params.latent_size, scope="label"
        )
        Zglobal_distribition = [zglobal_mu, zglobal_logvar]

    return Zsent_distribution, zsent_sample, Zglobal_distribition, zglobal_sample, zsent_pre_out, zglobal_pre_out


def decoder_model(
    word_input,
    label_input,
    zl,
    zc,
    batch_size,
    label_vocab_size,
    word_vocab_size,
    max_len_word,
    max_len_label,
    word_embed,
    label_embed,
    gen_mode=False,
    label_cell_state_inp=None,
    word_cell_state=None,
):
    label_cell_state = label_cell_state_inp

    ## the two LSTM cells
    with tf.variable_scope("label"):
        label_cell = tf.contrib.rnn.LSTMCell(
            params.decoder_hidden, dtype=tf.float64, state_is_tuple=False
        )
    with tf.variable_scope("word"):
        word_cell = tf.contrib.rnn.LSTMCell(
            params.decoder_hidden, dtype=tf.float64, state_is_tuple=False
        )

    ## will be batch_size x sentences for training
    ## will be single word for gen_mode
    with tf.variable_scope("word"):
        word_input = tf.nn.embedding_lookup(word_embed, word_input)
    with tf.variable_scope("label"):
        label_input = tf.nn.embedding_lookup(label_embed, label_input)
    if not gen_mode:
        ## 'time' major tensors
        word_input_t = tf.transpose(word_input, [1, 0, 2])
        label_input_t = tf.transpose(label_input, [1, 0, 2])

    ## Fully Connected layers for logits
    with tf.variable_scope("label"):
        label_dense_layer = tf.layers.Dense(label_vocab_size, activation=None)
    with tf.variable_scope("word"):
        word_dense_layer = tf.layers.Dense(word_vocab_size, activation=None)

    ## zero initial state
    if label_cell_state is None:
        label_cell_state = label_cell.zero_state(batch_size, tf.float64)

    if word_cell_state is None:
        word_cell_state = word_cell.zero_state(batch_size, tf.float64)

    word_logits_arr = []
    label_logits_arr = []
    word_state_arr = []
    label_state_arr = []

    ## compute the LSTM outputs
    for i in range(max_len_label):
        ## run the label decoder LSTM
        # label_embedding = tf.nn.embedding_lookup(label_embed, pred_label)
        with tf.variable_scope("label"):
            if gen_mode:
                label_embedding_input = label_input
            else:
                label_embedding_input = label_input_t[i]
            label_cell_input = tf.concat([zl, label_embedding_input], axis=-1)
            label_cell_output, label_cell_state = label_cell(
                label_cell_input, label_cell_state
            )
            ## get the label logits
            label_logits = label_dense_layer(label_cell_output)
            label_logits_softmax = tf.nn.softmax(label_logits)

            label_logits_arr.append(label_logits)
            #label_logits_arr.append(label_logits)
            label_state_arr.append(label_cell_state)

    if gen_mode:
        label_cell_state = label_cell_state_inp

    for i in range(max_len_word):
        # word_embedding = tf.nn.embedding_lookup(word_embed, pred_word)
        with tf.variable_scope("word"):
            if gen_mode:
                word_embedding_input = word_input
            else:
                word_embedding_input = word_input_t[i]

            word_cell_input = tf.concat(
                [zc, label_cell_state, word_embedding_input], axis=-1
            )

            word_cell_output, word_cell_state = word_cell(
                word_cell_input, word_cell_state
            )

            ## get word logits
            word_logits = word_dense_layer(word_cell_output)
            # word_logits_softmax = tf.nn.softmax(word_logits)

            word_logits_arr.append(word_logits)
            word_state_arr.append(word_cell_state)

    # word_logits_all = tf.concat(word_logits_arr, axis=0)
    word_state_all = tf.concat(word_state_arr, axis=0)
    # label_logits_all = tf.concat(label_logits_arr, axis=0)
    label_state_all = tf.concat(label_state_arr, axis=0)

    word_logits_all = tf.stack(word_logits_arr, axis=1)
    word_logits_all = tf.reshape(word_logits_all, [-1, word_vocab_size])
    label_logits_all = tf.stack(label_logits_arr, axis=1)
    label_logits_all = tf.reshape(label_logits_all, [-1, label_vocab_size])

    if params.beam_search:
        with tf.variable_scope("word"):
            word_sample = tf.nn.softmax(word_logits_all)
        with tf.variable_scope("label"):
            label_sample = tf.nn.softmax(label_logits_all)
    else:
        with tf.variable_scope("word"):
            word_sample = tf.multinomial(
                word_logits_all / params.temperature, 10
            )[0]
        with tf.variable_scope("label"):
            label_sample = tf.multinomial(
                label_logits_all / params.temperature, 10
            )[0]

    return word_logits_all, label_logits_all, word_sample, label_sample, word_state_all, label_state_all


def decoder(
    word_input,
    label_input,
    z_global_sample,
    batch_size,
    word_vocab_size,
    label_vocab_size,
    max_len_word,
    max_len_label,
    word_embed,
    label_embed,
    z_sent_sample=None,
    gen_mode=False,
    label_cell_state=None,
    word_cell_state=None
):
    with tf.variable_scope("decoder") as sc:
        Zglobal_dec_distribution = [
            tf.cast(0, dtype=tf.float64),
            tf.cast(1.0, dtype=tf.float64)
        ]
        # Zsent_dec_distribution = [0.0, 1.0]

        zs_dec_mu, zs_dec_logvar, zs_dec_sample = gauss_layer(
            z_global_sample, params.latent_size, scope="word"
        )
        # if zsent_dec_sample is None:
        if not gen_mode:  ## training mode
            zsent_dec_sample = z_sent_sample
        else:
            indicator = tf.sign(tf.reduce_sum(tf.abs(z_sent_sample)))
            zsent_dec_sample = z_sent_sample + zs_dec_sample * (1 - indicator)
        Zsent_dec_distribution = [zs_dec_mu, zs_dec_logvar]

        decoder_output = decoder_model(
            word_input,
            label_input,
            z_global_sample,
            zsent_dec_sample,
            batch_size,
            label_vocab_size,
            word_vocab_size,
            max_len_word,
            max_len_label,
            word_embed,
            label_embed,
            gen_mode,
            label_cell_state_inp=label_cell_state,
            word_cell_state=word_cell_state,
        )

    word_logits, label_logits, w_sample, l_sample, w_cell_state, l_cell_state = decoder_output
    return word_logits, label_logits, Zsent_dec_distribution, Zglobal_dec_distribution, l_sample, w_sample, w_cell_state, l_cell_state
