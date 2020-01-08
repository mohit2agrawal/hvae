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
        # return mu, logvar

        eps = tf.random_normal(tf.shape(logvar), name='eps', dtype=tf.float64)
        sample = mu + tf.exp(0.5 * logvar) * eps
        return mu, logvar, sample


def word_encoder(word_input, batch_size, len_word):

    with tf.variable_scope("word"):
        ## LSTM cells
        # word_cell_fw = tf.contrib.rnn.LSTMCell(
        #     params.encoder_hidden, dtype=tf.float64, state_is_tuple=False
        # )
        # word_cell_bw = tf.contrib.rnn.LSTMCell(
        #     params.encoder_hidden, dtype=tf.float64, state_is_tuple=False
        # )

        word_cell_fw = tf.contrib.rnn.GRUCell(
            params.encoder_hidden, dtype=tf.float64
        )
        word_cell_bw = tf.contrib.rnn.GRUCell(
            params.encoder_hidden, dtype=tf.float64
        )

        _, word_output_states = tf.nn.bidirectional_dynamic_rnn(
            word_cell_fw,
            word_cell_bw,
            word_input,
            sequence_length=len_word,
            dtype=tf.float64
        )

    return word_output_states
    # return tf.concat(word_output_states, -1)


def encoder(encoder_input, batch_size, seq_len):
    with tf.variable_scope("encoder"):
        wc_state = word_encoder(
            encoder_input, batch_size, tf.cast(seq_len, tf.int64)
        )

        zsent_mu, zsent_logvar, zsent_sample = gauss_layer(
            wc_state, params.num_topics * params.latent_size, scope="word"
        )
        # bs x T*LS
        zsent_mu = zsent_mu.reshape(
            batch_size, params.num_topics, params.latent_size
        )
        zsent_logvar = zsent_logvar.reshape(
            batch_size, params.num_topics, params.latent_size
        )
        zsent_sample = zsent_sample.reshape(
            batch_size, params.num_topics, params.latent_size
        )

        # bs x T
        enc_topic_dist = fully_connected(
            wc_state,
            params.num_topics,
            activation_fn=None,
            weights_initializer=xavier_initializer(),
            biases_initializer=tf.zeros_initializer(),
        )
        enc_topic_dist = tf.nn.softmax(enc_topic_dist)

        # ## sum(t_i x mu_i)
        # zsent_mu = tf.matmul(enc_topic_dist, zsent_mu)
        # zsent_logvar = tf.matmul(enc_topic_dist, zsent_logvar)
        zsent_sample = tf.matmul(enc_topic_dist, zsent_sample)

        # Zsent_distribution = [zsent_mu, zsent_logvar]
        return zsent_mu, zsent_logvar, zsent_sample, enc_topic_dist


def word_decoder_model(
    word_input,
    z,
    batch_size,
    word_vocab_size,
    seq_len,
    word_embed,
    gen_mode=False,
    word_cell_state=None,
):
    with tf.variable_scope("word"):
        word_cell = tf.contrib.rnn.LSTMCell(
            params.decoder_hidden, dtype=tf.float64, state_is_tuple=False
        )
        ## will be batch_size x sentences for training
        ## will be single word for gen_mode
        word_input = tf.nn.embedding_lookup(word_embed, word_input)

        # if not gen_mode:
        #     ## 'time' major tensors
        #     word_input_t = tf.transpose(word_input, [1, 0, 2])

        ## Fully Connected layers for logits
        word_dense_layer = tf.layers.Dense(word_vocab_size, activation=None)

        if word_cell_state is None:
            word_cell_state = word_cell.zero_state(batch_size, tf.float64)

        # word_logits_arr = []
        # word_state_arr = []

        max_sl = tf.shape(word_input)[1]
        z = tf.reshape(
            tf.tile(tf.expand_dims(z, 1), (1, max_sl, 1)),
            [batch_size, -1, params.latent_size]
        )
        word_input = tf.concat([word_input, z], 2)

        outputs, final_state = tf.nn.dynamic_rnn(
            word_cell,
            inputs=word_input,
            sequence_length=seq_len,
            initial_state=word_cell_state,
            swap_memory=True,
            dtype=tf.float64
        )

        # word_logits_all = tf.stack(word_logits_arr, axis=1)
        # word_logits_all = tf.reshape(word_logits_all, [-1, word_vocab_size])

        logits = word_dense_layer(outputs)

    return logits


def two_layer_mlp(inputs, hidden_units_1, hidden_units_2, activation_fn=None):
    layer_1_out = fully_connected(
        inputs,
        hidden_units_1,
        activation_fn=activation_fn,
        weights_initializer=xavier_initializer(),
        biases_initializer=tf.zeros_initializer(),
    )
    layer_2_out = fully_connected(
        layer_1_out,
        hidden_units_2,
        activation_fn=activation_fn,
        weights_initializer=xavier_initializer(),
        biases_initializer=tf.zeros_initializer(),
    )
    return layer_2_out


def doc_encoder(doc_bow):
    doc_mu = two_layer_mlp(
        doc_bow, params.ntm_hidden, params.num_topics, activation_fn=tf.nn.relu
    )
    doc_logvar = two_layer_mlp(
        doc_bow, params.ntm_hidden, params.num_topics, activation_fn=tf.nn.relu
    )

    doc_eps = tf.random_normal(
        tf.shape(doc_logvar), name='doc_eps', dtype=tf.float64
    )
    doc_sample = doc_mu + tf.exp(0.5 * doc_logvar) * doc_eps

    return doc_mu, doc_logvar, doc_sample


def doc_decoder(doc_sample, topic_word_dist):

    ## TODO: Linear Transformation on topic_sample
    topic_dist = fully_connected(
        doc_sample,
        params.ntm_hidden,
        activation_fn=tf.nn.relu,
        weights_initializer=xavier_initializer(),
        biases_initializer=tf.zeros_initializer(),
    )
    topic_dist_sm = tf.nn.softmax(topic_dist)

    ## T: num_topics
    ## D: vocab_size
    ## topic_word_dist: bs x T x D
    ## topic_dist_sm: bs x T
    ## want, decoded_doc: bs x D

    ## bs x 1 x T
    topic_dist_expanded = tf.expand_dims(topic_dist_sm, axis=1)
    ## bs x 1 x D -> squeeze -> bs x D
    decoded_doc = tf.squeeze(tf.matmul(topic_dist_expanded, topic_word_dist))

    return topic_dist_sm, decoded_doc


def get_word_priors(topic_word_dist, topic_dist):
    ## topic_word_dist are the beta

    all_dec_mu, all_dec_logvar, all_dec_samples = [], [], []
    for i in range(params.num_topics):
        _mu, _logvar, _sample = gauss_layer(
            topic_word_dist[:, i, :], params.latent_size
        )
        all_dec_mu.append(_mu)
        all_dec_logvar.append(_logvar)
        all_dec_samples.append(_sample)
    dec_mu = tf.stack(all_dec_mu, axis=1)
    dec_logvar = tf.stack(all_dec_logvar, axis=1)
    dec_samples = tf.stack(all_dec_samples, axis=1)  ## bs x T x latent_size

    ## sum ( t_i * z_i)
    dec_z = tf.matmul(topic_dist, dec_samples)

    return dec_mu, dec_logvar, dec_z


## TODO do t_i * sample_i, instead of sample after t_i * mu_i