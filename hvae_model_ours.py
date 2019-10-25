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
    with tf.variable_scope(scope + "/gauss") as sc:
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


def zglobal_encoder(label_input, zsent_sample, seq_len, batch_size):
    """
    Pre-stochastic layer encoder for z1 (latent segment variable)
    Args:
        x(tf.Tensor): tensor of shape (bs, T, F)
        z2(tf.Tensor): tensor of shape (bs, D1)
        rhus(list): list of numbers of LSTM layer hidden units
    Return:
        out(tf.Tensor): concatenation of hidden states of all LSTM layers
    """

    # prepare input
    bs, T = tf.shape(label_input)[0], tf.shape(label_input)[1]
    zsent_sample = tf.tile(tf.expand_dims(zsent_sample, 1), (1, T, 1))
    x_z2 = tf.concat([label_input, zsent_sample], axis=-1)
    encoder_input = x_z2

    if params.base_cell == 'lstm':
        base_cell = tf.contrib.rnn.LSTMCell
    elif params.base_cell == 'rnn':
        base_cell = tf.contrib.rnn.RNNCell
    else:
        base_cell = tf.contrib.rnn.GRUCell

    cell = model.make_rnn_cell(
        [params.encoder_hidden for _ in range(params.decoder_rnn_layers)],
        base_cell=base_cell
    )

    initial = cell.zero_state(batch_size, dtype=tf.float64)

    if params.keep_rate < 1:
        encoder_input = tf.nn.dropout(encoder_input, params.keep_rate)
    outputs, final_state = tf.nn.dynamic_rnn(
        cell,
        inputs=encoder_input,
        sequence_length=seq_len,
        initial_state=initial,
        swap_memory=True,
        dtype=tf.float64,
        scope="zglobal_encoder_rnn"
    )
    final_state = tf.concat(final_state[0], 1)
    return final_state


def zsent_encoder(encoder_input, seq_len, batch_size):
    """
    Pre-stochastic layer encoder for z2 (latent sequence variable)
    Args:
        x(tf.Tensor): tensor of shape (bs, T, F)
        rhus(list): list of numbers of LSTM layer hidden units
    Return:
        out(tf.Tensor): concatenation of hidden states of all LSTM layers
    """
    # construct lstm
    # cell = tf.nn.rnn_cell.BasicLSTMCell(params.cell_hidden_size)
    # cells = tf.nn.rnn_cell.MultiRNNCell([cell]*params.rnn_layers)
    if params.base_cell == 'lstm':
        base_cell = tf.contrib.rnn.LSTMCell
    elif params.base_cell == 'rnn':
        base_cell = tf.contrib.rnn.RNNCell
    else:
        base_cell = tf.contrib.rnn.GRUCell

    cell = model.make_rnn_cell(
        [params.encoder_hidden for _ in range(params.decoder_rnn_layers)],
        base_cell=base_cell
    )
    initial = cell.zero_state(batch_size, dtype=tf.float64)

    if params.keep_rate < 1:
        encoder_input = tf.nn.dropout(encoder_input, params.keep_rate)
    # print(encoder_input.shape)
    # 'final_state' is a tensor of shape [batch_size, cell_state_size]
    outputs, final_state = tf.nn.dynamic_rnn(
        cell,
        inputs=encoder_input,
        sequence_length=seq_len,
        initial_state=initial,
        swap_memory=True,
        dtype=tf.float64,
        scope="zsent_encoder_rnn"
    )
    final_state = tf.concat(final_state[0], 1)
    return final_state


def encoder_model(word_input, label_input, batch_size, max_sent_len):

    ## LSTM cells
    with tf.variable_scope("word"):
    	word_cell = tf.contrib.rnn.LSTMCell(
        	params.encoder_hidden, dtype=tf.float64, state_is_tuple=False
    	)
    	word_cell_state = word_cell.zero_state(batch_size, dtype=tf.float64)
    	word_input_t = tf.transpose(word_input, [1, 0, 2])

    with tf.variable_scope("label"):
    	label_cell = tf.contrib.rnn.LSTMCell(
        	params.encoder_hidden, dtype=tf.float64, state_is_tuple=False
    	)
    	label_cell_state = word_cell.zero_state(batch_size, dtype=tf.float64)
    	label_input_t = tf.transpose(label_input, [1, 0, 2])

    ## initial (zero) states
    
    

    ## 'time' major tensors
    
    

    ## word_input.shape: [batch_size, time_steps, latent_dim]
    for i in range(max_sent_len):
    	with tf.variable_scope("word"):
        	word_cell_output, word_cell_state = word_cell(
            word_input_t[i], word_cell_state
        	)
        with tf.variable_scope("label"):
        	label_cell_input = tf.concat([label_input_t[i], word_cell_state], -1)
        	label_cell_output, label_cell_state = label_cell(
            	label_cell_input, label_cell_state
        	)

    return word_cell_state, label_cell_state


def encoder(encoder_input, label_input, batch_size, max_sent_len):
    with tf.variable_scope("encoder"):
        zsent_pre_out, zglobal_pre_out = encoder_model(
            encoder_input, label_input, batch_size, max_sent_len
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


def lstm_decoder_labels(
    z,
    d_inputs,
    d_seq_l,
    batch_size,
    embed,
    vocab_size,
    gen_mode=False,
    scope=None
):

    with tf.variable_scope(scope, "decoder") as sc:
        with tf.device("/cpu:0"):
            dec_inps = tf.nn.embedding_lookup(embed, d_inputs)
        # turn off dropout for generation:
        if params.dec_keep_rate < 1 and not gen_mode:
            dec_inps = tf.nn.dropout(dec_inps, params.dec_keep_rate)

        max_sl = tf.shape(dec_inps)[1]
        # define cell
        if params.base_cell == 'lstm':
            base_cell = tf.contrib.rnn.LSTMCell
        elif params.base_cell == 'rnn':
            base_cell = tf.contrib.rnn.RNNCell
        else:
            # not working for now
            base_cell = tf.contrib.rnn.GRUCell
        cell = model.make_rnn_cell(
            [params.decoder_hidden for _ in range(params.decoder_rnn_layers)],
            base_cell=base_cell
        )

        if params.decode == 'hw':
            # Higway network [S.Sementiuta et.al]
            for i in range(params.highway_lc):
                with tf.variable_scope("hw_layer_dec{0}".format(i)) as scope:
                    z_dec = fully_connected(
                        z,
                        params.decoder_hidden * 2,
                        activation_fn=tf.nn.sigmoid,
                        weights_initializer=xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        scope="decoder_inp_state"
                    )

            inp_h, inp_c = tf.split(z_dec, 2, axis=1)
            initial_state = rnn_placeholders(
                (tf.contrib.rnn.LSTMStateTuple(inp_c, inp_h), )
            )
        elif params.decode == 'concat':
            z_out = tf.reshape(
                tf.tile(tf.expand_dims(z, 1), (1, max_sl, 1)),
                [batch_size, -1, params.latent_size]
            )
            dec_inps = tf.concat([dec_inps, z_out], 2)
            initial_state = rnn_placeholders(
                cell.zero_state(tf.shape(dec_inps)[0], tf.float64)
            )
        elif params.decode == 'mlp':
            # z->decoder initial state
            w1 = tf.get_variable(
                'whl', [params.latent_size, params.highway_ls],
                tf.float64,
                initializer=tf.truncated_normal_initializer()
            )
            b1 = tf.get_variable(
                'bhl', [params.highway_ls],
                tf.float64,
                initializer=tf.ones_initializer()
            )
            z_dec = tf.matmul(z, w1) + b1
            inp_h, inp_c = tf.split(
                tf.layers.dense(z_dec, params.decoder_hidden * 2), 2, axis=1
            )
            initial_state = rnn_placeholders(
                (tf.contrib.rnn.LSTMStateTuple(inp_c, inp_h), )
            )

        outputs, final_state = tf.nn.dynamic_rnn(
            cell,
            inputs=dec_inps,
            sequence_length=d_seq_l,
            initial_state=initial_state,
            swap_memory=True,
            dtype=tf.float64
        )
        # define decoder network
        if gen_mode:
            # only interested in the last output
            outputs = outputs[:, -1, :]
        # print(outputs.shape)
        outputs_r = tf.reshape(outputs, [-1, params.decoder_hidden])
        # print(outputs_r.shape,     "===============")
        x_logits = tf.layers.dense(outputs_r, units=vocab_size, activation=None)
        print(x_logits)
        if params.beam_search:
            sample = tf.nn.softmax(x_logits)
        else:
            sample = tf.multinomial(x_logits / params.temperature, 10)[0]
        print(sample)
        return x_logits, (initial_state, final_state), sample


def lstm_decoder_words(
    z_in,
    d_inputs,
    label_logits,
    d_seq_l,
    batch_size,
    embed,
    vocab_size,
    gen_mode=False,
    zsent=None,
    scope=None
):

    with tf.variable_scope(scope, "decoder") as sc:
        with tf.device("/cpu:0"):
            dec_inps = tf.nn.embedding_lookup(embed, d_inputs)
        # turn off dropout for generation:
        if params.dec_keep_rate < 1 and not gen_mode:
            dec_inps = tf.nn.dropout(dec_inps, params.dec_keep_rate)

        label_logits = tf.nn.softmax(label_logits)
        dep = int(label_logits.shape[1])
        bs, T = tf.shape(dec_inps)[0], tf.shape(dec_inps)[1]
        print(bs, T)
        label_logits = tf.reshape(label_logits, [bs, T, dep])
        print(label_logits)
        print(dec_inps)
        dec_inps = tf.concat([dec_inps, label_logits], axis=-1)
        print(dec_inps)
        # exit()
        max_sl = tf.shape(dec_inps)[1]
        # define cell
        if params.base_cell == 'lstm':
            base_cell = tf.contrib.rnn.LSTMCell
        elif params.base_cell == 'rnn':
            base_cell = tf.contrib.rnn.RNNCell
        else:
            # not working for now
            base_cell = tf.contrib.rnn.GRUCell

        cell = model.make_rnn_cell(
            [params.decoder_hidden for _ in range(params.decoder_rnn_layers)],
            base_cell=base_cell
        )

        if gen_mode:
            z = zsent
        else:
            z = z_in
        if params.decode == 'hw':
            # Higway network [S.Sementiuta et.al]
            for i in range(params.highway_lc):
                with tf.variable_scope("hw_layer_dec{0}".format(i)) as scope:
                    z_dec = fully_connected(
                        z,
                        params.decoder_hidden * 2,
                        activation_fn=tf.nn.sigmoid,
                        weights_initializer=xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        scope="decoder_inp_state"
                    )

            inp_h, inp_c = tf.split(z_dec, 2, axis=1)
            initial_state = rnn_placeholders(
                (tf.contrib.rnn.LSTMStateTuple(inp_c, inp_h), )
            )
        elif params.decode == 'concat':
            z_out = tf.reshape(
                tf.tile(tf.expand_dims(z, 1), (1, max_sl, 1)),
                [batch_size, -1, params.latent_size]
            )
            dec_inps = tf.concat([dec_inps, z_out], 2)
            initial_state = rnn_placeholders(
                cell.zero_state(tf.shape(dec_inps)[0], tf.float64)
            )
        elif params.decode == 'mlp':
            # z->decoder initial state
            w1 = tf.get_variable(
                'whl', [params.latent_size, params.highway_ls],
                tf.float64,
                initializer=tf.truncated_normal_initializer()
            )
            b1 = tf.get_variable(
                'bhl', [params.highway_ls],
                tf.float64,
                initializer=tf.ones_initializer()
            )
            z_dec = tf.matmul(z, w1) + b1
            inp_h, inp_c = tf.split(
                tf.layers.dense(z_dec, params.decoder_hidden * 2), 2, axis=1
            )
            initial_state = rnn_placeholders(
                (tf.contrib.rnn.LSTMStateTuple(inp_c, inp_h), )
            )

        outputs, final_state = tf.nn.dynamic_rnn(
            cell,
            inputs=dec_inps,
            sequence_length=d_seq_l,
            initial_state=initial_state,
            swap_memory=True,
            dtype=tf.float64
        )
        # define decoder network
        if gen_mode:
            # only interested in the last output
            outputs = outputs[:, -1, :]
        # print(outputs.shape)
        outputs_r = tf.reshape(outputs, [-1, params.decoder_hidden])
        # print(outputs_r.shape,     "===============")
        x_logits = tf.layers.dense(outputs_r, units=vocab_size, activation=None)
        print(x_logits)
        if params.beam_search:
            sample = tf.nn.softmax(x_logits)
        else:
            sample = tf.multinomial(x_logits / params.temperature, 10)[0]
        print(sample)
        return x_logits, (initial_state, final_state), sample


def decoder_model(
    word_input,
    label_input,
    zl,
    zc,
    batch_size,
    label_vocab_size,
    word_vocab_size,
    max_sent_len,
    word_embed,
    label_embed,
    gen_mode=False,
    label_cell_state=None,
    word_cell_state=None,
):
    ## the two LSTM cells
    label_cell = tf.contrib.rnn.LSTMCell(
        params.decoder_hidden, dtype=tf.float64, state_is_tuple=False
    )
    word_cell = tf.contrib.rnn.LSTMCell(
        params.decoder_hidden, dtype=tf.float64, state_is_tuple=False
    )

    ## will be batch_size x sentences for training
    ## will be single word for gen_mode
    word_input = tf.nn.embedding_lookup(word_embed, word_input)
    label_input = tf.nn.embedding_lookup(label_embed, label_input)
    if not gen_mode:
        ## 'time' major tensors
        word_input_t = tf.transpose(word_input, [1, 0, 2])
        label_input_t = tf.transpose(label_input, [1, 0, 2])

    ## Fully Connected layers for logits
    label_dense_layer = tf.layers.Dense(label_vocab_size, activation=None)
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
    for i in range(max_sent_len):
        ## run the label decoder LSTM
        ## input for the label decoder LSTM
        ##     takes word_cell_state from prev iter
        # label_embedding = tf.nn.embedding_lookup(label_embed, pred_label)
        if gen_mode:
            label_embedding_input = label_input
        else:
            label_embedding_input = label_input_t[i]
        label_cell_input = tf.concat(
            [zl, word_cell_state, label_embedding_input], axis=-1
        )
        label_cell_output, label_cell_state = label_cell(
            label_cell_input, label_cell_state
        )
        ## get the label logits
        label_logits = label_dense_layer(label_cell_output)
        label_logits_softmax = tf.nn.softmax(label_logits)

        label_logits_arr.append(label_logits_softmax)
        #label_logits_arr.append(label_logits)
        label_state_arr.append(label_cell_state)

        ## concat zc and label logits and run the word decoder LSTM
        # word_embedding = tf.nn.embedding_lookup(word_embed, pred_word)
        if gen_mode:
            word_embedding_input = word_input
        else:
            word_embedding_input = word_input_t[i]
        word_cell_input = tf.concat(
            [zc, label_logits_softmax, word_embedding_input], axis=-1
        )
        word_cell_output, word_cell_state = word_cell(
            word_cell_input, word_cell_state
        )

        ## get word logits
        word_logits = word_dense_layer(word_cell_output)
        # word_logits_softmax = tf.nn.softmax(word_logits)

        word_logits_arr.append(word_logits)
        word_state_arr.append(word_cell_state)

    word_logits_all = tf.concat(word_logits_arr, axis=0)
    word_state_all = tf.concat(word_state_arr, axis=0)
    label_logits_all = tf.concat(label_logits_arr, axis=0)
    label_state_all = tf.concat(label_state_arr, axis=0)

    if params.beam_search:
        word_sample = tf.nn.softmax(word_logits_all)
        label_sample = tf.nn.softmax(label_logits_all)
    else:
        word_sample = tf.multinomial(word_logits_all / params.temperature,
                                     10)[0]
        label_sample = tf.multinomial(
            label_logits_all / params.temperature, 10
        )[0]

    return word_logits_all, label_logits_all, word_sample, label_sample, word_state_all, label_state_all


def decoder(
    word_input,
    label_input,
    zglobal_sample,
    batch_size,
    word_vocab_size,
    label_vocab_size,
    max_sent_len,
    word_embed,
    label_embed,
    gen_mode=False,
    label_cell_state=None,
    word_cell_state=None,
    zsent_dec_mu=None,
    zsent_dec_logvar=None,
    zsent_dec_sample=None
):
    with tf.variable_scope("decoder") as sc:
        ## compute zc
        zs_dec_mu, zs_dec_logvar, zs_dec_sample = gauss_layer(
            zglobal_sample, params.latent_size, scope="zsent_dec_gauss"
        )
        # if zsent_dec_sample is None:
        if not gen_mode:
            zsent_dec_mu, zsent_dec_logvar, zsent_dec_sample = zs_dec_mu, zs_dec_logvar, zs_dec_sample
        else:
            ## for sampling case
            ## for first time sample, we pass in zero
            ## and would like to keep the values returned by above gauss
            ## later on, we would like to use the input values
            ## which will be non-zero, and not use the gauss outputs
            indicator = tf.sign(
                tf.reduce_sum(tf.abs(zsent_dec_sample))
            )
            zsent_dec_mu += zs_dec_mu * indicator
            zsent_dec_logvar += zs_dec_logvar * indicator
            zsent_dec_sample += zs_dec_sample * indicator

        zsent_dec_distribution = [zsent_dec_mu, zsent_dec_logvar]

        Zglobal_dec_distribution = [0., np.log(1.0**2).astype(np.float64)]
        decoder_output = decoder_model(
            word_input,
            label_input,
            zglobal_sample,
            zsent_dec_sample,
            batch_size,
            label_vocab_size,
            word_vocab_size,
            max_sent_len,
            word_embed,
            label_embed,
            gen_mode,
            label_cell_state=label_cell_state,
            word_cell_state=word_cell_state,
        )

    word_logits, label_logits, w_sample, l_sample, w_cell_state, l_cell_state = decoder_output
    return word_logits, label_logits, zsent_dec_distribution, Zglobal_dec_distribution, l_sample, w_sample, zsent_dec_sample, w_cell_state, l_cell_state
