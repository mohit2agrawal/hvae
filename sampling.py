from __future__ import print_function, division, absolute_import
import tensorflow as tf
import numpy as np

import utils.data as data_
import utils.label_data as label_data_
import utils.model as model
from utils.ptb import reader
from utils import parameters
from utils.beam_search import beam_search

from tensorflow.python import debug as tf_debug
from tensorflow.python.util.nest import flatten
import os
from tensorflow.python.tools import inspect_checkpoint as chkp
from sivae_model_mod import encoder, decoder

from tqdm import tqdm


# PTB input from tf tutorial
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
    ## if zero_probs


def zero_pad(sentences, max_len):
    return np.array([sent + [0] * (max_len - len(sent)) for sent in sentences])


def main(params):
    data_folder = './DATA/' + params.name

    train_data_raw, train_label_raw, val_data_raw, val_label_raw, vocab = data_.read_data(
        data_folder
    )
    data, encoder_data, val_data, encoder_val_data, embed_arr, data_dict, labels, val_labels, label_embed_arr = data_.prepare_data(
        train_data_raw, train_label_raw, val_data_raw, val_label_raw, vocab,
        params, data_folder
    )

    # max_len_word = max(max(map(len, data)), max(map(len, val_data)))
    # print('max len word:', max_len_word)

    word_vocab_size = data_dict.vocab_size
    label_vocab_size = len(label_embed_arr)

    encoder_data = np.array(encoder_data)

    ## one word at a time for decoder
    dec_word_len = 1
    ## one sentence at a time
    batch_size = 1

    word_bos_index = data_dict.word2idx[data_dict.bos]
    word_eos_index = data_dict.word2idx[data_dict.eos]
    word_pad_index = data_dict.word2idx[data_dict.pad]
    word_unk_index = data_dict.word2idx[data_dict.unk]

    with tf.Graph().as_default() as graph:

        zglobal_sample = tf.placeholder(
            dtype=tf.float64, shape=[None, params.latent_size]
        )
        d_word_inputs = tf.placeholder(
            dtype=tf.int32, shape=[1], name="d_word_inputs"
        )

        word_cell_state = tf.placeholder(
            dtype=tf.float64,
            shape=[1, 2 * params.decoder_hidden],
            name="word_cell_state"
        )
        zsent_dec_sample = tf.placeholder(
            dtype=tf.float64,
            shape=[1, params.latent_size],
            name="zsent_dec_sample"
        )

        word_embedding = tf.Variable(
            embed_arr,
            trainable=params.fine_tune_embed,
            name="word_embedding",
            dtype=tf.float64
        )
        label_embedding = tf.Variable(
            label_embed_arr,
            trainable=params.fine_tune_embed,
            name="label_embedding",
            dtype=tf.float64
        )

        ## for encoder
        label_inputs = tf.placeholder(
            dtype=tf.int32, shape=[None], name="lablel_inputs"
        )
        word_inputs = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name="word_inputs"
        )
        vect_inputs = tf.nn.embedding_lookup(
            word_embedding, word_inputs, name="word_lookup"
        )
        label_inputs_1 = tf.nn.embedding_lookup(
            label_embedding, label_inputs, name="label_lookup"
        )

        # Zsent_distribution, zsent_sample_enc, Zglobal_distribition, zglobal_sample, enc_word_cell_state = encoder(
        #     vect_inputs, label_inputs_1, batch_size, max_len_word
        # )

        word_logits, label_softmax, Zsent_dec_sample, Zsent_dec_distribution, Zglobal_dec_distribution, _, dec_word_states = decoder(
            d_word_inputs,
            zglobal_sample,
            batch_size,
            word_vocab_size,
            label_vocab_size,
            dec_word_len,
            word_embedding,
            label_input=label_inputs_1,
            z_sent_sample=zsent_dec_sample,
            word_cell_state=word_cell_state,
            gen_mode=True
        )

        saver = tf.train.Saver()
        config = tf.ConfigProto(device_count={'GPU': 0})
        with tf.Session(config=config) as sess:
            sess.run(
                [
                    tf.global_variables_initializer(),
                    tf.local_variables_initializer()
                ]
            )
            try:

                # path = "./models_ckpts_"+params.name+"/vae_lstm_model-11900"
                path = params.ckpt_path
                print('*** Loading checkpoint:', path)
                # chkp.print_tensors_in_checkpoint_file(path, tensor_name='', all_tensors=True)
                saver.restore(sess, path)
                print("*** Model Restored")
            except:
                print("-----exception occurred while loading checkpoints-----")
                exit()
                # traceback.print_exc()

            number_of_samples = params.num_samples
            out_positive = "./generated_positive.txt"
            out_negative = "./generated_negative.txt"

            debug = False
            debug_label = debug or True
            debug_sentence = debug or False
            biased_sampling = False  ##  to also have first generated word same
            # no_word_repetition = False
            # PARA_NUM = 1000

            label_extremes = [0, label_vocab_size - 1]
            pos_sampled = 0
            neg_sampled = 0
            need_pos = True
            need_neg = True

            with open(out_positive,
                      'w+') as pos_f, open(out_negative, 'w+') as neg_f, tqdm(
                          total=2 * number_of_samples
                      ) as pbar:
                while need_neg or need_pos:
                    # got_sentiment = False
                    # while not got_sentiment:

                    ## get sentiment
                    while True:
                        zs_sample = np.zeros((1, params.latent_size))

                        pred_word_idx = data_dict.word2idx[data_dict.bos]
                        zg_sample = np.random.normal(
                            0, 1, (1, params.latent_size)
                        )

                        l_softmax, zs_sample = sess.run(
                            [label_softmax, Zsent_dec_sample],
                            feed_dict={
                                d_word_inputs: [pred_word_idx],
                                zglobal_sample: zg_sample,
                                zsent_dec_sample:
                                    zs_sample,  ## not used for label
                            }
                        )
                        l_softmax = l_softmax[0]

                        if debug_label:
                            print('got sentiment softmax', l_softmax)
                        ## pred sentiment
                        if need_pos and l_softmax[label_extremes[1]] > 0.7:
                            pred_label_idx = label_extremes[1]
                            # got_sentiment = True
                            break

                        if need_neg and l_softmax[label_extremes[0]] > 0.7:
                            pred_label_idx = label_extremes[0]
                            break
                    ## end while

                    ## sentiment obtained, now sample sentence
                    if debug_label:
                        print('got sentment', pred_label_idx)

                    wc_state = np.zeros([1, 2 * params.decoder_hidden])
                    words = []
                    word_sent_len = 0

                    ## till EOS
                    while True:

                        w_logit, wc_state = sess.run(
                            [word_logits, dec_word_states],
                            feed_dict={
                                zsent_dec_sample: zs_sample,
                                label_inputs: [pred_label_idx],
                                d_word_inputs: [pred_word_idx],
                                zglobal_sample: zg_sample,  ## not used for word
                                word_cell_state: wc_state
                            }
                        )

                        w_logit = w_logit[0]

                        ## logit for <BOS> should be zero
                        w_logit[word_bos_index] = 0
                        w_logit[word_pad_index] = 0
                        if word_sent_len == 0:
                            w_logit[word_eos_index] = 0

                        w_softmax = softmax(w_logit)
                        pred_word_idx = np.random.choice(
                            word_vocab_size, size=1, p=w_softmax
                        )[0]
                        pred_word = data_dict.idx2word[pred_word_idx]

                        if debug_sentence:
                            print()
                            print('sent len:', word_sent_len)
                            print(
                                'predicted: {}  {}  {}'.format(
                                    pred_word, pred_word_idx,
                                    softmax(w_logit)[pred_word_idx]
                                )
                            )

                            # argmax
                            wsm = softmax(w_logit)
                            word_argmax_idx = np.argmax(wsm)
                            word_argmax = data_dict.idx2word[word_argmax_idx]
                            print(
                                'argmax: {}  {}  {}'.format(
                                    word_argmax, word_argmax_idx,
                                    wsm[word_argmax_idx]
                                )
                            )

                        if pred_word == data_dict.eos:
                            break
                        words.append(pred_word)
                        word_sent_len += 1

                    if debug_label:
                        print('sentence length', word_sent_len)
                    ## end while True ## word sampling

                    if pred_label_idx == 0:
                        neg_f.write(' '.join(words))
                        neg_f.write('\n')
                        neg_sampled += 1
                        need_neg = neg_sampled < number_of_samples
                    else:
                        pos_f.write(' '.join(words))
                        pos_f.write('\n')
                        pos_sampled += 1
                        need_pos = pos_sampled < number_of_samples

                    # total_sampled+=1
                    pbar.update(1)

                ## end while total_sampled < num_samples


if __name__ == "__main__":
    params = parameters.Parameters()
    params.parse_args()
    main(params)
