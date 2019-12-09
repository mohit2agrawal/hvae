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
from hvae_model1 import encoder, decoder

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
    # data_folder = './DATA/parallel_data_10k/'
    data_folder = './DATA/' + params.name
    # data in form [data, labels]
    train_data_raw, train_label_raw, val_data_raw, val_label_raw = data_.ptb_read(
        data_folder
    )
    word_data, encoder_word_data, word_labels_arr, word_embed_arr, data_dict, label_data, label_labels_arr, label_embed_arr, encoder_val_data, encoder_val_data_shifted, val_labels_arr, decoder_val_words, decoder_val_labels = data_.prepare_data(
        train_data_raw, train_label_raw, val_data_raw, val_label_raw, params,
        data_folder
    )

    encoder_word_data = np.array(encoder_word_data)
    label_labels_arr = np.array(label_labels_arr)

    ## for encoder
    max_sent_len_enc = max(
        max(map(len, word_data)), max(map(len, encoder_val_data))
    )

    ## one word at a time for decoder
    max_sent_len = 1
    ## one sentence at a time
    batch_size = 1

    word_vocab_size = data_dict.vocab_size
    label_vocab_size = data_dict.label_vocab_size
    labels_set = data_dict.labels_set

    label_bos_index = data_dict.l_word2idx[data_dict.bos]
    label_eos_index = data_dict.l_word2idx[data_dict.eos]
    label_pad_index = data_dict.l_word2idx[data_dict.pad]

    word_bos_index = data_dict.word2idx[data_dict.bos]
    word_eos_index = data_dict.word2idx[data_dict.eos]
    word_pad_index = data_dict.word2idx[data_dict.pad]

    sizes = data_dict.sizes
    ranges = []
    i = 0
    s = 0
    for i in range(len(sizes)):
        s += sizes[i]
        ranges.append(s)

    ## ranges stores the index(+1) of the label's last word
    ## sizes are vocab size of the labels
    ##       including size(specials) at [0]
    ## sizes : 4  10   5   7
    ## ranges: 4  14  19  26
    ## but, the labels_set will be of length len(ranges) - 1
    ##      it does not have "specials"

    with tf.Graph().as_default() as graph:

        zglobal_sample = tf.placeholder(
            dtype=tf.float64, shape=[None, params.latent_size]
        )
        d_word_inputs = tf.placeholder(
            dtype=tf.int32, shape=[1], name="d_word_inputs"
        )
        d_label_inputs = tf.placeholder(
            dtype=tf.int32, shape=[1], name="d_label_inputs"
        )

        label_cell_state = tf.placeholder(
            dtype=tf.float64, shape=[1, 2 * 16], name="label_cell_state"
        )
        word_cell_state = tf.placeholder(
            dtype=tf.float64,
            shape=[1, 2 * params.decoder_hidden],
            name="word_cell_state"
        )
        zsent_dec_mu = tf.placeholder(
            dtype=tf.float64,
            shape=[1, params.latent_size],
            name="zsent_dec_mu"
        )
        zsent_dec_logvar = tf.placeholder(
            dtype=tf.float64,
            shape=[1, params.latent_size],
            name="zsent_dec_logvar"
        )
        zsent_dec_sample = tf.placeholder(
            dtype=tf.float64,
            shape=[1, params.latent_size],
            name="zsent_dec_sample"
        )

        word_embedding = tf.Variable(
            word_embed_arr,
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
            dtype=tf.int32, shape=[None, None], name="lable_inputs"
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

        Zsent_distribution, zsent_sample, Zglobal_distribition, zglobal_sample_enc, zsent_state, zglobal_state = encoder(
            vect_inputs, label_inputs_1, batch_size, max_sent_len_enc
        )

        _, _, word_logits, label_logits, zsent_dec_distribution, _, _, _, zsent_dec_sample_out, w_cell_state, l_cell_state = decoder(
            d_word_inputs,
            d_label_inputs,
            zglobal_sample,
            batch_size,
            word_vocab_size,
            label_vocab_size,
            max_sent_len,
            word_embedding,
            label_embedding,
            gen_mode=True,
            label_cell_state=label_cell_state,
            word_cell_state=word_cell_state,
            zsent_dec_mu=zsent_dec_mu,
            zsent_dec_logvar=zsent_dec_logvar,
            zsent_dec_sample=zsent_dec_sample
        )

        saver = tf.train.Saver()
        with tf.Session() as sess:
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
            out_sentence_file = "./generated_sentences_posterior.txt"
            out_labels_file = "./generated_labels_posterior.txt"
            # input_sents_file = "input_sentences.txt"

            biased_sampling = False
            no_word_repetition = False
            PARA_NUM = 100

            with open(out_sentence_file,
                      'w+') as sent_f, open(out_labels_file, 'w+') as label_f:

                for num in tqdm(range(number_of_samples)):

                    sent_l_batch = encoder_word_data[[num]]
                    label_l_batch = label_labels_arr[[num]]
                    sent_l_batch = zero_pad(sent_l_batch, max_sent_len_enc)
                    label_l_batch = zero_pad(label_l_batch, max_sent_len_enc)

                    zg_dist = sess.run(
                        Zglobal_distribition,
                        feed_dict={
                            word_inputs: sent_l_batch,
                            label_inputs: label_l_batch,
                        }
                    )

                    zg_mu, zg_logvar = zg_dist
                    zg_mu = zg_mu[0]
                    zg_logvar = zg_logvar[0]

                    ## for biasing in sampling
                    first_word = sent_l_batch[0][0]
                    first_label = label_l_batch[0][0]

                    ## print the sentence being paraphrased
                    encoder_sent = list(
                        map(lambda x: data_dict.idx2word[x], sent_l_batch[0])
                    )
                    sent_f.write(
                        ' '.join(
                            encoder_sent[:encoder_sent.index(data_dict.eos)]
                        )
                    )
                    sent_f.write('\n')

                    encoder_labels = list(
                        map(
                            lambda x: data_dict.l_idx2word[x], label_l_batch[0]
                        )
                    )
                    label_f.write(
                        ' '.join(
                            encoder_labels[:encoder_labels.index(data_dict.eos)]
                        )
                    )
                    label_f.write('\n')

                    for para_num in range(PARA_NUM):
                        ## for decoder init

                        eps = np.random.normal(size=np.shape(zg_logvar))
                        z = zg_mu + np.exp(0.5 * zg_logvar) * eps
                        z = np.reshape(z, [1, params.latent_size])

                        # z = np.random.normal(0, 1, (1, params.latent_size))
                        lc_state = np.zeros([1, 2 * 16])
                        wc_state = np.zeros([1, 2 * params.decoder_hidden])

                        zs_mu = np.zeros(
                            [1, params.latent_size], dtype=np.float64
                        )
                        zs_logvar = np.zeros(
                            [1, params.latent_size], dtype=np.float64
                        )
                        zs_sample = np.zeros(
                            [1, params.latent_size], dtype=np.float64
                        )

                        pred_word_idx = data_dict.word2idx[data_dict.bos]
                        pred_label_idx = data_dict.l_word2idx[data_dict.bos]

                        words = []
                        labels = []

                        sent_len = 0

                        if no_word_repetition:
                            appeared_words = dict()
                            for lbl in labels_set:
                                ls_idx = labels_set.index(lbl)
                                appeared_words[lbl] = np.zeros(
                                    sizes[ls_idx + 1]
                                )

                        ## sample until <EOS>
                        while True:
                            w_logit, l_logit, wc_state, lc_state, zs_dist, zs_sample = sess.run(
                                [
                                    word_logits, label_logits, w_cell_state,
                                    l_cell_state, zsent_dec_distribution,
                                    zsent_dec_sample_out
                                ],
                                feed_dict={
                                    zglobal_sample: z,
                                    d_word_inputs: [pred_word_idx],
                                    d_label_inputs: [pred_label_idx],
                                    label_cell_state: lc_state,
                                    word_cell_state: wc_state,
                                    zsent_dec_sample: zs_sample,
                                    zsent_dec_mu: zs_mu,
                                    zsent_dec_logvar: zs_logvar,
                                }
                            )

                            l_logit = l_logit[0]
                            w_logit = w_logit[0]

                            zs_mu, zs_logvar = zs_dist[0], zs_dist[1]

                            ## logit for <BOS> should be zero
                            l_logit[label_bos_index] = 0
                            l_logit[label_pad_index] = 0
                            if sent_len == 0:
                                l_logit[label_eos_index] = 0

                            ## biased sampling
                            ## sample first label only from ['NOUN', 'DET']
                            # if biased_sampling and sent_len == 0:
                            #     start_labels = ['NOUN', 'DET']
                            #     # start_labels = [first_word_label]
                            #     start_label_idxs = [
                            #         data_dict.l_word2idx[l]
                            #         for l in start_labels
                            #     ]
                            #     start_logit = l_logit[start_label_idxs]
                            #     start_softmax = softmax(start_logit)

                            ## calc softmax
                            l_softmax = softmax(l_logit)

                            ## biased sampling (contd...)
                            # if biased_sampling and sent_len == 0:
                            #     l_softmax[:] = 0
                            #     l_softmax[start_label_idxs] = start_softmax

                            pred_label_idx = np.random.choice(
                                label_vocab_size, size=1, p=l_softmax
                            )[0]
                            pred_label = data_dict.l_idx2word[pred_label_idx]

                            if pred_label == data_dict.bos:
                                print(label_bos_index)
                                print(l_logit)
                                print(l_softmax)
                                print(pred_label_idx)

                            if pred_label in data_dict.specials:
                                pred_word = pred_label
                            else:
                                # w_logit[word_bos_index] = 0
                                # w_logit[word_eos_index] = 0
                                # w_logit[word_pad_index] = 0
                                # pred_word_idx = np.random.choice(
                                #     word_vocab_size, size=1, p=softmax(w_logit)
                                # )[0]
                                # pred_word = data_dict.idx2word[pred_word_idx]

                                ## given a label
                                ## will have a range [idx1, idx2]
                                ## in which words can fall

                                ## labels_set index
                                ls_idx = labels_set.index(pred_label)
                                ## sample the word
                                pred_word_idx = np.random.choice(
                                    range(ranges[ls_idx], ranges[ls_idx + 1]),
                                    size=1,
                                    p=softmax(
                                        w_logit[ranges[ls_idx]:ranges[ls_idx +
                                                                      1]]
                                    )
                                )[0]
                                pred_word = data_dict.idx2word[pred_word_idx]

                                # if no_word_repetition:
                                #     shifted_idx = pred_word_idx - ranges[ls_idx]
                                #     appeared_words[pred_label][shifted_idx] = 1
                                #     ## in case all have appeared, mark all as not appeared
                                #     if not any(appeared_words[pred_label] == 0):
                                #         appeared_words[pred_label][:] = 0
                                #     # if sum(appeared_words[pred_label] == 1
                                #     #        ) == len(appeared_words[pred_label]):
                                #     #     appeared_words[pred_label][:] = 0

                            if biased_sampling and sent_len == 0:
                                pred_label_idx = first_label
                                pred_word_idx = first_word

                                pred_label = data_dict.l_idx2word[pred_label_idx
                                                                  ]
                                pred_word = data_dict.idx2word[pred_word_idx]

                            if pred_label == data_dict.eos:
                                break
                            labels.append(pred_label)
                            words.append(pred_word)
                            sent_len += 1

                        sent_f.write(' '.join(words))
                        sent_f.write('\n')
                        label_f.write(' '.join(labels))
                        label_f.write('\n')

                    ## separate paraphrases by a newline
                    sent_f.write('\n')
                    label_f.write('\n')


if __name__ == "__main__":
    params = parameters.Parameters()
    params.parse_args()
    main(params)