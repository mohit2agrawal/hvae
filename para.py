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
from sivae_model import encoder, decoder

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
    train_data_raw, train_label_raw, val_data_raw, val_label_raw = data_.read_data(
        data_folder
    )
    data, encoder_data, val_data, encoder_val_data, word_embed_arr, data_dict, labels, encoder_labels, val_labels, encoder_val_labels, label_embed_arr, label_dict = data_.prepare_data(
        train_data_raw, train_label_raw, val_data_raw, val_label_raw, params,
        data_folder
    )

    print(label_dict.word2idx)
    # exit()

    encoder_data = np.array(encoder_data)
    encoder_labels = np.array(encoder_labels)

    ## for encoder
    max_len_word = max(max(map(len, data)), max(map(len, val_data)))
    max_len_label = max(max(map(len, labels)), max(map(len, val_labels)))

    print('max_len_label', max_len_label)
    print('max_len_word', max_len_word)

    ## one word at a time for decoder
    dec_word_len = 1
    dec_label_len = 1
    ## one sentence at a time
    batch_size = 1

    word_vocab_size = data_dict.vocab_size
    label_vocab_size = label_dict.vocab_size

    label_bos_index = label_dict.word2idx[label_dict.bos]
    label_eos_index = label_dict.word2idx[label_dict.eos]
    label_pad_index = label_dict.word2idx[label_dict.pad]
    label_unk_index = label_dict.word2idx[label_dict.unk]

    word_bos_index = data_dict.word2idx[data_dict.bos]
    word_eos_index = data_dict.word2idx[data_dict.eos]
    word_pad_index = data_dict.word2idx[data_dict.pad]
    word_unk_index = data_dict.word2idx[data_dict.unk]

    with tf.Graph().as_default() as graph:

        zglobal_sample = tf.placeholder(
            dtype=tf.float64, shape=[None, params.latent_size]
        )
        zsent_sample = tf.placeholder(
            dtype=tf.float64, shape=[None, params.latent_size]
        )
        d_word_inputs = tf.placeholder(
            dtype=tf.int32, shape=[1], name="d_word_inputs"
        )
        d_label_inputs = tf.placeholder(
            dtype=tf.int32, shape=[1], name="d_label_inputs"
        )

        label_cell_state = tf.placeholder(
            dtype=tf.float64,
            shape=[1, 2 * params.decoder_hidden],
            name="label_cell_state"
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
        d_seq_length_label = tf.placeholder(shape=[None], dtype=tf.float64)
        d_seq_length_word = tf.placeholder(shape=[None], dtype=tf.float64)

        Zsent_distribution, zsent_sample_enc, Zglobal_distribition, zglobal_sample_enc, zsent_state, zglobal_state = encoder(
            vect_inputs, label_inputs_1, batch_size, d_seq_length_word,
            d_seq_length_label
        )

        word_logits, label_logits, Zsent_dec_distribution, Zglobal_dec_distribution, _, _, w_cell_state, l_cell_state = decoder(
            d_word_inputs,
            d_label_inputs,
            zglobal_sample,
            zsent_sample,
            batch_size,
            word_vocab_size,
            label_vocab_size,
            dec_word_len,
            dec_label_len,
            word_embedding,
            label_embedding,
            gen_mode=True,
            label_cell_state=label_cell_state,
            word_cell_state=word_cell_state,
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

            # number_of_samples = params.num_samples
            out_sentence_file = "./generated_sentences_test.txt"
            out_labels_file = "./generated_labels_test.txt"
            input_sents_file = "input_sentences_test.txt"
            input_labels_file = "input_templates_test.txt"
            NUM_TRIES=100

            sent_num = 0

            biased_sampling = False  ##  to also have first generated word same
            # no_word_repetition = False
            # PARA_NUM = 1000

            with open(out_sentence_file, 'w+') as sent_f, open(
                out_labels_file, 'w+'
            ) as label_f, open(input_sents_file
                               ) as inp_sf, open(input_labels_file) as inp_lf:

                for inp_sent in inp_sf:
                    sent_num+=1

                    inp_sent = inp_sent.strip().split(' ') + [data_dict.eos]
                    inp_sent_enc = [
                        data_dict.word2idx.get(x, word_unk_index)
                        for x in inp_sent
                    ]
                    print('sent num', sent_num)
                    print('got sent:', inp_sent)

                    sent_l_batch = [inp_sent_enc]
                    sent_len_ = [len(x) for x in sent_l_batch]
                    sent_l_batch = zero_pad(sent_l_batch, max_len_word)
                    ## print the sentence being paraphrased
                    # _encoder_sent = list(
                    #     map(lambda x: data_dict.idx2word[x], sent_l_batch[0])
                    # )
                    # sent_f.write(
                    #     ' '.join(
                    #         _encoder_sent[:_encoder_sent.index(data_dict.eos)]
                    #     )
                    # )
                    # sent_f.write('\n')

                    for template in inp_lf:
                        if not template or not template.strip(
                        ):  ## got a new line
                            break
                        template = template.lower().strip().split(' ') + [
                            label_dict.eos
                        ]
                        # template_enc = [label_dict.word2idx[x] for x in template]
                        print('\tgot template:', template)
                        try:
                            template_enc = [
                                label_dict.word2idx.get(x, label_unk_index)
                                for x in template
                            ]
                        except KeyError as e:
                            print(e)
                            continue

                        label_l_batch = [template_enc]
                        label_len_ = [len(x) for x in label_l_batch]
                        label_l_batch = zero_pad(label_l_batch, max_len_label)

                        zs_dist, zg_dist = sess.run(
                            [Zsent_distribution, Zglobal_distribition],
                            feed_dict={
                                word_inputs: sent_l_batch,
                                label_inputs: label_l_batch,
                                d_seq_length_word: sent_len_,
                                d_seq_length_label: label_len_
                            }
                        )
                        zs_mu, zs_logvar = zs_dist[0], zs_dist[1]
                        zg_mu, zg_logvar = zg_dist[0], zg_dist[1]

                        # zg_sample = zg_sample[0]
                        # zg_sample = np.reshape(
                        #     zg_sample, (1, params.latent_size)
                        # )

                        # zs_sample = zs_sample[0]
                        # zs_sample = np.reshape(
                        #     zs_sample, (1, params.latent_size)
                        # )

                        ## for biasing in sampling
                        first_word = sent_l_batch[0][0]
                        first_label = label_l_batch[0][0]

                        # print('\nobtained z values from encoder')

                        # _encoder_labels = list(
                        #     map(
                        #         lambda x: label_dict.idx2word[x],
                        #         label_l_batch[0]
                        #     )
                        # )
                        ## write the label template also to sent file
                        # sent_f.write(
                        #     ' '.join(
                        #         _encoder_labels[:_encoder_labels.
                        #                         index(label_dict.eos)]
                        #     )
                        # )
                        # sent_f.write('\n')

                        # print('\ngenerating paraphrases')

                        # for para_num in range(PARA_NUM):
                        ## for decoder init

                        # z = np.random.normal(0, 1, (1, params.latent_size))


                        # zg_sample = np.loadtxt('data/zg_sample')
                        # zg_sample = np.reshape(
                        #     zg_sample, (1, params.latent_size)
                        # )

                        # zs_sample = np.loadtxt('data/zs_sample')
                        # zs_sample = np.reshape(
                        #     zs_sample, (1, params.latent_size)
                        # )

                        for try_idx in tqdm(range(NUM_TRIES)):

                            eps = np.random.normal(size=np.shape(zg_logvar))
                            zg_sample = zg_mu + np.exp(0.5 * zg_logvar) * eps
                            zg_sample = np.reshape(zg_sample, [1, params.latent_size])

                            eps = np.random.normal(size=np.shape(zs_logvar))
                            zs_sample = zs_mu + np.exp(0.5 * zs_logvar) * eps
                            zs_sample = np.reshape(zs_sample, [1, params.latent_size])

                            lc_state = np.zeros([1, 2 * params.decoder_hidden])
                            pred_label_idx = label_dict.word2idx[label_dict.bos]
                            labels = []
                            label_sent_len = 0

                            pred_word_idx = data_dict.word2idx[data_dict.bos]
                            ## label sampling to get the last label_cell_state
                            ## sample until <EOS>
                            # print('doing label sampling')
                            while label_sent_len<max_len_label:
                                l_logit, lc_state = sess.run(
                                    [label_logits, l_cell_state],
                                    feed_dict={
                                        zglobal_sample: zg_sample,
                                        zsent_sample: zs_sample,
                                        d_word_inputs: [pred_word_idx],
                                        d_label_inputs: [pred_label_idx],
                                        label_cell_state: lc_state,
                                    }
                                )

                                l_logit = l_logit[0]

                                ## logit for <BOS> should be zero
                                l_logit[label_bos_index] = 0
                                l_logit[label_pad_index] = 0
                                # if sent_len == 0:
                                #     l_logit[label_eos_index] = 0
                                l_softmax = softmax(l_logit)

                                # pred_label_idx = np.random.choice(
                                #     label_vocab_size, size=1, p=l_softmax
                                # )[0]

                                # if label_sent_len < len(template):
                                #     pred_label_idx = template_enc[label_sent_len]
                                pred_label_idx = label_l_batch[0][label_sent_len]

                                pred_label = label_dict.idx2word[pred_label_idx]
                                # print()
                                # print('label_sent_len', label_sent_len)
                                # print('pred_label_idx', pred_label_idx)
                                # print('pred_label', pred_label)
                                # if label_sent_len < len(template):
                                #     print(
                                #         'template[label_sent_len]',
                                #         template[label_sent_len]
                                #     )
                                # print(
                                #     'label argmax',
                                #     label_dict.idx2word[np.argmax(l_softmax)]
                                # )
                                #assert pred_label == template[label_sent_len]

                                # if pred_label == label_dict.eos:
                                #     break

                                labels.append(pred_label)
                                label_sent_len += 1
                            # print('sampled labels:', ' '.join(labels))

                            wc_state = np.zeros([1, 2 * params.decoder_hidden])
                            pred_word_idx = data_dict.word2idx[data_dict.bos]
                            words = []
                            word_sent_len = 0

                            # np.savetxt('data_sampled/last_label_state', lc_state[0])

                            ## word sampling
                            ## sample until <EOS>
                            # print('doing word sampling')
                            while True:
                                w_logit, wc_state = sess.run(
                                    [word_logits, w_cell_state],
                                    feed_dict={
                                        zglobal_sample: zg_sample,
                                        zsent_sample: zs_sample,
                                        d_word_inputs: [pred_word_idx],
                                        d_label_inputs: [pred_label_idx],
                                        label_cell_state: lc_state,
                                        word_cell_state: wc_state,
                                    }
                                )

                                # np.savetxt('data_sampled/word_state_'+str(word_sent_len), wc_state[0])
                                # np.savetxt('data_sampled/word_logit_'+str(word_sent_len), w_logit[0])

                                w_logit = w_logit[0]


                                ## logit for <BOS> should be zero
                                # w_logit[word_bos_index] = 0
                                # w_logit[word_pad_index] = 0
                                if word_sent_len == 0:
                                    w_logit[word_eos_index] = 0
                                w_softmax = softmax(w_logit)

                                # pred_word_idx = np.random.choice(
                                #     word_vocab_size, size=1, p=w_softmax
                                # )[0]
                                pred_word_idx = np.argmax(w_softmax)

                                # if word_sent_len < len(sent_l_batch[0]):
                                #     pred_word_idx = sent_l_batch[0][word_sent_len]

                                # if biased_sampling and word_sent_len == 0:
                                #     pred_word_idx = first_word
                                # if word_sent_len == 0:
                                #     pred_word_idx = first_word

                                pred_word = data_dict.idx2word[pred_word_idx]

                                # print()
                                # print('sent len:', word_sent_len)
                                # if len(sent_l_batch[0]) > word_sent_len:
                                #     orig_word_idx = sent_l_batch[0][word_sent_len]
                                #     orig_word = data_dict.idx2word[orig_word_idx]
                                #     orig_word_logit = softmax(w_logit
                                #                               )[orig_word_idx]

                                #     as_sm = list(np.argsort(w_softmax))[::-1]
                                #     rank = as_sm.index(orig_word_idx)
                                #     print(
                                #         'original: {}  {}  {} rank:{}'.format(
                                #             orig_word, orig_word_idx,
                                #             orig_word_logit, rank
                                #         )
                                #     )

                                # print(
                                #     'predicted: {}  {}  {}'.format(
                                #         pred_word, pred_word_idx,
                                #         softmax(w_logit)[pred_word_idx]
                                #     )
                                # )

                                # argmax
                                # wsm = softmax(w_logit)
                                # word_argmax_idx = np.argmax(wsm)
                                # word_argmax = data_dict.idx2word[word_argmax_idx]
                                # print(
                                #     'argmax: {}  {}  {}'.format(
                                #         word_argmax, word_argmax_idx,
                                #         wsm[word_argmax_idx]
                                #     )
                                # )

                                if pred_word == data_dict.eos:
                                    break
                                words.append(pred_word)

                                # if word_sent_len < len(sent_l_batch[0]):
                                #     pred_word_idx = sent_l_batch[0][word_sent_len]

                                word_sent_len += 1

                            sent_f.write(' '.join(words))
                            sent_f.write('\n')
                            ## labels are not being predicted, just used for history
                            # label_f.write(' '.join(labels))
                            # label_f.write('\n')

                        ## end of one trial
                        sent_f.write('\n')


                    ## separate paraphrases by a newline
                    # sent_f.write('\n')
                    # label_f.write('\n')


if __name__ == "__main__":
    params = parameters.Parameters()
    params.parse_args()
    main(params)
