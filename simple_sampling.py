from __future__ import print_function, division, absolute_import
import tensorflow as tf
import numpy as np

# import zhusuan as zs
# from zhusuan import reuse

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
from hvae_model import encoder, decoder

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


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        return np.exp(x) / np.sum(np.exp(x))
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def main(params):
    # data_folder = './DATA/parallel_data_10k/'
    data_folder = {
        'ptb': './DATA/ptb/',
        'ptb_ner': './DATA/ptb_ner'
    }.get(params.name)
    # data in form [data, labels]
    train_data_raw, train_label_raw, val_data_raw, val_label_raw = data_.ptb_read(
        data_folder
    )
    word_data, encoder_word_data, word_labels_arr, word_embed_arr, word_data_dict, encoder_val_data = data_.prepare_data(
        train_data_raw, train_label_raw, val_data_raw, val_label_raw, params,
        data_folder
    )

    train_label_raw, val_label_raw, test_label_raw = label_data_.ptb_read(
        data_folder
    )
    label_data, label_labels_arr, label_embed_arr, label_data_dict, val_labels_arr = label_data_.prepare_data(
        train_label_raw, val_label_raw, params
    )

    max_sent_len = max(
        max(map(len, word_data)), max(map(len, encoder_word_data))
    )

    with tf.Graph().as_default() as graph:
        word_vocab_size = max(word_data_dict.sizes)
        label_vocab_size = label_data_dict.vocab_size

        zglobal_sample = tf.placeholder(
            dtype=tf.float64, shape=[None, params.latent_size]
        )

        word_logits, label_logits, _, _, _, _, _, _, _ = decoder(
            zglobal_sample,
            1,  # batch_size
            word_vocab_size,
            label_vocab_size,
            max_sent_len,
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

            batch_size = 1
            number_of_samples = params.num_samples
            same_context_sentences = 1
            out_sentence_file = "./generated_sentences.txt"
            out_labels_file = "./generated_labels.txt"

            with open(out_sentence_file,
                      'w+') as sent_f, open(out_labels_file, 'w+') as label_f:

                for num in tqdm(range(number_of_samples)):
                    params.is_training = False

                    z = np.random.normal(0, 1, (1, params.latent_size))
                    w_logits, l_logits = sess.run(
                        [word_logits, label_logits],
                        feed_dict={zglobal_sample: z}
                    )
                    w_logits = w_logits.reshape((max_sent_len, word_vocab_size))
                    l_logits = l_logits.reshape(
                        (max_sent_len, label_vocab_size)
                    )

                    # w_softmax = softmax(w_logits)
                    l_softmax = softmax(l_logits)

                    labels_idx = [
                        np.random.choice(label_vocab_size, size=1, p=smax)[0]
                        for smax in l_softmax
                    ]
                    labels = [label_data_dict.idx2word[i] for i in labels_idx]

                    sizes = word_data_dict.sizes
                    b1 = sizes[0]
                    b2 = sizes[0] + sizes[1]
                    b3 = sizes[0] + sizes[1] + sizes[2]
                    b4 = sizes[0] + sizes[1] + sizes[2] + sizes[3]
                    b5 = sizes[0] + sizes[1] + sizes[2] + sizes[3] + sizes[4]

                    # words_idx = []
                    words = []
                    for i, label in enumerate(labels):
                        if label == '0':  # other(O)
                            word_idx = np.random.choice(
                                range(b1, b2),
                                size=1,
                                p=softmax(w_logits[i][:b2 - b1])
                            )[0]
                        elif label == '1':  # LOCATION
                            word_idx = np.random.choice(
                                range(b2, b3),
                                size=1,
                                p=softmax(w_logits[i][:b3 - b2])
                            )[0]
                        elif label == '2':  # PERSON
                            word_idx = np.random.choice(
                                range(b3, b4),
                                size=1,
                                p=softmax(w_logits[i][:b4 - b3])
                            )[0]
                        elif label == '3':  # ORGANIZATION
                            word_idx = np.random.choice(
                                range(b4, b5),
                                size=1,
                                p=softmax(w_logits[i][:b5 - b4])
                            )[0]
                        elif label == '4':
                            words.append('<BOS>')
                            continue
                        elif label == '5':
                            words.append('<EOS>')
                            continue
                        elif label == '6':
                            words.append('<UNK>')
                            continue
                        elif label == '7':
                            words.append('<PAD>')
                            continue
                        else:
                            print('got unwanted label:', label)
                            words.append('UNK')
                            continue

                        # words_idx.append(word_idx)
                        words.append(word_data_dict.idx2word[word_idx])

                    sent_f.write(' '.join(words))
                    sent_f.write('\n')
                    label_f.write(' '.join(labels))
                    label_f.write('\n')


if __name__ == "__main__":
    params = parameters.Parameters()
    params.parse_args()
    main(params)
