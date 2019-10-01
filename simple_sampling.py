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
    data_folder = './DATA/' + params.name
    # data in form [data, labels]
    train_data_raw, train_label_raw = data_.ptb_read(data_folder)
    word_data, encoder_word_data, word_labels_arr, word_embed_arr, data_dict, label_data, label_labels_arr, label_embed_arr, decoder_words, decoder_labels = data_.prepare_data(
        train_data_raw, train_label_raw, params, data_folder
    )

    max_sent_len = max(map(len, word_data))

    d_word_inputs = tf.placeholder(
        dtype=tf.int32, shape=[None, None], name="d_word_inputs"
    )
    d_label_inputs = tf.placeholder(
        dtype=tf.int32, shape=[None, None], name="d_label_inputs"
    )

    class_vocab_sizes = [1] * data_dict.sizes[0] + data_dict.sizes[1:]

    with tf.Graph().as_default() as graph:
        word_vocab_size = max(data_dict.sizes)
        label_vocab_size = data_dict.label_vocab_size

        zglobal_sample = tf.placeholder(
            dtype=tf.float64, shape=[None, params.latent_size]
        )
        word_embedding = tf.Variable(
            word_embed_arr,
            trainable=params.fine_tune_embed,
            name="word_embedding",
            dtype=tf.float64
        )  # creates a variable that can be used as a tensor
        label_embedding = tf.Variable(
            label_embed_arr,
            trainable=params.fine_tune_embed,
            name="label_embedding",
            dtype=tf.float64
        )

        _, _, _, _, _, _, _, _, _, pred_label_indices, pred_word_indices = decoder(
            d_word_inputs,
            d_label_inputs,
            zglobal_sample,
            1,  # batch_size
            word_vocab_size,
            label_vocab_size,
            max_sent_len,
            word_embedding,
            label_embedding,
            class_vocab_sizes=class_vocab_sizes,
            label_bos_idx=data_dict.l_word2idx[data_dict.bos],
            word_bos_idx=data_dict.word2idx[data_dict.bos],
            gen_mode=True,
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

            batch_size = 1
            number_of_samples = params.num_samples
            out_sentence_file = "./generated_sentences.txt"
            out_labels_file = "./generated_labels.txt"

            with open(out_sentence_file,
                      'w+') as sent_f, open(out_labels_file, 'w+') as label_f:

                for num in tqdm(range(number_of_samples)):
                    params.is_training = False

                    z = np.random.normal(0, 1, (1, params.latent_size))
                    l_indices, w_indices = sess.run(
                        [pred_label_indices, pred_word_indices],
                        feed_dict={zglobal_sample: z}
                    )

                    labels = [data_dict.l_idx2word[i] for i in l_indices]
                    words = [data_dict.idx2word[i] for i in l_indices]

                    sent_f.write(' '.join(words))
                    sent_f.write('\n')
                    label_f.write(' '.join(labels))
                    label_f.write('\n')


if __name__ == "__main__":
    params = parameters.Parameters()
    params.parse_args()
    main(params)
