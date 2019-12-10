import os
import tqdm
import nltk
import multiprocessing
import pickle
import numpy as np
from collections import defaultdict, Counter
from utils import parameters
from fasttext import FastVector

params = parameters.Parameters()


def _read_file(corpus_file, sent_file):
    if os.path.exists(sent_file):
        print("Loading sentences file")
        with open(sent_file, 'r') as rf:
            sentences = pickle.load(file=rf)
        return sentences

    if not os.path.exists("./trained_embeddings_" + params.name):
        os.makedirs("./trained_embeddings_" + params.name)
    sentences = []
    with open(corpus_file) as rf:
        for line in rf:
            sentences.append(['<BOS>'] + line.strip().split(' ') + ['<EOS>'])
    with open(sent_file, 'w') as wf:
        pickle.dump(sentences, file=wf)
    return sentences


def _read_file_labels(corpus_file, sent_file):
    if os.path.exists(sent_file):
        print("Loading sentences file")
        with open(sent_file, 'r') as rf:
            sentences = pickle.load(file=rf)
        return sentences

    if not os.path.exists("./trained_embeddings_" + params.name):
        os.makedirs("./trained_embeddings_" + params.name)
    sentences = []
    with open(corpus_file) as rf:
        for line in rf:
            sentences.append(line.strip())
    with open(sent_file, 'w') as wf:
        pickle.dump(sentences, file=wf)
    return sentences


def read_data(data_path):
    sentences_data = _read_file(
        os.path.join(data_path, 'data.txt'),
        sent_file="./trained_embeddings_" + params.name + "/sentences.pickle"
    )

    label_data = _read_file_labels(
        os.path.join(data_path, 'labels.txt'),
        sent_file="./trained_embeddings_" + params.name + "/labels.pickle"
    )

    val_data = _read_file(
        os.path.join(data_path, 'val_data.txt'),
        sent_file="./trained_embeddings_" + params.name +
        "/val_sentences.pickle"
    )

    val_label_data = _read_file_labels(
        os.path.join(data_path, 'val_labels.txt'),
        sent_file="./trained_embeddings_" + params.name + "/val_labels.pickle"
    )

    vocab = _read_file_labels(
        os.path.join(data_path, 'vocab'),
        sent_file="./trained_embeddings_" + params.name + "/vocab.pickle"
    )

    return sentences_data, label_data, val_data, val_label_data, vocab


class Dictionary(object):
    def __init__(self, sentences, val_sentences, vocab):
        # sentences - array of sentences
        self.unk = "<UNK>"
        self.bos = "<BOS>"
        self.eos = "<EOS>"
        self.pad = "<PAD>"
        ## keep pad as the first element, word2idx['PAD'] := 0
        self.specials = [self.pad, self.bos, self.eos, self.unk]
        self.specials_ = self.specials + [x.lower() for x in self.specials]

        ## add specials to vocab
        self._vocab = self.specials + [x.lower() for x in vocab]

        self._sentences = sentences
        self._val_sentences = val_sentences

        self._word2idx = dict(zip(self._vocab, range(len(self._vocab))))
        self._idx2word = dict(zip(range(len(self._vocab)), self._vocab))
        assert self._idx2word[0] == self.pad
        assert self._word2idx[self.pad] == 0

        # self._words.append('<unk>')
        ## modify sentences to mark the dropped words as UNK
        self._mod_sentences(self._sentences)
        self._mod_sentences(self._val_sentences)

    @property
    def vocab_size(self):
        return len(self._idx2word)

    @property
    def vocab(self):
        return self._vocab

    @property
    def sentences(self):
        return self._sentences

    @property
    def val_sentences(self):
        return self._val_sentences

    @property
    def word2idx(self):
        return self._word2idx

    @property
    def idx2word(self):
        return self._idx2word

    def __len__(self):
        return len(self.idx2word)

    def seq2dx(self, sentence):
        return [self.word2idx[wd] for wd in sentence]

    def _mod_sentences(self, sentences):
        # for every sentence, if word not in vocab set to <UNK>
        for i, sent in enumerate(sentences):
            for j in range(len(sent)):
                if sent[j] in self.specials_:  # + ['N', 'n']:
                    sent[j] = sent[j].upper()
                else:
                    sent[j] = sent[j].lower()
                if sent[j] not in self._vocab:
                    sent[j] = self.unk
        return


def save_data(sentences, pkl_file, text_file):

    with open(pkl_file, 'w') as wf:
        pickle.dump(sentences, file=wf)

    with open(text_file, 'w') as wf:
        for sent in sentences:
            line = ' '.join(
                [word for word in sent if word not in ['<BOS>', '<EOS>']]
            )
            wf.write(line)
            wf.write("\n")


def prepare_data(
    data_raw, labels_raw, val_data_raw, val_labels_raw, vocab, params, data_path
):
    # get embeddings, prepare data
    print("building dictionary")
    data_dict = Dictionary(data_raw, val_data_raw, vocab)
    # label_dict = Dictionary(labels_raw, val_labels_raw, params.vocab_drop)

    save_data(
        data_dict.sentences,
        "./trained_embeddings_" + params.name + "/sentences_mod.pickle",
        os.path.join(data_path, 'data_mod.txt')
    )
    save_data(
        data_dict.val_sentences,
        "./trained_embeddings_" + params.name + "/val_sentences_mod.pickle",
        os.path.join(data_path, 'val_data_mod.txt')
    )
    # save_data(
    #     label_dict.sentences,
    #     "./trained_embeddings_" + params.name + "/labels_mod.pickle",
    #     os.path.join(data_path, 'labels_mod.txt')
    # )
    # save_data(
    #     label_dict.val_sentences,
    #     "./trained_embeddings_" + params.name + "/val_labels_mod.pickle",
    #     os.path.join(data_path, 'val_labels_mod.txt')
    # )

    model_path = "./trained_embeddings_" + params.name
    filename = os.path.join(model_path, "embedding_file.pkl")
    # label_filename = os.path.join(model_path, "embedding_file.label.pkl")

    if os.path.exists(filename):
        with open(filename, 'r') as rf:
            embed_arr = pickle.load(rf)

    else:
        en_align_dictionary = FastVector(
            vector_file='updated.yelp.embed.10epochs.vec'
        )
        print("loaded the files..")

        embed_arr = np.zeros([data_dict.vocab_size, params.embed_size])
        for i in range(1, embed_arr.shape[0]):
            # print(i)
            try:
                embed_arr[i] = en_align_dictionary[data_dict.idx2word[i]]
                # print(str(i), "english")
            except:
                embed_arr[i] = en_align_dictionary["<UNK>"]
                print("using <UNK>'s embedding for", data_dict.idx2word[i], i)

        print("Embedding created")
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        with open(filename, 'w') as wf:
            pickle.dump(embed_arr, wf)

    # if os.path.exists(label_filename):
    #     with open(label_filename, 'r') as rf:
    #         label_embed_arr = pickle.load(rf)

    # else:
    #     label_en_align_dictionary = FastVector(
    #         vector_file='w2v.labels.tree.embed'
    #     )
    #     print("loaded the files..")

    #     label_embed_arr = np.zeros([label_dict.vocab_size, 32])
    #     for i in range(1, label_embed_arr.shape[0]):
    #         # print(i)
    #         try:
    #             label_embed_arr[i] = label_en_align_dictionary[
    #                 label_dict.idx2word[i]]
    #             # print(str(i), "english")
    #         except:
    #             label_embed_arr[i] = label_en_align_dictionary["<UNK>"]
    #             print(
    #                 i, label_dict.idx2word[i], ": set to embedding of \"<UNK>\""
    #             )

    #     print("Embedding created")
    #     if not os.path.exists(model_path):
    #         os.makedirs(model_path)

    #     with open(label_filename, 'w') as wf:
    #         pickle.dump(label_embed_arr, wf)

    data = [
        [data_dict.word2idx[word] for word in sent[:-1]]
        for sent in data_dict.sentences
        # if len(sent) < params.sent_max_size - 2
    ]

    encoder_data = [
        [data_dict.word2idx[word] for word in sent[1:]]
        for sent in data_dict.sentences
        # if len(sent) < params.sent_max_size - 2
    ]

    val_data = [
        [data_dict.word2idx[word] for word in sent[:-1]]
        for sent in data_dict.val_sentences
        # if len(sent) < params.sent_max_size - 2
    ]

    encoder_val_data = [
        [data_dict.word2idx[word] for word in sent[1:]]
        for sent in data_dict.val_sentences
        # if len(sent) < params.sent_max_size - 2
    ]

    ## for LABELS
    labels_set = set(labels_raw + val_labels_raw)
    print('labels set', labels_set)
    label_embed_arr = np.eye(len(labels_set))
    # labels = [
    #     [label_dict.word2idx[word] for word in sent[:-1]]
    #     for sent in label_dict.sentences
    #     # if len(sent) < params.sent_max_size - 2
    # ]
    # encoder_labels = [
    #     [label_dict.word2idx[word] for word in sent[1:]]
    #     for sent in label_dict.sentences
    #     # if len(sent) < params.sent_max_size - 2
    # ]

    # val_labels = [
    #     [label_dict.word2idx[word] for word in sent[:-1]]
    #     for sent in label_dict.val_sentences
    #     # if len(sent) < params.sent_max_size - 2
    # ]
    # encoder_val_labels = [
    #     [label_dict.word2idx[word] for word in sent[1:]]
    #     for sent in label_dict.val_sentences
    #     # if len(sent) < params.sent_max_size - 2
    # ]

    print(
        "----Corpus_Information--- \n "
        "Raw data size: {} sentences \n Vocabulary size {}"
        "\n Limited data size {} sentences \n"
        "Label levels: {}\n".format(
            len(data_raw), data_dict.vocab_size, len(data), len(labels_set)
        )
    )

    return data, encoder_data, val_data, encoder_val_data, embed_arr, data_dict, labels_raw, val_labels_raw, label_embed_arr
