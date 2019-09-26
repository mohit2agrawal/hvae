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


def ptb_data_read(corpus_file, sent_file):
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


def ptb_read(data_path):
    sentences_data = ptb_data_read(
        os.path.join(data_path, 'data.txt'),
        sent_file="./trained_embeddings_" + params.name + "/sentences.pickle"
    )

    label_data = ptb_data_read(
        os.path.join(data_path, 'labels.txt'),
        sent_file="./trained_embeddings_" + params.name + "/labels.pickle"
    )

    return sentences_data, label_data


class Dictionary(object):
    def __init__(self, sentences, labels, vocab_drop):
        # sentences - array of sentences
        self._vocab_drop = vocab_drop
        if vocab_drop < 0:
            raise ValueError
        self.unk = "<UNK>"
        self.bos = "<BOS>"
        self.eos = "<EOS>"
        self.pad = "<PAD>"
        ## keep pad as the first element, word2idx['PAD'] := 0
        self.specials = [self.pad, self.bos, self.eos, self.unk]
        self.specials_ = self.specials + [x.lower() for x in self.specials]
        self._sentences = sentences
        self._labels = labels

        # self.get_words()
        ## word frequency stored per label
        counts = defaultdict(Counter)
        for i, sent in enumerate(sentences):
            for j, word in enumerate(sent):
                if word not in self.specials_:
                    counts[self._labels[i][j]][word.lower()] += 1

        self._labels_set = sorted(counts.keys())
        ## drop words less frequent than vocab drop
        words_per_label = {
            label:
            [w for w, c in counts[label].items() if c >= self._vocab_drop]
            for label in self._labels_set
        }
        ## vocab drop on the label itself
        words_per_label = {
            label: word_list
            for label, word_list in words_per_label.items()
            if len(word_list) >= self._vocab_drop
        }
        ## update label list
        self._labels_set = sorted(words_per_label.keys())

        all_words = self.specials[:]
        sizes = [len(self.specials)]
        for label in self._labels_set:
            all_words.extend(words_per_label[label])
            sizes.append(len(words_per_label[label]))
        print('sizes:', sizes)

        self._vocab = all_words
        self._sizes = sizes

        ## for words
        self._word2idx = dict(zip(all_words, range(len(all_words))))
        self._idx2word = dict(zip(range(len(all_words)), all_words))
        assert self._idx2word[0] == self.pad
        assert self._word2idx[self.pad] == 0

        ## for labels
        all_labels = self.specials + self._labels_set
        self._l_word2idx = dict(zip(all_labels, range(len(all_labels))))
        self._l_idx2word = dict(zip(range(len(all_labels)), all_labels))
        assert self._l_idx2word[0] == self.pad
        assert self._l_word2idx[self.pad] == 0

        # self._words.append('<unk>')
        ## modify sentences to mark the dropped words as UNK
        ## also modify the labels
        self._mod_sentences()

    @property
    def vocab_size(self):
        return len(self._idx2word)

    @property
    def label_vocab_size(self):
        return len(self._l_idx2word)

    @property
    def sizes(self):
        return self._sizes

    @property
    def vocab(self):
        return self._vocab

    @property
    def labels_set(self):
        return self._labels_set

    @property
    def sentences(self):
        return self._sentences

    @property
    def labels(self):
        return self._labels

    @property
    def word2idx(self):
        return self._word2idx

    @property
    def idx2word(self):
        return self._idx2word

    @property
    def l_word2idx(self):
        return self._l_word2idx

    @property
    def l_idx2word(self):
        return self._l_idx2word

    def seq2dx(self, sentence):
        return [self.word2idx[wd] for wd in sentence]

    def get_words(self):
        for i in range(len(self.sentences)):
            sent = self.sentences[i]
            for j in range(len(sent)):
                word = sent[j]
                if word in ["<EOS>", "<BOS>", "<PAD>", "<UNK>"]:
                    self._words.append(word)
                elif (self._labels[i][j] == '0'):
                    self._other_words.append(word.lower())
                elif (self._labels[i][j] == '1'):
                    self._location_words.append(word.lower())
                elif (self._labels[i][j] == '2'):
                    self._person_words.append(word.lower())
                elif (self._labels[i][j] == '3'):
                    self._org_words.append(word.lower())

    def _mod_sentences(self):
        all_labels = self.specials + self._labels_set
        # for every sentence, if word not in vocab set to <UNK>
        for i, sent in enumerate(self._sentences):
            for j in range(len(sent)):
                if sent[j] in self.specials_ + ['N', 'n']:
                    sent[j] = sent[j].upper()
                else:
                    sent[j] = sent[j].lower()
                sent_invalid = sent[j] not in self.word2idx
                label_invalid = self._labels[i][j] not in all_labels
                if sent_invalid or label_invalid:
                    sent[j] = self.unk
                    self._labels[i][j] = self.unk
            # self._sentences[i] = sent
            # self._labels[i] = lab

    def build_vocabulary(self):
        counter_words = collections.Counter(self._words)
        # words, that occur less than 5 times dont include
        sorted_dict_words = sorted(
            counter_words.items(), key=lambda x: (-x[1], x[0])
        )
        # keep n words to be included in vocabulary
        sorted_dict_words = [
            (wd, count) for wd, count in sorted_dict_words
            if count >= self._vocab_drop or wd in ['<unk>', '<BOS>', '<EOS>']
        ]

        counter_other_words = collections.Counter(self._other_words)
        # words, that occur less than 5 times dont include
        sorted_dict_other_words = sorted(
            counter_other_words.items(), key=lambda x: (-x[1], x[0])
        )
        # keep n words to be included in vocabulary
        sorted_dict_other_words = [
            (wd, count) for wd, count in sorted_dict_other_words
            if count >= self._vocab_drop or wd in ['<unk>', '<BOS>', '<EOS>']
        ]

        counter_location_words = collections.Counter(self._location_words)
        # words, that occur less than 5 times dont include
        sorted_dict_location_words = sorted(
            counter_location_words.items(), key=lambda x: (-x[1], x[0])
        )
        # keep n words to be included in vocabulary
        sorted_dict_location_words = [
            (wd, count) for wd, count in sorted_dict_location_words
            if count >= self._vocab_drop or wd in ['<unk>', '<BOS>', '<EOS>']
        ]

        counter_person_words = collections.Counter(self._person_words)
        # words, that occur less than 5 times dont include
        sorted_dict_person_words = sorted(
            counter_person_words.items(), key=lambda x: (-x[1], x[0])
        )
        # keep n words to be included in vocabulary
        sorted_dict_person_words = [
            (wd, count) for wd, count in sorted_dict_person_words
            if count >= self._vocab_drop or wd in ['<unk>', '<BOS>', '<EOS>']
        ]

        counter_org_words = collections.Counter(self._org_words)
        # words, that occur less than 5 times dont include
        sorted_dict_org_words = sorted(
            counter_org_words.items(), key=lambda x: (-x[1], x[0])
        )
        # keep n words to be included in vocabulary
        sorted_dict_org_words = [
            (wd, count) for wd, count in sorted_dict_org_words
            if count >= self._vocab_drop or wd in ['<unk>', '<BOS>', '<EOS>']
        ]

        # print(sorted_dict_words)
        # print(sorted_dict_e_words)
        # print(sorted_dict_h_words)
        # after sorting the dictionary, get ordered words
        all_words = []
        words, _ = list(zip(*sorted_dict_words))
        other_words, _ = list(zip(*sorted_dict_other_words))
        location_words, _ = list(zip(*sorted_dict_location_words))
        person_words, _ = list(zip(*sorted_dict_person_words))
        org_words, _ = list(zip(*sorted_dict_org_words))

        all_words.extend(words)
        all_words.extend(other_words)
        all_words.extend(location_words)
        all_words.extend(person_words)
        all_words.extend(org_words)

        sizes = [
            len(words) + 1,
            len(other_words),
            len(location_words),
            len(person_words),
            len(org_words)
        ]
        print(sizes)
        # print(all_words)
        # print(sizes)
        # print(len(all_words))
        self._word2idx = dict(zip(all_words, range(1, len(all_words) + 1)))
        self._idx2word = dict(zip(range(1, len(all_words) + 1), all_words))
        # add <PAD> as zero
        # print(words)
        self._idx2word[0] = '<PAD>'
        self._word2idx['<PAD>'] = 0
        # print(self._idx2word)
        all_words = ['<PAD>'] + all_words
        self._vocab = all_words
        self._sizes = sizes

    def __len__(self):
        return len(self.idx2word)


def train_w2vec(
    embed_fn,
    embed_size,
    w2vec_it=5,
    tokenize=True,
    sentences=None,
    model_path="./trained_embeddings_" + params.name
):
    from gensim.models import KeyedVectors, Word2Vec
    embed_fn += '.embed'
    print(os.path.join(model_path, embed_fn))
    print(
        "Corpus contains {0:,} tokens".format(
            sum(len(sent) for sent in sentences)
        )
    )
    if os.path.exists(os.path.join(model_path, embed_fn)):
        print("Loading existing embeddings file")
        return KeyedVectors.load_word2vec_format(
            os.path.join(model_path, embed_fn)
        )
    # sample parameter-downsampling for frequent words
    w2vec = Word2Vec(
        sg=0,
        workers=multiprocessing.cpu_count(),
        size=embed_size,
        min_count=0,
        window=5,
        iter=w2vec_it
    )  # CBOW MODEL IS USED AND Embed_size default
    w2vec.build_vocab(sentences=sentences)
    print("Training w2vec")
    w2vec.train(
        sentences=sentences,
        total_examples=w2vec.corpus_count,
        epochs=w2vec.iter
    )
    # Save it to model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    w2vec.wv.save_word2vec_format(os.path.join(model_path, embed_fn))
    return KeyedVectors.load_word2vec_format(os.path.join(model_path, embed_fn))


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


def prepare_data(data_raw, labels_raw, params, data_path):
    # get embeddings, prepare data
    print("building dictionary")
    data_dict = Dictionary(data_raw, labels_raw, params.vocab_drop)
    save_data(
        data_dict.sentences,
        "./trained_embeddings_" + params.name + "/sentences_mod.pickle",
        os.path.join(data_path, 'data_mod.txt')
    )
    save_data(
        data_dict.labels,
        "./trained_embeddings_" + params.name + "/labels_mod.pickle",
        os.path.join(data_path, 'labels_mod.txt')
    )

    model_path = "./trained_embeddings_" + params.name
    filename = os.path.join(model_path, "embedding_file.pkl")

    if os.path.exists(filename):
        with open(filename, 'r') as rf:
            embed_arr = pickle.load(rf)

    else:
        vector_file = 'wiki.en.align.vec'
        en_align_dictionary = FastVector(vector_file=vector_file)
        print("loaded:", vector_file)

        embed_arr = np.zeros([data_dict.vocab_size, params.embed_size])
        for i in range(1, embed_arr.shape[0]):
            # print(i)
            try:
                embed_arr[i] = en_align_dictionary[data_dict.idx2word[i]]
                # print(str(i), "english")
            except:
                embed_arr[i] = en_align_dictionary["unk"]
                print(str(i), "unk")

        print("Embedding created")
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        with open(filename, 'w') as wf:
            pickle.dump(embed_arr, wf)

    # if params.pre_trained_embed:
    #     w2_vec = train_w2vec(params.input, params.embed_size,
    #                         w2vec_it=5,
    #                         sentences=data_dict.sentences,
    #                         model_path="./trained_embeddings_"+params.name)
    #     embed_arr = np.zeros([data_dict.vocab_size, params.embed_size])
    #     for i in range(embed_arr.shape[0]):
    #         if i == 0:
    #             continue
    #         try:
    #             embed_arr[i] = w2_vec.word_vec(unicode(data_dict.idx2word[i], "utf-8"))
    #             # print(data_dict.idx2word[i])

    #         except:
    #             ax=2
    #             # embed_arr[i] = w2_vec.word_vec('<unk>')
    data = [
        [data_dict.word2idx[word] for word in sent[:-1]]
        for sent in data_dict.sentences if len(sent) < params.sent_max_size - 2
    ]

    encoder_data = [
        [data_dict.word2idx[word] for word in sent[1:]]
        for sent in data_dict.sentences if len(sent) < params.sent_max_size - 2
    ]

    sizes = data_dict.sizes
    to_subtract = [0]
    i = 0
    s = 0
    for i in range(len(sizes) - 1):
        s += sizes[i]
        to_subtract.append(s)
    to_subtract = to_subtract[::-1]

    encoder_data_adjusted_idx = []
    for i, sent in enumerate(data_dict.sentences):
        a = []
        for word in sent[1:]:
            index = data_dict.word2idx[word]
            for s in to_subtract:
                if s <= index:
                    break
            a.append(index - s)
        encoder_data_adjusted_idx.append(a)

    ## label encodings
    # label_embed_arr = np.zeros(
    #     [len(data_dict.l_word2idx.keys()), params.label_embed_size]
    # )
    # for i in range(len(data_dict.l_word2idx.keys())):
    #     label_embed_arr[i][i] = 1
    label_embed_arr = np.eye(len(data_dict.l_word2idx.keys()))
    labels = [
        [data_dict.l_word2idx[word] for word in sent[:-1]]
        for sent in data_dict.labels if len(sent) < params.sent_max_size - 2
    ]
    encoder_labels = [
        [data_dict.l_word2idx[word] for word in sent[1:]]
        for sent in data_dict.labels if len(sent) < params.sent_max_size - 2
    ]

    filename = os.path.join(model_path, "data_dict.pkl")
    with open(filename, 'w') as wf:
        pickle.dump(data_dict, wf)

    print(
        "----Corpus_Information--- \n "
        "Raw data size: {} sentences \n Vocabulary size {}"
        "\n Limited data size {} sentences \n".format(
            len(data_raw), data_dict.vocab_size, len(data)
        )
    )
    return data, encoder_data, encoder_data_adjusted_idx, embed_arr, data_dict, labels, encoder_labels, label_embed_arr
