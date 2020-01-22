import json
import os
import pickle
from collections import Counter

import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm

from fasttext import FastVector


def read_lines(fname):
    lines = open(fname).readlines()
    lines = map(lambda x: x.strip(), lines)
    return lines


def modify_sentences(data, vocab):
    # for every sentence, if word not in vocab set to <UNK>
    for sentences in tqdm(data):
        for sent in sentences:
            for j in range(len(sent)):
                if sent[j] not in vocab:
                    sent[j] = '<UNK>'
    return


def save_pickle(var, fname):
    with open(fname, 'wb') as f:
        pickle.dump(var, f)


def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def get_data(folder, embed_size):

    encoder_sentences_pkl_fname = os.path.join(
        folder, 'pickles', 'encoder_sentences.pkl'
    )
    decoder_sentences_pkl_fname = os.path.join(
        folder, 'pickles', 'decoder_sentences.pkl'
    )
    documents_pkl_fname = os.path.join(folder, 'pickles', 'documents.pkl')
    embed_arr_pkl_fname = os.path.join(folder, 'pickles', 'embed_arr.pkl')
    word2idx_pkl_fname = os.path.join(folder, 'pickles', 'word2idx.pkl')
    idx2word_pkl_fname = os.path.join(folder, 'pickles', 'idx2word.pkl')
    topic_word2idx_pkl_fname = os.path.join(
        folder, 'pickles', 'topic_word2idx.pkl'
    )

    pickle_files = [
        encoder_sentences_pkl_fname, decoder_sentences_pkl_fname,
        documents_pkl_fname, embed_arr_pkl_fname, word2idx_pkl_fname,
        idx2word_pkl_fname, topic_word2idx_pkl_fname
    ]
    if all(map(os.path.isfile, pickle_files)):
        print('data.py: loading from pickle files')
        return list(map(load_pickle, pickle_files))

    MIN_WORD_FREQ = 4
    if 'imdb' in folder:
        MIN_WORD_FREQ = 10
    print('MIN_WORD_FREQ',MIN_WORD_FREQ)

    sent_file = os.path.join(folder, 'train.txt')
    vector_file = 'updated.' + os.path.basename(folder) + '.embed.10epochs.vec'
    ## each line contains tab separated sentences
    data = read_lines(sent_file)
    if 'imdb' in folder:
        data = data[:5000]

    ## list of texts
    ## each text is a list of sentences
    ## each sentence is a list of words
    data = [[x.lower().split(' ') for x in line.split('\t')] for line in data]
    print(len(data))
    print(sum(map(len, data)))

    c = Counter()
    for text in tqdm(data):
        for sentence in text:
            c.update(sentence)  ## sentence is a list of words

    vocab = [k for k, v in c.items() if v >= MIN_WORD_FREQ]
    if 'amazon' in folder:
        for x in ['vga', 'hal', 'recognizing', 'avery', 'nurses', 'stressed', 'family.i', 'greg', 'bad.i', 'tightened', 'guards', 'invite', 'offical', 'mingus', 'c.d', 'faucets', 'exceed', 'schwinn', 'pieza', 'rosa', 'rural', 'murders', '1971', 'usado', 'colourful', 'spills', 'insect', 'itch', 'pots', 'den', 'stroke', 'punches', 'thankfully', 'handlebars', '2.50', 'psychiatric', 'sandwich', 'tendency', 'perfectamente', 'profesional', 'upstairs', 'banking', 'dressed', 'betta', 'tenth', 'accompaniment', 'cumplio', 'brat', 'arena', 'godd', 'iraq', 'prone', 'hotmail.com', 'appearances', 'hombre', 'ufo', 'fitment', 'scriptures', 'tyson', 'good.but', "n'roll", 'exclusive', 'bogged', 'amen', 'intricacies', 'dreamer', 'transformers', 'salesperson', 'tart', 'withstand', 'specialy', 'clearing', 'freeway', 'brewer', '.worth', 'unsatisfactory', 'grabbing', 'annoy', 'rape']:
            if x in vocab:
                vocab.remove(x)
    print('vocab_size', len(vocab) + 4)
    modify_sentences(data, vocab)

    ## remove 0.1% most frequent words
    all_words = set(vocab)
    # most_freq = set(k for k in c.most_common(len(vocab) // 1000))
    most_freq = set(k for k, v in c.items() if v > 500)
    print('len(most_freq)', len(most_freq))
    topic_vocab = all_words - most_freq
    topic_vocab.discard('<UNK>')

    ## remove words that appear in less than 100 docs
    print('calculating doc freq')
    doc_c = Counter()
    for doc in tqdm(data):
        doc_c.update(set(word for sentence in doc for word in sentence))
    less_freq = set(k for k, v in doc_c.items() if v < 50)
    print('len(less_freq)', len(less_freq))
    topic_vocab -= less_freq

    ## remove stop words
    topic_vocab -= set(stopwords.words('english'))
    print('len(topic_vocab)', len(topic_vocab))

    print('removing non alpha words')
    topic_vocab -= set([k for k in topic_vocab if not k.isalpha()])

    topic_vocab_size = len(topic_vocab)
    print('len(topic_vocab)', topic_vocab_size)

    topic_vocab = sorted(topic_vocab)
    topic_word2idx = dict(zip(topic_vocab, range(topic_vocab_size)))

    vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'] + sorted(vocab)
    vocab_size = len(vocab)
    word2idx = dict(zip(vocab, range(vocab_size)))
    idx2word = dict(zip(range(vocab_size), vocab))

    ## write modified sentences so that w2v can be trained
    # with open(os.path.join(folder, 'train_mod.txt'), 'w') as f:
    #     for sentences in data:
    #         for sentence in sentences:
    #             f.write(' '.join(['<BOS>'] + sentence + ['<EOS>']))
    #             f.write('\n')

    vec = FastVector(vector_file=vector_file)

    embed_arr = np.zeros([vocab_size, embed_size])
    for i in range(1, vocab_size):
        # print(i)
        try:
            embed_arr[i] = vec[idx2word[i]]
            # print(str(i), "english")
        except:
            embed_arr[i] = vec["<UNK>"]
            print("using <UNK>'s embedding for", idx2word[i], i)

    ## sentences, bow
    print('generating document vectors (bag of words)')
    documents = []
    for document in data:
        c = Counter()
        for sentence in document:
            c.update(sentence)
        doc = np.zeros(topic_vocab_size)
        for k, v in c.items():
            if k in topic_vocab:
                doc[topic_word2idx[k]] = v
        for _ in range(len(document)):
            documents.append(doc)

    ## limit sentence length to 50
    if 'imdb' in folder:
        for document in data:
            for i in range(len(document)):
                document[i] = document[i][:50]

    sentences = []
    for document in data:
        for sentence in document:
            sentences.append(
                [word2idx[x] for x in ['<BOS>'] + sentence + ['<EOS>']]
            )

    encoder_sentences = [x[1:] for x in sentences]
    decoder_sentences = [x[:-1] for x in sentences]

    print('data.py: saving pickle files')
    if not os.path.exists(os.path.dirname(encoder_sentences_pkl_fname)):
        os.makedirs(os.path.dirname(encoder_sentences_pkl_fname))
    save_pickle(encoder_sentences, encoder_sentences_pkl_fname)
    save_pickle(decoder_sentences, decoder_sentences_pkl_fname)
    save_pickle(documents, documents_pkl_fname)
    save_pickle(embed_arr, embed_arr_pkl_fname)
    save_pickle(word2idx, word2idx_pkl_fname)
    save_pickle(idx2word, idx2word_pkl_fname)
    save_pickle(topic_word2idx, topic_word2idx_pkl_fname)
    print('data.py: saving pickle files\t..done')

    return encoder_sentences, decoder_sentences, documents, embed_arr, word2idx, idx2word, topic_word2idx
