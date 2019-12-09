from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from datetime import datetime

# define training data
file_paths = ['ptb_pos_all/data_mod.txt', 'ptb_pos_all/val_data_mod.txt']

all_sentences = []
for file_path in file_paths:
    with open(file_path) as f:
        data = f.readlines()
    sentences = [['<BOS>'] + x.strip().split(' ') + ['<EOS>'] for x in data]
    all_sentences.extend(sentences)
print('sentences loaded.')

# sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
#             ['this', 'is', 'the', 'second', 'sentence'],
#             ['yet', 'another', 'sentence'],
#             ['one', 'more', 'sentence'],
#             ['and', 'the', 'final', 'sentence']]
# train model
# model_1 = Word2Vec(sentences, size=300, min_count=1)

model_2 = Word2Vec(size=300, min_count=4)
model_2.build_vocab(all_sentences)
total_examples = model_2.corpus_count

vocab = model_2.wv.vocab
print('<UNK> in vocab:', '<UNK>' in vocab)
print('<BOS> in vocab:', '<BOS>' in vocab)
print('<EOS> in vocab:', '<EOS>' in vocab)

# model = KeyedVectors.load_word2vec_format("glove.6B.300d.txt", binary=False)
# print('loading wiki...')
# model = KeyedVectors.load_word2vec_format("wiki.en.align.vec", binary=False)
# print('loaded wiki...')
# model_2.build_vocab([list(model.vocab.keys())], update=True)
# model_2.intersect_word2vec_format("glove.6B.300d.txt", binary=False, lockf=1.0)
print(datetime.now())
print('intersecting...')
model_2.intersect_word2vec_format("wiki.en.align.vec", binary=False, lockf=1.0)
print(datetime.now())
print('starting training')
epochs = model_2.iter
model_2.train(all_sentences, total_examples=total_examples, epochs=10)
print(datetime.now())
print('training completed.')
print('saving word vectors')
model_2.wv.save_word2vec_format('updated.embed.vec')
print(datetime.now())

# print('loading wiki...')
# model = KeyedVectors.load_word2vec_format("wiki.en.align.vec", binary=False)
# print('loaded wiki...')
# vocab_keys = model.vocab.keys()
# print('<UNK> in vocab:', '<UNK>' in vocab_keys)
# print('<BOS> in vocab:', '<BOS>' in vocab_keys)
# print('<EOS> in vocab:', '<EOS>' in vocab_keys)
# model.build_vocab(all_sentences, update=True)
# print('updated vocab')
# vocab_keys = model.vocab.keys()
# print('<UNK> in vocab:', '<UNK>' in vocab_keys)
# print('<BOS> in vocab:', '<BOS>' in vocab_keys)
# print('<EOS> in vocab:', '<EOS>' in vocab_keys)
# model.train(all_sentences, total_examples=model_2.corpus_count, epochs=5)
# model.wv.save_word2vec_format('updated.embed')
