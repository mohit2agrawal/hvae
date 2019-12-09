from gensim.models import KeyedVectors, Word2Vec
import os
import multiprocessing

file_paths = ['ptb_pos_all/data_mod.txt', 'ptb_pos_all/val_data_mod.txt']
all_sentences = []
for file_path in file_paths:
    with open(file_path) as f:
        data = f.readlines()
    sentences = [['<BOS>'] + x.strip().split(' ') + ['<EOS>'] for x in data]
    all_sentences.extend(sentences)
print('sentences loaded.')
print(all_sentences[0])

embed_fp = 'w2v.embed'
embed_size = 300
w2vec_it = 5

print(
    "Corpus contains {0:,} tokens".format(
        sum(len(sent) for sent in all_sentences)
    )
)
# sample parameter-downsampling for frequent words
w2vec = Word2Vec(
    sg=0,
    workers=multiprocessing.cpu_count(),
    size=embed_size,
    min_count=4,
    window=5,
    iter=w2vec_it
)  #CBOW MODEL IS USED AND Embed_size default
w2vec.build_vocab(sentences=all_sentences)
print("Training w2vec")
w2vec.train(
    sentences=all_sentences,
    total_examples=w2vec.corpus_count,
    epochs=w2vec.iter
)
# Save it to model_path
w2vec.wv.save_word2vec_format(embed_fp)
# return KeyedVectors.load_word2vec_format(embed_fp)
