"""train a pos tagger using input sentences and tags
and get accuracy on test set (the second input)
"""
import sys
import re

train_sent_file, train_label_file = sys.argv[1:3]
test_sent_file, test_label_file = sys.argv[3:5]


def read_data(sent_file, label_file):
    tagged_data = []
    with open(sent_file) as sf, open(label_file) as lf:
        while True:
            s = sf.readline()
            if not s:
                break
            s = s.strip().split(' ')
            l = lf.readline().strip().split(' ')

            tagged_data.append(zip(s, l))
    return tagged_data


train_set = read_data(train_sent_file, train_label_file)
print("Number of Tagged Sentences (train):", len(train_set))

test_set = read_data(test_sent_file, test_label_file)
print("Number of Tagged Sentences (test):", len(test_set))


def features(sentence, index):
    ### sentence is of the form [w1,w2,w3,..], index is the position of the word in the sentence
    return {
        'is_first_capital':
            int(sentence[index][0].isupper()),
        'is_first_word':
            int(index == 0),
        'is_last_word':
            int(index == len(sentence) - 1),
        'is_complete_capital':
            int(sentence[index].upper() == sentence[index]),
        'prev_word':
            '' if index == 0 else sentence[index - 1],
        'next_word':
            '' if index == len(sentence) - 1 else sentence[index + 1],
        'is_numeric':
            int(sentence[index].isdigit()),
        'is_alphanumeric':
            int(
                bool(
                    (re.match('^(?=.*[0-9]$)(?=.*[a-zA-Z])', sentence[index]))
                )
            ),
        'prefix_1':
            sentence[index][0],
        'prefix_2':
            sentence[index][:2],
        'prefix_3':
            sentence[index][:3],
        'prefix_4':
            sentence[index][:4],
        'suffix_1':
            sentence[index][-1],
        'suffix_2':
            sentence[index][-2:],
        'suffix_3':
            sentence[index][-3:],
        'suffix_4':
            sentence[index][-4:],
        'word_has_hyphen':
            1 if '-' in sentence[index] else 0
    }


def untag(sentence):
    return [word for word, tag in sentence]


def prepareData(tagged_sentences):
    X, y = [], []
    for sentences in tagged_sentences:
        X.append(
            [
                features(untag(sentences), index)
                for index in range(len(sentences))
            ]
        )
        y.append([tag for word, tag in sentences])
    return X, y


X_train, y_train = prepareData(train_set)
X_test, y_test = prepareData(test_set)
from sklearn_crfsuite import CRF
crf = CRF(
    algorithm='lbfgs',
    c1=0.01,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
print('Starting training')
crf.fit(X_train, y_train)
print('Training Complete')

## METRICS
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
y_pred = crf.predict(X_test)
print("F1 score on Test Data ")
print(
    metrics.flat_f1_score(
        y_test, y_pred, average='weighted', labels=crf.classes_
    )
)
print("F score on Training Data ")
y_pred_train = crf.predict(X_train)
metrics.flat_f1_score(
    y_train, y_pred_train, average='weighted', labels=crf.classes_
)

### Look at class wise score
print(
    metrics.flat_classification_report(
        y_test, y_pred, labels=crf.classes_, digits=3
    )
)

print("Flat accuracy score")
print(metrics.flat_accuracy_score(y_test, y_pred))
