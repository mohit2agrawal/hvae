## Data Required
- DATA/{name}/data.txt
- DATA/{name}/labels.txt
- DATA/{name}/val_data.txt
- DATA/{name}/val_labels.txt
- wiki.en.align.vec

[Dropbox Link](https://www.dropbox.com/sh/5bgmlo33bpjam2f/AAA39m2wM0Tj5s1FpvokpsACa?dl=0)

## How to run

- with default configuration
  - monotonous scheduling
  - linear increase
  - no beta lag (alpha = beta)

  `python vae_lstm-lstm.py`

- options

  `python vae_lstm-lstm.py --fn tanh --cycles 5 --cycle_proportion 0.5 --beta_lag 0.1 --zero_start 0.75`
  - --fn (scheduling function)
    - linear (default)
    - tanh
    - cosine
  - --cycles (number of cycles for cyclic schedule)
    - 1 (default) implies a monotonous schedule
  - --cycle_proportion (the proportion of the cycle to be used for the increase of alpha and beta)
    - 0.5 (default) implies that alpha reaches 1 in half cycle
  - --beta_lag (proportion of cycle beta lags behing alpha)
    - 0 (default) implies no lag, hence alpha = beta
    - 0.1 would imply that beta would lag behind alpha by 0.1*cycle
    - beta would be zero for the lagged proportion of first cycle
  - --zero_start (proportion of the time period for which the cycle remains at zero)

- sampling
  by default, both 'biased sampling' and 'no word repetition' are used.
  The 'no word repetition' is per label per sentence. If all words for a label have appeared in a sentence, we consider that none has occured, so that next word can be sampled.

  `python simple_sampling.py --num_samples 10000 --ckpt_path models_ckpts_ptb_ner/vae_lstm_model-23300`

## Output
- Following values will be saved in `test_plot.txt`
  - alpha
  - beta
  - ELBO
  - weighted sum of KL
  - KL_global
  - KL_sentence

- Following values will be saved in `test_plot_ppl.txt`
  - label PPL
  - sentence PPL

- To generate graph (`test_plot.png`) from the values stored in `test_plot.txt`

  `python create_plot_iters.py`

- The checkpoints (to
 be used for sampling/ generation) are stored in directory `models_ckpts_<params.name>`

- sampling output
  - generated_labels.txt
  - generated_sentences.txt

  If not sampling using the latest sampling code, use `python clip_by_eos.py <sent/label file>` to get the sentences and labels before the {EOS} tag.

## Utilities
### utils/pos
- tag.py

  `python tag.py sentenes.txt labels.txt`
  - Input: sentences.txt
  - Output: labels.txt, the POS tags for sentences.txt

- accuracy.py

  `python accuracy.py sentences.txt labels.txt test_sents.txt test_labels.txt`
  Train a CRF model to do POS tagging learnt from `sentences.txt` and `labels.txt`
  Then, report accuracy and other metrics on `test_sents.txt` and `test_labels.txt`
  
  **requires sklearn_crfsuite:**
  `pip install sklearn-crfsuite`

### utils/ner
- accuracy.py

  `python accuracy.py labels.txt true_labels.txt`
  - Input: labels.txt and true_labels.txt
  - Output: accuracy score
  
  Get the accuracy score considering `true_labels.txt` as the true value.

- tag.py

  `python tag.py sentences.txt labels.txt <model.ser.gz>`
  - Input: sentences.txt
  - Output: labels.txt, the NER tags for sentences.txt
  
  Find NER tags using the Stanford model.
  Provide an optional `model.ser.gz` to be used.
  
  NOTE: verify the path to `stanford-ner` in the code

- to_tsv.py

  `python to_tsv.py sentences.txt labels.txt output.tsv`
  - Input: sentences.txt and labels.txt
  - Output: output.tsv
  
  Convert the `sentences.txt` and `labels.txt` to a single `output.tsv` to be used for training a stanford NER model

- transform.py

  `python transform.py labels_ner.txt labels_numeric.txt`
  - Input: labels_ner.txt, the labels file with NER word tags like 'PERSON', 'LOCATION'
  - Output: labels_numeric.txt, the labels file with numeric labels (0,1,2,3)
  
  Convert NER word labels to numeric labels according to
  'O': '0', 'LOCATION': '1', 'PERSON': '2', 'ORGANIZATION': '3'
  
  NOTE: This may not be required if using the latest `data.py` and latest `sampling.py`

- verify_lengths.py

  `python verify_lengths.py data.txt labels.txt`
  verifies if every word has a label
  checks if the length of list after splitting by <space> is equal for both files per line

## Training a (Stanford) NER model
  - We will need sentences and labels in tsv format (use utils/ner/to_tsv.py).
  - A properties file. Modify following in utils/ner/sample.prop
    - trainFile = sample.tsv (input tsv path)
    - serializeTo = ner-sample.ser.gz (output model path)

  - Run following to train
  ```bash
  java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier \
    -prop train.prop
  ```

  - We may optionally use the following command
    (not tested, might have to remove trainFile nad serializeTo from the .prop file)
  ```bash
  java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier \
    -trainFile train.tsv \
    -serializeTo ner-model.ser.gz \
    -prop train.prop
  ```

  For testing the model,
  ```bash
  java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier \
  -loadClassifier ner-model.ser.gz \
  -testFile text.tsv
  ```
