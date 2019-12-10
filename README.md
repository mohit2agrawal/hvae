## Data Required
- DATA/{name}/data.txt
- DATA/{name}/labels.txt
- DATA/{name}/val_data.txt
- DATA/{name}/val_labels.txt
- wiki.en.align.vec

[Dropbox Link](https://www.dropbox.com/sh/5bgmlo33bpjam2f/AAA39m2wM0Tj5s1FpvokpsACa?dl=0)

## How to run


- for gated lagging inference scheme

`python sivae_mod_gated.py`

- for cyclical schedule, where alpha goes to zero when either of KL_zl or KL_zc goes less than 0.8.

`python sivae_mod.py`

- for SiVAE-i implementation

`python sivae.py`

- Pharaphrase generation

  To generate `NUM_TRIES` sentences for a given sentence and syntax tree template, for first 250 sentences in `input_sents_file`.

  `python para.py --num_samples 250`

  Following may be edited in `para.py`.

  ```python
  out_sentence_file = "./generated_sentences_test.txt"
  out_labels_file = "./generated_labels_test.txt" ## not being used
  input_sents_file = "input_sentences_test.txt"
  input_labels_file = "input_templates_test.txt"
  NUM_TRIES=100
  ```

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

## Utils

- bleu score

  `python utils/extras/bleu_paraphrases.py <input_sentences_file> <paraphrases_file>`

  ```python
  output_fn = input_para_fn.replace('.txt', '.selected.txt') ## output filename
  NUM_SENTS = 250 * 5 ## number of sentences
  NUM_PARA = 100      ## paraphrases per sentence
  OUPUT_NUM = 2       ## top N to save in output file
  ```

  - calculates BLEU-2 after removing stop words and lemmatizing the words
  - saves only the unique paraphrases that are also not same as input
