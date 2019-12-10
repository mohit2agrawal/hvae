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

  - for cyclical schedule where alpha resets when either of KL_zl or KL_zc drops below 0.8.

  `python sivae_mod.py`


- sampling
  Sample `num_samples` positive and negative sentiment sentences. The sentences are saved in `generated_positive.txt` and `generated_negative.txt`.

  `python simple_sampling.py --num_samples 2000 --ckpt_path models_ckpts_ptb_ner/vae_lstm_model-23300`


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
