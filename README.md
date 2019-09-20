## Data Required
- DATA/ptb_ner/train.txt
- DATA/ptb_ner/labels.txt
- wiki.en.align.vec

[Dropbox Link](https://www.dropbox.com/sh/5bgmlo33bpjam2f/AAA39m2wM0Tj5s1FpvokpsACa?dl=0)

## How to run

- with default configuration
  - monotonous scheduling
  - linear increase
  - no beta lag (alpha = beta)

  `python vae_lstm-lstm.py`

- options

  `python vae_lstm-lstm.py --fn tanh --cycles 5 --cycle_proportion 0.5 --beta_lag 0.1`
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

- sampling

  `python simple_sampling.py --num_samples 10000 --ckpt_path models_ckpts_ptb_ner/vae_lstm_model-23300`

## Output
- Following values will be saved in `test_plot.txt`
  - alpha
  - beta
  - ELBO
  - weighted sum of KL
  - KL_global
  - KL_sentence

- To generate graph (`test_plot.png`) from the values stored in `test_plot.txt`

  `python create_plot_iters.py`

- The checkpoints (to be used for sampling/ generation) are stored in directory `models_ckpts_ptb_ner`

- sampling output
  - generated_labels.txt
  - generated_sentences.txt
