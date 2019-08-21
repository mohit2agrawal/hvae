### hvae_model.py
#### encoder
- `zsent_encoder`, and `zglobal_encoder` now combined into `encoder_model`
- `zglobal_encoder` required `zsent_sample`, which is no longer required in the new architecture
- output of `zsent_encoder` concatenated with labels and fed as input to `zglobal_encoder`
#### decoder
- modified decoder architecture, `decoder_model` is the new func, `lstm_decoder_labels` and `lstm_decoder_words` are not used anymore.
- previously, labels and words were being fed as input to decoder LSTMs as well, they are now not fed as input in the new code