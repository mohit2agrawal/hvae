### hvae_model.py
- zsent_encoder, and zglobal_encoder now combined into encoder_model
- zglobal_encoder required zsent_sample, which is no longer required in the new architecture
- output of zsent_encoder concatenated with labels and fed as input to zglobal_encoder
