from speechbrain.pretrained import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="jfreiwa/asr-crdnn-german", savedir="pretrained_models/asr-crdnn-german")
asr_model.transcribe_file("jfreiwa/asr-crdnn-german/example-de.wav")
