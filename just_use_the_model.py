from speechbrain.pretrained import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="jfreiwa/asr-crdnn-german", savedir="pretrained_models/asr-crdnn-german")
transcript = asr_model.transcribe_file("jfreiwa/asr-crdnn-german/example-de.wav")
print(transcript)
# >> hier tragen die mitgliedstaaten eine entsprechend grosse verantwortung
