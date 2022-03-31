# Performance

This model has a test performance of 7.24% WER.


## Using the model
```
from speechbrain.pretrained import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="jfreiwa/asr-crdnn-german", savedir="pretrained_models/asr-crdnn-german")
asr_model.transcribe_file("jfreiwa/asr-crdnn-german/example-de.wav")

```

# How to run the training?

1. Download the databases.

  - https://nats.gitlab.io/swc/
  - https://commonvoice.mozilla.org/de/datasets
  - https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/

2. Extract them in different folders in the same root folder

//<root folder>/source/mcv
//<root folder>/source/swc
//<root folder>/source/mai
  
3. Run the preprocessing scripts

  - ./preprocessing/mai/convert.py --root_dir <root folder>/source/mai --save_dir <root folder>/processed/mai
  - ./preprocessing/mcv/convert.py --root_dir <root folder>/source/mcv --save_dir <root folder>/processed/mcv
  - ./preprocessing/swc/convert.py --root_dir <root folder>/source/swc --save_dir <root folder>/processed/swc
  
4. Set your database paths in following files:
  - ./tokenizer/hparams/unigram_5000.yaml
  - ./tokenizer/run.sh
  - ./seq2seq/hparams/<setup>.yaml

5. Run the tokenizer (this includes the splitting of the databases in train-, test- and validation-sets)
  - ./tokenizer/run.sh
  
6. Run ./seq2seq/train.py hparams/<setup>.yml


# Limitations
We do not provide any warranty on the performance achieved by this model when used on other datasets.

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/

# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.
```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```

# **Citing our paper**
```bibtex
@inproceedings{freiwald2022,
  author={J. Freiwald and P. Pracht and S. Gergen and D. Kolossa},
  title={Open-Source End-To-End Learning for Privacy-Preserving German {ASR}},
  year=2022,
  booktitle={DAGA 2022}
}
```
