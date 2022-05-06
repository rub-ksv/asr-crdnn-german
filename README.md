# Performance

This model has a test performance of 7.24% WER.


# Using the model
You can find the model on Huggingface:
https://huggingface.co/jfreiwa/asr-crdnn-german


```python
from speechbrain.pretrained import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="jfreiwa/asr-crdnn-german", savedir="pretrained_models/asr-crdnn-german")
transcript = asr_model.transcribe_file("jfreiwa/asr-crdnn-german/example-de.wav")
print(transcript)
# >> hier tragen die mitgliedstaaten eine entsprechend grosse verantwortung
```

# How to run the training?

0. Install prerequisite software

You might need to install python3.7, sox and ffmpeg on your system, as well as speechbrain and some utilities.
You may also want to use a virtual environment (or conda):


```bash
sudo apt-get install sox libsox-fmt-mp3 ffmpeg python3-venv
python3 -m venv ~/venvs/asr-crdnn-german
source ~/venvs/asr-crdnn-german/bin/activate

pip install wheel
pip install -r requirements.txt

```

If you need more information about the installation of speechbrain, please consider their installation guide:

https://speechbrain.readthedocs.io/en/latest/installation.html

If you want to run the complete training, it is recommended to use CUDA:

https://developer.nvidia.com/cuda-downloads

Here you can find more information on how to verfiy your CUDA installation:

https://pytorch.org/get-started/locally/

Alternatively, in some cases you might want to use nvidia-docker:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html




1. Download the databases.

  - https://corpora.uni-hamburg.de/hzsk/de/islandora/object/spoken-corpus:swc-2.0
  - https://commonvoice.mozilla.org/de/datasets  (We used version 6.1)
  - https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/

2. Extract them in different folders in the same root folder

```
//<root folder>/source/mcv/cv-corpus-6.1-2020-12-11
//<root folder>/source/swc/german
//<root folder>/source/mai/de_DE
```
These folder should contain the first extracted path, that has files or more then 2 subfolders in it.

3. Run the preprocessing scripts


```
./preprocessing/mai/convert.py --root_dir <root folder>/source/mai/de_DE --save_dir <root folder>/processed/mai
./preprocessing/mcv/convert.py --root_dir <root folder>/source/mcv/cv-corpus-6.1-2020-12-11/de --save_dir <root folder>/processed/mcv
./preprocessing/swc/convert.py --root_dir <root folder>/source/swc/german --save_dir <root folder>/processed/swc
```
This step takes some time, so grab a coffee. You can skip the generation of wav files by adding the "--dry_run" option to each line, if you want to regenerate only the json files after the first run.

(You can generate training-data with german umlauts if you replace the symbollink to "utils.py" with "utils_umlaute.py".)

4. Set your database paths in following files:
```
./tokenizer/hparams/unigram_5000.yaml
./tokenizer/run.sh
./seq2seq/hparams/<setup>.yaml
```

5. Run the tokenizer (this includes the splitting of the databases in train-, test- and validation-sets)
```
./tokenizer/run.sh
```

6. Run the training.

```
./seq2seq/train.py hparams/<setup>.yml
```

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

# Acknowledgements
This work was funded by the German Federal Ministry of Education and Research (BMBF)
within the “Innovations for Tomorrow’s Production, Services, and
Work” Program (02L19C200), a project that is implemented by
the Project Management Agency Karlsruhe (PTKA). The authors
are responsible for the content of this publication.
