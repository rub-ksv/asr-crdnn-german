import pandas as pd
from glob2 import glob
from os import system, makedirs
from os.path import join, exists
from utils import normalize_text, get_preprocessing_arguments
from scipy.io.wavfile import read as audio_read
from tqdm import tqdm
tqdm.pandas()

class ToWav():
    def __init__(self, args):
        self.args = args
        if not args.dry_run:
            makedirs(join(args.save_dir, 'clips'), exist_ok=True)

    def __call__(self, filename):
        infile = join(self.args.root_dir, 'clips', filename)
        outfile = join(self.args.save_dir, 'clips', filename.replace('.mp3', '.wav'))
        syscall = f"sox -G {infile} -b 32 {outfile} rate 16k channels 1"
        if not args.dry_run and not exists(outfile):
            system(syscall)
        return outfile.replace(self.args.save_dir, '.')

class GetLength():
    def __init__(self, args):
        self.args = args

    def __call__(self, filename):
        tmp = join(self.args.save_dir, filename)
        rate, signal = audio_read(tmp)
        assert rate == 16000 # rate should match
        assert len(signal) > 0
        return len(signal)

if __name__ == '__main__':
    # Parse Arguments
    args = get_preprocessing_arguments('MCV')

    # Load sets
    train = pd.read_csv(join(args.root_dir, 'train.tsv'), sep='\t')
    test = pd.read_csv(join(args.root_dir, 'test.tsv'), sep='\t')
    dev = pd.read_csv(join(args.root_dir, 'dev.tsv'), sep='\t')
    # Clean sentences
    for table in [train, test, dev]:
        table['clean_sentence'] = table['sentence'].apply(normalize_text)
    # concat
    complete = pd.concat([train, test, dev])

    # convert to wav
    print("Converting wavs... (This may take a while)")
    to_wav = ToWav(args)
    complete['file_path'] = complete['path'].progress_apply(to_wav)
    # get length and duration

    get_length = GetLength(args)
    complete['length'] = complete['file_path'].apply(get_length)
    complete['duration'] = complete['length'] / 16000
    # convert db column names
    complete.rename({'clean_sentence': 'words', 'client_id': 'spkID'}, axis='columns', inplace=True)
    # drop unnecessary columns
    complete.drop(['path', 'locale', 'segment', 'accent', 'up_votes', 'down_votes', 'age'], axis=1, inplace=True)
    # create new indexing by utterance id
    to_index = lambda x: x.replace('clips/','').replace('.wav','')
    complete['utt'] = complete['file_path'].apply(to_index)
    complete.set_index('utt', inplace=True)
    complete.to_json(join(args.save_dir, 'complete.json'), orient='index', indent=2)
