import pandas as pd
from glob2 import glob
from os import system, makedirs
from os.path import join, dirname, sep, exists
from utils import normalize_text, get_preprocessing_arguments
import re
from scipy.io.wavfile import read as audio_read

def add_meta_gendered(filename, args):
    table = pd.read_json(filename, orient='index')
    match = re.search(r'.*/by_book/(\w+)/(\w+)/', filename)
    table['gender'] = match.group(1)
    #table['gender'] = table['gender'].replace('mix', None)
    table['spkID'] = match.group(2)
    table['src_dir'] = join(dirname(filename), 'wavs') + sep
    return table


def add_meta(filename, args):
    table = pd.read_json(filename, orient='index')
    table['gender'] = 'other'
    table['spkID'] = None
    table['src_dir'] = join(dirname(filename), 'wavs') + sep
    return table

def toWav(args, files):
    infile = files[0]
    outfile = join(args.save_dir, files[1])
    syscall = f"sox -G {infile} -b 32 {outfile} rate 16k channels 1"
    if not args.dry_run and not exists(outfile):
        assert not exists(outfile)
        system(syscall)

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
    args = get_preprocessing_arguments('m-ailabs')
    if not args.dry_run:
        makedirs(join(args.save_dir, 'clips'))
    # handle "male" and "female" folder
    files1 = glob(join(args.root_dir, 'by_book', '*', '*', '*', '*.json'))
    tables1 = [add_meta_gendered(f, args) for f in files1]
    # handle "mix" folder: It has no gender or spkID
    files2 = glob(join(args.root_dir, 'by_book', '*', '*','*.json'))
    tables2 = [add_meta(f, args) for f in files2]
    
    # concat tables
    tables = tables1 + tables2
    complete = pd.concat(tables)
    
    #normalize
    complete['words'] = complete['clean'].apply(normalize_text)
    complete.reset_index(inplace=True)
    complete.rename({'index': 'filename', 'clean': 'sentence'}, axis='columns', inplace=True)
    complete['src_file'] = complete['src_dir'] + complete['filename']
    
    complete['utt'] = complete['filename'].apply(lambda x: x.replace('.wav', ''))
    complete.set_index('utt', inplace=True)
    complete['file_path'] = 'clips' + sep + complete['filename']
    
    
    complete[['src_file','file_path']].apply(lambda x: toWav(args, x), axis=1)
    complete.drop(['original', 'filename', 'src_dir', 'src_file'], axis=1, inplace=True)

    # get length and duration
    get_length = GetLength(args)
    complete['length'] = complete['file_path'].apply(get_length)
    complete['duration'] = complete['length'] / 16000

    complete.to_json(join(args.save_dir, 'complete.json'), orient='index', indent=2)

