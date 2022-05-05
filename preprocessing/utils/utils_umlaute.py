from unidecode import unidecode
from argparse import ArgumentParser
import re
from os.path import join, dirname, abspath, relpath

def replace_symbols(text):
    # replace specific symbols with common pronunciation
    text = text.replace('$', ' dollar ')
    text = text.replace('§', ' paragraph ')
    text = text.replace('&', ' und ')
    text = text.replace('*', ' mal ')
    text = text.replace('^', ' hoch ')
    text = text.replace('@', ' et ')
    text = text.replace('%', ' prozent ')
    text = text.replace('<=', ' kleiner gleich ')
    text = text.replace('>=', ' groesser gleich ')
    text = text.replace('<', ' kleiner ')
    text = text.replace('>', ' groesser ')
    text = text.replace('=', ' gleich ')
    text = text.replace('1/2', ' einhalb ')
    text = text.replace('3/4', ' dreiviertel ')
    text = text.replace('2/3', ' zweidrittel ')
    return text

def replace_umlaute_1(text):
    text = text.replace('ä', 'AE')
    text = text.replace('ö', 'OE')
    text = text.replace('ü', 'UE')
    text = text.replace('ß', 'SS')
    return text

def replace_umlaute_2(text):
    text = text.replace('AE', 'ä')
    text = text.replace('OE', 'ö')
    text = text.replace('UE', 'ü')
    text = text.replace('SS', 'ß')
    return text

def remove_punctuation(text):
    # replace symbols with spaces
    for punc in '_<>.,:;-?!\\/[]{}()"~#`\'|':
        text = text.replace(punc, ' ')
    
    # remove multiple spaces
    text = ' '.join(text.split())
    return text

def normalize_text(text):
    # You can check whether this works well in the output of the tokenizer.
    # make everything lowercase
    text = str(text).lower()
    # convert symbols
    text = replace_symbols(text)
    # replace german umlaute
    text = replace_umlaute_1(text)
    # use unidecode to transliterate the rest
    text = unidecode(text)
    # replace german umlaute back
    text = replace_umlaute_2(text)
    text = text.lower()
    # remove punctuation
    text = remove_punctuation(text)
    return text

class UpdateFilePath():
    def __init__(self, root_dir, save_dir):
        self.root_dir = root_dir
        self.save_dir = save_dir
    def __call__(self, filename):
        abs_root = join(abspath(self.root_dir), filename)
        abs_dest = abspath(self.save_dir)
        filename = relpath(abs_root, abs_dest)
        return filename

def save_db_file(df, infile, save_dir, name, prefix="", json=True):
    ''' This function saves the new database file with updated file_path. '''
    rep = UpdateFilePath(dirname(infile), save_dir)
    print(dirname(infile), save_dir)
    df['file_path'] = df['file_path'].apply(rep)
    if json:
        outname = join(save_dir, f'{prefix}{name}.json')
        df.to_json(outname, orient='index', indent=2)
    else:
        outname = join(save_dir, f'{prefix}{name}.csv')
        df.to_csv(outname)

def get_preprocessing_arguments(db_name):
    parser = ArgumentParser(description=f'Preprocess the {db_name} database.')
    parser.add_argument('--root_dir', required=True, help='Path to the root directory.')
    parser.add_argument('--save_dir', required=True, help='Path to the save directory.')
    parser.add_argument('--dry_run', action='store_true', help='Generates no audio files if set.')
    return parser.parse_args()

