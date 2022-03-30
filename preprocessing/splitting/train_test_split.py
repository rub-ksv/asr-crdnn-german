from argparse import ArgumentParser
from os.path import join, dirname, abspath, relpath
import pandas as pd
from utils import save_db_file


    
def split(df, group):
    # https://stackoverflow.com/questions/23691133/split-pandas-dataframe-based-on-groupby
    gb = df.groupby(group)
    return [gb.get_group(x) for x in gb.groups]

if __name__=='__main__':
    parser = ArgumentParser(description=f'Split complete.json into train.json, dev.json and test.json.')
    parser.add_argument('--infile', required=True, help='Path to the source file (complete.json).')
    parser.add_argument('--save_dir', default=None, help='Path to the output directory. If not set the input directory will be used.')
    parser.add_argument('--prefix', default="", help='Prefix to the output file')
    parser.add_argument('--train_ratio', default=0.9, help='How many percent of utterances will be in the train set?')
    parser.add_argument('--test_dev_ratio', default=0.5, help='How many percent of the test set will be moved to the dev set?')

    parser.add_argument('--unique', action='store_true', help='If set, only unique utterances (text) will be used.')
    parser.add_argument('--drop_numerals', action='store_true', help='If set, only utterances without numbers will be used.')
    parser.add_argument('--drop_long', action='store_true', help='If set, very long utterances will be dropped.')
    parser.add_argument('--disjunct_speakers', action='store_true', help='If set, all splits will have different speakers.')

    args = parser.parse_args()

    complete = pd.read_json(args.infile, orient='index')

    if args.unique:
        length = len(complete)
        complete.drop_duplicates(subset='words', inplace=True)
        print(f"unique: dropped {length - len(complete)} utterances.")

    if args.drop_numerals:
        def contains_number(st):
            for ch in st:
                if ch.isdigit():
                    return True
            return False
        num_ids = complete["words"].apply(lambda x: contains_number(x))
        complete = complete[~num_ids]
        
        print(f"drop_numerals: dropped {num_ids.sum()} utterances.")

    if args.drop_long:
        # Drop utternaces that are longer then 30 seconds
        raise NotImplementedError()

    if args.disjunct_speakers:
        raise NotImplementedError()
    else:
        train = complete.sample(frac=args.train_ratio, random_state=42)
        test = complete.drop(train.index)
        dev = test.sample(frac=args.test_dev_ratio, random_state=42)
        test = test.drop(dev.index)

    if args.save_dir is None:
        args.save_dir = dirname(args.infile)

    save_db_file(train, args.infile, args.save_dir, 'train', args.prefix, json=True)
    save_db_file(test, args.infile, args.save_dir, 'test', args.prefix, json=True)
    save_db_file(dev, args.infile, args.save_dir, 'dev', args.prefix, json=True)
