from glob2 import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
from slugify import slugify
from os.path import join, dirname, exists
from os import system, makedirs
from utils import normalize_text, get_preprocessing_arguments
import pandas as pd
from scipy.io.wavfile import read as audio_read

def parse_file(alignfile, args):
    # This function reads the xml alignment files
    tree = ET.parse(alignfile)
    root = tree.getroot()
    spk = None
    for prop in root.findall('.//prop'):
        if prop.attrib['key'] == 'reader.name':
            spk = slugify(prop.attrib['value'])
    for sentence in root.findall('.//s'):
        start = None
        end = None
        words = []
        for word in sentence.findall('.//n'):
            try:
                if start is None:
                    start = float(word.attrib['start'])/1000.0
                end = float(word.attrib['end'])/1000.0
                words.append(word.attrib['pronunciation'])
            except:
                if start != None and end != None:
                    yield(spk, start, end, " ".join(words))
                start = None
                end = None
                words = []
        if start != None and end != None:
            yield(spk, start, end, " ".join(words))
    return


if __name__ == '__main__':
    # Parse Arguments
    args = get_preprocessing_arguments('SWC')
    makedirs(join(args.save_dir, 'clips'),exist_ok=True)
    data = []
    for alignfile in tqdm(sorted(glob(join(args.root_dir, '**/aligned.swc')))):
        # gather audio files to join them
        oggs = list(glob(alignfile.replace("aligned.swc", "audio*.ogg")))
        oggs = sorted(oggs)
        oggs = [f'"{ogg}"' for ogg in oggs]
        oggs = " ".join(oggs)

        # cut sentences
        for ids, sentence in enumerate(parse_file(alignfile, args)):
            spk = str(sentence[0])

            utt = spk + "-" + slugify(dirname(alignfile).split("/")[-1]) + "-"  + str(ids)
            cutfile = join(args.save_dir, 'clips', utt) + ".wav"

            start, end = sentence[1], sentence[2]
            syscall = f"sox -G {oggs} -b 32 {cutfile} trim {start} ={end} rate 16k channels 1"

            if not args.dry_run and not exists(cutfile):
                #print(syscall)
                system(syscall)

            try:
                rate = None
                signal = None
                rate, signal = audio_read(cutfile)
                assert rate == 16000 # rate should match
                assert abs((end - start) * 16000 - len(signal)) < 160 # 100 ms difference are ok
                assert len(signal) > 0
                # spk, file, duration, gender, transcript
                data.append([utt,
                   sentence[0],
                   cutfile.replace(args.save_dir,'.'),
                   float(len(signal)) / 16000.0, None,
                   normalize_text(sentence[3]),
                   sentence[3],
                   len(signal)])
            except Exception as e:
                print(e)
                print(f"Skipped file {cutfile}, because an assertion was not matched")
                print(f"Sentence is: <<{sentence[3]}>>")
                print(f"File should be {(end - start) * 16000} samples and is {len(signal)} samples.")
                print(system(f"soxi {cutfile}"))


    df = pd.DataFrame(data, columns=['utt', 'spkID', 'file_path', 'duration', 'gender', 'words', 'sentence', 'length'])
    df = df.set_index('utt')
    df.to_json(join(args.save_dir, 'complete.json'), orient='index', indent=2)

