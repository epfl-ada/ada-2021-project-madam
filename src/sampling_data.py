import os
from os import listdir
from os.path import isfile, join#, getsize
import bz2
import json
import random
from tqdm import tqdm

def write_stream(out_file, quote_files):
    with bz2.open(out_file, 'wb') as d_file:
        for path_to_in in quote_files:
            print("Reading " + path_to_in + " ...")
            with bz2.open(path_to_in, 'rb') as s_file:
                for instance in tqdm(s_file, total=os.path.getsize(path_to_in)):
                    instance = json.loads(instance)
                    # only write if the random is <=0.05 (sampling 1/20 lines of the datasets)
                    if random.random() <= 0.05:
                        d_file.write((json.dumps(instance)+'\n').encode('utf-8'))

if __name__ == '__main__':

    # quotebanks folder
    path_to_files = '/content/drive/MyDrive/Phase2/Quotebank/'

    # output file path
    path_to_out = '/content/drive/MyDrive/Phase2/quotes-sample.json.bz2'

    # list of paths to all the quotebank datasets
    quote_files = [path_to_files + f for f in listdir(path_to_files) if isfile(join(path_to_files, f))]

    write_stream(path_to_out, quote_files)
