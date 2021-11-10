import os
from os import listdir
from os.path import isfile, join
import bz2
import json
import random
from tqdm import tqdm

def write_stream(out_file, quote_files):
    """
    Function to extract a random sample from the files provided.
    
    Parameters
    ----------
    outfile : str
        File where to store the sample.
    quote_files : list
        List of files where the data is stored.

    Returns
    -------
    None
    """
    with bz2.open(out_file, 'wb') as d_file:
        # we iterate through each one of the input files
        for path_to_in in quote_files:
            print("Reading " + path_to_in + " ...")
            with bz2.open(path_to_in, 'rb') as s_file:
                # go through all the input lines
                # tqdm is simply a pretty progress bar :)
                for instance in tqdm(s_file, total=os.path.getsize(path_to_in)):
                    instance = json.loads(instance)
                    # only write if the random is <=0.05 (sampling 1/20 lines of the datasets)
                    if random.random() <= 0.05:
                        d_file.write((json.dumps(instance)+'\n').encode('utf-8'))

if __name__ == '__main__':

    # quotebanks folder
    path_to_files = '../data/Quotebank/'

    # output file path
    path_to_out = '../data/quotes-sample.json.bz2'

    # list of paths to all the quotebank datasets
    quote_files = [path_to_files + f for f in listdir(path_to_files) if isfile(join(path_to_files, f))]

    write_stream(path_to_out, quote_files)
