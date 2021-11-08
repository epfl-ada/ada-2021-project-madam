# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:17:39 2021

@author: tsfei
"""

from text_scores import dale_chall_score
from thematic_processing import prepare_docs, topic_cluster, grid_search
import pandas as pd
import re

if __name__ == '__main__':

    # tests for the complexity scoring
    # test_str1 = 'This seems like an easy sentence. Now I can add a new sentence that I still feel will be easy. One more just for volume.'
    # score1 = dale_chall_score(test_str1)
    # score_converted1 = dale_chall_score(test_str1, level = True)

    # test_str2 = 'Furthermore, I desired to test a fairly more advanced sentence. Following that line of thought, I expressed myself as eloquently as possible.'
    # score2 = dale_chall_score(test_str2)
    # score_converted2 = dale_chall_score(test_str2, level = True)

    # print(f'The simpler sentence \n\n\t{test_str1}\n\nhas a Dale-Chall score of {score1}, which converted to school levels is {score_converted1}.')
    # print(f'\nThe more advanced sentence \n\n\t{test_str2}\n\nhas a Dale-Chall score of {score2}, which converted to school levels is {score_converted2}.')

    # tests for thematic clustering

    df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
    # Convert to list
    data = df.content.values.tolist()
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    data = prepare_docs(data[::100])

    grid_search(data, [10, 15, 20, 25, 30], [.5, .7, .9])

    # topic_cluster(data, 10)
