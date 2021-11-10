# -*- coding: utf-8 -*-
#This file includes pipeline to prepare the document for analysis
#functions used here can be found in prep_utilities.py file

#Pipeline for data prep

from src.prep_utilities import *

#Pipeline for data prep
#Pipeline for data prep
def prep_docs(doc, speaker_attributes, fix_contract = True, del_stop = True, lemmatize = True):

    # Delete rows with 'None' speaker
    print("Deleting rows with 'None' speaker...")
    copy_doc = doc[doc['speaker'] != 'None']

    # get date in YYYY-MM format
    print("Simplifying date column...")
    copy_doc['date'] = copy_doc['date'].apply(lambda x: get_yyyy_mm(x))

    # prepare clean tokens
    print("Tokenizing quotes...")
    copy_doc['tokens'] = copy_doc['quotation'].apply(
        lambda x: prep_tokens_row(x, fix_contract, del_stop, lemmatize))

    # filter out unnecessary rows (by number of words/true words)
    print("Filtering rows...")
    copy_doc = filter_quotes(copy_doc)

    # get domain names
    print("Getting url domains...")
    copy_doc['websites'] = copy_doc['urls'].apply(lambda x: get_website(x))
    copy_doc.drop(columns='urls', inplace=True)

    # get the gender of the speaker
    print("Getting genders...")
    copy_doc['gender'] = copy_doc['qids'].apply(lambda x: find_gender(x, speaker_attributes))

    # Drop rows with gender = 'None'
    copy_doc = copy_doc[copy_doc['gender'].apply(lambda x: type(x)) != type(None)]

    return copy_doc