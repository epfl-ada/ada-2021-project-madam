# -*- coding: utf-8 -*-
#This file includes pipeline to prepare the document for analysis
#functions used here can be found in prep_utilities.py file

from src.prep_utilities import *

#Pipeline for data prep
def prep_docs(doc, speaker_attributes, fix_contract = True, del_stop = True, lemmatize = True, print_progress = True):
    """
    This function performs the full pipeline to prepare the data taken from Quotebank
    
    Parameters
    ----------
    doc : pandas.DataFrame
        Dataframe with the data to pre-process.
    speaker_attributes : pandas.DataFrame
        WHAT IS IN HERE ?? SOMEBODY FILL THIS IN PLEASE.
    fix_contract : bool
        If true, expand contractions (don't -> do not; I'm -> I am;...)
    del_stop : bool
        If true, remove all stopwords.
    lemmatize : bool
        If true, lemmatize all words.
    
    Returns
    -------
    copy_doc : pandas.DataFrame
        Result of the data provided after having passed through the whole pipeline.
    """
    
    # COMMENT ----
    #
    # Are there any more columns that we can drop during all of this?
    #
    # ------------
    
    # Delete rows with 'None' speaker
    if print_progress: print("Deleting rows with 'None' speaker...")
    copy_doc = doc[doc['speaker'] != 'None']

    # get date in YYYY-MM format
    if print_progress: print("Simplifying date column...")
    copy_doc['date'] = copy_doc['date'].apply(lambda x: get_yyyy_mm(x))

    # prepare clean tokens
    if print_progress: print("Tokenizing quotes...")
    copy_doc['tokens'] = copy_doc['quotation'].apply(
        lambda x: prep_tokens_row(x, fix_contract, del_stop, lemmatize))

    # filter out unnecessary rows (by number of words/true words)
    if print_progress: print("Filtering rows...")
    copy_doc = filter_quotes(copy_doc)

    # get domain names
    if print_progress: print("Getting url domains...")
    copy_doc['websites'] = copy_doc['urls'].apply(lambda x: get_website(x))
    copy_doc.drop(columns='urls', inplace=True)

    # get the gender of the speaker
    if print_progress: print("Getting genders...")
    copy_doc['gender'] = copy_doc['qids'].apply(lambda x: find_gender(x, speaker_attributes))

    # Drop rows with gender = 'None'
    copy_doc = copy_doc[copy_doc['gender'].apply(lambda x: type(x)) != type(None)]

    return copy_doc