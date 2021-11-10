# -*- coding: utf-8 -*-
#Utiliy functions in this file include:
#date transformation
#token preparation
#filtering unnecessary rows
#extracting domain names from urls
#replacing none speakers with next highest probable speaker
#finding qids for replaced speakers
#finding speakers' gender


####double check unnecessary imports###

import numpy as np
#import pandas as pd
import nltk, gensim, spacy, contractions#, string

# nltk.download(['words', 'averaged_perceptron_tagger', 'stopwords'])

from nltk.corpus import stopwords, words
#from nltk.stem import WordNetLemmatizer
#from nltk import pos_tag, word_tokenize
from urllib.parse import urlparse

# nltk.download(['words', 'averaged_perceptron_tagger'])
# nltk.download('stopwords')
# spacy.cli.download("en_core_web_sm")

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

#Save real words in to wordlist
wordlist = set(words.words())

def get_yyyy_mm(date):
    """
    For a given date format this function returns YYYY-MM format
    """
    yyyy_mm=date.to_period('m')
    return yyyy_mm

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    doc = nlp(" ".join(texts))
    texts_out = [token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]
    return texts_out

def prep_tokens_row(doc, fix_contract = True, del_stop = True, lemmatize = True):
    """
    This function takes a doc (list of strings), and prepares it for analysis.
    Preparation includes:
        - Contractions expansion
        - Tokenization of sentences
        - Removal of stopwords
        - Lemmatization of words
    Parameters
    ----------
    doc : list
        Document to prepare.
    Returns
    -------
    clean_doc : list
        Document prepared for analysis.
    """

    # iterate through all the docs to process them
    if fix_contract:
      # expand contractions
      doc = contractions.fix(doc)
      # tokenize doc and remove punctuation
      token_doc = gensim.utils.simple_preprocess(str(doc), deacc=True) # deacc=True removes punctuations
    if del_stop:
      # remove stopwords
      token_doc = [word for word in token_doc if not word in set(stopwords.words('english'))]
    if lemmatize:
      # Do lemmatization keeping only Noun, Adj, Verb, Adverb
      token_doc = lemmatization(token_doc, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    return token_doc

# Test filter
def count_true_words(tokens):
  """ Counts the number of true words in a list of tokens. """
  true_words = [1 if token in wordlist else 0 for token in tokens]
  return sum(true_words)

def filter_quotes(doc, min_size = 1, min_true_size=1):
    """
    This function receives a document and deletes rows which have
        - less than min_size words in the selected column
        - less than min_true_size real words in the selected columns

    Parameters
    ----------
    doc : Dataframe
        Quotes dataframe


    Returns
    -------
    doc_new : Dataframe
        Filtered version of original doc.
    """

    #delete rows which have less than min_size words and reset index
    doc_new=doc[doc['tokens'].apply(lambda x: len(x)) >= min_size].reset_index(drop=True)

    #delete rows which have less than min_true_size real words and reset index
    doc_new = doc_new[doc_new['tokens'].apply(lambda x: count_true_words(x)) >= min_true_size]

    return doc_new.reset_index(drop=True)

def get_website(doc):
    """
    This function extracts the domain names from the listed urls
    """
    web_list=[]
    for j in range(len(doc)):
        #get website as www.blabla.com or as blabla.com according to provided url and split using '.'
        core=urlparse(doc[j]).netloc.split('.')
        #get wrid of www. and .com parts
        if len(core)>2:
            web_list.append(core[1])
        else :
            web_list.append(core[0])
    return web_list

"""def replace_none_speaker(doc):
 
    This function replaces speakers originally assigned 'None' with the speaker that have second highest probability

    doc_copy=doc.copy()

    doc_copy['speaker'][doc_copy['speaker'] == 'None'] = doc_copy['probas'][doc_copy['speaker'] == 'None'].apply(lambda x: x[1][0])
    return doc_copy"""

def find_qids(speaker, doc_speaker_attributes):
    """
    This function finds qids for missing rows
    Since we replaced None speakers those rows have [] as qids
    """
    qid_list=[]
    for ind, lines in doc_speaker_attributes['aliases'].items():
        if speaker in str(lines):
            qid_list.append(ind)
    return qid_list

def find_gender(qids, doc_speaker_attributes):
    """
    This function finds the gender of the speaker according to speaker_attributes
    If there's multiple qids with different genders, returns None
    """
    if len(qids)==0:
        return None

    else:
    # Check every qid, if some has gender != None, then we choose that one
        for i in range(len(qids)):
            try:
                gender = doc_speaker_attributes.loc[qids[i]]['gender_label']
            except:
                gender = None

    return gender
