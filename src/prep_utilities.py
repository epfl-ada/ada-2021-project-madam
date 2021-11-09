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
import pandas as pd
import nltk, gensim, spacy, contractions, string
from nltk.corpus import stopwords, words
#from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize 
from urllib.parse import urlparse
nltk.download(['words','averaged_perceptron_tagger'])

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

def prep_tokens(docs, fix_contract = True, lemmatize = True):
    """
    This function takes a list of docs, and prepares them for analysis.
    Preparation includes:
        - Contractions expansion
        - Tokenization of sentences
        - Removal of stopwords
        - Lemmatization of words
    Parameters
    ----------
    docs : list
        List of docs to prepare.
    Returns
    -------
    clean_docs : list
        List of docs already prepared for analysis.
    """
    clean_docs = []

    # iterate through all the docs to process them
    for doc in docs:
        if fix_contract:
            # expand contractions
            doc = contractions.fix(doc)
        # tokenize doc and remove punctuation
        token_doc = gensim.utils.simple_preprocess(str(doc), deacc=True) # deacc=True removes punctuations

        if lemmatize:
            # Do lemmatization keeping only Noun, Adj, Verb, Adverb
            token_doc = lemmatization(token_doc, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        clean_docs.append(token_doc)
    return clean_docs

# Test filter
def filter_quotes(doc, min_size = 1, min_true_size=1, none_prob_threshold=0.9):
    """
    This function receives a document and deletes rows which have
        - less than min_size words in the selected column
        - less than min_true_size real words in the selected columns
        - more than none_prob_threshold probability assigned to speaker as 'None'
    
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
    doc_new=doc[doc['tokens'].map(len)>min_size].reset_index(drop=True)
    
    #delete rows which have less than min_true_size real words and reset index
    true_word_count=[]
    for words in doc_new['tokens']:
        true_word=[]
        for w in words:
            true_word.append(w in wordlist)
        true_word_count.append(sum(true_word))
    doc_new['true']=true_word_count
    doc_new=doc_new[doc_new['true']>min_true_size].drop(columns='true').reset_index(drop=True)
    
    #delete rows which have more than none_prob_threshold probability assigned to speaker as 'None'
    """ind_list=[]
    for ind, i in enumerate(doc_new['probas']):
        for j in range(len(i)):
            if i[j][0]=='None':
                if float(i[j][1])>none_prob_threshold:
                    ind_list.append(ind)
    doc_new.drop(doc_new.index[ind_list], axis=0, inplace=True)"""
    none_rows = doc_new[(doc_new['speaker'] == 'None') & (doc_new['probas'].apply(lambda x: float(x[0][1])) > none_prob_threshold)]
    
    doc_new = doc_new.drop(none_rows.index, axis=0)

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

def replace_none_speaker(doc):
    """
    This function replaces speakers originally assigned 'None' with the speaker that have second highest probability  
    """
    doc_copy=doc.copy()
    for i in range(len(doc_copy)):
        if doc_copy['speaker'][i]=='None':
            doc_copy['speaker'][i]=doc_copy['probas'][i][1][0]
    doc_copy.drop(columns='probas', inplace=True)
    return doc_copy      

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