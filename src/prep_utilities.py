# -*- coding: utf-8 -*-
#Utiliy functions in this file include:
#date transformation
#token preparation
#filtering unnecessary rows
#extracting domain names from urls
#finding speakers' gender


import gensim, spacy
import re
from nltk.corpus import stopwords, words
from urllib.parse import urlparse
from src.contractions import CONTRACTION_MAP

# Initialize spacy 'en_core_web_sm' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

#Save real words in to wordlist
wordlist = set(words.words())

# save stopwords in to stop_words
stop_words = set(stopwords.words('english'))

def get_yyyy_mm(date):
    """
    For a given date format this function returns YYYY-MM format
    """
    yyyy_mm=date.to_period('m')
    return yyyy_mm

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """
    This function lemmatizes a given sentence, and return the sentence already lemmatized.
    For detailed information on spacy see https://spacy.io/api/annotation
    
    Parameters
    ----------
    texts : list
        (Already tokenized) words to lemmatize.
    allowed_postags : list
        Types of words we wish to allow to pass to the processed output.
        
    Returns
    -------
    texts_out : list
        List with lemmatized input.
    """
    # turn the text into something spacy can process. Namely, this saves
    # Text, Lemma, POS, Tag, Dep, Shape, alpha, stop
    # for all the words
    doc = nlp(" ".join(texts))
    # now we only save the lemmas of the words with allowed postags
    texts_out = [str(token.lemma_).lower() if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]
    return texts_out

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    """
    This function expands all the contractions within a text, for e.g. I've -> I have.
    This code was taken from this source: 
    https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72
    """
    
    # regex to spot contractions
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def prep_tokens_row(doc, fix_contract=True, del_stop=True, lemmatize=True):
    """
    This function takes a doc, and prepares it for analysis.
    Preparation includes:
        - Contractions expansion
        - Tokenization of sentences
        - Removal of stopwords
        - Lemmatization of words
    
    Parameters
    ----------
    doc : str
        Document to prepare.
    fix_contract : bool
        If true, expand contractions (don't -> do not; I've -> I have,...).
    del_stop : bool
        If true, remove stopwords.
    lemmatize : bool
        If true, lemmatize words.
    Returns
    -------
    clean_doc : list
        Document prepared for analysis.
    """
    # iterate through all the docs to process them
    if fix_contract:
      # expand contractions
      doc = expand_contractions(doc)
        
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
  """
  Counts the number of true words in a list of tokens.
  
  Parameters
    ----------
    tokens : list
        List of tokens to measure the number of true words.
    
    Returns
    -------
    true_words : int
        Number of true words in input of tokens.

    """
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
    min_size : int
        Minimum number of words in quote
    min_true_size : int
        Minimum number of real words in quote

    Returns
    -------
    doc_new : pandas.DataFrame
        Filtered version of original doc.
    """

    #delete rows which have less than min_size words and reset index
    doc_new=doc[doc['tokens'].apply(lambda x: len(x)) >= min_size].reset_index(drop=True)

    #delete rows which have less than min_true_size real words and reset index
    doc_new = doc_new[doc_new['tokens'].apply(lambda x: count_true_words(x)) >= min_true_size].reset_index(drop=True)

    return doc_new

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

def find_gender(qids, doc_speaker_attributes):
    """
    This function finds the gender of the speaker according to speaker_attributes
    If there's multiple qids with different genders, returns None
    
    Parameters
    ----------
    qids : list
        List of qids to use to find the gender
    doc_speaker_attributes : pandas.DataFrame
        Dataframe with information on the speakers

    Returns
    -------
    gender : str
        Gender of the speaker determined from the qids.
    """
    if len(qids)==0:
        return None

    else:
    # Check every qid, if some has gender != None, then we choose that one
        for qid in qids:
            try:
                gender = doc_speaker_attributes.loc[qid]['gender_label']
            except:
                gender = None

    return gender
