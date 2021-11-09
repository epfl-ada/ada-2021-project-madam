# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:07:44 2021

@author: tsfei
"""
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.feature_extraction.text import TfidfVectorizer

# packages needed for LDA grouping
from gensim import corpora, models


def words_cleaner(words, min_size = 0, del_nums = True):
    """
    This function takes a list of words, and cleans it based on the parameters provided.

    Parameters
    ----------
    words : list
        List of words to clean.
    min_size : int
        All words with size < min_size are removed.
    del_nums : bool
        If True, removes all words which can be converted to numbers.

    Returns
    -------
    words_clean : list
        List with the clean words obeying the parameters provided
    """
    words_clean = []
    for word in words:
        if del_nums:
            # if we want to remove numbers, then we try to convert the word to number
            try:
                float(word)
            # if it raises an exception then it's fine
            except:
                pass
            # if no exceptions are raised then we skip it
            else:
                continue

        # we also only want words bigger than min_size
        if len(word) >= min_size:
            words_clean.append(word)

    # return clean word list
    return words_clean

def keywords_extractor_sklearn(docs, num_words = 100, min_size = 0):
    """
    This function takes a series of documents and extracts the 'num_words'
    most important keywords from it, using TF-IDF to spot these words.

    Parameters
    ----------
    docs : list
        List of str, contains the docs we wish to analyze.
    num_words : int
        Number of words to return.
    min_size : int
        All words with len(word) < min_size are automatically excluded from
        the output. If None does nothing.

    Returns
    -------
    top_words : list
        List containing the most relevant keywords from the docs.
    """

    # Generate a vectorizer to analyze the documents via TF-IDF
    # Automatically strip word accents if possible and exclude all stop_words in english language
    vectorizer = TfidfVectorizer(strip_accents = 'ascii', stop_words = 'english')
    # Get the TF-IDF Matrix from the docs provided
    tf_idf_matrix = vectorizer.fit_transform(docs)
    # And get the words from the same docs
    words = np.array(vectorizer.get_feature_names_out())
    # Then sort the words according to their TF-IDF weights
    tfidf_sorting = np.argsort(tf_idf_matrix.toarray()).flatten()[::-1]

    # Clean the words and return only the most relevant
    top_words = words_cleaner(words[tfidf_sorting], min_size = min_size)[:num_words]

    return top_words

def topic_cluster(docs, num_topics = 1):
    """
    This function takes a set of documents and clusters them into topics.

    Parameters
    ----------
    docs : list
        List of tokenized documents to cluster into categories.
    num_topics : int
        Number of topics to model.

    Returns
    -------
    ???
    """
    docs = [doc.split() for doc in docs]
    dictionary_LDA = corpora.Dictionary(docs)

    corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in docs]

    lda_model = models.LdaModel(corpus,
                                num_topics=num_topics,
                                id2word=dictionary_LDA,
                                passes=5,
                                alpha=[0.01]*num_topics,
                                eta=[0.01]*len(dictionary_LDA.keys())
                                )

    for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=10):
        print(str(i)+": "+ topic)
        print()

    for corpse in corpus:
        print(lda_model[corpse])
