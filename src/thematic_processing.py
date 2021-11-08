# packages needed for LDA grouping
from gensim import corpora, models

# patch_sklearn is an Intel® patch that allows to optimize sklearn operations
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# packages for data pre-processing
import contractions
import nltk, gensim, spacy
from nltk.corpus import stopwords

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    doc = nlp(" ".join(texts))
    texts_out = [token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]
    return texts_out

def prepare_docs(docs, fix_contract = True, lemmatize = True):
    """
    This function takes a list of docs, and prepares them for LDA grouping.
    Preparation includes:
        - Contractions expansion
        - Tokenization of sentences
        - Removal of stopwords
        - Lemmatization of words (NEEDS IMPROVEMENT)

    Parameters
    ----------
    docs : list
        List of docs to prepare.

    Returns
    -------
    clean_docs : list
        List of docs already prepared for LDA.
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

def gen_docword_matrix(data):
    vectorizer = CountVectorizer(analyzer='word',
                                 min_df=10,                        # minimum reqd occurences of a word
                                 stop_words='english',             # remove stop words
                                 lowercase=True,                   # convert all words to lowercase
                                 token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                 # max_features=50000,             # max number of uniq words
                                 )

    data_vectorized = vectorizer.fit_transform(data)

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
    dictionary_LDA = corpora.Dictionary(docs)

    corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in docs]

    lda_model = models.LdaModel(corpus,
                                num_topics=num_topics,
                                id2word=dictionary_LDA,
                                passes=5,
                                alpha=[0.01]*num_topics,
                                eta=[0.01]*len(dictionary_LDA.keys())
                               )

    for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=5):
        print(str(i)+": "+ topic)
        print()

    for corpse in corpus:
        print(lda_model[corpse])
