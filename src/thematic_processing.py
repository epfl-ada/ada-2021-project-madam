# view https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/ for some guidelines


# packages needed for LDA grouping
from gensim import corpora, models

# patch_sklearn is an Intel® patch that allows to optimize sklearn operations
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt

# packages for data pre-processing
import contractions
import nltk, gensim, spacy
from nltk.corpus import stopwords

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    doc = nlp(" ".join(texts))
    texts_out = ' '.join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags])
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
        # # remove stopwords
        # token_doc = [word for word in token_doc if not word in set(stopwords.words('english'))]
        if lemmatize:
            # Do lemmatization keeping only Noun, Adj, Verb, Adverb
            token_doc = lemmatization(token_doc, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        clean_docs.append(token_doc)
    return clean_docs

def topic_cluster(docs, num_topics = 10):
    vectorizer = CountVectorizer(analyzer='word',
                                  min_df=10,                        # minimum reqd occurences of a word
                                  stop_words='english',             # remove stop words
                                  lowercase=True,                   # convert all words to lowercase
                                  token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                  # max_features=50000,             # max number of uniq words
                                  )

    docword_matrix = vectorizer.fit_transform(docs)

    # Build LDA Model
    lda_model = LatentDirichletAllocation(n_components=num_topics,               # Number of topics
                                          max_iter=10,               # Max learning iterations
                                          learning_method='online',
                                          random_state=100,          # Random state
                                          batch_size=128,            # n docs in each learning iter
                                          evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                          n_jobs = -1,               # Use all available CPUs
                                          )
    lda_output = lda_model.fit_transform(docword_matrix)

    # Log Likelyhood: Higher the better
    print("Log Likelihood: ", lda_model.score(docword_matrix))

    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    print("Perplexity: ", lda_model.perplexity(docword_matrix))

    # See model parameters
    pprint(lda_model.get_params())

def grid_search(docs, num_topics, decay_vals):

    vectorizer = CountVectorizer(analyzer='word',
                                  min_df=10,                        # minimum reqd occurences of a word
                                  stop_words='english',             # remove stop words
                                  lowercase=True,                   # convert all words to lowercase
                                  token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                  # max_features=50000,             # max number of uniq words
                                  )

    docword_matrix = vectorizer.fit_transform(docs)

    search_params = {'n_components': num_topics, 'learning_decay': decay_vals}

    # Init the Model
    lda = LatentDirichletAllocation()

    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params)

    # Do the Grid Search
    model.fit(docword_matrix)

    # Best Model
    best_lda_model = model.best_estimator_

    # Model Parameters
    print("Best Model's Params: ", model.best_params_)

    # Log Likelihood Score
    print("Best Log Likelihood Score: ", model.best_score_)

    # Perplexity
    print("Model Perplexity: ", best_lda_model.perplexity(docword_matrix))

    log_likelihood = [[round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['learning_decay']==val] for val in decay_vals]

    # Show graph
    plt.figure(figsize=(12, 8))
    for index, plot in enumerate(log_likelihood):
        plt.plot(num_topics, plot, label=str(decay_vals[index]))
    plt.title("Choosing Optimal LDA Model")
    plt.xlabel("Num Topics")
    plt.ylabel("Log Likelyhood Scores")
    plt.legend(title='Learning decay', loc='best')
    plt.show()


# def topic_cluster(docs, num_topics = 1):
#     """
#     This function takes a set of documents and clusters them into topics.

#     Parameters
#     ----------
#     docs : list
#         List of tokenized documents to cluster into categories.
#     num_topics : int
#         Number of topics to model.

#     Returns
#     -------
#     ???
#     """
#     docs = [doc.split() for doc in docs]
#     dictionary_LDA = corpora.Dictionary(docs)

#     corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in docs]

#     lda_model = models.LdaModel(corpus,
#                                 num_topics=num_topics,
#                                 id2word=dictionary_LDA,
#                                 passes=5,
#                                 alpha=[0.01]*num_topics,
#                                 eta=[0.01]*len(dictionary_LDA.keys())
#                                 )

#     for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=10):
#         print(str(i)+": "+ topic)
#         print()

#     for corpse in corpus:
#         print(lda_model[corpse])
