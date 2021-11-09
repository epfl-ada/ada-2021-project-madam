# view https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/ for some guidelines

import numpy as np
import modin.pandas as pd
from IPython.display import display

# patch_sklearn is an Intel® patch that allows to optimize sklearn operations
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.decomposition import LatentDirichletAllocation#, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer#, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt

# packages for data pre-processing
import contractions
import gensim, spacy
from nltk.corpus import stopwords

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    doc = nlp(" ".join(texts))
    texts_out = ' '.join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags])
    return texts_out

def prepare_docs(docs, fix_contract = True, del_stop = True, lemmatize = True):
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

    for doc in docs:
    # for doc in nlp.pipe(docs, batch_size=20):
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
        clean_docs.append(token_doc)

    return clean_docs

def topic_cluster(docs, num_topics = 10, print_res = False):
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
                                          batch_size=50,             # n docs in each learning iter
                                          evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                          n_jobs = -1,               # Use all available CPUs
                                          )
    lda_output = lda_model.fit_transform(docword_matrix)

    if print_res:
        # Log Likelihood: Higher the better
        print("Log Likelihood: ", lda_model.score(docword_matrix))
        # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
        print("Perplexity: ", lda_model.perplexity(docword_matrix))
        # See model parameters
        pprint(lda_model.get_params())

    return vectorizer, docword_matrix, lda_output

def grid_search(docs, num_topics, decay_vals, print_res = False, plot_res = False):

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

    if print_res:
        # Model Parameters
        print("Best Model's Params: ", model.best_params_)
        # Log Likelihood Score
        print("Best Log Likelihood Score: ", model.best_score_)
        # Perplexity
        print("Model Perplexity: ", best_lda_model.perplexity(docword_matrix))

    if plot_res:
        log_likelihood = [[round(model.cv_results_['mean_test_score'][index]) for index, gscore in enumerate(model.cv_results_['params']) if gscore['learning_decay']==val] for val in decay_vals]
        # Show graph
        plt.figure(figsize=(12, 8))
        for index, plot in enumerate(log_likelihood):
            plt.plot(num_topics, plot, label=str(decay_vals[index]))
        plt.title("Choosing Optimal LDA Model")
        plt.xlabel("Num Topics")
        plt.ylabel("Log Likelihood Scores")
        plt.legend(title='Learning decay', loc='best')
        plt.show()

    return vectorizer, docword_matrix, best_lda_model

def topics_docs_matrix(lda_model, docword_matrix, show_mat = False, show_dist = False):
    # Create Document - Topic Matrix
    lda_output = lda_model.transform(docword_matrix)
    # column names
    topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
    # index names
    docnames = ["Doc" + str(i) for i in range(docword_matrix.shape[0])]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic

    # Styling
    def color_green(val):
        color = 'green' if val > .3 else 'black'
        return 'color: {col}'.format(col=color)

    def make_bold(val):
        weight = 700 if val > .3 else 400
        return 'font-weight: {weight}'.format(weight=weight)

    if show_mat:
        # Apply Style
        display(df_document_topic.head(15).style.applymap(color_green).applymap(make_bold))

    df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
    df_topic_distribution.columns = ['Topic Num', 'Num Documents']

    if show_dist:
        display(df_topic_distribution)

    return df_document_topic, df_topic_distribution

def show_intertopic_distance(lda_model, docword_matrix, vectorizer):
    pyLDAvis.enable_notebook()
    panel = pyLDAvis.sklearn.prepare(lda_model, docword_matrix, vectorizer, mds='tsne')

    # if on an IDE, use this
    pyLDAvis.save_html(panel, 'lda.html')

    # if on a notebook, use this
    # panel

def get_topics_words(lda_model, vectorizer, show_words = False):
    # Topic-Keyword Matrix
    df_topic_keywords = pd.DataFrame(lda_model.components_)

    # Assign Column and Index
    df_topic_keywords.columns = vectorizer.get_feature_names()
    topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
    df_topic_keywords.index = topicnames

    if show_words:
        # View
        display(df_topic_keywords.head(10))

    return df_topic_keywords

def get_top_words_per_topic(lda_model, vectorizer, n_words = 15):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))

    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]

    return df_topic_keywords

def make_prediction(new_docs, lda_model, vectorizer, df_topic_keywords):

    clean_docs = prepare_docs(new_docs, del_stop = False)

    clean_docs = vectorizer.transform(clean_docs)
    topics_probs = lda_model.transform(clean_docs)

    topics = []
    for dist in topics_probs:
        topics.append(df_topic_keywords.iloc[np.argmax(dist), :].values.tolist())
    return topics, topics_probs
