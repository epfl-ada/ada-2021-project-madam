# view https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/ for some guidelines

# standard packages to import
import numpy as np
import ray
ray.init(log_to_driver=False, ignore_reinit_error=True)
import pandas as pd
from IPython.display import display

# patch_sklearn is an Intel® patch that allows to optimize sklearn operations
from sklearnex import patch_sklearn
patch_sklearn()
# sklearn is used for the LDA theme clustering
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# tune_sklearn using cutting-edge methods to increase the speed of sklearn
# here we use it to optimize the grid search for LDA
from tune_sklearn import TuneGridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt

def topic_cluster(docs, num_topics = 10, print_res = False):
    """
    Function to perform the topic clustering for a set of documents.
    
    Parameters
    ----------
    docs : list
        List containing all the documents to cluster
    num_topics : int
        Number of topics to cluster the docs into. This
        hyperparameter NEEDS TO BE OPTIMIZED PREVISOULY.
        (See grid_search below)
    print_res : bool
        If true, print the results after algorithm
        completion.
        
    Returns
    -------
    vectorizer : sklearn.feature_extraction.text.CountVectorizer
        Vectorizer containing the trained model
    docword_matrix : np.ndarray
        Document-term array generated from the input docs
    lda_output : np.darray
        Array containing the samples and the features
    """
    # create the vectorizer to apply LDA
    vectorizer = CountVectorizer(analyzer='word',
                                  min_df=10,                        # minimum required occurences of a word
                                  stop_words='english',             # remove stop words
                                  lowercase=True,                   # convert all words to lowercase
                                  token_pattern='[a-zA-Z0-9]{3,}'   # num chars > 3
                                  )

    # Generate the document-term matrix from the input data
    docword_matrix = vectorizer.fit_transform(docs)

    # Build LDA Model
    lda_model = LatentDirichletAllocation(n_components=num_topics,   # Number of topics
                                          max_iter=10,               # Max learning iterations
                                          learning_method='online',
                                          random_state=100,          # Random state
                                          batch_size=50,             # n docs in each learning iter
                                          evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                          n_jobs = -1,               # Use all available CPUs
                                          )
    
    # find the document term matrix for the documents provided
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
    """
    Function to perform grid search and find the optimal hyperparameters.

    Parameters
    ----------
    docs : list
        List containing all the documents to cluster
    num_topics : list
        List with the values of num_topics to run 
        through grid search
    decay_vals : list
        List with the values of decay rate to run
        through grid search
    print_res : bool
        If true, print the results after grid search
    plot_res : bool
        If true, plot the results after grid search
        
    Returns
    -------
    vectorizer : sklearn.feature_extraction.text.CountVectorizer
        Vectorizer containing the trained model
    docword_matrix : np.ndarray
        Document-term array generated from the input docs
    best_lda_output : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. 
    """
    vectorizer = CountVectorizer(analyzer='word',
                                  min_df=10,                        # minimum reqd occurences of a word
                                  stop_words='english',             # remove stop words
                                  lowercase=True,                   # convert all words to lowercase
                                  token_pattern='[a-zA-Z0-9]{3,}'   # num chars > 3
                                  )

    # Generate the document-term matrix from the input data
    docword_matrix = vectorizer.fit_transform(docs)

    # define the search parameters for grid search
    search_params = {'n_components': num_topics, 'learning_decay': decay_vals}

    # Init the Model
    lda = LatentDirichletAllocation()
    
    # define the model for the GridSearch
    # NOTE : TuneGridSearchCV provides a tremendous speed boost over GridSearchCV
    # For more info see
    #     https://github.com/ray-project/tune-sklearn
    #     https://towardsdatascience.com/5x-faster-scikit-learn-parameter-tuning-in-5-lines-of-code-be6bdd21833c
    model = TuneGridSearchCV(lda,
                             param_grid = search_params,
                             early_stopping = "MedianStoppingRule",
                             max_iters = 5)

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
        # get the log_likelihoods for all the values of learning decay to graph
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
    """
    This function allows us to retrieve the most dominant topics for each document
    
    Parameters
    ----------
    lda_model : estimator
        THE OUTPUT best_lda_model FROM grid_search SHOULD BE USED HERE
    docword_matrix : np.ndarray
        Document-term matrix.
    show_mat : bool
        If true, display the df_document_topic matrix after its creation
    show_dist : bool
        If true, display the df_topic_distribution matrix after its creation
        
    Returns
    -------
    df_document_topic : pandas.DataFrame
        DataFrame containing the main topics per document
    df_topic_distribution : pandas.DataFrame
        DatFrame containing the distribution of topics over-all
    """
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

def show_intertopic_distance(lda_model, docword_matrix, vectorizer, output_name = 'lda.html'):
    """
    Function to create a visual representation of the intertopic distance,
    which allows to visually gauge the correctness of the model. It also
    allows to easily label the topics generated.
    
    Parameters
    ----------
    lda_model : estimator
        THE OUTPUT best_lda_model FROM grid_search SHOULD BE USED HERE
    docword_matrix : np.ndarray
        Document-term matrix.
    vectorizer : sklearn.feature_extraction.text.CountVectorizer
        THE OUTPUT vectorizer FROM grid_search SHOULD BE USED HERE
    output_name : str
        Sets the name of the output file.
        
    Returns
    -------
    None
    """
    pyLDAvis.enable_notebook()
    # create the visual representation
    panel = pyLDAvis.sklearn.prepare(lda_model, docword_matrix, vectorizer, mds='tsne')

    pyLDAvis.save_html(panel, output_name)

def get_topics_words(lda_model, vectorizer, show_words = False):
    """
    Function to get the main words associated with each topic.
    
    Parameters
    ----------
    lda_model : estimator
        THE OUTPUT best_lda_model FROM grid_search SHOULD BE USED HERE
    vectorizer : sklearn.feature_extraction.text.CountVectorizer
        THE OUTPUT vectorizer FROM grid_search SHOULD BE USED HERE
    show_words : bool
        If true, display df_topic_keywords after its generation
        
    Returns
    -------
    df_topic_keywords : pd.DataFrame
        DataFrame containing the main keywords for each topic
    """
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
    """
    THIS FUNCTION GOT KINDA SCREWED UP WHEN WE CHANGED PREP_TOKENS, I NEED TO CORRECT IT.
    """

    clean_docs = vectorizer.transform(clean_docs)
    topics_probs = lda_model.transform(clean_docs)

    topics = []
    for dist in topics_probs:
        topics.append(df_topic_keywords.iloc[np.argmax(dist), :].values.tolist())
    return topics, topics_probs
