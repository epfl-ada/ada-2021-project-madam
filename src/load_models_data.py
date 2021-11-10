import nltk
import spacy

def nltk_download(data, quiet=True):
    """
    This function serves to automatically download all the necessary contents for nltk.
    
    Parameters
    ----------
    data : list
        nltk contents to download
    quiet : bool
        If true, don't show download messages.
        
    Returns
    -------
    None
    """
    for d in data:
        nltk.download(d, quiet=quiet)

def spacy_model_download(models):
    """
    This function serves to automatically download all the necessary contents for spacy.
    
    Parameters
    ----------
    data : list
        spacy contents to download
        
    Returns
    -------
    None
    """
    for model in models:
        spacy.cli.download(model)

if __name__=='__main__':

    # these are all the needed extras for nltk and spacy
    nltk_data = ['words', 'averaged_perceptron_tagger', 'stopwords']
    spacy_models = ['en_core_web_sm']

    nltk_download(nltk_data)
    spacy_model_download(spacy_models)