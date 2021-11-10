import nltk
import spacy

def nltk_download(data, quiet=True):
    for d in data:
        nltk.download(d, quiet=quiet)

def spacy_model_download(models):
    for model in models:
        spacy.cli.download(model)

if __name__=='__main__':

    nltk_data = ['words', 'averaged_perceptron_tagger', 'stopwords']
    spacy_models = ['en_core_web_sm']

    nltk_download(nltk_data)
    spacy_model_download(spacy_models)