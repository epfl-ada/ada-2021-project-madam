## MIND THE (GENDER) GAP

### ABSTRACT

Gender equality is a long-fought battle, and one that has already taken long strides. With this project we set ourselves out to analyze whether there is still a fundamental difference in the way quotes from men vs women are handled by various news sources. More than simply looking at the representativity of genders, we want to study the content of the quotes. Is there a difference in the topics men or women are cited on? What is the overall sentiment (positive or negative) associated with the quotes? What kind of language is cited (simpler or more complex)?
News sources have the power to influence society on a large scale. Biases can be easily propagated, via the choice of quotes, by their content or sentiment, it is entirely possible to reflect an image (potentially distorted) of a group of people. Is that happening? That's what we want to discover.

### Research Questions

With this project we hope to answer the following questions.

- What is the representativity of quotes by gender, and how has it evolved over time?
- Which are the main topics in the quotes from different genders?
- What is the sentiment (negative, neutral, positive) associated to the quotes by gender, potentially distributed by themes? *
- What is the complexity of the speech from the quoted individuals, by gender and by theme?

Then, we want to study a couple of very influential websites (which ones is still TBD), and compare the sentiment and portion of the quotes by gender, to decide which ones have a more equalitarian roster of quotes.

\*While quotes are inherently unchangeable, the context in which they are used, and the predominance of their sentiment, can reveal information about the sources' predisposition towards the quoted. For example, if a newspaper tendentially selects quotes with a negative sentiment for women, while mainly neutral/positive for men, this could be a display of an internal bias.

---

### Proposed additional datasets

From the current outlook of the project, there is no prospect of any additional datasets, the Quotebanks and WikiData should be enough to conduct our research.

---

### Methods

Before any study could be conducted, there had to be a data pre-processing, which sent the raw data through a pipeline to clean it and make analyzable.
- **Data Sampling :** As the original dataset is very large, the data was sampled to be used during data explorations. This sampling is necessary to capture any correlations of the quotes throughout the years;
- **Structural Changes :** (TALK ABOUT REMOVAL OF ROWS, REPLACEMENT OF GENDERS,...);
- **Content Changes :** The quotes were manipulated into a form easier to analyze. For that, the sentences were tokenized, the stopwords were removed and the words were lemmatized. These allowed to reduce the variance of words, creating a more meaningful bag of words (BoG).

To study the different aspects mentioned in the research questions, several methods were used.

- **Characteristics Identification :** Data extracted from Quotebank or Wikidata (via QIds);
- **Topic Identification :** To identify the main topics in a set of quotes, we used 'Latent Dirichlet Allocation' (or [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)). This model clusters the BoG provided into a set of *n* topics. Then, it provides the main words associated to each topic and, by human interpretation, it is possible to label them. This model, after being trained, can be used to identify the main topics from any given quote;
- **Text Sentiment :** The sentiment analysis was done via a polarity score, which ranges from -1 (negative) to 1 (positive);
- **Text Readability / Complexity :** Using the [Dale-Chall Readability Formula](https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula) we can measure the 'difficulty' of any given text.

In order to conduct this analysis and use these methods, several assumptions were made. 

- **Topic generalization :** Since unsupervised learning is computationally expensive, it is not necessary to train the model with the whole dataset to get the classification of the topics. We assumed that, with a large enough sample, we can classify the topics presented with high accuracy. The topics of the remaining quotes can be classified using the model trained from this sampled training set, as they will most likely contain repetitions / variations of the topics observed before.
- **Relationship between complexity and education level :** The base assumption used to establish this connection is that people who write more 'difficult' text will, tendentially, have a higher level of education. In fact the Dale-Chall Readability Formula can be converted directly into an indication of the school-level required to *understand* the text, which we assume can be correlated with the education level required to *generate* the text.

**_Note :_** These methods required the usage of external packages. For a more detailed exposition of the packages used, see [requirements.txt](https://github.com/epfl-ada/ada-2021-project-madam/blob/main/requirements.txt).

---

### Proposed Timeline

---

### Internal Organization

- Data Extraction : André, Khanh, Medya
- Pre-processing pipeline : Medya, Tomás, André
- Study of adequate NLP methods : Khanh, Tomás
- Implementation of NLP methods : Medya, Tomás
- Runtime Optimization : André, Tomás
- Results Interpretation : 
- Data Story Construction : 

---

### Questions for TAs

---

### Repo Architecture
<pre>
├─── src
│   ├─── contractions.py : table necessary to expand contractions
│   ├─── load_models_data.py : functions to automatically download all packages from nltk and spacy
│   ├─── prep_pipeline.py : pipeline to process the quotes and create dataframe containing the features needed for analysis
│   ├─── prep_utilities.py : functions to do NLP tasks and engineer features for the dataset 
│   ├─── sampling_data.py : functions to generate a 1/20 sample of all the quotes
│   ├─── text_scores.py : functions used to measure the complexity of the quotes
│   └─── thematic_processing.py : functions used for clustering quotes by topics
├─── Data_Prep.ipynb
├─── README.md : [ERROR] Infinite Recursion :)
└─── requirements.txt : file showing the packages, and version, used
</pre>
