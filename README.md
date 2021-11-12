## MIND THE (GENDER) GAP

### ABSTRACT

Gender equality is a long-fought battle, one that has already taken long strides. But has it been enough? On this project we will analyse whether there is still a fundamental difference in the way quotes from men vs women are handled by various news sources. More than simply the representativity of genders, we want to study the content of the quotes. Is there a difference in the topics men or women are cited on? What is the overall sentiment (positive or negative) associated with the quotes? What kind of language is cited?
News sources have the power to influence society on a large scale, and biases can be easily propagated. Via the choice of quotes, by their content or sentiment, it is entirely possible to reflect an image (potentially distorted) of a group of people. Is that happening?

### Research Questions

We hope to answer the following questions.

- What is the representativity of quotes by gender, and how has it evolved over time?
- Which are the main topics in the quotes from different genders?
- What is the sentiment (negative, neutral, positive) associated to the quotes by gender, distributed by themes? *
- What is the complexity of the speech quoted, by gender and by theme?

Furthermore, we want to study a couple of very influential websites (which ones is still TBD), and compare the sentiment and portion of the quotes by gender, to decide which ones have a more equalitarian roster of quotes.

\*While quotes are inherently unchangeable, the context in which they are used, and the predominance of their sentiment, can reveal information about the sources' predisposition towards the quoted. Concretely, if a newspaper tendentially selects quotes with a negative sentiment for women, while mainly neutral/positive for men, this could be a display of an internal bias which is being propagated to the reader.

---

### Proposed additional datasets

At the moment, there is no prospect of any additional datasets. Quotebanks and WikiData should be enough to conduct our research.

---

### Methods

Before anything else, there had to be data pre-processing, which sent the raw data through a pipeline to clean it.

- **Data Sampling :** As the original dataset is very large, the data was randomly sampled (around 5%) from the 2015-2020 quotes to be used during data explorations. This sampling is necessary to capture any correlations of the quotes throughout the years;
- **Structural Changes :** We removed quotes: 1) With `None` speaker; 2) Whose number of "true words" were below a certain threshold; 3) Whose speaker gender is `None`.
- **Content Changes :** The quotes were manipulated into an analysable form. The sentences were tokenized, the stopwords were removed and the words were lemmatized. These transformations reduced the variance of words, creating a more meaningful bag of words (BoG).

To study the different aspects mentioned in the research questions, several methods were considered.

- **Characteristics Identification :** Data extracted from Quotebank or Wikidata;
- **Topic Identification :** To identify the topics in a set of quotes, we used 'Latent Dirichlet Allocation' (or [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)). This model clusters the BoG provided into a set of *n* topics. Then, it provides the words associated to each topic and, by human interpretation, we can label them. After being trained, it can be used to identify the main topics in any given quote;
- **Text Sentiment :** The sentiment analysis was done via a polarity score, which ranges from -1 (negative) to 1 (positive);
- **Text Readability / Complexity :** Using the [Dale-Chall Readability Formula](https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula) we can measure the 'difficulty' of any given text.

In order to conduct this analysis and use these methods, several assumptions were made. 

- **Topic repetition (aka everybody says the same) :** As unsupervised learning is computationally expensive, it's wasteful to train the model with the whole dataset to get the classification of topics. We assumed that, with a large enough sample, we can classify all the topics presented with high accuracy. The topics of the remaining quotes can be classified using the trained model, as they will most likely contain repetitions / variations of the topics observed before.
- **Text complexity and education level :** To relate these two concepts we assumed that people who write more 'difficult' text will, tendentially, have a higher level of education. In fact, the Dale-Chall Readability Formula can be converted directly into an indication of the school-level required to *understand* the text, which we assume is correlated with the education level required to *generate* the text.

**_Note :_** These methods required the usage of external packages. For a detailed list see [requirements.txt](https://github.com/epfl-ada/ada-2021-project-madam/blob/main/requirements.txt).

---

### Proposed Timeline

01/11 - 07/11 : Exploratory data analysis; Data extraction (samples); Study and implementation of NLP algorithms

08/11 - 12/11 : Creation of notebook; Phase 2 README; Repo clean-up; (PHASE 2 DELIVERY)

15/11 - 21/11 : Runtime optimizations (pandas, spaCy, numpy, sklearn)

22/11 - 28/11 : Run full dataset through pipeline; Rerun methods

29/11 - 05/12 : Temporal analysis on full dataset; Interpret results

06/12 - 12/12 : Crafting 'Data Story'; Working on GitHub Pages

13/12 - 17/12 : Finalize GitHub Pages; Final repo clean-up; (PHASE 3 DELIVERY)

---

### Internal Organization

- Data Extraction : André, Khanh, Medya
- Pre-processing pipeline : Medya, Tomás, André
- Study of NLP methods : Khanh, Tomás
- Implementation of NLP methods : Medya, Tomás
- Runtime Optimization : André, Tomás
- Results Interpretation : Collective
- Website + Data Story : Collective

---

### Repo Architecture
<pre>
├─── data : (Content .gitignored) Folder to store the data
├─── src
│   ├─── contractions.py : table necessary to expand contractions
│   ├─── load_models_data.py : automatically download all packages from nltk and spaCy
│   ├─── prep_pipeline.py : pipeline to process quotes and create dataframe with the features for analysis
│   ├─── prep_utilities.py : NLP tasks and engineer features for the dataset 
│   ├─── sampling_data.py : generate a 1/20 sample of all quotes
│   ├─── text_scores.py : measure the complexity of the quotes
│   └─── thematic_processing.py : clustering quotes by topics
├─── Data_Prep.ipynb : Phase 2 notebook
├─── README.md : [ERROR] Infinite Recursion :)
└─── requirements.txt : packages, and versions, used
</pre>
