## MIND THE (GENDER) GAP

### ABSTRACT

---

### Research Questions

On this project we set ourselves out to analyze whether there is any fundamental difference in the way quotes from men vs women and handled by various news sources. For all the available quotes, we want to study:

- What is the representativity of quotes by gender, and how has it evolved over time?
- Which are the main topics in the quotes from different genders?
- What is the sentiment (negative, neutral, positive) associated to the quotes by gender, potentially distributed by themes? *
- What is the complexity of the speech from the quoted individuals, by gender and by theme?

Then, we want to study a couple of very influencial websites [WE STILL HAVE TO DECIDE WHICH], and compare the sentiment and portion of the quotes by gender, to decide which ones have a more gender equal roster of quotes.

\*While quotes are inherently unchangeable, the context in which they are used and the predominance of their sentiment can reveal information about the sources predisposition towards the quoted. For example, if a newspaper tendentially selects quotes with a negative sentiment for women, while mainly neutral/positive for men, this could be a display of an internal bias which would reflect a different image for each gender.

---

### Proposed additional datasets

From the current outlook of the project, there is no prospect of any additional datasets.

---

### Methods

Before any study could be conducted, there had to be a data pre-processing, which sent the raw data through a pipeline until it was in a state where it could be processed.
- **Data Sampling :** As the orginal dataset is enormous, the data was sampled to be utilized during data explorations. This sample is necessary to capture any correlations of the quotes throughout the years;
- **Structural Changes :** ;
- **Content Changes :** The quotes were manipulated into a form easier to analyze. For that, the sentences were tokenized, the stopwords were removed and the words were lemmatized. These allowed to reduce the variance of words, creating a more meaningful bag of words (BoG).

To study the different aspects mentioned in the research questions, several methods were used.

- **Characteristics Identification :** Data extracted from Quotebank or Wikidata (via QIds);
- **Topic Identification :** To identify the main topics in a set of quotes, we used 'Latent Dirichlet Allocation' (or [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)). This model clusters the BoG provided into a set of *n* topics (where *n* is a hyperparameter). Then, it provides the main words associated to each topic and, by human interpretation, it is possible to label them. This model, after being trained, can be used to identify the main topics from any given quote;
- **Text Sentiment :** The sentiment analysis was done via a polarity score, which ranges from -1 (negative) to 1 (positive). ;
- **Text Readability / Complexity :** Using the [Dale-Chall Readability Formula](https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula) we can measure the 'difficulty' of any given text (the base assumption is that people who write more 'difficult' text will, tendentially, have a higher level of education).

In order to conduct this analysis and use this methods, several assumptions had to be made. STILL NEED TO WRITE THIS MORE CAREFULLY

- **Topic generalization :** Since unsupervised learning is computationally expensive, it is not necessary to train the LDA the model with the whole dataset to get the classification of the topics. We assumed that, with a large enough sample, we can classify the topics presented with hig accuracy. The topics of the remaining quotes can be classified using the model generated from this sampled training set.
- **Relationship between complexity and education level :** The base assumption used to establish this connection is that people who write more 'difficult' text will, tendentially, have a higher level of education. In fact the Dale-Chall Readability Formula can be converted directly into an indication of the school-level required to understand the text, which we assume can be correlated with the education level required to generate the text.

**_Note :_** These methods required the usage of external packages. For a more detailed exposition of the packages used, please see [requirements.txt](https://github.com/epfl-ada/ada-2021-project-madam/blob/main/requirements.txt).

---

### Proposed Timeline

---

### Internal Organization

- Data Extraction : André, Khanh, Medya
- Pre-processing pipeline : Medya, Tomás
- Study of best methods : Khanh, Tomás
- Implementation of best methods : Medya, Tomás
- Runtime Optimization : André, Tomás
- Results Interpretation : 
- Website Construction (for the 'Data Story') : 

---

### Questions for TAs

---

### Repo Architecture
<pre>
├─── src
│   ├─── prep_pipeline.py : pipeline to precessed the quotes to create dataframe that contains all the needed features for analysis
│   ├─── prep_utilities.py : helper functions used to do NLP tasks and enfineer features for the dataset 
│   ├─── sampling_data.py : functions to generate a 1/20 sample of all the quotes
│   ├─── text_scores.py : functions used to measure the complexity of the quotes
│   └─── thematic_processing.py : functions used for the clustering of quotes by topics
├─── Initial_Analysis.ipynb
├─── README.md : [ERROR] Infinite Recursion :)
└─── requirements.txt
</pre>
