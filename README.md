## MIND THE (GENDER) GAP

### ABSTRACT

-- Fill in abstract here --

---

### Research Questions

On this project we set ourselves out to analyze whether there is any fundamental difference in the way quotes from men vs women and handled by various news sources. To study this we propose the following research questions.

- What is the representativity of quotes by men vs women, and how has it evolved over time?
- Which are the main themes in the quotes from men and from women?
- What is the sentiment (negative, neutral, positive) associated to the quotes by gender, potentially distributed by themes? *
- What is the overall level of education from the quoted individuals, by gender and by theme?

Afterwards, we restrict our analysis to the main generators of quotes (i.e. the sites with the highest number of quotes counts), as these will tend to be the most influential, and we conduct a similar analysis within this subset.

\*While quotes are inherently unchangeable, the context in which they are used and the predominance of their sentiment can reveal information about the sources predisposition towards the quoted. For example, if a newspaper tendentially selects quotes with a negative sentiment for women, while mainly neutral/positive for men, this could be a display of an internal bias which would reflect a different image for each gender.

---

### Proposed additional datasets

From the current outlook of the project, there is no prospect of any additional datasets.

---

### Methods

Before any study could be conducted, there had to be a data pre-processing, which sent the raw data through a pipeline until it was in a state where it could be processed.
- **Structural Changes :** ;
- **Content Changes :** The quotes themselves were manipulated into a form easier to analyze. For that, the contractions were expanded (don't -> do not, yall -> you all), the sentences were tokenized, the stopwords were removed and the words were lemmatized. This latter manipulation allowed to reduce the variance of words, creating a more representative bag of words (BoG).

To study the different aspects mentioned in the research questions, several methods were used.

- **Characteristics Identification :** Data extracted from Quotebank or Wikidata (via QIds);
- **Theme Identification :** To identify the main topics in a set of quotes, we used 'Latent Dirichlet Allocation' (or [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)). This model clusters the BoG provided into a set of *n* topics (where *n* is a hyperparameter). Then, it provides the main words associated to each topic and, by human interpretation, it is possible to label them. This model, after being trained, can be used to identify the main topics from any given quote;
- **Text Sentiment :** TBD;
- **Text Readability / Complexity :** Using the [Dale-Chall Readability Formula](https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula) we can measure the 'difficulty' of any given text (the base assumption is that people who write more 'difficult' text will, tendentially, have a higher level of education).

In order to conduct this analysis and use this methods, several assumptions had to be made. STILL NEED TO WRITE THIS MORE CAREFULLY

- **Homogeneity of data between chuncks :** (regarding the removal of rows);
- **Theme repetition :** It is not necessary to train the LDA the model with all the quotes to get all of the topics. Using a large enough sample we can capture all of the topics present, and in the remaining quotes there will only be repetitions (or slight variations) of these topics.
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
│   ├─── prep_pipeline.py
│   ├─── prep_utilities.py
│   ├─── sampling_data.py
│   ├─── text_scores.py
│   └─── thematic_processing.py
├─── Initial_Analysis.ipynb
└─── requirements.txt
</pre>
