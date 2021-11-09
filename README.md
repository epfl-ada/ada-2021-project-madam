## MIND THE GAP

### ABSTRACT

-- Fill in abstract here --

### Research Questions

-- Write down research questions here --

### Proposed additional datasets

From the current outlook at the project, there is no prospect of any additional datasets required.

### Methods

Before any study could be conducted, there had to be a data pre-processing, which sent the raw data through a pipeline until it was in a state where it could be processed.
- <u>Structural Changes :</u> ;
- <u>Content Changes :</u> The quotes themselves were manipulated into a form easier to analyze. For that, the contractions were expanded (don't -> do not, yall -> yall), the sentences were tokenized, the stopwords were removed and the words were lemmatized. This latter manipulation allowed to reduce the variance of words, creating a more representative bag of words (BoG).

To study the different aspects mentioned in the research questions, several methods were used.

- <u>Characteristics Identification :</u> Data extracted from Quotebank or Wikidata (via QIds);
- <u>Theme Identification :</u> To identify the main topics in a set of quotes, we used 'Latent Dirichlet Allocation' (or [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)). This model clusters the BoG provided into a set of <u>n</n> topics (where <u>n</u> is a hyperparameter). Then, it provides the main words associated to each topic and, by human interpretation, it is possible to label it. This model, after trained, can be used to identify the main topic in any given quote;
- <u>Text Sentiment :</u> TBD;
- <u>Text Readability / Complexity :</u> Using the [Dale-Chall Readability Formula](https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula) we can measure the 'difficulty' of any given text (the base assumption is that people who write more 'difficult' text will, tendentially, have a higher level of education).

<b>Note :</b> These methods required the usage of external packages. For a more detailed exposition of the packages used, please see [requirements.txt](https://github.com/epfl-ada/ada-2021-project-madam/blob/main/requirements.txt).

### Proposed Timeline


### Internal Organization

- Data Extraction : André, Khahn, Medya
- Pre-processing pipeline : Medya, Tomás
- Study of best methods : Khahn, Tomás
- Implementation of best methods : Medya, Tomás
- Runtime Optimization : André, Tomás
- Results Interpretation : 
- Website Construction (for the 'Data Story') : 

### Questions for TAs


### Repo Architecture
'''
└───src

    └─── prep_pipeline.py
    
    └─── prep_utilities.py
    
    └─── sampling_data.py
    
    └─── text_scores.py
    
    └─── thematic_processing.py
    
└─── Initial_Analysis.ipynb

└─── requirements.txt
'''


