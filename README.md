## MIND THE (GENDER) GAP - Milestone 3 Update

### Data Story
The data story for this project is in [this repo](https://github.com/khanhnguyen15/project-madam-website.git). It was made using [Jekyll](https://jekyllrb.com/) and the [TeXt Theme](https://tianqi.name/jekyll-TeXt-theme/).

### ABSTRACT

Gender equality is a long-fought battle, one that has already taken long strides. But has it been enough? In this project we will analyze whether there is still a fundamental difference in the way quotes from men vs women are handled by various news sources. More than simply the representativity of genders, we want to study the content of the quotes. Is there a difference in the topics men or women are cited on? What is the overall sentiment (positive or negative) associated with the quotes? What kind of language is cited?
News sources have the power to influence society on a large scale, and biases can be easily propagated. Via the choice of quotes, by their content or sentiment, it is entirely possible to reflect an image (potentially distorted) of a group of people. Is that happening?

### Research Questions

We hope to answer the following questions.

- What is the representativity of quotes by gender, and how has it evolved?
- Which are the main topics in the quotes from different genders?
- What is the sentiment (negative, neutral, positive) associated with the quotes by gender, distributed by themes? *
- What is the complexity of the speech quoted?

Furthermore, we want to study a couple of very influential websites, some liberal some conservative, and compare the sentiment and portion of the quotes by gender, to decide which ones have a more equalitarian roster of quotes.

\*While quotes are inherently unchangeable, the context in which they are used, and the predominance of their sentiment, can reveal information about the sources' predisposition towards the quoted. Concretely, if a newspaper tendentially selects quotes with a negative sentiment for women, while mainly neutral/positive for men, this could be a display of an internal bias that is being propagated to the reader.

---

### Analyses
The analyses present in the website, made in the notebook `Data_story_analysis.ipynb`, are the following (all of them for each gender):
 1. Percentage of quote occurrences and speakers;
 2. Most-quoted speakers;
 3. Topic distribution in quotes;
 4. Sentiment scores in general, and in conservative vs liberal news websites;
 5. Text complexity in conservative vs liberal news websites.

---

### Methods

Before anything else, there had to be data pre-processing, which sent the raw data through a pipeline to clean it.

- **Structural Changes:** We removed quotes: 1) With `None` speaker; 2) With less than 5 words; 3) Whose speaker gender is `None`.
- **Content Changes:** The quotes were manipulated into an analyzable form. The sentences were tokenized and the words were lemmatized. For the topic analysis, the stopwords were also removed. These transformations reduced the variance of words, creating a more meaningful bag of words (BoG).

To study the different aspects mentioned in the research questions, several methods were considered.

- **Characteristics Identification:** Data extracted from Quotebank or Wikidata;
- **Topic Identification:** To identify the main topics in a set of quotes we used the package [empath](https://pypi.org/project/empath/). This package has a list of topics and the main words associated with each of them, and thus allows us to count how many words per topic there are in the quotes. The more words of a given topic there, the most likely it is that quote is about that topic.
- **Text Sentiment:** The sentiment analysis was done via a polarity score, which ranges from -1 (negative) to 1 (positive);
- **Text Readability / Complexity:** Using the [Dale-Chall Readability Formula](https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula) we can measure the 'difficulty' of any given text.

**_Note:_** These methods required the usage of external packages. For a detailed list see [requirements.txt](https://github.com/epfl-ada/ada-2021-project-madam/blob/main/requirements.txt).

---

### Internal Organization
André: Analyses 1 and 2. Writing data story. Data Extraction. Runtime optimizations.
Medya: Analysis 4. Writing data story. Data Extraction.
Tomás: Analyses 3 and 5. Writing data story. Repo cleaning. Study of NLP methods.
Khahn: Setting up website structure, repo, and theme. Data Extraction. Study of NLP methods.


### Timeline

01/11 - 07/11: Exploratory data analysis; Data extraction (samples); Study and implementation of NLP algorithms

08/11 - 12/11: Creation of notebook; Phase 2 README; Repo clean-up; (MILESTONE 2 DELIVERY)

15/11 - 21/11: Runtime optimizations (pandas, spaCy, NumPy)

22/11 - 28/11: Run full dataset through the pipeline; Rerun methods

29/11 - 05/12: Temporal analysis on the full dataset; Interpret results

06/12 - 12/12: Crafting 'Data Story'; Working on GitHub Pages

13/12 - 17/12: Finalize GitHub Pages; Final repo clean-up; (MILESTONE 3 DELIVERY)

---

### Repo Architecture (updated for Milestone 3)
<pre>
├─── data: (Content .gitignored) Folder to store the processed quotes
├─── data_processed: Folder to store the data from the various analyses in Data_story_analysis.ipynb
├─── plotly: Folder containing all the .html dynamic plots generated for the website
├─── src
│   ├─── contractions.py: table necessary to expand contractions (ended up not using it for Milestone 3)
│   ├─── load_models_data.py: automatically download all packages from nltk and spaCy
│   ├─── prep_pipeline.py: a pipeline to process quotes and create a DataFrame with the features for analysis
│   ├─── prep_utilities.py: NLP tasks and engineer features for the dataset 
│   ├─── sampling_data.py: generate a 1/20 sample of all quotes
│   ├─── text_scores.py: measure the complexity of the quotes
│   └─── thematic_processing.py: clustering quotes by topics with LDA (ended up not using it for Milestone 3)
├─── Data_Prep.ipynb: Milestone 2 notebook
├─── Data_story_analysis.ipynb: Notebook with all the analyses conducted for Milestone 3
├─── Quotes_prep.ipynb: Notebook used to generate the pre-processed quotes from Quotebank
├─── README.md: [ERROR] Infinite Recursion :)
└─── requirements.txt: packages and versions used
</pre>

---

### Additional Notes
The processed quotes for each year `quotes-XXXX-prep` files were generated in the same way as in Milestone 2, but saved to `json.bz2` files (inside the `/data/` folder), instead of `.parquet` since it allows for reading in chunks.
