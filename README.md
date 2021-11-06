## [TITLE]

### ABSTRACT

-- Fill in abstract here --

### Research Questions

-- Write down research questions here --

### Proposed additional datasets

-- If we use any additional datasets, write them down here

### Methods

To study the different aspects mentioned above, several methods need to be used.

- <u>Characteristics Identification (Gender, Race, Age) :</u> Data extracted from Wikidata (WHAT TO DO WHEN NOT POSSIBLE? GENDER CAN BE TAKEN FROM NLTK, BUT RACE AND AGE DOESN'T SEEM FEASIBLE)
- <u>Theme Identification :</u> For a set of quotes, the main themes identification is performed in two steps. First, 'Term Frequency–Inverse Document Frequency' (or [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)) allows to find the main keywords among the quotes. Then, 'Latent Dirichlet Allocation' (or [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)) clusters these keywords into topics. It is necessary to manually label the topics afterwards, by human interpretation of the results.
- <u>Text Sentiment :</u> (STILL NEED TO LOOK UP FOR SOME PACKAGES FOR THIS. PROBABLY NLTK HAS SOMETHING)
- <u>Text Readability / Complexity :</u> Using the [Dale-Chall Readability Formula](https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula) we can measure the 'difficulty' of any given text (the base assumption is that people who write more 'difficult' text will, tendentially, have a higher level of education)
- <u>Some kind of policital analysis :</u> (GOING TO LOOK INTO THIS, BUT SEEMS FUNDAMENTALLY THE SAME AS THEME IDENTIFICATION)


### Proposed Timeline

-- Write down timeline here --

### Internal Organization

-- Define internal organization here --

### Questions for TAs

-- If we have questions for the TAs, write them down here --
