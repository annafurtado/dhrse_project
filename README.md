#Linguistic analysis of Literary texts written by Shakespeare
### Dataset description 

This dataset contains the complete text of "Hamlet," a tragedy written by William Shakespeare. The play, which is among Shakespeare's most famous and frequently performed works, explores themes of treachery, revenge, incest, and moral corruption. It is structured in five acts, with a series of scenes within each act.
The dataset is organized as follows:

###File structure

The dataset is organized as follows:

**hamlet.txt**: This plain text file contains the entire play "Hamlet," organized by acts and scenes. Each act and scene are clearly marked, and character dialogues are presented in the order they appear in the play.

### Method and the process

**Stopwords removal**
Stop word removal is a process in natural language processing (NLP) where common, non-essential words (such as "and," "the," "is," etc.) are removed from the text. These words are often filtered out because they do not carry significant meaning and can clutter the analysis. Removing stop words helps in focusing on the more important terms and improves the efficiency of text processing tasks.

**Lemmatisation**
Lemmatization is a process in natural language processing (NLP) that reduces words to their base or root form, known as a "lemma." Unlike stemming, which merely removes prefixes or suffixes to produce a root form, lemmatization considers the context and morphological analysis of the words, ensuring the base form is a valid word. This makes lemmatization more sophisticated and accurate compared to stemming.

**N-gram Extraction**

The project employs n-gram techniques to extract key terms from the targeted texts. Specifically, bigram and trigram approaches have been employed to analyze the text and uncover significant word patterns.
### Visulising the data


### Required libraries 

- pandas
- gensim
- spacy
- nltk
- plotly
- re
- math

