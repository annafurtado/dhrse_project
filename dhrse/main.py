import os
import spacy
import nltk
import operator
import re
import math

from collections import defaultdict
from nltk.corpus import stopwords
import pandas as pd
from gensim.models.phrases import Phrases, Phraser
from gensim.utils import simple_preprocess
import plotly as py
import plotly.express as px



nltk.download('stopwords')

# Descriptive comment
source_dir = os.path.abspath(os.getcwd())
data_path = os.path.join(source_dir, 'data') # TODO Inputs
output_dir = os.path.join(source_dir, 'results') # TODO Inputs

# Set variables for processing
lang_model = "en_core_web_sm"
error_handling = "ignore"


def get_stop_wordlist(stop_word_file, previous_stop_wordlist, words_to_add):
    """
    Reads a stop word file and add its words into the stop_wordlist
    :param stop_word_file: a txt file containing one word per line
    :previous_stop_wordlist: the list containing extended and nltk stop words
    :words_to_add: a list of words you would like to add to the stop wordlist
    :return: a custom stop wordlist
    """
    stop_word_custom = []
    with open(stop_word_file, "r", encoding="utf-8") as flh:
        for word in flh.readlines():
            stop_word_custom.append(word.strip())
    previous_stop_wordlist.extend(words_to_add)
    return previous_stop_wordlist.extend(stop_word_custom)


# Create the list of stop words, or words that are not important for our analysis and should be excluded
stop_words = []
stop_words.extend(stopwords.words("english")) # Add nltk stop words
stop_words_file = os.path.join(data_path, "earlyModernStopword.txt")
more_stopwords = ['would', 'said', 'says', 'also', 'good', 'lord', 'come', 'let','say', 'speak', 'know', 'hamlet']
# call the function for stop words
final_stop_wordlist = get_stop_wordlist(stop_words_file, stop_words, more_stopwords)
print(final_stop_wordlist)
"""
# Variables for making Trigrams
min_count = 5
threshold = 100

def make_trigrams(document):
    
    Builds bigram and trigram models and extract them given a document
    :param document: document containing the text
    :return: a list of bi- and trigrams
    
    bigram = Phrases(document, min_count=min_count, threshold=threshold) # higher threshold indicates fewer phrases
    bigram_model = Phraser(bigram) # Removes model state from Phrases to reduce memory use
    trigram = Phrases(bigram[document], threshold=threshold)
    trigram_model = Phraser(trigram)
    result_ngrams = []
    for token in document:
        bigram_result = bigram_model[token]
        trigram_result = trigram_model[bigram_result]
        result_ngrams.append(trigram_result)
    return result_ngrams


def lemmatization(document):
    n_grams = make_trigrams(document)
    
    Takes a document and do lemmatization
    :document: the document to lemmatized
    :return: a list of lemmatized_text
    
    lemmatized_text = []
    # Initialize spacy language model, removing the parser and ner components
    nlp = spacy.load(lang_model, disable=['parser', 'ner'])
    for sent in document:
        doc = nlp(" ".join(sent))
        lemmatized_text.append([token.lemma_ for token in doc if token.lemma_ != '-PRON-'])
    # Do lemmatization
    lemmatized_data = lemmatization(n_grams)
    return lemmatized_data


document_name = 'Hamlet.txt'
top_ten_output_file = "topTenPlainText"

n = 10
file_format = '.html'
fig_x_label = "Word"
fig_y_label = "Count"
fig_z_label = "Percent"
fig_width = 750
fig_height = 550
axis_angle = -45
fig_title = 'Top 10 Words, Hamlet'
colors = px.colors.qualitative.Dark24
pallette = "crimson"

text_path = os.path.join(data_path, document_name) # TODO Inputs

docs=[]
with open(text_path, "r", encoding="utf-8", errors=error_handling) as flh:
    for line in flh:
        singleLine = line.strip()
        if len(singleLine) == 0:
            continue
        docs.append(singleLine.split())
# remove punctuation from sentences and parse into words

words = []
for sentence in docs:
    words.append(simple_preprocess(str(sentence), deacc=True)) # deacc=True removes punctuations
# remove stop words
stop=[]
for doc in words:
    processed_doc = []
    for word in simple_preprocess(str(doc)):
        if word not in stopWords:
            processed_doc.append(word)
    stop.append(processed_doc)
lemma = makeLemma(stop)
tokens=[]
for sublist in  lemma:
    for item in sublist:
        tokens.append(item)
# get frequency
freq = defaultdict(int)

for t in tokens:
    freq[t] += 1
# sort frequency in descending order
freq = sorted(freq.items(), key = operator.itemgetter(1), reverse = True)
imgFilepath = os.path.join(resultsPath, top_ten_output_file + file_format)

# plot top ten words in a histogram
topn = n
df = pd.DataFrame(freq, columns = ["Words", "Count"])
df["Pct"] = ((df["Count"]/df["Count"].sum())*100).round(3)
df["Pct"] = df["Pct"].astype(str) + "%"
# TODO DRY:
dfPct = df[0:1]
dfPct = pd.concat([dfPct,df[1:2]])
dfPct = pd.concat([dfPct,df[2:3]])
dfPct = pd.concat([dfPct,df[3:4]])
dfPct = pd.concat([dfPct,df[4:5]])
dfPct = pd.concat([dfPct,df[5:6]])
dfPct = pd.concat([dfPct,df[6:7]])
dfPct = pd.concat([dfPct,df[7:8]])
dfPct = pd.concat([dfPct,df[8:9]])
dfPct = pd.concat([dfPct,df[9:10]])
# Descriptive comment
high = max(df["Count"])
low = 0
fig = px.bar(dfPct, x = "Words", y = "Count", hover_data=[dfPct["Pct"]], text = "Count", color = "Words",
             title = fig_title, color_discrete_sequence=colors,
             labels = {"Words":fig_x_label, "Count":fig_y_label, "Pct":fig_z_label})
fig.update_layout(title={'y':0.90, 'x':0.5, 'xanchor': 'center', 'yanchor':'top'},
                  font={"color": pallette}, width = fig_width, height = fig_height, showlegend=False)
fig.update_xaxes(tickangle = axis_angle)
fig.update_yaxes(range = [low,math.ceil(high + 0.1 * (high - low))])
py.offline.plot(fig, filename=imgFilepath, auto_open = False)
fig.show()
"""
