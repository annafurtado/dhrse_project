import os

import nltk
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords


nltk.download('stopwords')


# Descriptive comment
# TODO Naming
source_dir = os.path.abspath(os.getcwd())
data_path = os.path.join(source_dir, 'data') # TODO Inputs
output_dir = os.path.join(source_dir, 'results') # TODO Inputs

# Set variables for processing
# TODO Naming
lang_model = "en_core_web_sm"
error_handling = "ignore"

# Create the list of stop words, or words that are not important for our analysis and should be excluded
stop_words = []
stop_words.extend(stopwords.words("english")) # Add nltk stop words
# Add custom stop words
stop_words.extend(['would', 'said', 'says', 'also', 'good', 'lord', 'come', 'let','say', 'speak', 'know', 'hamlet'])
stop_words_file = os.path.join(data_path, "earlyModernStopword.txt")


def get_stop_wordlist(stop_word_file):
    """
    Reads a stop word file and add its words into the stop_wordlist
    :param stop_word_file: a txt file containing one word per line
    :return: a custom stop wordlist
    """
    stop_word_custom = []
    with open(stop_word_file, "r", encoding="utf-8") as flh:
        for word in flh.readlines():
            stop_word_custom.append(word.strip())
    return stop_word_custom


stopWords.extend(stopWordsCustom)

# Variables for making Trigrams
minCount = 5
thresHold = 100

def makeTrigrams(tokens):
    # Build the bigram and trigram models
    # get bigram phrases
    bigram = Phrases(tokens, min_count=minCount, threshold=thresHold) # higher threshold indicates fewer phrases
    bigram_model = Phraser(bigram) # Removes model state from Phrases to reduce memory use
    # get trigram phrases
    bigram = Phrases(tokens, min_count=minCount, threshold=thresHold) # TODO DRY
    trigram = Phrases(bigram[tokens], threshold=thresHold)
    trigram_model = Phraser(trigram)
    result = []
    for doc in tokens:
        bigram_result = bigram_model[doc]
        trigram_result = trigram_model[bigram_result]
        result.append(trigram_result)
    return result


# Descriptive comment
def makeLemma(tokens, spacy=None):
    dataWordsNgrams = makeTrigrams(tokens)

    def lemmatization(tokens):
        """https://spacy.io/api/annotation"""
        textsOut = []
        for sent in tokens:
            doc = nlp(" ".join(sent))
            textsOut.append([token.lemma_ for token in doc if token.lemma_ != '-PRON-'])
        return textsOut

    # Initialize spacy language model, removing the parser and ner components
    nlp = spacy.load(lemLang_model, disable=['parser', 'ner'])
    # Do lemmatization
    dataLemmatized = lemmatization(dataWordsNgrams)

    return dataLemmatized

# TODO Naming:
n = 10
documentName = 'Hamlet.txt' # TODO Inputs
outputFile = "topTenPlainText" # TODO Inputs
fmt = '.html'
figure_Xlabel = "Word"
figure_Ylabels = "Count"
figureZlabel = "Percent"
figureWidth = 750
figure_height = 550
figure_Axisangle = -45
figureTitle = 'Top 10 Words, Hamlet'
colors = px.colors.qualitative.Dark24
labCol = "crimson"

textFilepath = os.path.join(data_path, documentName) # TODO Inputs
# Descriptive comment
docs=[]
with open(textFilepath, "r", encoding = encoding, errors = how_to_handleErrors) as f:
    for line in f:
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
imgFilepath = os.path.join(resultsPath, outputFile + fmt)

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
fig = px.bar(dfPct, x = "Words", y = "Count",hover_data=[dfPct["Pct"]],text = "Count", color = "Words",
             title = figureTitle, color_discrete_sequence=colors,
             labels = {"Words":figure_Xlabel,"Count":figure_Ylabels,"Pct":figureZlabel})
fig.update_layout(title={'y':0.90, 'x':0.5, 'xanchor': 'center', 'yanchor':'top'},
                  font={"color": labCol}, width = figureWidth, height = figure_height, showlegend=False)
fig.update_xaxes(tickangle = figure_Axisangle)
fig.update_yaxes(range = [low,math.ceil(high + 0.1 * (high - low))])
py.offline.plot(fig, filename=imgFilepath, auto_open = False)
fig.show()
