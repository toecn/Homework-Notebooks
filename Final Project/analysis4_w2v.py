# Word2Vec

print('Lets start!')

# Packages
import pandas as pd
import re
import sklearn
import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.datasets
import sklearn.cluster
import sklearn.decomposition
import sklearn.metrics

import gensim
from gensim.matutils import kullback_leibler
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm
import seaborn as sns
import spacy 

## Helper functions
import util_analysis1_word_count as util


nlp = spacy.load("es_core_news_sm")

def normalize(text):
    '''
    Normalizes: takes out punctuation and stopwords (defined by SpaCy) 
    Input:
        text (str): text made of string
    
    Output:
        lexical_tokens(list): list of strings. Text has been clean of puntuation, numbers,
                        stop words, and made lower case
    '''
    test = nlp(text)
    words = [t.orth_ for t in test if not t.is_punct | t.is_stop] #here is filtering
    lexical_tokens = [t.lower() for t in words if t.isalpha()] #make lowercase and leave nonnumerical
    
    return lexical_tokens

def sent_tokenize(word_list, model=nlp):
    '''
    Preserves the structure of sentences
    
    Input:
        word_liststr: text
        model(SpaCy model): defaults to a Spacy model named nlp
    
    Output:
        sentences(list): list of litst that represent tokenized sentences
    '''
    doc = model(word_list)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences


def normalizeTokens(word_list, extra_stop=[], model=nlp, lemma=True, 
                        MAX_LEN=1500000):
    '''
    #We can use a generator here as we just need to iterate over it
    
    Input:
        word_list (list)
        extra_stop
        model: nlp model object from spacy library (here is where the 
        language gets set)
        lemma
        MAX_LEN

    Outpur:
        normalized (list)

    '''

    normalized = []
    if type(word_list) == list and len(word_list) == 1:
        word_list = word_list[0]

    if type(word_list) == list:
        word_list = ' '.join([str(elem) for elem in word_list]) 

    # since we're only normalizing, I remove RAM intensive operations 
    # and increase max text size

    model.max_length = MAX_LEN
    doc = model(word_list.lower(), disable=["parser", "tagger", "ner"])

    if len(extra_stop) > 0:
        for stopword in extra_stop:
            lexeme = nlp.vocab[stopword]
            lexeme.is_stop = True

    # we check if we want lemmas or not earlier to avoid checking every time we loop
    if lemma:
        for w in doc:
            # if it's not a stop word or punctuation mark, add it to our article
            if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and len(w.text.strip()) > 0:
            # we add the lematized version of the word
                normalized.append(str(w.lemma_))
    else:
        for w in doc:
            # if it's not a stop word or punctuation mark, add it to our article
            if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and len(w.text.strip()) > 0:
            # we add the lematized version of the word
                normalized.append(str(w.text.strip()))

    return normalized

print('functions loaded!')
'''
## Data
data = pd.read_csv('data/2021_03_data.csv')
corpus = data

## Text preprocesing:
## Delete duplicates
corpus = corpus.drop_duplicates(subset=['Text'])


# Create a dataset in which each politician is mentioned and 
# the source comes from the politician‘s country
santos = corpus[(corpus["Text"].str.contains("Santos")) & (corpus["Country"]=="CO")]
uribe = corpus[(corpus["Text"].str.contains("Uribe")) & (corpus["Country"]=="CO")]
pena_nieto = corpus[(corpus["Text"].str.contains("Peña Nieto")) & (corpus["Country"]=="MX")]
correa = corpus[(corpus["Text"].str.contains("Correa")) & (corpus["Country"]=="EC")]
morales = corpus[corpus["Text"].str.contains("Evo") & (corpus["Country"]=="BO")]
chavez = corpus[corpus["Text"].str.contains("Chávez") & (corpus["Country"]=="VE")]
maduro = corpus[corpus["Text"].str.contains("Maduro") & (corpus["Country"]=="VE")]
kirchner = corpus[corpus["Text"].str.contains("Kirchner") & (corpus["Country"]=="AR")]
ortega = corpus[corpus["Text"].str.contains("Ortega") & (corpus["Country"]=="NI")]
bachelet = corpus[corpus["Text"].str.contains("Bachelet") & (corpus["Country"]=="CL")]
mujica = corpus[corpus["Text"].str.contains("Mujica") & (corpus["Country"]=="UY")]

print('data frames for each president created')

list_df = [santos, uribe, pena_nieto, correa,\
            morales, chavez, maduro, kirchner, ortega, bachelet, mujica]

list_names = ['santos', 'uribe', 'pena_nieto', 'correa',\
              'morales', 'chavez', 'maduro', 'kirchner', 'ortega', 
              'bachelet', 'mujica']

i = 0
for df in list_df:
    df.name = list_names[i]
    i += 1

for df in list_df:
    print(df.name, 'df:', df.shape[0], len(df['Text']))

c = 0
for df in df_list:
    c = 0
    for text in df['Text']:
        c += len(text)
    print(df.name, 'total words', c)
    

stp_wrds = ['me', 
 'mi', 
 'yo', 
 'era', 
 'había', 
 'muy', 
 'estaba',
 'qué', 
 'he', 
 'día', 
 'tnn', 
 'me',
 'qué',
 'ni', 
 'gente', #I don't think you want to take this word out. 
 'muy', 
 'yo', 
 'bien', #I don't think you want to take this word out.
 'decir',  
 'puede', 
 'esa', 
 'te', 
 'usted']
 

for df in list_df:
    df['normalized_tokens'] = df['Text'].apply(lambda x: \
                                    [normalize(s) \
                                    for s in sent_tokenize(x)])
    df['normalized_sents'] = df['normalized_tokens'].apply(lambda x: \
                                    [normalizeTokens(s, lemma=False) \
                                    for s in x])

for df in list_df:
    df.to_csv('results/word2vec/data/{}_data.csv'.format(df.name))
    

santos_w2v_300 = gensim.models.word2vec.Word2Vec(santos['normalized_sents'].sum(), 
                                                        size=300)
uribe_w2v_300 = gensim.models.word2vec.Word2Vec(uribe['normalized_sents'].sum(), 
                                                        size=300)
pena_nieto_w2v_300 = gensim.models.word2vec.Word2Vec(pena_nieto['normalized_sents'].sum(), 
                                                        size=300)
pena_nieto_w2v_300 = gensim.models.word2vec.Word2Vec(pena_nieto['normalized_sents'].sum(), 
                                                        size=300)

columns = santos.columns
df_all = pd.DataFrame(columns=columns)
for df in list_df:
    df['president_name'] = df.name
    df_all = df_all.append(df)
    
'''    
#########################################################################################################
#########################################################################################################
#########################################################################################################

def dimension(model, positives, negatives):
    diff = sum([normalize(model[x]) for x in positives]) - sum([normalize(model[y]) for y in negatives])
    return diff
  
Politicians_spectrum = ["correa","chávez","morales", "kirchner", "ortega", 
                        "uribe", "santos", "nieto", 
                        "mujica", "bachelet"]
                        
# Let's calculate three dimensions of populism

Nationalism = dimension(full_corpusW2V_300, ['soberanía', 'democracia', 'patria', 'bolívar'], 
                                        ['interferencia', 'dictadura', 'extranjeros', 'dependencia'])
Anti_capital = dimension(full_corpusW2V_300, ['élite', 'capital', 'riqueza', 'enriquecen'], 
                                        ['interferencia', 'dictadura', 'pobreza', 'miseria'])
Anti_elite = dimension(full_corpusW2V_300, ['élite', 'políticos', 'corrupción'], 
                                        ['pueblo', 'ciudadanos', 'leyes'])

def makeDF(w2vmodel, word_list, dim1, dim2, dim3):
    g = []
    r = []
    c = []
    for word in word_list:
        g.append(sklearn.metrics.pairwise.cosine_similarity(w2vmodel[word].reshape(1,-1), dim1.reshape(1,-1))[0][0])
        r.append(sklearn.metrics.pairwise.cosine_similarity(w2vmodel[word].reshape(1,-1), dim2.reshape(1,-1))[0][0])
        c.append(sklearn.metrics.pairwise.cosine_similarity(w2vmodel[word].reshape(1,-1), dim3.reshape(1,-1))[0][0])
    df = pd.DataFrame({'Nationalism': g, 'Pro-Anti Capitalism': r, 'Pro-Anti Pol Elite': c}, index = word_list)
    return df

# Get the projections:
Politicians_n_ac_ae = makeDF(full_corpusW2V_300, Politicians_spectrum, Nationalism, Anti_capital, Anti_elite)

# Plot left in each dimenssion:
fig = plt.figure(figsize = (12,4))
ax1 = fig.add_subplot(131)
PlotDimension(ax1, Politicians_n_ac_ae, 'Nationalism')
ax2 = fig.add_subplot(132)
PlotDimension(ax2, Politicians_n_ac_ae, 'Pro-Anti Capitalism')
ax3 = fig.add_subplot(133)
PlotDimension(ax3, Politicians_n_ac_ae, 'Pro-Anti Pol Elite')
plt.savefig('results/word2vec/dimenssion.pdf')
