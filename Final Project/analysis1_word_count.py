## Counting words

import pandas as pd
import re


## Helper functions
import util_analysis1_word_count as util

## Data
data = pd.read_csv('data/2021_03_data.csv')
corpus = data

## Text preprocesing:
## Delete duplicates
corpus = corpus.drop_duplicates(subset=['Text'])


# Lematize 
# corpus['lemmatized_text'] = \
#    corpus['normalized_text'].apply(lambda x: lematize_list(x))


# Create a dataset in which each politician is mentioned and 
# the source comes from the politician‘s country
santos = corpus[(corpus["Text"].str.contains("Santos")) & (corpus["Country"]=="CO")]
uribe = corpus[(corpus["Text"].str.contains("Uribe")) & (corpus["Country"]=="CO")]
pena_nieto = corpus[(corpus["Text"].str.contains("Peña Nieto")) & (corpus["Country"]=="MX")]
calderon = corpus[(corpus["Text"].str.contains("Calderon")) & (corpus["Country"]=="MX")]

correa = corpus[(corpus["Text"].str.contains("Correa")) & (corpus["Country"]=="EC")]
morales = corpus[corpus["Text"].str.contains("Evo") & (corpus["Country"]=="BO")]
chavez = corpus[corpus["Text"].str.contains("Chávez") & (corpus["Country"]=="VE")]
kirchner = corpus[corpus["Text"].str.contains("Kirchner") & (corpus["Country"]=="AR")]
ortega = corpus[corpus["Text"].str.contains("Ortega") & (corpus["Country"]=="NI")]

bachelet = corpus[corpus["Text"].str.contains("Bachelet") & (corpus["Country"]=="CL")]
mujica = corpus[corpus["Text"].str.contains("Mujica") & (corpus["Country"]=="UY")]

print('data frames for each president created')

list_df = [santos, uribe, pena_nieto, calderon, correa,\
            morales, chavez, kirchner, ortega, bachelet, mujica]

list_names = ['santos', 'uribe', 'pena_nieto', 'calderon', 'correa',\
            'morales', 'chavez', 'kirchner', 'ortega', 'bachelet', 'mujica']

i = 0
for df in list_df:
    df.name = list_names[i]
    i += 1


for df in list_df:
    df['text2'] = df['Text'].apply(lambda x: re.sub('[¡!@#$:).;,¿?&]', '', x.lower()))
    df['normalized_text'] = \
            df['text2'].apply(lambda x: util.normalize(x, ['y', 'a', 'o', 'e']))


list_count_words = []
for df in list_df:
    countsDict = {}
    for word in df['normalized_text'].sum():
        if word in countsDict:
            countsDict[word] += 1
        else:
            countsDict[word] = 1
    word_counts = sorted(countsDict.items(), key = lambda x : x[1], reverse = True)
    print(word_counts[:10])
    for tpl in word_counts:
        t = (df.name, tpl[0], tpl[1])
        list_count_words.append(t)

df_word_counts = pd.DataFrame(list_count_words, columns = ['Name', 'Word', 'Count'])

df_word_counts.to_csv('03_15_word_counts.csv', index=False)

