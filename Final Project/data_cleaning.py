#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning for CCA21 Project
# # Working with El Corpus del Español
# 
# This notebook creates the pipeline to do the text preprocesing steps for:
# 
# 1. Topic Models
# 2. Dynamic Topic Models
# 3. Word2Vec
# 4. Diachronic Word Embeddings
# 
# `1` and `2` require the data in a different format from `3` and `4`. 
# 
# Notebook index:
# 1. Libraries
# 2. Helper functions
# 3. Pipeline
# 4. Test

# # 1. Libraries 

# In[1]:


import re
import zipfile
import os
import sys
import pandas as pd


# # 2. Helper functions

# In[4]:


def loadcorpus(corpus_name, corpus_style="text"):
    '''
    Iterates through the files in the folder, and 
    unzips the files, storing them in a dictionary with 
    each zip file mapping to a list of the texts.
    
    Input:
        corpus_name (str): indicates the working directory and the name
                        of the foldet that contains the corpus

    Output:
        text_raw (dict):
            key - name of the enclosing folder
            value - string that corresponds to that folder
    '''
    texts_raw = {}
    for file in os.listdir(corpus_name + "/"):
        if corpus_style in file:
            print(file)
            zfile = zipfile.ZipFile(corpus_name + "/" + file)
            for file in zfile.namelist():
                texts_raw[file] = []
                with zfile.open(file) as f:
                    for line in f:
                        texts_raw[file].append(line)
    return texts_raw


# In[5]:


def att_sources(corpus_name, source_name):
    '''
    Returns list of the sources (websites) the text comes from source_name
    
    Input:
        corpus_name (str)
        source_name (str)
    
    Output:
        list_source_name (list)
    '''
    zfile = zipfile.ZipFile(corpus_name + "/" + source_name)
    list_source_name = []

    for file in zfile.namelist():
        with zfile.open(file) as f:
            for line in f:
                list_source_name.append(line)

    return list_source_name


# In[40]:


def clean_raw_text(raw_texts):
    '''
    Decodes and removes some reg expresssions from text
    Reg expressions removed: [¡!@#$:);,¿?&]
    Notice that I don't remove dots (.) to be able to mark sentences
    
    Input:
        raw_texts (str): text

    
    Output:
        clean text(list): list with clean texts
        
    '''
    clean_texts = []
    for text in raw_texts:
        try:
            text = text.decode("utf-8")
            text = re.sub('[¡!@#$:);,¿?&]', '', text)
            clean_texts.append(text)
        except AttributeError:
            print("ERROR CLEANING", "Text:")
            print(text)
            continue
        except UnicodeDecodeError:
            print("Unicode Error, Skip")
            continue
    return clean_texts


# In[51]:

def dic_match_key_text(raw_dic_texts, max_num_loops, max_texts, max_onetext_length):
    '''
    Creates dictionary of text to match text and sources
    
    Input:
        raw_dic_texts (dict):
            key - name of the enclosing folder
            value - string that corresponds to that folder

        max_num_loops(int): number that determines the overall
                             number of loops
        
        max_texts(int): number that detemines the number of texts 
                        included in the list
        
        max_onetext_length(int): number that blocks larger than
                                 n character texts
        

    Output:
        websites_text(dict):
            key - id that matches the text and the source
            value - (str) text
    '''
    websites_text = {}
    i=0
    
    for key in raw_dic_texts:
        i =+ 1

        if len(websites_text) > max_texts:
            break
        texts_for_key = clean_raw_text(raw_dic_texts[key])
        for one_text in texts_for_key:
            if len(one_text) >= max_onetext_length:
                break
            key_text = one_text.split()[0]
            try:
                websites_text[key_text] = one_text[6:]
            except IndexError:
                continue
        if i==max_num_loops:
                break
    return websites_text


# In[58]:


## THIS FUNCTION IS NOT WORKING

def merge_text_sources(source_list, websites_text, span_df, max_num_loops):
    '''
    Merges the list of sources and the text comming from those sources 
    into a pandas dataframe
    
    Input:
        source_list (list): list of url sources 
        websites_text (dict): 
            key (int)- id to source
            value (str)- text 
        span_df (pandas dr): pandas dataframe that has only the maes of the columns
        max_num_loops (int): number to break the loop and get smaller 
                             pandas dataframes
                             
    Output:
        
    '''
    i = 0
    for website in source_list[3:]:
        '''
        Loops over the list of url sources
        '''
        i =+ 1
        try:
            textID, Number_of_words, Genre, Country,                 Website, URL, Title = website.decode("utf-8").split("\t")
        except UnicodeDecodeError:
            continue
        try:
            span_df.loc[textID.strip()] =                         [Title.strip(), Genre.strip(), Country.strip(), 
                        Website.strip(), URL.strip(), Number_of_words.strip(),  
                        websites_text[textID.strip()]]
        except KeyError:
            continue
        if i==max_num_loops:
            break
        
        return span_df


# # 3. Pipeline - Test

# In[6]:


# loads corpus as a dictionary

raw_span = loadcorpus("data/SPAN")


# In[7]:


# loads the url where the texts come from as a list

source_list = att_sources("data/SPAN", "span_sources.zip")


# In[8]:


span_df = pd.DataFrame(columns=["Title", "Genre", "Country",
                                    "Website", "URL", "Number of words",
                                    "Text"])


# In[52]:

# Number of loops 20 000,000 # there should be much less than that
# Max number of text  20 000,000 # there are only 2M texts
# Max number of words per text 200,000

websites_text = dic_match_key_text(raw_span, max_num_loops=20000000, max_texts=20000000, max_onetext_length=200000)


# In[59]:

for website in source_list[3:]:
    '''
    Loops over the list
    '''
    try:
        textID, Number_of_words, Genre, Country, \
            Website, URL, Title = website.decode("utf-8").split("\t")
    except UnicodeDecodeError:
        continue
    try:
        span_df.loc[textID.strip()] = \
                        [Title.strip(), Genre.strip(), Country.strip(), \
                        Website.strip(), URL.strip(), Number_of_words.strip(), \
                        websites_text[textID.strip()]]
    except KeyError:
        continue    


# In[60]:


span_df.shape


# In[ ]:


span_df.to_csv("data/2021_03_data.csv")

