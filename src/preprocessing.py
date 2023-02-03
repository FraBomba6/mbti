#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import nltk
# from transformers import BertForSequenceClassification, BertTokenizer

# FUNCTIONS
def remove_urls(text):
    return re.sub(r'http\S+', '', text)

# %%
# IF YOU RUN USING CHUNKS
mbti_dataset = pd.read_csv("data/dataset.csv")
print(mbti_dataset.head(5))
# ELSE
#mbti_dataset = pd.read_csv("../data/dataset.csv")

#print(mbti_dataset)
#TODO: decide how to hand the words/sentences to the model: 1 sentence, 2 sentences, half-half,
# tokenization: tokener will split sentence

# %% LOWERCASE
mbti_dataset['posts'] = mbti_dataset['posts'].str.lower()

# %% REMOVE HYPERLINKS, SPECIAL, CHARACTERS, AND MBTI TYPES -> STOPWORDS WILL BE REMOVED AND LEMMATIZATION WILL BE DONE WITH THE TOKENIZER
words_to_remove = [r'http\S+', '|', '_', 'infp', 'infj', 'intp', 'intj', 'entp', 'enfp', 'istp', 'isfp', 'entj', 'istj', 'enfj', 'isfj', 'estp', 'esfp', 'esfj', 'estj']
for word in words_to_remove:
    mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(word, '', regex=True)


# %% CONVERT SMILEY FACES INTO WORDS
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r":-\)|:\)", 'smile', regex=True)
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r":-\)|:\)", 'frown', regex=True)
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r";-\)|;\)", 'wink', regex=True)
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r":-D|:D", 'grin', regex=True)
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r":-P|:P", 'tongue_out', regex=True)
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r":-\||:\|", 'neutral', regex=True)
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r":-S|:S", 'confused', regex=True)




