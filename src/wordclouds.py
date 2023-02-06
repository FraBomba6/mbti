# %%

import os
import nltk
from os import path
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.probability import FreqDist
from PIL import Image
import numpy as np
import pandas as pd
import sklearn

nltk.download('punkt')

text_file = "../data/dataset.csv" # ../ for going up one folder
text = open(text_file, "r", encoding="utf-8").read()
text = text.lower()

# %%
# ALTERNATIVE 1 (probably worse)
# taken from https://opendatascience.com/creating-word-clouds-from-text/?utm_campaign=Newsletters&utm_medium=email&_hsmi=2&_hsenc=p2ANqtz--JaDC53T21O5oExiZFIFC_2OUcsfsrKerobHldEhvccO4prGCg6ooQtjlAP-WZfR1XB_Moc1qGtaVlLgrd4VWt9t6b7A&utm_content=2&utm_source=hs_email



# probably not needed because stopwords were already taken away
#stopwords = compile_stopwords_list_frequency(text)
#stopwords.remove("MBTI")

#output_filename = "odsc_wordcloud.png"
#wordcloud = WordCloud(min_font_size=10, max_font_size=100, stopwords=stopwords, width=1000, height=1000, max_words=1000, background_color="white").generate(text)
#wordcloud.to_file(output_filename)

#plt.figure()
#plt.imshow(wordcloud, interpolation="bilinear")
#plt.axis("off")
#plt.show()


# %%
# taken from https://donche.github.io/en/2017/12/27/mbti_blog.html, example with E-I
#ALTERNATIVE 2 (probably better because applied to the same task)

data = "../data/dataset.csv" # ../ for going up one folder
text = open(text_file, "r", encoding="utf-8").read()
text = text.lower()
type_quote = data.groupby('type').sum()

e_posts = ''
i_posts = ''
n_posts = ''
s_posts = ''
f_posts = ''
t_posts = ''
j_posts = ''
p_posts = ''

for _type in type_quote.index:
    if 'E' in _type:
        e_posts += type_quote.loc[_type].posts
    else:
        i_posts += type_quote.loc[_type].posts

for _type in type_quote.index:
    if 'S' in _type:
        s_posts += type_quote.loc[_type].posts
    else:
        n_posts += type_quote.loc[_type].posts

for _type in type_quote.index:
    if 'T' in _type:
        t_posts += type_quote.loc[_type].posts
    else:
        f_posts += type_quote.loc[_type].posts

for _type in type_quote.index:
    if 'J' in _type:
        j_posts += type_quote.loc[_type].posts
    else:
        p_posts += type_quote.loc[_type].posts


# Generate wordcloud

#stopwords = set(STOPWORDS)
#stopwords.add("think")
#stopwords.add("people")
#stopwords.add("thing")
def compile_stopwords_list_frequency(text, freq_percentage=0.02):
    words = nltk.tokenize.word_tokenize(text)
    freq_dist = FreqDist(word.lower() for word in words)
    words_with_frequencies = [(word, freq_dist[word]) for word in freq_dist.keys()]
    sorted_words = sorted(words_with_frequencies, key=lambda tup: tup[1])
    length_cutoff = int(freq_percentage*len(sorted_words))
    stopwords = [tuple[0] for tuple in sorted_words[-length_cutoff:]]
    return stopwords

stopwords = compile_stopwords_list_frequency(text)
my_wordcloud = WordCloud(width=800, height=800, stopwords=stopwords, background_color='white')

# %%

my_wordcloud_infj = my_wordcloud.generate(i_posts, n_posts, f_posts, j_posts)
plt.subplots(figsize = (15,15))
plt.imshow(my_wordcloud_infj)
plt.axis("off")
plt.title('INFJ', fontsize = 30)
plt.show()



# %%
# Introvert
my_wordcloud_i = my_wordcloud.generate(i_posts)
plt.subplots(figsize = (15,15))
plt.imshow(my_wordcloud_infj)
plt.axis("off")
plt.title('Introvert', fontsize = 30)
plt.show()

#Extrovert
my_wordcloud_e = my_wordcloud.generate(e_posts)
plt.subplots(figsize = (15,15))
plt.imshow(my_wordcloud_infj)
plt.axis("off")
plt.title('Extrovert', fontsize = 30)
plt.show()

#Sensing
my_wordcloud_s = my_wordcloud.generate(s_posts)
plt.subplots(figsize = (15,15))
plt.imshow(my_wordcloud_infj)
plt.axis("off")
plt.title('Sensing', fontsize = 30)
plt.show()

#Intuition
my_wordcloud_i = my_wordcloud.generate(i_posts)
plt.subplots(figsize = (15,15))
plt.imshow(my_wordcloud_infj)
plt.axis("off")
plt.title('Intuition', fontsize = 30)
plt.show()

#Thinking
my_wordcloud_t = my_wordcloud.generate(t_posts)
plt.subplots(figsize = (15,15))
plt.imshow(my_wordcloud_infj)
plt.axis("off")
plt.title('Thinking', fontsize = 30)
plt.show()

#Feeling
my_wordcloud_f = my_wordcloud.generate(f_posts)
plt.subplots(figsize = (15,15))
plt.imshow(my_wordcloud_infj)
plt.axis("off")
plt.title('Feeling', fontsize = 30)
plt.show()

#Judging
my_wordcloud_j = my_wordcloud.generate(j_posts)
plt.subplots(figsize = (15,15))
plt.imshow(my_wordcloud_infj)
plt.axis("off")
plt.title('Judging', fontsize = 30)
plt.show()

#Perception
my_wordcloud_p = my_wordcloud.generate(e_posts)
plt.subplots(figsize = (15,15))
plt.imshow(my_wordcloud_infj)
plt.axis("off")
plt.title('Perception', fontsize = 30)
plt.show()