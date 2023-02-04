# %%
# taken from https://opendatascience.com/creating-word-clouds-from-text/?utm_campaign=Newsletters&utm_medium=email&_hsmi=2&_hsenc=p2ANqtz--JaDC53T21O5oExiZFIFC_2OUcsfsrKerobHldEhvccO4prGCg6ooQtjlAP-WZfR1XB_Moc1qGtaVlLgrd4VWt9t6b7A&utm_content=2&utm_source=hs_email

import os
import nltk
from os import path
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.probability import FreqDist
from PIL import Image
import numpy as np

nltk.download('punkt')

text_file = "../data/dataset.csv" # ../ for going up one folder
text = open(text_file, "r", encoding="utf-8").read()
text = text.lower()

def compile_stopwords_list_frequency(text, freq_percentage=0.02):
    words = nltk.tokenize.word_tokenize(text)
    freq_dist = FreqDist(word.lower() for word in words)
    words_with_frequencies = [(word, freq_dist[word]) for word in freq_dist.keys()]
    sorted_words = sorted(words_with_frequencies, key=lambda tup: tup[1])
    length_cutoff = int(freq_percentage*len(sorted_words))
    stopwords = [tuple[0] for tuple in sorted_words[-length_cutoff:]]
    return stopwords

# probably not needed because stopwords were already taken away
stopwords = compile_stopwords_list_frequency(text)
stopwords.remove("MBTI")

output_filename = "odsc_wordcloud.png"
wordcloud = WordCloud(min_font_size=10, max_font_size=100, stopwords=stopwords, width=1000, height=1000, max_words=1000, background_color="white").generate(text)
wordcloud.to_file(output_filename)

plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
