import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from tqdm import tqdm
import pandas as pd


text_file = "../data/dataset.csv" # ../ for going up one folder
text = open(text_file, "r", encoding="utf-8").read()
text = text.lower()
dataset = pd.read_csv("../data/dataset.csv")

posts_by_type = {}

for row in dataset.iterrows():
    row = row[1]
    if row[0] not in posts_by_type.keys():
        posts_by_type[row[0]] = ""
    posts_by_type[row[0]] += row[1]

# %%

stopwords = set(STOPWORDS)
stopwords.add('|')
stopwords.add('_')
stopwords.add(r'http\S+')
stopwords.add('infp')
stopwords.add('infj')
stopwords.add('intp')
stopwords.add('intj')
stopwords.add('entp')
stopwords.add('istp')
stopwords.add('isfp')
stopwords.add('entj')
stopwords.add('istj')
stopwords.add('enfj')
stopwords.add('isfj')
stopwords.add('estp')
stopwords.add('esfj')
stopwords.add('esfp')
stopwords.add('enfp')
stopwords.add('estj')
stopwords.add('people')
stopwords.add('think')
stopwords.add('know')
stopwords.add('one')
stopwords.add('thing')
stopwords.add('really')
stopwords.add('well')
stopwords.add('type')

my_wordcloud = WordCloud(width=800, height=800, stopwords=stopwords, background_color='white')

for type in tqdm(posts_by_type.keys()):
    my_wordcloud_type = my_wordcloud.generate(posts_by_type[type])
    plt.subplots(figsize = (15,15))
    plt.imshow(my_wordcloud_type)
    plt.axis("off")
    plt.title(type, fontsize = 30)
    plt.savefig('../img/'+type+'.png')
    plt.show()
