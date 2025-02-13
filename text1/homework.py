import multiprocessing
import re
from collections import defaultdict
from time import time

import gensim
import matplotlib.pyplot as plt

import numpy as np
import spacy
from gensim import models
from gensim.models import Word2Vec
import nltk
import pandas as pd


data = pd.read_csv('data/simpsons_script_lines.csv', on_bad_lines='skip')
data.dropna(subset=["spoken_words" ], inplace=True)
data = data.dropna().reset_index(drop=True)


nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
def cleaning(doc):
    txt = [token.lemma_ for token in doc if not token.is_stop]
    if len(txt) > 2:
        return ' '.join(txt)
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in data['spoken_words'])
txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000)]
df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()
df_clean.shape


from gensim.models.phrases import Phrases, Phraser
sent = [row.split() for row in df_clean['clean']]
phrases = Phrases(sent, min_count=30, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[sent]

print(txt)
exit()
