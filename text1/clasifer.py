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
data.dropna(subset=["normalized_text" ], inplace=True)

data = data[(data['raw_character_text'] =='Lisa Simpson') | (data['raw_character_text'] =='Bart Simpson')]

data_bart = data[(data['raw_character_text'] =='Bart Simpson')]
data_bart_text = list(data_bart['normalized_text'].values)

data_lisa = data[(data['raw_character_text'] =='Lisa Simpson')]
data_lisa_text = list(data_bart['normalized_text'].values)

w2v_model = Word2Vec.load("word2vec.model")


def get_phrase_embedding(model, secs):
    collect = []
    for sec in secs:
        collect_sec = model.wv.get_mean_vector(sec)
        # collect_sec = []
        # for word in sec:
        #     if word in  model.wv.key_to_index.keys():
        #         index_word = model.wv.key_to_index[word]
        #         vector = np.array(model.wv[index_word])
        #         collect_sec.append(vector)
        #
        # collect_sec = np.mean(collect_sec, axis=0)
        # collect_sec = collect_sec / np.linalg.norm(collect_sec)
        collect.append(collect_sec)

    return collect
data_bart_text = get_phrase_embedding(w2v_model, data_bart_text)
data_bart_text = get_phrase_embedding(w2v_model, data_lisa_text)


print(len(data_bart_text))
exit()

