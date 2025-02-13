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
data = data[(data['raw_character_text'] =='Lisa Simpson') | (data['raw_character_text'] =='Bart Simpson')]

data_bart = data[(data['raw_character_text'] =='Bart Simpson')]
data_bart_text = list(data_bart['normalized_text'].values)
w2v_model = Word2Vec.load("word2vec.model")


from sklearn.feature_extraction.text import TfidfVectorizer

text = ["She sells seashells by the seashore","The sea.","The seashore"]
vectorizer = TfidfVectorizer()
vectorizer.fit(data_bart_text)
print(vectorizer)
exit()

def get_phrase_embedding(model, secs):
    collect = []
    for sec in secs:
        collect = []
        text_vector = model.wv.get_mean_vector(sec)

        for word in sec:
            if word in  model.wv.key_to_index.keys():
                index_word = model.wv.key_to_index[word]

                vector = np.array(model.wv[index_word])
                collect.append(vector)

        ret = np.mean(np.array(collect), axis=0)
        ret = ret / np.linalg.norm(ret)
        print(sec)
        print(ret[:50])
        print(text_vector[:50])
        exit()

    print(ret.shape)
    exit()



    vector = np.zeros([model.vector_size], dtype='float32')

    return vector
get_phrase_embedding(w2v_model, data_bart_text)

