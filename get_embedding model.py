import pandas as pd
import torch
import random
import numpy as np

train_data=pd.read_csv("/home/mohit/akshay/train.csv",
                 lineterminator='\n')

test_data=pd.read_csv("/home/mohit/akshay/test.csv",
                 lineterminator='\n')



dataset = pd.concat([train_data, test_data], ignore_index=True, sort=False)

#d1=dataset['headline']

d2=dataset['summary']

d3=dataset['article']

#d2.columns=['article']

#d1.columns=['article']

data=pd.concat([d2,d3],ignore_index=True,sort=True)

data=data.dropna()

import re
def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, ' ', data)

def preprocess_tokenize(text):
      # for removing punctuation from sentencesc
    text = str(text)
    text = re.sub(r'(\d+)', r'', text)

    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\u200d', '')
    text=re.sub("(__+)", ' ', str(text)).lower()   #remove _ if it occors more than one time consecutively
    text=re.sub("(--+)", ' ', str(text)).lower()   #remove - if it occors more than one time consecutively
    text=re.sub("(~~+)", ' ', str(text)).lower()   #remove ~ if it occors more than one time consecutively
    text=re.sub("(\+\++)", ' ', str(text)).lower()   #remove + if it occors more than one time consecutively
    text=re.sub("(\.\.+)", ' ', str(text)).lower()   #remove . if it occors more than one time consecutively
    text=re.sub(r"[<>()|&©@#ø\[\]\'\",;:?.~*!]", ' ', str(text)).lower() #remove <>()|&©ø"',;?~*!
    text = re.sub(r"[‘’।:]", " ", str(text)) #removing other special characters
    text = re.sub("([a-zA-Z])",' ',str(text)).lower()
    text = re.sub("(\s+)",' ',str(text)).lower()
    text = remove_emojis(text)
    return text

def tokenize_text(text):
	sepp=text.split(" ")
	return sepp


print('applying preprocessing')
data = data.apply(preprocess_tokenize)
print('done preprocessing')
data=data.to_frame()
data.columns=['data']
print('tokenising...')
data = data.apply(lambda col: col.apply(tokenize_text), axis=1)
print('done tokenizing...')
from gensim.models import Word2Vec
print('training model')
model = Word2Vec(data['data'],vector_size = 256, min_count=1)
model.build_vocab(data['data'])

model.train(data['data'], total_examples=model.corpus_count, epochs=50)
print('done training model')
print(model.wv.index_to_key)
model.save("/media/mohit/custom_word2vec.bin")
print(model.wv["संस्कृति"])
model1 = Word2Vec.load("/media/mohit/my_word2vec_model.bin")
print(model1.wv.most_similar("संस्कृति", topn=3))

