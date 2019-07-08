from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def build_word_cloud(text):
	wordcloud = WordCloud().generate(text)
	plt.figure(figsize=(16,10))
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.show()

def get_max_word_length(df, text_col):
	# input: dataframe, column name that require preprocessing
	# out:  maximum word in text corpus
	len_max = 0
	for idx,row in df.iterrows():
	    sentence = row['headline'].split(' ')
	    if len(sentence)>len_max:
	        len_max=len(sentence)
	return len_max

def create_embedding_index():
	embeddings_index = dict()
	f = open('glove.6B.50d.txt')
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()
	return embeddings_index


def construct_embedding_matrix(df, text_col):
	embeddings_index = create_embedding_index()
	# prepare tokenizer
	t = Tokenizer()
	t.fit_on_texts(df[text_col])
	vocab_size = len(t.word_index) + 1
	# integer encode the documents
	encoded_docs = t.texts_to_sequences(df[text_col])
	len_max = get_max_word_length(df, text_col)
	# pad documents to a max length of 4 words
	padded_docs = pad_sequences(encoded_docs, maxlen=len_max, padding='post')

	embedding_matrix = np.zeros((vocab_size, 50))
	for word, i in t.word_index.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        embedding_matrix[i] = embedding_vector
	return padded_docs, embedding_matrix
