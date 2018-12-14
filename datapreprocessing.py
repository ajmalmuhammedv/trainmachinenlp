
from __future__ import division
import numpy as np
import pickle

import csv
import nltk as nlp
from nltk import word_tokenize
import string
filename = "glove.6B.50d.txt"
# (glove data set from: https://nlp.stanford.edu/projects/glove/)
print("hello")
vocab = []
embd = []

def loadGloVe():

    file = open("trail.txt", "r+")
    print("Name of the file: ", file.name)
# file = open("glove.6B.50d.txt", "r+")
#lines = file.readlines()
# print("hello",lines)
    """"
    for line in file.readlines(20):
        row = line.strip().split(' ')
        print(row)
        vocab.append(row[0])
        embd.append(row[1:])
    print('GloVe Loaded.')
    file.close()
    return vocab,embd
"""
    for i in range(1400):
        lines=file.readline()
        row = lines.strip().split(' ')
        print(row)
        vocab.append(row[0])
        embd.append(row[1:])


# Pre-trained GloVe embedding
loadGloVe()
print(vocab)
print(embd)

embedding = np.asarray(embd)

#print(embedding)

summaries = []
texts = []


def clean(text):
    text = text.lower()
    printable = set(string.printable)
    return filter(lambda x: x in printable, text)  # filter funny characters, if any.


with open('Reviews.csv', newline='') as csvfile:
    Reviews = csv.DictReader(csvfile)
    for row in Reviews:
        print(row['Text'])
        clean_text = clean(row['Text'])
        clean_summary = clean(row['Summary'])
        summaries.append(word_tokenize(clean_summary))
        texts.append(word_tokenize(clean_text))



print("SAMPLE CLEANED & TOKENIZED TEXT: \n\n"+str(texts[2]))
print("\nSAMPLE CLEANED & TOKENIZED SUMMARY: \n\n"+str(summaries[2]))

"""
vocab,embd = loadGloVe()
print(vocab)
print("hello",embd)


embedding = embedding.astype(np.float32)

word_vec_dim = len(embd[0]) # word_vec_dim = dimension of each word vectors



summaries = []
texts = []


def clean(text):
    text = text.lower()
    printable = set(string.printable)
    return filter(lambda x: x in printable, text)  # filter funny characters, if any.


with open('Reviews.csv', 'rb') as csvfile:  # Data from https://www.kaggle.com/snap/amazon-fine-food-reviews
    Reviews = csv.DictReader(csvfile)
    for row in Reviews:
        clean_text = clean(row['Text'])
        clean_summary = clean(row['Summary'])
        summaries.append(word_tokenize(clean_summary))
        texts.append(word_tokenize(clean_text))




import random

index = random.randint(0,len(texts)-1)

print("SAMPLE CLEANED & TOKENIZED TEXT: \n\n"+str(texts[index]))
print("\nSAMPLE CLEANED & TOKENIZED SUMMARY: \n\n"+str(summaries[index]))


def np_nearest_neighbour(x):
    # returns array in embedding that's most similar (in terms of cosine similarity) to x

    xdoty = np.multiply(embedding, x)
    xdoty = np.sum(xdoty, 1)
    xlen = np.square(x)
    xlen = np.sum(xlen, 0)
    xlen = np.sqrt(xlen)
    ylen = np.square(embedding)
    ylen = np.sum(ylen, 1)
    ylen = np.sqrt(ylen)
    xlenylen = np.multiply(xlen, ylen)
    cosine_similarities = np.divide(xdoty, xlenylen)

    return embedding[np.argmax(cosine_similarities)]


def word2vec(word):  # converts a given word into its vector representation
    if word in vocab:
        return embedding[vocab.index(word)]
    else:
        return embedding[vocab.index('unk')]


def vec2word(vec):  # converts a given vector representation into the represented word
    for x in xrange(0, len(embedding)):
        if np.array_equal(embedding[x], np.asarray(vec)):
            return vocab[x]
    return vec2word(np_nearest_neighbour(np.asarray(vec)))


word = "unk"
print("Vector representation of '"+str(word)+"':\n")
print(word2vec(word))

# REDUCE DATA (FOR SPEEDING UP THE NEXT STEPS)

MAXIMUM_DATA_NUM = 50000

texts = texts[0:MAXIMUM_DATA_NUM]
summaries = summaries[0:MAXIMUM_DATA_NUM]

vocab_limit = []
embd_limit = []

i = 0
for text in texts:
    for word in text:
        if word not in vocab_limit:
            if word in vocab:
                vocab_limit.append(word)
                embd_limit.append(word2vec(word))

for summary in summaries:
    for word in summary:
        if word not in vocab_limit:
            if word in vocab:
                vocab_limit.append(word)
                embd_limit.append(word2vec(word))

if 'eos' not in vocab_limit:
    vocab_limit.append('eos')
    embd_limit.append(word2vec('eos'))
if 'unk' not in vocab_limit:
    vocab_limit.append('unk')
    embd_limit.append(word2vec('unk'))

null_vector = np.zeros([word_vec_dim])

vocab_limit.append('<PAD>')
embd_limit.append(null_vector)

vec_summaries = []

for summary in summaries:

    vec_summary = []

    for word in summary:
        vec_summary.append(word2vec(word))

    vec_summary.append(word2vec('eos'))

    vec_summary = np.asarray(vec_summary)
    vec_summary = vec_summary.astype(np.float32)

    vec_summaries.append(vec_summary)

vec_texts = []

for text in texts:

    vec_text = []

    for word in text:
        vec_text.append(word2vec(word))

    vec_text = np.asarray(vec_text)
    vec_text = vec_text.astype(np.float32)

    vec_texts.append(vec_text)

# Saving processed data in another file.



with open('vocab_limit', 'wb') as fp:
    pickle.dump(vocab_limit, fp)
with open('embd_limit', 'wb') as fp:
    pickle.dump(embd_limit, fp)
with open('vec_summaries', 'wb') as fp:
    pickle.dump(vec_summaries, fp)
with open('vec_texts', 'wb') as fp:
    pickle.dump(vec_texts, fp)
"""