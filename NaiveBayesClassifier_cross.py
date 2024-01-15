import nltk
from nltk.classify.scikitlearn import SklearnClassifier
import linecache
import random
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np


# 假设 pos_reviews 和 neg_reviews 分别包含正面和负面评论的文本
pos_reviews = [line.strip() for line in open('data/pos.txt', 'r')]
neg_reviews = [line.strip() for line in open('data/neg.txt', 'r')]

reviews = pos_reviews + neg_reviews
labels = ['pos'] * len(pos_reviews) + ['neg'] * len(neg_reviews)


Sum_Line=5000
Devide_Part=0.8
data=[]
dict_num={}

def line_clean(line):
    words=line.lower().strip().split()
    for word in words:
        if word in ',.;:':
            words.remove(word)
    return ' '.join(words)

def preprocess1(s):
    return {word : True for word in line_clean(s).split()}

for i in range(int(Sum_Line*Devide_Part)):
    data.append([preprocess1(line_clean(linecache.getline('data/pos.txt',i))),'pos'])
    data.append([preprocess1(line_clean(linecache.getline('data/neg.txt',i))),'neg'])

random.shuffle(data)
training_data=data[:int(Sum_Line*Devide_Part)]
test_data=data[int(Sum_Line*Devide_Part):]
model = nltk.NaiveBayesClassifier.train(training_data)


model = make_pipeline(CountVectorizer(), MultinomialNB())
scores = cross_val_score(model, reviews, labels, cv=5)  # 5折交叉验证
print("平均准确率: ", np.mean(scores))
