import numpy
from sklearn.feature_extraction.text import CountVectorizer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import linecache
import random

pos_text=open('data/pos.txt','r')
neg_text=open('data/neg.txt','r')

Sum_Line=5000
Devide_Part=0.7
training_data=[]
dict_num={}

def line_clean(line):
    words=line.lower().strip().split()
    for word in words:
        if word in ',.?!;:':
            words.remove(word)
    return ' '.join(words)

for i in range(int(Sum_Line*Devide_Part)):
    new_line=[word for word in line_clean(linecache.getline('data\\rt-polarity.pos',i)).split()]
    for word in new_line:
        if word in dict_num.keys():
            dict_num[word]+=1.0
        else:
            dict_num[word]=1.0
for i in range(int(Sum_Line*Devide_Part)):
    new_line=[word for word in line_clean(linecache.getline('data\\rt-polarity.neg',i)).split()]
    for word in new_line:
        if word in dict_num.keys():
            dict_num[word]+=1.0
        else:
            dict_num[word]=1.0
max_num=0.0
for word in dict_num.keys():
    max_num=max(max_num,dict_num[word])
for word in dict_num.keys():
    dict_num[word]=float(1-dict_num[word]/max_num)

size=25
for pro_lim1 in range(0,size,1):
    for pro_lim2 in range(0,size,1):

        pro_lim1=float(pro_lim1)/size
        pro_lim2=float(pro_lim2)/size
        def preprocess1(s):
            return {word : (dict_num[word]>pro_lim1) for word in line_clean(s).split()}
        def preprocess2(s):
            return {word : (word in dict_num.keys() and dict_num[word]>pro_lim2) for word in line_clean(s).split()}

        for i in range(int(Sum_Line*Devide_Part)):
            training_data.append([preprocess1(line_clean(linecache.getline('data\\rt-polarity.pos',i))),'pos'])

        for i in range(int(Sum_Line*Devide_Part)):
            training_data.append([preprocess1(line_clean(linecache.getline('data\\rt-polarity.neg',i))),'neg'])

        random.shuffle(training_data)
        model = NaiveBayesClassifier.train(training_data)

        Cor_Num_Pos=0.0
        for i in range(int(Sum_Line*Devide_Part),Sum_Line):
            Cor_Num_Pos+=(model.classify(preprocess2(line_clean(linecache.getline('data\\rt-polarity.pos',i))))=='pos')

        Cor_Num_Neg=0.0
        for i in range(int(Sum_Line*Devide_Part),Sum_Line):
            Cor_Num_Neg+=(model.classify(preprocess2(line_clean(linecache.getline('data\\rt-polarity.neg',i))))=='neg')

        print(round((Cor_Num_Pos/(Sum_Line*(1.0-Devide_Part))+Cor_Num_Neg/(Sum_Line*(1.0-Devide_Part)))/2,5),)
    print