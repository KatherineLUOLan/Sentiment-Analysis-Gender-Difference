import numpy
from nltk.classify import NaiveBayesClassifier
import linecache
import random
import os
import re
import pandas as pd
from pdfminer.converter import LTChar, TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox
from io import StringIO
from io import open


pos_text=open('data/pos.txt','r')
neg_text=open('data/neg.txt','r')

Sum_Line=5000
Devide_Part=1
training_data=[]
dict_num={}

def line_clean(line):
    words=line.lower().strip().split()
    for word in words:
        if word in ',.?!;:':
            words.remove(word)
    return ' '.join(words)


#IDF预处理
for i in range(int(Sum_Line*Devide_Part)):
    new_line=[word for word in line_clean(linecache.getline('data/pos.txt',i)).split()]
    for word in new_line:
        if word in dict_num.keys():
            dict_num[word]+=1.0
        else:
            dict_num[word]=1.0
for i in range(int(Sum_Line*Devide_Part)):
    new_line=[word for word in line_clean(linecache.getline('data/neg.txt',i)).split()]
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


pro_lim1=0.01
pro_lim2=0.02

#特殊特征标记
def preprocess1(s):
    return {word : (dict_num[word]>pro_lim1) for word in line_clean(s).split()}
def preprocess2(s):
    return {word : (word in dict_num.keys() and dict_num[word]>pro_lim2) for word in line_clean(s).split()}

for i in range(int(Sum_Line*Devide_Part)):
    training_data.append([preprocess1(line_clean(linecache.getline('data/pos.txt',i))),1])

for i in range(int(Sum_Line*Devide_Part)):
    training_data.append([preprocess1(line_clean(linecache.getline('data/neg.txt',i))),-1])

model = NaiveBayesClassifier.train(training_data)

#建立输出excel
df = pd.DataFrame(None,columns=['txtfile','sum'])

path = 'data/assignment3'
txtList = os.listdir(path)
#批量读取存储
pdf_num = 0

for li in txtList: #遍历文件夹
    txtfile = path + '/' + li   #构造绝对路径
    with open(txtfile, 'r' ,encoding='ISO-8859-1' ) as f:
        lines=f.readlines()
        sum=0  
        for line in lines:   #遍历文件夹的每行
            probdist = model.prob_classify(preprocess2(line_clean(line)))  #计算每行的正负向
            sum=sum+probdist.max()  #计算总的正负向
        df = df.append(pd.DataFrame([[txtfile,sum]], columns=df.columns))
    pdf_num = pdf_num + 1
print(df)
df.to_excel('output.xlsx', sheet_name='output', index=False)    
print('number of done-article:',end = "")
print(pdf_num)