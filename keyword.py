#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import datetime

from tqdm import tqdm
from gensim.models import KeyedVectors
from konlpy.tag import Okt

import warnings
warnings.filterwarnings('ignore')


# In[2]:


path = os.getcwd() # 절대 경로 설정
stw_df = pd.read_excel(path + '/data/stopword_dictionary.xlsx', usecols = [0,4]) 
stw_df = stw_df[stw_df['stopword'] == 1]
stw_lst = stw_df['word'].tolist()


# In[3]:


# get yesterday
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)
yesterday = str(yesterday)
yesterday = yesterday.replace("-","")


# In[4]:


from ckonlpy.tag import Twitter ### Okt가 아니라 Twitter라는 이름으로 씀
twitter = Twitter()


# In[31]:


def extract_10keywords() : 
    global title_series
    
#     raw_df = pd.read_csv(path + f'/data/crawling/{yesterday}_naver_finance_news.csv', encoding = 'utf-8-sig')
    raw_df = pd.read_csv(path + '/data/crawling/20220324_naver_finance_news.csv')
    title_series = raw_df['제목']


    def drop_null_n_duplicates(untokenized_texts) :  #결측치, 중복값 제거 
        untokenized_texts = untokenized_texts.drop_duplicates()
        untokenized_texts = untokenized_texts.dropna()
        
        title_series = untokenized_texts

        tokenize(untokenized_texts)

    def tokenize(untokenized_texts) : # 토크나이징하기(Nouns)
        
        global tokenized_title
        
        tokenized_texts = untokenized_texts.map(lambda x : twitter.nouns(x))
        
        tokenized_title = tokenized_texts

        count_n_sort(untokenized_texts, tokenized_texts)

    def count_n_sort(untokenized_texts, tokenized_texts) : # 단어-횟수 count하고 내림차순으로 정렬
        dic = {}
        for row in tokenized_texts : 
            for word in row : 
                if word not in dic : 
                    dic[word] = 1
                else : 
                    dic[word] += 1

        res_dict = dict(sorted(dic.items(), key=lambda item: item[1], reverse = True))
        word_rank_df = pd.DataFrame(list(res_dict.items()),columns=['word', 'count'])

        filter_stopword(untokenized_texts, tokenized_texts, word_rank_df)

    def filter_stopword(untokenized_texts, tokenized_texts, word_rank_df) : # 불용어 처리 후 출력
        global word_df
        for i in tqdm(range(len(word_rank_df))) : 
            if word_rank_df['word'][i] in stw_lst :
                word_rank_df.drop(i, inplace = True, axis = 0)

        word_rank_df.reset_index(inplace = True, drop = True)
        
        word_df = word_rank_df.head(10)

        find_text_from_word(word_df['word'], tokenized_texts, untokenized_texts)
    
    def find_text_from_word(words, tokenized_texts, untokenized_texts): # 단어가 본문에서 어떻게 쓰였는지 찾아주는 함수

        global word_n_article
        
        tmp = pd.DataFrame(columns = ['word','text'])
        for word in words : 
            for i in range(len(tokenized_texts)) : 
                if word in tokenized_texts.iloc[i] : 
                    res = {'word':word, 'text' : untokenized_texts.iloc[i], 'index' : str(i)}
                    tmp = tmp.append(res, ignore_index = True)
                    
        word_n_article = tmp
                    


#         return words.head(10), tmp
    
    drop_null_n_duplicates(title_series)


# In[28]:


extract_10keywords()


# In[29]:


word_df


# In[30]:


word_n_article


# In[ ]:




