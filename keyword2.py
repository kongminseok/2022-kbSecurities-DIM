from settingDatetime import *

import numpy as np
import pandas as pd
import re
import os

from tqdm import tqdm
from gensim.models import KeyedVectors
from konlpy.tag import Okt

import warnings
warnings.filterwarnings('ignore')

import w2v

path = os.getcwd() # 절대 경로 설정
stw_df = pd.read_excel(path + '/data/stopword_dictionary.xlsx', usecols = [0,4]) 
stw_df = stw_df[stw_df['stopword'] == 1]
stw_lst = stw_df['word'].tolist()


from ckonlpy.tag import Twitter ### Okt가 아니라 Twitter라는 이름으로 씀
twitter = Twitter()


def extract_10keywords() : 

    raw_df = pd.read_csv(path + f'/data/crawling/{target}_naver_finance_news.csv', encoding = 'utf-8-sig') # 이게 찐
    #raw_df = pd.read_csv(path + '/data/crawling/20220324_naver_finance_news.csv') # test용
    raw_df['내용']=raw_df[['제목','본문']].apply(lambda x:'.'.join(x.values.astype(str)), axis=1)
    title_series = raw_df['내용']
    #title_series = raw_df['본문']
    
    def drop_null_n_duplicates(untokenized_texts) :  #결측치, 중복값 제거 
        untokenized_texts = untokenized_texts.drop_duplicates()
        untokenized_texts = untokenized_texts.dropna()

        tokenize(untokenized_texts)

    def tokenize(untokenized_texts) : # 토크나이징하기(Nouns)
        
        tokenized_texts = untokenized_texts.map(lambda x : twitter.nouns(x))

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
        
        for i in tqdm(range(len(word_rank_df))) : 
            if word_rank_df['word'][i] in stw_lst :
                word_rank_df.drop(i, inplace = True, axis = 0)

        word_rank_df.reset_index(inplace = True, drop = True)
        
        word_df = word_rank_df.head(10)

        #untokenized_texts.to_csv('./data/untokenized_texts.csv',encoding='utf-8-sig',index=False)
        #tokenized_texts.to_csv('./data/tokenized_texts.csv',encoding='utf-8-sig',index=False)
        #word_df.to_csv('./data/word_df.csv',encoding='utf-8-sig',index=False)

        print("keywords have been extracted!")
        
        w2v.get_stock_name(word_df['word'], word_df['count'])
        
        w2v_res = pd.read_csv(f'./data/rank/{target}_rank.csv',encoding='utf-8-sig')
        
        find_text_from_word(w2v_res['stock'], tokenized_texts, untokenized_texts)

    
    def find_text_from_word(words, tokenized_texts, untokenized_texts): # 단어가 본문에서 어떻게 쓰였는지 찾아주는 함수

        global word_n_article

        tmp = pd.DataFrame(columns = ['word','text'])
        for word in words : 
            for i in range(len(tokenized_texts)) : 
                if word in tokenized_texts.iloc[i] : 
                    res = {'word':word, 'text' : untokenized_texts.iloc[i], 'index' : str(i)}
                    tmp = tmp.append(res, ignore_index = True)

        word_n_article = tmp
        
        word_n_article['date'] = target
        word_n_article['day'] = target_day

        word_n_article[['date','day','word','text']].to_csv(f'./data/input/{target}_input.csv',encoding='utf-8-sig',index=False)

        return print('articles have been extracted and saved!')
    
    drop_null_n_duplicates(title_series)
