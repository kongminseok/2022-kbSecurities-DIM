from pycode.settingDatetime import *

import numpy as np
import pandas as pd
import re
import os
from tqdm import tqdm

from gensim.models import KeyedVectors
from gensim.models import Word2Vec

from ckonlpy.tag import Twitter


twitter = Twitter()

# 절대경로 설정
path = os.getcwd()

# 뉴스 데이터 불러오기
df = pd.read_csv(path + '/data/naver_finance_news_concat.csv')

# 불용어 사전 불러오기
stw_df = pd.read_excel(path + '/data/stopword_dictionary.xlsx', usecols = [0,4])
stw_df = stw_df[stw_df['stopword'] == 1]
stw_lst = stw_df['word'].tolist()

# 토크나이징 할 데이터 변수지정(제목 + 본문)
data = df['제목'] + df['본문']
data = data.dropna()
data = data.drop_duplicates()
data = data.reset_index(drop= True)

# 토크나이징
tokenized_data = data.map(lambda x : twitter.nouns(x))

# w2v 모델 fitting
# model = Word2Vec(sentences = tokenized_data) # 모델 처음부터 학습시키기
model = Word2Vec.load('word2vec.model') # 저장된 모델 불러오기

# 상장기업 이름 리스트 불러오기
with open(path + "/data/twitter/stock.txt", "r", encoding='utf-8-sig') as f:
    entre = f.read() #entre : 기업이름
    
entre_lst = entre.split('\n')[:-1]

# 붙어서 나와야 하는 단어들 추가하는 함수
def append_new_words(word) : # '메타버스', '트래블룰' 처럼 기업명은 아니지만 짤려서 추출되면 안되는 단어들 추가(표제어 느낌으로 생각해주셈)
    with open(path + '/data/twitter/new_words.txt', 'r') as f:
        new_words = f.read()
    
    new_words_lst = new_words.split('\n')
    
    if word not in new_words_lst : 
        new_words += (word+'\n')
        
        with open(path + '/data/twitter/new_words.txt', 'w') as f:
            f.write(new_words)
        
    else : 
        print('This word has already been appended')
        
    return


# 종목명 추출 함수

# stock_lst = []
def get_stock_name(keyword_lst) : 
    stock_lst = []
    is_keyword_lst = []
        
    for keyword in keyword_lst : 
        try : 
            if keyword in entre_lst : 
                stock_lst.append(keyword)
                is_keyword_lst.append(keyword)
                continue
            else : 
                words = model.wv.most_similar(keyword, topn = 100)
                for word in words : 
                    if word[0] in entre_lst : 
                        stock_lst.append(word[0])
                        is_keyword_lst.append(keyword)
                        break

                    else : 
                        words = model.wv.most_similar(keyword, topn = 100)
                        continue
                    
        except KeyError : 
            continue
    
    #for keyword in keyword_lst : 
    #    if keyword in entre_lst : 
    #        stock_lst.append(keyword)
    #        continue
    #    else : 
    #        words = model.wv.most_similar(keyword, topn = 100)
    #        for word in words : 
    #            if word[0] in entre_lst : 
    #                stock_lst.append(word[0])
    #                break
    #
    #            else : 
    #                words = model.wv.most_similar(keyword, topn = 100)
    #                continue
                    
    res = pd.DataFrame()
    res['word'] = is_keyword_lst
    res['stock'] = stock_lst
    res.to_csv(f'./data/rank/{target}_rank.csv', encoding='utf-8-sig', index = False)


    return print('stock list has been extracted and saved!')