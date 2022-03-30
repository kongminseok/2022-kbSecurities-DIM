from pycode.settingDatetime import *

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

import easydict
import pandas as pd

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair) 

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]
        #self.sentences = [transform([i]) for i in dataset[dataset.columns[sent_idx]]]
        #self.labels = [np.int32(i) for i in dataset[dataset.columns[label_idx]]]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=3, # 3가지로 분류
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


# sentences : 예측하고자 하는 텍스트 데이터 리스트
def getLabelValue(sentences, tok, max_len, batch_size, device, model):
    textList = [] # 텍스트 데이터를 담을 리스트
    labelList = [] # 라벨을 담을 리스트
    for s in sentences: # 모든 문장
        textList.append([s,5]) # [문장, 임의의 양의 정수값] 설정
    
    # print(textList)
    pdData = pd.DataFrame(textList, columns=[['text', 'label']])
    pdData = pdData.values
    test_set = BERTDataset(pdData, 0, 1, tok, max_len, True, False)
    test_input = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=5)

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_input)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        out = model(token_ids, valid_length, segment_ids)
        #print(out)
        for s in out:
            if (s[0]>=s[1])&(s[0]>=s[2]):
                value = -1
            elif (s[1]>s[0])&(s[1]>s[2]):
                value = 0
            else:
                value = 1
            labelList.append(value)
    
    result = pd.DataFrame({'text':sentences,'label':labelList})
    
    return result

# deviceName는 'cpu' or 'cuda:0'
def get_output(deviceName):
    # 파라미터1 데이터셋 불러오기
    data = pd.read_csv(f'./data/input/{target}_input.csv',encoding='utf-8-sig')
    word = data['word']
    input_data = list(data['text'])
    
    
    # 파라미터2 tok 설정
    bertmodel, vocab = get_pytorch_kobert_model()
    ## 기본 Bert tokenizer 사용
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    
    
    # 파라미터6 모델 불러오기
    ## 모델 클래스는 어딘가에 반드시 선언되어 있어야 한다.
    model = torch.load(f"./model/model_{deviceName}.pt")
    model.eval()
    
    #jupter notebook에서는 사용할 수 없음
    #parser = argparse.ArgumentParser(description='Process some integers.')

    #parser.add_argument("--tok", type=nlp.data.transforms.BERTSPTokenizer, default=tok)
    #parser.add_argument("--max_len", type=int, default=200) # 해당 길이를 초과하는 단어에 대해선 bert가 학습하지 않음
    #parser.add_argument("--batch_size", type=int, default=16)
    #parser.add_argument("--device", type=torch.device, default=device)
    #parser.add_argument("--checkpoint_path", type=str, default="./")
    #arges = parser.parse_args()
    
    args = easydict.EasyDict({
        "sentences" : input_data,
        "tok" : tok,
        "max_len" : 200,
        "batch_size" : 16,
        "device" : torch.device(deviceName),
        "model" : model
    })

    # get output data
    output = getLabelValue(args.sentences, args.tok, args.max_len, args.batch_size, args.device, args.model)
    output = pd.concat([data['date'],data['day'],word,output],axis=1)
    output.to_csv(f'./data/output/{target}_output.csv',encoding='utf-8-sig',index=False)
    
def get_result():
    output = pd.read_csv(f'./data/output/{target}_output.csv',encoding='utf-8-sig')
    dic = {'date' : target,'days': target_day,'word':[], 'score':[], 'expect':[],'recommend':[]}
    for company in output.word.unique():
        score = output[output.word==company].label.sum()
        if score>0:
            expect = 'up'
            recommend = 'buy'
        elif score==0:
            expect = 'stay'
            recommend =  'stay'
        elif score<0:
            expect = 'down'
            recommend = 'sell'
        dic['word'].append(company)
        dic['score'].append(score)
        dic['expect'].append(expect)
        dic['recommend'].append(recommend)
    result = pd.DataFrame(dic)
    result = result.sort_values('score',ascending=False)
    result.to_csv(f'./data/result/{target}_result.csv',encoding='utf-8-sig',index=False)
