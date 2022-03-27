import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from model import BERTClassifier
from dataset import BERTDataset
from evaluate import calc_accuracy

import argparse
from os import TMP_MAX
import os

def train(args, model, train_dataloader, e  ): #e = current epoch
    train_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader),
                                                                        total=len(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % args.log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
                                                                     train_acc / (batch_id + 1)))
    print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))


def validate(args, model,test_dataloader, epoch):
    model.eval()
    test_acc = 0.0
    for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(test_dataloader),
                                                                        total=len(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(epoch + 1, test_acc / (batch_id + 1)))
    torch.save(model, os.path.join(args.checkpoint_path,"model_{epoch}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    # 파라미터 설정
    parser.add_argument("--max_len", type=int, default=200) # 해당 길이를 초과하는 단어에 대해선 bert가 학습하지 않음 default=64
    parser.add_argument("--batch_size", type=int, default=64) # default=64
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--checkpoint_path", type=str, default="/content/drive/MyDrive/colab")

    args = parser.parse_args()

    print(torch.cuda.is_available())
    ## CPU
    if torch.cuda.is_available()==False:
        device = torch.device("cpu")
    ## GPU
    else:
        device = torch.device("cuda:0")

    bertmodel, vocab = get_pytorch_kobert_model()

    # 학습용 데이터셋 불러오기

    use_data = pd.read_csv('./fineTuningData2.csv')
    use_data = use_data
    use_data.tail()


    # 라벨링
    encoder = LabelEncoder()
    encoder.fit(use_data['label'])
    use_data['label'] = encoder.transform(use_data['label'])

    mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))

    # Train / Test set 분리
    train, test = train_test_split(use_data, test_size=0.2, random_state=42)
    print("train shape is:", len(train))
    print("test shape is:", len(test))

    # 기본 Bert tokenizer 사용
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    train = train.values
    test = test.values

    data_train = BERTDataset(train, 0, 1, tok, args.max_len, True, False)
    data_test = BERTDataset(test, 0, 1, tok, args.max_len, True, False)

    # pytorch용 DataLoader 사용
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, num_workers=5)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, num_workers=5)

    #load model
    model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # 옵티마이저 선언
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    t_total = len(train_dataloader) * args.num_epochs
    warmup_step = int(t_total * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    # start train
    for e in range(args.num_epochs):
        train(args, model, train_dataloader, e)
        validate(args, model, test_dataloader, e)


