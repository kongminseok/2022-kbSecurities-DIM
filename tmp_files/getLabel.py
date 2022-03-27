from train import *
from train import tok,device

# sentences : 예측하고자 하는 텍스트 데이터 리스트
def getLabelValue(sentences, tok, max_len, batch_size, device, model):
    textList = [] # 텍스트 데이터를 담을 리스트
    labelList = [] # 라벨을 담을 리스트
    for s in sentences: # 모든 문장
        textList.append([s,5]) # [문장, 임의의 양의 정수값] 설정
    
    print(textList)
    pdData = pd.DataFrame(textList, columns=[['sentence', 'label']])
    pdData = pdData.values
    test_set = BERTDataset(pdData, 0, 1, tok, max_len, True, False)
    test_input = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=5)

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_input)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        out = model(token_ids, valid_length, segment_ids)
        print(out)

    for s in out:
        if (s[0]>=s[1])&(s[0]>=s[2]):
            value = 0
        elif (s[1]>s[0])&(s[1]>s[2]):
            value = 1
        else:
            value = 2
        labelList.append(value)
    
    result = pd.DataFrame({'title':sentences,'label':labelList})
    
    return result


if __name__ == "__main__":
    # 파라미터 설정
    # train.py에서 설정한 tok, max_len, batch_size, device를 그대로 입력
    parser.add_argument("--tok", type=gluonnlp.data.transforms.BERTSPTokenizer, default=tok)
    parser.add_argument("--max_len", type=int, default=200) # 해당 길이를 초과하는 단어에 대해선 bert가 학습하지 않음 default=64
    parser.add_argument("--batch_size", type=int, default=64) # default=64
    parser.add_argument("--device", type=torch.device, default=device)

    arges = parser.parse_args()
    
    model = torch.load(os.path.join(args.checkpoint_path,"model_{?????}.pth")
    model.eval()

    # 데이터셋 불러오기
    data = pd.read_csv('./????',encoding='utf-8-sig')

    # start test
    result = test(data, args, model)
    result.to_csv('./result.csv',encoding='utf-8-sig',index=False)
    display(result)
    
                       
