from pycode.settingDatetime import *
import FinanceDataReader as fdr
import pandas as pd


stocklist = fdr.StockListing('KRX') #나중에 stocklist 변경하기
stocks = pd.concat([stocklist.Name, stocklist.Symbol], axis = 1)

def get_answer():
    result = pd.read_csv(f'./data/result/{target}_result.csv',encoding='utf-8-sig')
    #display(result)
    companylist = list(result.word)
    #companylist = ['삼성전자','NAVER','카카오','LG에너지솔루션','SK하이닉스','삼성바이오로직스','삼성SDI', '현대차', 'LG화학','기아']
    answer = pd.DataFrame({'date':target, 'day':target_day,'word':companylist,'score':result.score,'expect':result.expect})
    changelst = []
    for element in companylist:
        tic = "".join(list(stocks[stocks.Name==element].Symbol.values))
        df = fdr.DataReader(tic,target_default,target_default)
        #print(df.Change.values)
        if df.Change.values>0:
            changelst.append('up')
        elif df.Change.values==0:
            changelst.append('stay')
        elif df.Change.values.any()==False:
            changelst.append('-')
        else:
            changelst.append('down')

    answer['change'] = changelst
    answer['service'] = ''
    
    for i in range(len(answer)):
        if answer.expect[i]==answer.change[i]:
            answer.service[i]='⭕'
        else:
            answer.service[i]='❌'
    answer.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)
   
    answer.to_csv(f'./data/final/{target}_final.csv',encoding='utf-8-sig',index=False)
    display(answer)