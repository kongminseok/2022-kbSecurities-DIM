from pycode.settingDatetime import *
import firebase_admin
import pandas as pd

from firebase_admin import credentials
from firebase_admin import db


#Firebase database 인증 및 앱 초기화
cred = credentials.Certificate('kb-app-23eb5-firebase-adminsdk-72lxa-1fddf52ea4.json')
firebase_admin.initialize_app(cred,{
    'databaseURL' : 'https://kb-app-23eb5-default-rtdb.firebaseio.com/'
})

#필요 데이터 로드
final = pd.read_csv(f'./data/final/{target}_final.csv',encoding='utf-8-sig')

def todaystock_update():
    max_score = final[final.score==final.score.max()]
    min_score = final[final.score==final.score.min()]
    if max_score.score.values[0]>0 : 
        goodstock = max_score.word.values[0] 
    else:
        goodstock = '아쉽지만 찾을 수 없습니다'
    if min_score.score.values[0]<0:
        badstock = min_score.word.values[0]
    else:
        badstock = '다행히 없습니다'
    todaystockUpdate = {target+'-'+target_day:{'good':goodstock,'bad':badstock}}

    ref = db.reference()
    todaystock_ref = ref.child('todaystock')
    todaystock_ref.update(todaystockUpdate)


def ranking_update():
    companyRank = {}
    for i in range(len(final.index)):
        companyRank[i+1] = final.word[i]
    rankUpdate = {target+'-'+target_day:companyRank}

    ref = db.reference()
    rank_ref = ref.child('ranking')
    rank_ref.update(rankUpdate)


    
service ={}
user = {}
def checked_service_update():
    for i in range(len(final)):
        service[i+1] = final.service[i]
    checkedServiceUpdate = {target+'-'+target_day:{'service': service,'user': user}}

    ref = db.reference()
    checked_ref = ref.child('checked')
    checked_ref.update(checkedServiceUpdate)
    
# 4:00 pm에 진행됨
def checked_user_update():
    for i in range(len(final)):
        if final.expect[i] == final.user[i]:
            user[i+1] = '⭕'
        else:
            user[i+1] = '❌'
    checkedUserUpdate = {target+'-'+target_day:{'service': service, 'user': user}}
    
    ref = db.reference()
    checked_ref = ref.child('checked')
    checked_ref.update(checkedUserUpdate)
    
def loadUserToto():
    ref = db.reference().child('UserToTo')
    usertoto = pd.DataFrame(columns = ['id','date','day','user_expect','user'])
    date_day = []
    user_choice = []
    for key, val in ref.get().items():
        date_day.append(key)
        user_choice.append(val)

    for i in range(len(date_day)):
        user_answer = []
        user_date, user_day = date_day[i].split('-')
        user_info = {'id': 45782626, # 임의의 숫자
                     'date': user_date,
                     'day': user_day,
                     'user_expect': user_choice[i][1:]}
    
        for j in range(len(final)):
            if user_choice[i][1:][j]== final.change[i]:
                user_answer.append('⭕')
                final.user[j] = '⭕'
            elif user_choice[i][1:][j] != final.change[i]:
                user_answer.append('❌')
                final.user[j] = '❌'
            else:
                user_answer.append('-')
    
        user_info['user'] = user_answer
        usertoto = usertoto.append(user_info,ignore_index=True)

    usertoto.to_csv(f'./data/users/{user_date}_user.csv', encoding = 'utf-8-sig', index = False)
    final.to_csv(f'./data/final/{target}_final.csv', encoding = 'utf-8-sig', index = False)
    display(final)
    display(usertoto)
    
    
    
    
    

