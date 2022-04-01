from pycode.settingDatetime import *
import firebase_admin
import pandas as pd
import random

from firebase_admin import credentials
from firebase_admin import db


#Firebase database 인증 및 앱 초기화
cred = credentials.Certificate('kb-app-23eb5-firebase-adminsdk-72lxa-1fddf52ea4.json')
try:
    firebase_admin.initialize_app(cred,{
        'databaseURL' : 'https://kb-app-23eb5-default-rtdb.firebaseio.com/'
    })
except:
    pass

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

def overall_update():
    up_count = len(final[final.expect=='up'])
    stay_count = len(final[final.expect=='stay'])
    down_count = len(final[final.expect=='down'])

    if up_count>=6:
        maxim_box = [
            '매우 맑음, 3000만큼 투자해',
            '매우 맑음, 응 주가 계속 올라봐 투자하면 그만이야~',
            '매우 맑음, 주가 화성갈끄니까'
        ]
    elif up_count>down_count:
        maxim_box = [
            '맑음, 회사 갈 땐 가더라도 주식투자 정도는 괜찮잖아',
            '맑음, 거 투자하기 딱 좋은 날씨네',
            '맑음, 너는 다 투자계획이 있구나'
        ]
    elif down_count>=6:
        maxim_box = [
            '매우 흐림, 네가 선택한 주식이다. 악으로 깡으로 버텨라',
            '매우 흐림, 주가 오른다고 하니까 진짠줄 알더라',
            '매우 흐림, 꼭 그렇게...다 가져가야만 했냐!'
        ]
    elif down_count>up_count:
        maxim_box = [
            '흐림, 호황이 계속되면 그게 권리인 줄 알아',
            '흐림, 비와 주식의 공통점을 아시나요? 그건 바로 갑자기 떨어진다는 거...'
            '흐림, 저쪽 주가가 떨어졌다고 해서 가봤죠. 그런데 떨어진건 제 주가였어요'
        ]
    else:
        maxim_box = [
            '보통, 증시가 좋을수도 있고 안 좋을 수도 있습니다',
            '보통, 오늘 주가는 어떨지...몰?루'
            '보통, 살려는 드릴게'
        ]

    maxim = random.choice(maxim_box)
    overallUpdate = {target+'-'+target_day: maxim}

    ref = db.reference()
    overall_ref = ref.child('overall')
    overall_ref.update(overallUpdate)
    
def loadUserToto():
    ref = db.reference().child(f'UserToTo/{target}-{target_day}')
    usertoto = pd.DataFrame(columns = ['date','day','id','user_expect','user_answer'])
    user_id = [] 
    user_choice = []

    for key, val in ref.get().items():
        user_id.append(key)
        user_choice.append(val)

    for i in range(len(user_id)):
        user_answer_list = []
        user_info = {'date': target,
           'day': target_day,
            'id': user_id[i],
           'user_expect': user_choice[i][1:]}
    
        for j in range(len(final)):
            if user_choice[i][1:len(final.change)+1][j]== final.change[j]:
                user_answer_list.append('⭕')
            elif user_choice[i][1:len(final.change)+1][j] != final.change[j]:
                user_answer_list.append('❌')
            else:
                user_answer_list.append('-')
    
        user_info['user_answer'] = user_answer_list
        usertoto = usertoto.append(user_info,ignore_index=True)

    usertoto.to_csv(f'./data/users/{target}_users.csv', encoding = 'utf-8-sig', index = False)
    final.to_csv(f'./data/final/{target}_final.csv', encoding = 'utf-8-sig', index = False)
    display(usertoto)    
    
    

def checked_service_update():
    service ={}
    user = {}
    for i in range(len(final)):
        service[i+1] = final.service[i]
    checkedServiceUpdate = {target+'-'+target_day:{'service': service,'user': user}}

    ref = db.reference()
    checked_ref = ref.child('checked')
    checked_ref.update(checkedServiceUpdate)

    
# 4:00 pm에 진행됨
def checked_user_update():
    ref = db.reference().child(f'UserToTo/{target}-{target_day}')
    usertoto = pd.DataFrame(columns = ['date','day','id','user_expect','user_answer'])
    user_id = [] 
    user_choice = []

    for key, val in ref.get().items():
        user_id.append(key)
        user_choice.append(val)

    for i in range(len(user_id)):
        user_answer_list = []
        user_info = {'date': target,
           'day': target_day,
            'id': user_id[i],
           'user_expect': user_choice[i][1:]}
    
        for j in range(len(final)):
            if user_choice[i][1:len(final.change)+1][j]== final.change[j]:
                user_answer_list.append('⭕')
            elif user_choice[i][1:len(final.change)+1][j] != final.change[j]:
                user_answer_list.append('❌')
            else:
                user_answer_list.append('-')
    
        user_info['user_answer'] = user_answer_list
        usertoto = usertoto.append(user_info,ignore_index=True)
    
    service = {}
    for i in range(len(final)):
        service[i+1] = final.service[i]
    #checkedServiceUpdate = {target+'-'+target_day:{'service': service,'user': user}}
    
    #display(pd.DataFrame(usertoto.user_expect))
    
    user_id = {}
    for i in range(len(usertoto.id)):
        user = {}
        for j in range(len(final.change)):
            if final.expect[j] == usertoto.user_expect[i][j]:
                user[j+1] = '⭕'
            else:
                user[j+1] = '❌'
        user_id[usertoto.id[i]] = user
                
    checkedUserUpdate = {target+'-'+target_day:{'service': service, 'user': user_id}}
    
    ref = db.reference()
    checked_ref = ref.child('checked')
    checked_ref.update(checkedUserUpdate)
    

    
    
    
    
    

