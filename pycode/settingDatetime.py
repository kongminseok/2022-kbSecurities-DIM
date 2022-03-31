import datetime
from dateutil.relativedelta import relativedelta

# 형식 datetime.date(2022, 3, 28)
today_datetime = datetime.date.today()
yesterday_datetime = today_datetime - datetime.timedelta(days=1)
tomorrow_datetime = today_datetime + datetime.timedelta(days=1)

# 형식 '2022-03-28'
yesterday_default = str(yesterday_datetime)
today_default = str(today_datetime)
tomorrow_default = str(tomorrow_datetime)

# 형식 '20220328'
yesterday = yesterday_default.replace("-",'')
today = today_default.replace("-",'')
tomorrow = tomorrow_default.replace("-",'')

# 요일
days = ['월', '화', '수', '목', '금', '토', '일']

yesterday_index = (datetime.datetime.today() - datetime.timedelta(days=1)).weekday()
today_index = (datetime.datetime.today()).weekday()
tomorrow_index = (datetime.datetime.today() + datetime.timedelta(days=1)).weekday()

# 형식 '월'
yesterday_day = days[yesterday_index]
today_day = days[today_index]
tomorrow_day = days[tomorrow_index]

# 타겟 날짜 및 요일 설정
## 어제를 타겟 날짜 및 요일의 디폴트값으로 설정
target_datetime = yesterday_datetime
target_default = yesterday_default
target = yesterday
target_day = yesterday_day

# 크롤링에 쓰일 리스트
dates = [] # 공휴일이 끼어있을 경우 임의 지정

"""
공휴일이 없다는 가정하에,
크롤링하는 요일은 평일(월,화,수,목,금)
오전 12시가 되면 전날 기사를 크롤링하는 것으로 함.
주말이 끼어있을 경우, 월요일 오전 12시가 되면 전주 금,토,일 기사를 모두 크롤링하는 것으로 함.
"""

if target_day=='일':
    for i in range(0,3):
        dates.append(str(yesterday_datetime - datetime.timedelta(days=i)).replace("-",""))
    target_datetime = target_datetime - datetime.timedelta(days=2)
    target_default = str(target_datetime)
    target = dates[len(dates)-1]+'-'+dates[0]
    target_day = ",".join(days[4:])


elif target_day=='금' or target_day=='토':
    dates = []
    
else:
    dates= [yesterday]
    
    