import time
start = time.time()

from pycode.crawling import *
from pycode.input import *
from pycode.result import *
from pycode.final import *
from pycode.firebase import *

#12:00am에 실행
def runStockLuck():
    get_news()
    extract_10keywords()
    get_output('cuda:0') #인자는 'cpu' or 'cuda:0'
    get_result()
    get_answer()
    todaystock_update()
    ranking_update()
    checked_service_update()
    
    
if __name__ == '__main__' :
    runStockLuck() #12:00am에 실행
    end = time.time()
    print('===== Running Time : ', (end - start) / 60, ' minutes =====')
