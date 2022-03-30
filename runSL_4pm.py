import time
start = time.time()

from pycode.firebase import *


if __name__ == '__main__' :
    checked_user_update() #4:00pm에 실행
    end = time.time()
    print('===== Running Time : ', (end - start) / 60, ' minutes =====')

