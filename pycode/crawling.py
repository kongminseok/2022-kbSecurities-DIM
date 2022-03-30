from pycode.settingDatetime import *
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

import time
from tqdm import tqdm
import pandas as pd 
import warnings
import os
warnings.filterwarnings('ignore')

# create empty list
fail = []

def get_news() : 
    
    # create datafame
    df = pd.DataFrame(columns = ['제목','시간','언론사','본문'])
    
    # driver
    # path = ChromeDriverManager().install()
    driver = webdriver.Chrome(ChromeDriverManager().install())
    # driver.maximize_window() # 윈도우 창 최대화
    # driver.minimize_window() # 윈도우 창 최소화
    
    """
    # get url
    url = 'https://finance.naver.com/news/'
    driver.get(url)
    time.sleep(0.5)
    
    # click button(실시간 속보)
    btn = driver.find_element_by_css_selector('div > ul > li.frst > a')
    btn.click()
    
    # click button(날짜)
    elems = driver.find_elements_by_css_selector('div.pagenavi_day > a')
    for elem in elems:
        if elem.text == '02월08일(화)' : 
            btn = elem
    btn.click()
    """
    
    # set up page loop(페이지)
    for date in dates:
            print("====== DATE {0} CRAWLING START =====".format(date))
            # set up dates loop(날짜)
            for page in tqdm(range(1,100)):
                    try:
                            # next_path = '//*[@id="contentarea_left"]/table/tbody/tr/td/table/tbody/tr/td[{0}]/a'.format(page)
                            # driver.find_element_by_xpath(next_path).click()
                    
                            # get url
                            url = 'https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258&date={0}&page={1}'.format(date,page)
                            driver.get(url)
                            time.sleep(0.5)
                        
                            # click headline
                            elems = driver.find_elements_by_xpath('//dl/*[@class="articleSubject"]')
                            for i in range(len(elems)) :
                                    btns = driver.find_elements_by_xpath('//dl/*[@class="articleSubject"]/a')
                                    btns[i].click()
                                    time.sleep(1)
                            
                                    tmp_dic ={}
                            
                                    # title
                                    title_elem = driver.find_element_by_css_selector('div.article_info > h3')
                                    title = title_elem.text
                            #       print(title)
                                    tmp_dic['제목'] = title
                        
                                    # time
                                    time_elem = driver.find_element_by_css_selector('div.article_sponsor > span.article_date')
                                    time_txt = time_elem.text
                            #       print(time_txt)
                                    tmp_dic['시간'] = time_txt
                        
                                    # press
                                    press_elem = driver.find_element_by_css_selector('span.press > img')
                                    press = press_elem.get_attribute('title')
                            #       print(press)
                                    tmp_dic['언론사'] = press
                        
                                    # content
                                    content_elem = driver.find_element_by_css_selector('div.articleCont')
                                    content = content_elem.text.replace('\n', '')
                            #       print(content)
                                    tmp_dic['본문'] = content
                            
                                    df = df.append(tmp_dic, ignore_index = True)
                            
                                    time.sleep(1)
                                    driver.back()
                                    time.sleep(2)
                        
                    except:
                            print("====== PAGE {0} CRAWLING FAIL =====".format(page))
                            fail.append([date,page])
    
    
    for date, page in fail:
            url = 'https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258&date={0}&page={1}'.format(date, page)
            driver.get(url)
            time.sleep(0.5)
            
            # click headline
            elems = driver.find_elements_by_xpath('//dl/*[@class="articleSubject"]')
            for i in range(len(elems)) :
                btns = driver.find_elements_by_xpath('//dl/*[@class="articleSubject"]/a')
                btns[i].click()
                time.sleep(1)

                tmp_dic = {}

                # title
                title_elem = driver.find_element_by_css_selector('div.article_info > h3')
                title = title_elem.text
                # print(title)
                tmp_dic['제목'] = title

                # time
                time_elem = driver.find_element_by_css_selector('div.article_sponsor > span.article_date')
                time_txt = time_elem.text
                # print(time_txt)
                tmp_dic['시간'] = time_txt

                # press
                press_elem = driver.find_element_by_css_selector('span.press > img')
                press = press_elem.get_attribute('title')
                # print(press)
                tmp_dic['언론사'] = press

                # content
                content_elem = driver.find_element_by_css_selector('div.articleCont')
                content = content_elem.text.replace('\n', '')
                # print(content)
                tmp_dic['본문'] = content

                df = df.append(tmp_dic, ignore_index=True)

                time.sleep(1)
                driver.back()
                time.sleep(2)
            
    # close driver
    driver.close()
    
    df.to_csv("./data/crawling/{0}_naver_finance_news.csv".format(target), encoding = 'utf-8-sig', index=False)
    
    return df

