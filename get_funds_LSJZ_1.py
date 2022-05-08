'''
该模块获得基金的历史净值数据。
已测试可以获得。
'''

# 导入需要的模块
import time
import requests
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

#　获得爬取网站的url
def get_html(code, start_date, end_date, page=1, per=20):
    url = 'http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code={0}&page={1}\
        &sdate={2}&edate={3}&per={4}'.format(code, page, start_date, end_date, per)
    rsp = requests.get(url)
    html = rsp.text
    return html

# 获取基金的交易数据，返回DF数据
def get_fund(code, start_date, end_date, page=1, per=20):
    # 获取html
    html = get_html(code, start_date, end_date, page, per)
    soup = BeautifulSoup(html, 'html.parser')

    #print( soup )

    # 获取总页数
    pattern = re.compile('pages:(.*),')
    result = re.search(pattern, html).group(1)
    total_page = int(result)
    # 获取表头信息
    heads = []
    for head in soup.findAll("th"):
        heads.append(head.contents[0])

    # 数据存取列表
    records = []
    # 获取每一页的数据
    current_page = 1
    while current_page <= total_page:
        html = get_html(code, start_date, end_date, current_page, per)
        soup = BeautifulSoup(html, 'html.parser')
        print(soup)
        # 获取数据
        for row in soup.findAll("tbody")[0].findAll("tr"):
            row_records = []
            for record in row.findAll('td'):
                val = record.contents
                # 处理空值
                if val == []:
                    row_records.append(np.nan)
                else:
                    row_records.append(val[0])
            # 记录数据
            records.append(row_records)
        # 下一页
        current_page = current_page + 1

    # 将数据转换为Dataframe对象
    np_records = np.array(records)
    fund_df = pd.DataFrame()
    for col, col_name in enumerate(heads):
        fund_df[col_name] = np_records[:, col]
    # 按照日期排序
    fund_df['净值日期'] = pd.to_datetime(fund_df['净值日期'], format='%Y/%m/%d')
    fund_df = fund_df.sort_values(by='净值日期', axis=0, ascending=True).reset_index(drop=True)
    fund_df = fund_df.set_index('净值日期')
    # 数据类型处理
    fund_df['单位净值'] = fund_df['单位净值'].astype(float)
    fund_df['累计净值'] = fund_df['累计净值'].astype(float)
    fund_df['日增长率'] = fund_df['日增长率'].str.strip('%').astype(float)
    return fund_df


#　定义相关基金的代码
fundcode_train = ['519918', '001054', '000556', '001126']
fundcode_test = '360016'

# 获取训练数据
def get_training_data(fundcode_train):
    rows = []
    for i in fundcode_train:
        # 定义需要获取净值的基金代码，开始日期，结束日期，
        s_long = 200       # 最小超过多少个数据才计算
        fund_code = i[:6]
        start_date = '2018-03-01'
        end_date = '2022-05-01'
        try:
            fund_df = get_fund( fund_code, start_date=start_date, end_date=end_date )
        except:
            print(fund_code, '失败！！！！！！')
            continue
        print( fund_code, len(fund_df) )
        #print( fund_df )
        row = list( fund_df['累计净值'] )     # 日增长率转换为相邻交易日间的价格比值
        print(row[:20])

        # 将每一只股票收益率存入列表
        rows.append(row)

    #print( len(rows), rows )
    pd.DataFrame(rows).to_excel('data/funds_data_train.xlsx',index=False,header=None)


# 获取测试数据
def get_test_data(fundcode_test):

    # 定义需要获取净值的基金代码，开始日期，结束日期，
    s_long = 200  # 最小超过多少个数据才计算
    fund_code = fundcode_test
    start_date = '2018-03-01'
    end_date = '2022-05-01'
    try:
        fund_df = get_fund(fund_code, start_date=start_date, end_date=end_date)
    except:
        print(fund_code, '失败！！！！！！')

    print(fund_code, len(fund_df))
    # print( fund_df )
    row = [ list( fund_df['累计净值'] ) ]    # 日增长率转换为相邻交易日间的价格比值
    print(row[:20])

    # print( len(row), row )
    pd.DataFrame(row).to_excel('data/funds_data_test.xlsx', index=False, header=None)

if __name__ == '__main__':
    get_training_data(fundcode_train)
    get_test_data(fundcode_test)