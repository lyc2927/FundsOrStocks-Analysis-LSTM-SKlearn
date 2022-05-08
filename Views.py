
'''
此处为streamlit的ＷＥＢ显示与交互模块
'''


import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


import plotly.graph_objects as go

# clicked = st.button('Click me')  # 按钮
# st.radio('pick one',['单选一','单选二','单选三'])
# st.text_input('请输入基金代码：')   # 文本输入

# 设置网页名称
st.set_page_config(page_title='基金分析页--RR的分享网页')

# st.table(都用）不会拖尾？？？？？？

# 一年收益率排行前十名
def button_1(  ):
    st.subheader('基金排名结果')
    df = pd.read_excel('data/funs_1year_top10.xlsx')
    for i in range(10):
        df.iloc[i,2] = df.iloc[i,2][1:7]
        #df.iloc[i, 5] = df.iloc[i,5]*100
    st.table(df[['基金代码','基金简称','日期','近1年']])
    st.text('注：近1年指收益率倍数。')

def button_2( fund_code ):
    #fund_code = st.text_input('请输入基金代码：')
    #st.write('你输入的基金代码为：'+ fund_code )


    if not fund_code:
        #st.write('NO fund code')
        pass
    else:
        st.write(fund_code)

        import requests
        url = 'https://fund.10jqka.com.cn/data/client/myfund/' + fund_code
        try:
            res = requests.get(url).json()
            # 返回列表包含：代码，名称，净值，成立日期，基金类型
            st.text(res['data'][0]['code'] + res['data'][0]['name'] + res['data'][0]['net'] \
                    + res['data'][0]['clrq'] + res['data'][0]['fundtype'])
        except:
            st.write('未获得该基金数据。')

        col1,col2 = st.columns(2)
        col1.write('124')
        col2.write('125')



def button_3():
    st.button('预测基金（代码：360016）7天的走势')
    st.text('训练数据为519918, 001054, 000556, 001126四个基金的走势，预测的模型为LSTM模型。')

    #　转入模型，导出两组数字y_validation, predict_validation
    import lstm_model
    res = lstm_model.main()
    #print(res)
    y_validation = res[0]
    predict_validation = res[1]
    # 将tuple的两个组数据，转为两个列表
    #print(type(res),res)
    list1 = []
    for i in res[0]:
        list1.append(i[0])
    list2 = []
    for i in res[1]:
        list2.append(i[0])

    #　转为dataframe并由st的图表显示出来
    chart_data = pd.DataFrame([list1,list2])
    chart_data = pd.DataFrame(chart_data.values.T, columns=['基金每日涨幅','预测每日涨幅'])
    #print( chart_data )
    st.line_chart(chart_data)



def button_4( ):
    st.text('button_4')

def Layouts():
    st.sidebar.write('基金分析系统')

    # 用with进行Form的声明--------------------------------
    #with st.sidebar.form(key='my_form'):
        #uname_input = st.sidebar.text_input(label='用户名：')
        #text_input = st.sidebar.text_input(label='请输入基金代码：')
        #submit_button = st.sidebar.form_submit_button(label='提交')
    #try:
    #st.sidebar.write('你输入的基金代码为：' + text_input + uname_imput)
    #except:
    #    pass
    #uname = st.sidebar.text_input('请输入名字：')



    st.sidebar.button('  基金排名  ', on_click=button_1)

    fund_code = st.sidebar.text_input('请输入基金代码：')
    st.sidebar.button('基金数据查询', on_click=button_2(fund_code))

    st.sidebar.button('  基金预测  ', on_click=button_3)

    #st.sidebar.button('  个人中心  ', on_click=button_4)


    #fund_code = st.text_input('请输入基金代码：')
    #text_input.to_excel('342342.xlsx')
    #uname = st.text_input('请输入名字：')
    #fund_code = st.text_input('请输入基金代码：')
    #st.write(uname,fund_code
    #t.sidebar.write('你输入的基金代码为：'+ fund_code )

if __name__ == "__main__":

    Layouts()
    pass





