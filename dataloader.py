import pandas as pd
import numpy as np
from settings import filename, filepath

'''
用于为后续步骤准备数据，包括：
1. 从excel中读取原始数据存储在df_data中
2. 输出后续mc模拟需要的净值数据、占比、月收益率、月波动率, 存储在mc_imput.csv中
3. 输出后续风险平价需要的历史数据, 按照股基和债基, 分别存储在history_stock.csv和history_bond.csv中
'''


# 净值转换(适用于货币   市场基金)
def net_transform(df):
    df[df.columns[3]] = (df[df.columns[3]]+10000) / 10000
    for i in range(59):
        df[df.columns[4+i]] =(df[df.columns[3+i]]*(df[df.columns[4+i]]+10000)) / 10000
    return df


# 从excel中读取原始数据
def get_initial(path):
    cols=['年化收益率','年化波动率',' 2019-01-31 ',' 2019-02-28 ',' 2019-03-31 ',' 2019-04-30 ',' 2019-05-31 ',' 2019-06-30 ',' 2019-07-31 ',' 2019-08-31 ',' 2019-09-30 ',' 2019-10-31 ',' 2019-11-30 ',' 2019-12-31 ', ' 2020-01-31 ',' 2020-02-28 ',' 2020-03-31 ',' 2020-04-30 ',' 2020-05-31 ',' 2020-06-30 ',' 2020-07-31 ',' 2020-08-31 ',' 2020-09-30 ',' 2020-10-31 ',' 2020-11-30 ',' 2020-12-31 ',
        ' 2021-01-31 ',' 2021-02-28 ',' 2021-03-31 ',' 2021-04-30 ',' 2021-05-31 ',' 2021-06-30 ',' 2021-07-31 ',' 2021-08-31 ',' 2021-09-30 ',' 2021-10-31 ',' 2021-11-30 ',' 2021-12-31 ', ' 2022-01-31 ',' 2022-02-28 ',' 2022-03-31 ',' 2022-04-30 ',' 2022-05-31 ',' 2022-06-30',' 2022-07-31 ',' 2022-08-31 ',' 2022-09-30 ',' 2022-10-31 ',' 2022-11-30 ',' 2022-12-31 ',
        ' 2023-01-31 ',' 2023-02-28 ',' 2023-03-31 ',' 2023-04-30 ',' 2023-05-31 ',' 2023-06-30 ',' 2023-07-31 ',' 2023-08-31 ',' 2023-09-30 ',' 2023-10-31 ',' 2023-11-30 ',' 2023-12-31 ','基金规模2023-12-31']
    df1 = pd.read_excel(path, sheet_name='被动管理型股票基金', nrows=5, usecols=cols)
    df2 = pd.read_excel(path, sheet_name='主动管理型股票基金', nrows=5, usecols=cols)   
    df3 = pd.read_excel(path, sheet_name='商品型基金', nrows=1, usecols=cols)
    df4 = pd.read_excel(path, sheet_name='QDII基金', nrows=2, usecols=cols)
    df5 = pd.read_excel(path, sheet_name='债券型基金', nrows=10, usecols=cols)
    df6 = pd.read_excel(path, sheet_name='货币市场基金', nrows=2, usecols=cols)
    df6 = net_transform(df6)
    df_initial = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)  # 按行拼接数据
    # 仅读取历史月末净值
    cols1 = ['年化收益率','年化波动率','基金规模2023-12-31']
    df_his_stock = pd.concat([df1, df2, df3, df4], ignore_index=True)
    df_his_bond = pd.concat([df5], ignore_index=True)
    df_his_stock = df_his_stock.drop(columns=cols1,inplace=False)  
    df_his_bond = df_his_bond.drop(columns=cols1,inplace=False)
    return df_initial, df_his_stock, df_his_bond


# 根据原始数据计算历史数据(净值数据、占比、月收益率、月波动率)
def get_history(df_initial):
    df_history = pd.DataFrame()
    df_history['净值数据'] = df_initial[' 2023-12-31 ']
    df_history['年收益率'] = df_initial['年化收益率']
    df_history['年波动率'] = df_initial['年化波动率']
    sum = df_initial['基金规模2023-12-31'].sum()
    df_history['占比'] = df_initial['基金规模2023-12-31'] / sum
    df_temp = pd.DataFrame()  # 用于存储每支基金的对数月收益率
    for i in range(29):
        temp1 = df_initial[df_initial.columns[3+i]]  # 月末净值数据从第4列(索引3)开始
        temp2 = df_initial[df_initial.columns[4+i]]
        df_temp[f'{2019+(i+1)//12}年{(i+1)%12+1}月对数增长率'] = np.log(temp2 / temp1)  # 对数收益增长率
    df_history['月收益率'] = df_temp.mean(axis=1)  # 按行求每支基金的月收益率均值
    df_history['月收益率'] = ((df_initial['年化收益率']/100+1)**(1/12)) - 1
    df_history['月波动率'] = np.sqrt(df_temp.var(axis=1))  # 按行求每支基金的月波动率标准差
    return df_history

if __name__ == '__main__':
    
    path = r'C:\Users\24592\Desktop\TDF设计\基金池0730.xlsx'
    df_initial, df_his_stock, df_his_bond = get_initial(path)
    df_history = get_history(df_initial)

    df_history.to_csv(filepath[0], index=False)
    print(f' {filename[0]} 已输出, 尺寸为:{df_initial.shape}')
    df_his_stock.to_csv(filepath[1], index=False)
    print(f' {filename[1]} 已输出, 尺寸为:{df_his_stock.shape}')
    df_his_bond.to_csv(filepath[2], index=False)
    print(f' {filename[2]} 已输出, 尺寸为:{df_his_bond.shape}')










