import os

'''
设置共享模块,用于集中管理参数
'''


directory = 'data'  # 设置输出文件夹
if not os.path.exists(directory):
    os.makedirs(directory)
filename = ['mc_input.csv', 'history_stock.csv', 'history_bond.csv', 'mc_output.csv', 'result.xlsx']  # 文件名
filepath = [os.path.join(directory, name) for name in filename]  # 文件路径