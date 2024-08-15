import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from settings import filename, filepath

'''
使用 Monte Carlo 模拟基金未来净值变化,  并提供工具用于验证模拟的合理性
1、对多次模拟的结果取均值得到平均路径, 存储在mc_output.csv中
2、提供 plot_path 函数绘制模拟结果, 计算增值倍数
'''


# 使用几何布朗模型进行 mc 模拟
def gen_paths(S0,r,sigma,T,M,I):
    dt = T
    paths=np.zeros((M+1, I))
    paths[0]=S0
    for t in range(1,M+1):
        rand = np.random.standard_normal(I)  # 生成标准正态分布随机数
        paths[t]=paths[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*1.2*np.sqrt(dt)*rand)  # 几何布朗模型
    return paths


# 绘制指定数量基金的模拟路径并计算增长倍数
def plot_paths(predict, S0_list, r_list, num_fund=10, M=120):
    ref = np.power(np.exp(r_list), M)  # 参考值计算
    for i in range(num_fund):
        mean_path = predict[i]
        print(f'第{i+1}只增长倍数:{mean_path[M] / mean_path[0]}, 参考值:{ref[i]}')  # 打印增长倍数
        plt.figure(figsize=(10, 6))
        plt.plot(mean_path, color='red', linewidth=2, label='Mean Path')  # 绘制平均路径
        plt.title(f'Simulated Paths of fund {i+1}')
        plt.xlabel('Time Steps / mouth')
        plt.ylabel('fund net value')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':

    ## 设置mc模拟参数
    np.random.seed(10)  # 设置随机种子
    data = pd.read_csv(filepath[0])
    S0_list = data['净值数据']
    r_list = data['月收益率']
    sigma_list = data['月波动率']
    T, M, I = 1, 300, 100  # 设置时间步长、模拟次数、模拟路径数
    '''
    模拟未来25年共300个月的基金净值变化, 单只基金模拟1000次取均值作为预测值
    时间步长T根据使用的“收益率(波动率)时限”来设置：
    1、使用月收益率模拟月末净值, T=1
    2、使用年收益率模拟月末净值, T=1/12
    3、使用月收益率模拟年末净值, T=12
    '''

    num = data.shape[0]  # 待模拟基金数量
    predict = np.zeros((num, M+1))
    for i in range(num):
        S0 = S0_list[i]
        sigma = sigma_list[i]
        r = r_list[i]
        paths=gen_paths(S0, r, sigma, T, M, I)  # I条模拟路径
        mean_path = np.mean(paths, axis=1)  # I条模拟路径的均值
        predict[i] = mean_path
    # plot_paths(predict, S0_list, r_list, num_fund=25, M=300)
    df_predict = pd.DataFrame(predict)
    df_predict.to_csv(filepath[3], index=False)
    print(f' {filename[3]} 已输出, 尺寸为:{df_predict.shape}')


