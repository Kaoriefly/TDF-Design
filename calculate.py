import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from settings import filename, filepath

'''
根据人力资本理论和投资组合理论, 在考虑风险厌恶随年龄变化的情况下, 计算并绘制权益类资产的下滑曲线;
根据风险平价理论, 分股类和债类分别计算各年份各标的基金权重
输出result.xlsx:
1、A-H列包含各年份市场组合的年收益率、年波动率、风险偏好、R*、权益类占比、股类基金权重、债类基金权重
2、I列开始为各年份各标的基金权重, 依次为股类基金、分割列(-1列)、债类基金
'''


# 根据市场组合理论计算权益类资产下滑曲线
def cal_portfolio(df_predict, ratio, rf=0.025, xm=0.23 ,theta=0.3):
    '''
    rf: 无风险利率
    xm: 市场组合中的股票占比
    theta: 人力资本中的股类性质占比
    '''
    # 计算月收益率
    result = []  # 存储结果
    r_month_list = np.zeros((df_predict.shape[0], df_predict.shape[1]-1))
    for i in range(df_predict.shape[1]-1):
        r_month = df_predict[f'{i+1}'] / df_predict[f'{i}']  
        r_month_list[:,i] = r_month
    r_month_list = (r_month_list-1)*100  # 将收益率转换为百分比
    # 计算市场组合的年收益率、年波动率等
    ratio = ratio.values  # 将df转换为数组 
    for i in range(df_predict.shape[1]//12):
        r_year = df_predict[f'{12*(i+1)}'] / df_predict[f'{12*i}'] - 1  
        r = (r_year*ratio).sum()  ## 计算市场组合的年收益率 
        r_month_year = r_month_list[:, 12*i:12*(i+1)]  
        r_cov = np.cov(r_month_year)  
        sigma = np.sqrt(np.dot(np.dot(ratio.T, r_cov), ratio))  ## 计算市场组合年波动率
        a = (40 + i)/5  ## 计算不同年龄风险偏好（假设购买时投资者为40岁
        r_star = (r - rf) / (a*(sigma**2))  ## 计算投资组合中市场组合占比
        ht = -0.02*(i+40) + 1.4  # 计算不同年龄时人力资本占比（假设购买时投资者为40岁
        if r_star <= 0:
            alpha = 0
        elif r_star >= 1:
            alpha = xm
        else:
            alpha = (xm*r_star - theta*ht) / (1-ht)  ## 计算权益类资产占比
        # alpha = (xm*r_star - theta*ht) / (1-ht)  ## 计算权益类资产占比
        result.append([2025+i, r, sigma, a, r_star, alpha])
    
    cols = ['年份', '市场组合年收益率', '市场组合年波动率', '风险偏好', 'R*', '权益类占比']
    df_result = pd.DataFrame(result, columns=cols)
    return df_result


# 风险平价模型的优化函数
def risk_parity_objective(weights, cov_matrix):
    sigma = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # 计算投资组合的风险
    sigma_mar = np.dot(cov_matrix, weights) / sigma  # 计算各资产的边际风险贡献
    risk = weights * sigma_mar  # 计算各资产的真实风险贡献
    target = np.ones_like(weights) / len(weights)
    return np.sum((risk - target) ** 3)


# 计算风险平价权重
def risk_parity(df_history):
    returns= df_history.values
    volatility = np.std(returns, axis=1)
    cov_matrix = np.cov(returns)
    num = returns.shape[0]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num))
    initial_weights = np.ones(num) / num
    result = minimize(risk_parity_objective, initial_weights, args=(cov_matrix,),method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


# 计算各年份各标的基金权重
def subfund_weight(stock_w, bond_w, equity_ratio):
    equity_ratio = np.clip(equity_ratio, 0, 1).reshape(-1, 1)
    nonequity_ratio = 1 - equity_ratio.reshape(-1, 1)
    stock_weight = np.dot(equity_ratio, stock_w.reshape(1, -1))  # 各年份（行）各股票类基金（列）权重
    bond_weight = np.dot(nonequity_ratio, bond_w.reshape(1, -1))  # 各年份（行）各债券类基金（列）权重
    # 将股票类和债券类基金权重合并输出，中间加一列"-1"用以区分
    df_weight = pd.DataFrame(np.hstack((stock_weight, -np.ones((equity_ratio.shape[0],1)), bond_weight)))
    return df_weight


# FOF基金评估
def evaluate_fof(df_weight, df_history):
    df_weight = df_weight.drop(13, axis=1)  # 去掉"-1"列
    r_fof = np.dot(df_weight.values, df_history['年收益率'].values)
    sigma_fof = np.dot(df_weight.values, df_history['年波动率'].values)
    df_fof = pd.DataFrame({'投资组合年收益率': r_fof, '投资组合年波动率': sigma_fof})
    return r_fof, sigma_fof, df_fof


# 绘制权益类资产下滑曲线
def plot_curve(data):
    plt.figure(figsize=(10, 6))
    # data = data[0:10]
    x = np.arange(2025, 2050, 1)
    x_mod = np.linspace(min(x), max(x), 100)
    # 使用三次样条插值平滑曲线
    cs = CubicSpline(x, data)
    y_smooth = cs(x_mod)
    # 使用多项式拟合绘制趋势线
    coefficients = np.polyfit(x, data, 3)  # 选择拟合的多项式的阶数
    polynomial = np.poly1d(coefficients)   
    y_fit = polynomial(x_mod)
    plt.plot(x_mod, y_smooth, color='blue', linewidth=2, label='equity proportion') 
    plt.plot(x_mod, y_fit, color='green', linewidth=2, label='trend line')
    plt.scatter(x, data, color='red', s=30, label='point')
    plt.title('Equity Glide Curve')
    plt.xlabel('year')
    plt.ylabel('proportion')
    plt.xlim(min(x), max(x)) 
    plt.grid(True)
    plt.legend()
    plt.show()
   

if __name__ == '__main__':
    # 读取数据
    df_history = pd.read_csv(filepath[0])
    ratio = df_history['占比']
    df_stock = pd.read_csv(filepath[1])  
    df_bond = pd.read_csv(filepath[2])
    df_predict = pd.read_csv(filepath[3])  
    # 计算数据
    df_portfolio = cal_portfolio(df_predict, ratio, rf=0.025, xm=0.3 ,theta=0.2) 
    stock_w = risk_parity(df_stock)
    bond_w = risk_parity(df_bond)
    df_weight = subfund_weight(stock_w, bond_w, df_portfolio['权益类占比'].values)
    r_fof, sigma_fof, df_fof = evaluate_fof(df_weight, df_history)
    # 输出数据
    df_stock_w = pd.DataFrame(stock_w, columns=['stock_w'])
    df_bond_w = pd.DataFrame(bond_w, columns=['bond_w'])
    df_result = pd.concat([df_portfolio, df_stock_w, df_bond_w, df_weight, df_fof], axis=1)
    df_print = pd.concat([df_portfolio, df_fof], axis=1)
    df_result.to_excel(filename[4], index=False)
    # 打印数据
    print(df_print.to_string(index=False),'\n',f'股类权重{stock_w}','\n',f'债类权重{bond_w}')
    # plot_curve(df_portfolio['R*'])
    plot_curve(df_portfolio['权益类占比'])
    # plot_curve(r_fof)
    # plot_curve(sigma_fof)


