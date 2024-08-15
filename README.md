
# Target-Date Fund design
Combined with human capital, Ibbotson's portfolio theory and risk parity, I designed an target-date fund (fund of fund) to help clients make assets increased to better cope with people's old age. The project mainly teach you how to plot the equity  glide curve and allocate the weights of the underlying funds. Tha data used in the project could be got from financial database, such as Wind.


本项目共包含4个python脚本,首次使用仅需在dataloader.py中修改源数据路径，然后依次运行predict.py和calculate.py即可输出结果

关于参数修改：项目使用settings.py统一控制参数（主要是输出文件的名称和路径）
关于结果输出：中间结果以csv文件格式保存在文件夹 '/data' 中，最终结果以 'result.xlsx' 保存在项目目录下

以下是每个脚本实现功能的详细介绍：

dataloader.py  
用于为后续步骤准备数据，包括
1. 从excel中读取原始数据存储在df_data中
2. 输出后续mc模拟需要的净值数据、占比、月收益率、月波动率, 存储在mc_imput.csv中
3. 输出后续风险平价需要的历史数据, 按照股基和债基, 分别存储在history_stock.csv和history_bond.csv中

predict.py  
使用 Monte Carlo 模拟基金未来净值变化,  并提供工具用于验证模拟的合理性：  
1. 对多次模拟的结果取均值得到平均路径, 存储在mc_output.csv中  
2. 提供 plot_path 函数绘制模拟结果, 计算增值倍数

calculate.py  
根据人力资本理论和投资组合理论, 在考虑风险厌恶随年龄变化的情况下, 计算并绘制权益类资产的下滑曲线;  
根据风险平价理论, 分股类和债类分别计算各年份各标的基金权重，输出result.xlsx:  
1. A-H列包含各年份市场组合的年收益率、年波动率、风险偏好、R*、权益类占比、股类基金权重、债类基金权重  
2. I列开始为各年份各标的基金权重, 依次为股类基金、分割列(-1列)、债类基金

if you need further help such as initial fund data or project report, please contact me by email:ziheng-l22@mails.tsinghua.edu.cn
