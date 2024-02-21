import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('ETTh1.csv')

# Z-score标准化
data.iloc[:, 1:] = (data.iloc[:, 1:] - data.iloc[:, 1:].mean()) / data.iloc[:, 1:].std()
# scaler = StandardScaler()
# data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:])
print(data)

# 构建模型
lookback_window = 336 #历史回看窗口长度
forecast_window = 24  #预测窗口长度
model = LinearRegression()

# 滚动训练与预测
# 从0开始，每次滚动24个数据点
predictions = []
losses = []  # 用于存储每个轮次的损失值
flag = 0
for i in range(0, len(data)-lookback_window-forecast_window+1, 24):
    # 1 从数据中提取训练集和测试集。
    # 2 用训练集训练模型。
    # 3 对测试集进行预测，并将预测结果添加到predictions列表中。
    # 4 将测试集数据添加到训练集中，以准备下一次循环的训练。
    train_data = data.iloc[i:i+lookback_window]
    test_data = data.iloc[i+lookback_window:i+lookback_window+forecast_window]

    X_train = train_data.iloc[:, 1:]
    y_train = train_data['OT']

    X_test = test_data.iloc[:, 1:]
    y_test = test_data['OT']

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions.extend(y_pred)

    # 计算每个轮次的损失值
    loss = mean_squared_error(y_test, y_pred)
    losses.append(loss)
    print(f"Loss for round {i}: {loss}")

    # 将预测值加入训练集，继续滚动训练
    train_data = train_data.append(test_data)
    data.iloc[i:i+lookback_window+forecast_window] = train_data

    flag = i

# 如果不是24的倍数
if flag<(len(data)-lookback_window-forecast_window):
    train_data = data.iloc[flag+forecast_window:len(data)-lookback_window]
    test_data = data.iloc[flag+lookback_window+forecast_window:len(data)] # 注意最后flag=17040，+336+24=17400，共17420个，正好差20个，把这段if注释掉开debug就知道20是什么意思了

    X_train = train_data.iloc[:, 1:]
    y_train = train_data['OT']

    X_test = test_data.iloc[:, 1:]
    y_test = test_data['OT']

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions.extend(y_pred)

    loss = mean_squared_error(y_test, y_pred)
    losses.append(loss)
    print(f"Loss for round last: {loss}")


print(data['OT'][lookback_window:])
print(predictions)

# 评估模型效果
mse = mean_squared_error(data['OT'][lookback_window:], predictions)
mae = mean_absolute_error(data['OT'][lookback_window:], predictions)
print(f'MSE: {mse}, MAE: {mae}')

# 绘制预测结果展示图
plt.figure(figsize=(12, 6))
plt.plot(data['OT'][lookback_window:], label='True Value', color='blue')
plt.plot(predictions, label='Predicted Value', color='red')
plt.legend()
plt.xlabel('Time')
plt.ylabel('OT')
plt.title('Model Prediction Results')
plt.show()
