import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt

# 加载数据
train_data = pd.read_csv('train1_icu_data.csv')
train_labels = pd.read_csv('train1_icu_label.csv', header=None).values.ravel()
test_data = pd.read_csv('test1_icu_data.csv')
test_labels = pd.read_csv('test1_icu_label.csv', header=None).values.ravel()

# 数据标准化
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# 创建MLP分类器
mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', max_iter=1000, random_state=1)

# 训练MLP分类器
mlp.fit(train_data, train_labels[1:])

# 预测训练集
train_predictions = mlp.predict(train_data)
train_proba = mlp.predict_proba(train_data)

# 预测测试集
test_predictions = mlp.predict(test_data)
test_proba = mlp.predict_proba(test_data)

# 计算训练误差和测试误差
train_error = 1 - accuracy_score(train_labels[1:], train_predictions)
test_error = 1 - accuracy_score(test_labels[1:], test_predictions)

# 计算交叉验证误差
cv_error = mlp.loss_  # MLPClassifier自带交叉验证误差

# 打印误差
print(f'Training error: {train_error}')
print(f'Cross-validation error: {cv_error}')
print(f'Test error: {test_error}')

# 绘制学习曲线
plt.plot(mlp.loss_curve_)
plt.title('Learning Curve')
plt.xlabel('Number of iterations')
plt.ylabel('Loss')
plt.show()

# 可选：尝试不同的超参数
# 例如，改变隐藏层的大小
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=1000, random_state=1)
mlp.fit(train_data, train_labels[1:])
test_predictions_new = mlp.predict(test_data)
test_error_new = 1 - accuracy_score(test_labels[1:], test_predictions_new)
print(f'New test error with different hyperparameters: {test_error_new}')