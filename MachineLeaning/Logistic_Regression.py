import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np  # for matrix calculation
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

SCATTER_TITLE = 'data 3.0a scatter'
SCATTER_XLABEL = 'density'
SCATTER_YLABEL = 'sugar ratio'

# 加载对应数据
dataset = np.genfromtxt('/Users/admin/Desktop/CL/Python/data/3.0a.txt',encoding='utf8',delimiter=',')
# 将数据按照属性分离开
# X[:,0]就是取所有行的第0个数据
X = dataset[:, 1:3]
y = dataset[:, 3]
m, n = np.shape(X)
# 绘制原始数据散点图
f1 = plt.figure(1)
plt.title(SCATTER_TITLE)
plt.xlabel(SCATTER_XLABEL)
plt.ylabel(SCATTER_YLABEL)

plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=100, label='bad')
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=100, label='good')
plt.legend(loc='upper right')
plt.show()

# 生成测试和训练数据集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=0)
# 模型训练
log_model = LogisticRegression()  # using log-regression lib model
log_model.fit(X_train, y_train)  # fitting
# 模型验证
y_pred = log_model.predict(X_test)
# 总结模型的适用性
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
# 显示决策边界
f2 = plt.figure(2)
h = 0.001
x0_min, x0_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
x1_min, x1_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
x0, x1 = np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h))
# 这里“模型”是你模型的预测（分类）功能
z = log_model.predict(np.c_[x0.ravel(), x1.ravel()])
# 将结果标记出来
z = z.reshape(x0.shape)
plt.contourf(x0, x1, z, cmap=pl.cm.Paired)
plt.title(SCATTER_TITLE)
plt.xlabel(SCATTER_XLABEL)
plt.ylabel(SCATTER_YLABEL)
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=100, label='bad')
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=100, label='good')
plt.legend(loc='upper right')
plt.show()