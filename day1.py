# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# 设置中文字体支持 - 添加这部分代码解决显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 加载数据
data = pd.read_csv('train.csv')
df = data.copy()
print("原始数据形状:", df.shape)
print("数据样本:")
print(df.sample(5))

# 数据预处理
# 删除不相关的特征
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
print("\n删除列后的数据信息:")
print(df.info())

# 处理缺失值
print("\n处理前的缺失值统计:")
print(df.isnull().sum())
df.dropna(inplace=True)  # 删除包含缺失值的行
print("\n处理后的缺失值统计:")
print("剩余缺失值数量:", df.isnull().sum().sum())

# 特征工程
# 创建家庭成员数量特征
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# 创建是否独自一人特征
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
# 从姓名中提取称谓
df['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
# 规范化称谓
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                  'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# 转换分类数据为数值形式
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'])
print("\n特征工程和独热编码后的数据样本:")
print(df.sample(5))
print("\n数据形状:", df.shape)

# 分离特征和标签
X = df.drop(columns=['Survived'])
y = df['Survived']
print(f"\n特征形状: {X.shape}, 标签形状: {y.shape}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
print("训练集生存率:", y_train.mean())
print("测试集生存率:", y_test.mean())

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建和评估模型
models = {
    "SVM": SVC(kernel='rbf', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\n=== 训练 {name} 模型 ===")
    # 训练模型
    model.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm
    }
    
    print(f"{name} 准确率: {accuracy:.4f}")
    print("分类报告:")
    print(report)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['死亡', '生存'], 
                yticklabels=['死亡', '生存'])
    plt.title(f'{name} 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()  # 添加自动调整布局
    plt.savefig(f'{name}_confusion_matrix.png')  # 保存图片以防显示问题
    plt.show()

# 比较模型性能
print("\n=== 模型比较 ===")
for name, res in results.items():
    print(f"{name}: 准确率 = {res['accuracy']:.4f}")

# 选择最佳模型
best_model = max(results, key=lambda x: results[x]['accuracy'])
print(f"\n最佳模型: {best_model}, 准确率: {results[best_model]['accuracy']:.4f}")

# 随机森林特征重要性
if '随机森林' in models:
    rf_model = models['随机森林']
    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    sorted_importances = feature_importances.sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances.values, y=sorted_importances.index)
    plt.title('随机森林特征重要性')
    plt.xlabel('重要性分数')
    plt.ylabel('特征')
    plt.tight_layout()
    plt.savefig('random_forest_feature_importance.png')  # 保存图片
    plt.show()