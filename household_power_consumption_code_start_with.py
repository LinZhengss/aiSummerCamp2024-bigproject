# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)

# %%
# 加载数据
df = pd.read_csv('data/household_power_consumption.txt', sep=';', low_memory=False)
df.head()

# %%
# 检查数据
df.info()

# %%
# 创建合并日期时间列
df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
df.drop(['Date', 'Time'], axis=1, inplace=True)

# 处理缺失值（用前向填充处理）
df.replace('?', np.nan, inplace=True)
df = df.astype(float)
df.fillna(method='ffill', inplace=True)

# %%
print("Start Date: ", df['datetime'].min())
print("End Date: ", df['datetime'].max())

# %%
# 划分训练集和测试集
train = df.loc[df['datetime'] <= '2009-12-31']
test = df.loc[df['datetime'] > '2009-12-31']

# %%
# 数据归一化
scaler = StandardScaler()
cols_to_scale = [col for col in train.columns if col != 'datetime']

train_scaled = scaler.fit_transform(train[cols_to_scale])
test_scaled = scaler.transform(test[cols_to_scale])

# 转换回DataFrame
train_scaled_df = pd.DataFrame(train_scaled, columns=cols_to_scale)
test_scaled_df = pd.DataFrame(test_scaled, columns=cols_to_scale)

train_scaled_df['datetime'] = train['datetime'].values
test_scaled_df['datetime'] = test['datetime'].values

# %%
# 创建时间序列数据集
def create_sequences(data, seq_length, target_col='Global_active_power'):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, cols_to_scale.index(target_col)])
    return np.array(X), np.array(y)

# 设置序列长度
SEQ_LENGTH = 24 * 2  # 使用2天的数据预测下一天

# 准备数据
X_train, y_train = create_sequences(train_scaled, SEQ_LENGTH)
X_test, y_test = create_sequences(test_scaled, SEQ_LENGTH)

# %%
# 创建DataLoader
BATCH_SIZE = 64

# 转换为PyTorch张量
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%
# 构建LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 模型参数
INPUT_SIZE = len(cols_to_scale)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)

# %%
# 训练模型
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)

NUM_EPOCHS = 20

train_losses = []
test_losses = []

for epoch in range(NUM_EPOCHS):
    # 训练阶段
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * inputs.size(0)
    
    avg_train_loss = epoch_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)
    
    # 测试阶段
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            test_loss += loss.item() * inputs.size(0)
    
    avg_test_loss = test_loss / len(test_loader.dataset)
    test_losses.append(avg_test_loss)
    scheduler.step(avg_test_loss)
    
    print(f'Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}')

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.show()

# %%
# 在测试集上评估模型
model.eval()
all_targets = []
all_predictions = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(outputs.squeeze().cpu().numpy())

# 计算RMSE
rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
print(f'Test RMSE: {rmse:.4f}')

# %%
# 绘制预测结果与真实值对比图
plt.figure(figsize=(15, 6))

# 选择一段连续的时间展示（例如前500个样本）
plot_samples = 500
plt.plot(all_targets[:plot_samples], label='Actual', alpha=0.7)
plt.plot(all_predictions[:plot_samples], label='Predicted', linestyle='dashed')
plt.xlabel('Time Steps')
plt.ylabel('Scaled Global Active Power')
plt.title('Actual vs Predicted Global Active Power')
plt.legend()
plt.show()

# 绘制整个测试集的对比
plt.figure(figsize=(15, 6))
plt.plot(all_targets, label='Actual', alpha=0.5)
plt.plot(all_predictions, label='Predicted', alpha=0.7)
plt.xlabel('Time Steps')
plt.ylabel('Scaled Global Active Power')
plt.title('Full Test Set: Actual vs Predicted')
plt.legend()
plt.show()