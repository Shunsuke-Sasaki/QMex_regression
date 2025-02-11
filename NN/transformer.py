import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
from scipy.optimize import minimize

# ==========================================
# 1. Transformer モデルの定義
# ==========================================
class StockTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(StockTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        return self.fc_out(x[:, -1, :])  # 最後の時系列ステップの出力を使用

# ==========================================
# 2. ハイパーパラメータの設定
# ==========================================
input_dim = 4  # open, high, low, close
model_dim = 64
num_heads = 4
num_layers = 3
output_dim = 1
window_size = 100  # 過去100日間を入力
batch_size = 64
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer モデルの準備
model = StockTransformer(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)

# ==========================================
# 3. データ読み込みと前処理
# ==========================================
print("データを読み込んでいます...")

symbol_path = "tosho_price_30.csv"
stock_codes = pd.read_csv(symbol_path, header=None)[0].tolist()

all_data = []
for code in tqdm(stock_codes, desc="データ読み込み", unit="銘柄"):
    file_path = f'price_data_jp_1000_1d/{code}_jp_1d.csv'
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        continue

    data['code'] = code
    all_data.append(data)

df = pd.concat(all_data, ignore_index=True)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['code', 'timestamp']).reset_index(drop=True)

df['prev_close'] = df.groupby('code')['close'].shift(1)
df['next_open'] = df.groupby('code')['open'].shift(-1)
df['rate_of_change'] = (df['next_open'] - df['close']) / df['close']

df = df.dropna(subset=['prev_close', 'open', 'high', 'low', 'close', 'rate_of_change']).reset_index(drop=True)
df = df[abs(df['rate_of_change']) <= 0.4].copy()

df['date_only'] = df['timestamp'].dt.date
df['year'] = df['timestamp'].dt.year
df['period'] = df['date_only'].apply(lambda x: x.toordinal())

# 特徴量
feature_cols = ['open', 'high', 'low', 'close']

# データ正規化
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# `rate_of_change` の正規化
df['rate_of_change'] = (df['rate_of_change'] - df['rate_of_change'].mean()) / df['rate_of_change'].std()

# ==========================================
# 4. データセット作成
# ==========================================
def create_sequences(data, target, window_size):
    sequences, labels = [], []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i + window_size])
        labels.append(target[i + window_size])
    return np.array(sequences), np.array(labels)

X, y = create_sequences(df[feature_cols].values, df['rate_of_change'].values, window_size)

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(StockDataset(X, y), batch_size=batch_size, shuffle=True)

# ==========================================
# 5. 学習
# ==========================================
optimizer = optim.Adam(model.parameters(), lr=0.001)

def compute_s_loss(y_true, y_pred):
    """s を最大化するための損失関数 (-s を最小化)"""
    y_true = (y_true - y_true.mean()) / (y_true.std() + 1e-8)
    y_pred = (y_pred - y_pred.mean()) / (y_pred.std() + 1e-8)
    s = torch.dot(y_true, y_pred) / len(y_true)
    return -s

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()

        loss = compute_s_loss(y_batch, y_pred)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), "transformer_stock_model.pth")

# ==========================================
# 6. 予測の生成
# ==========================================
model.eval()
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
with torch.no_grad():
    df['predictions'] = np.nan
    df.loc[df.index[-len(X):], 'predictions'] = model(X_tensor).cpu().numpy().flatten()

df.dropna(subset=['predictions'], inplace=True)

# ==========================================
# 7. s の最適化と評価
# ==========================================
def optimize_coefficients(data):
    def objective_function(coefficients):
        y_true = (data['rate_of_change'].values - np.mean(data['rate_of_change'])) / np.std(data['rate_of_change'])
        y_pred = (data['predictions'].values - np.mean(data['predictions'])) / np.std(data['predictions'])
        return -np.dot(y_true, y_pred) * 10000

    return minimize(objective_function, np.ones(1), method='L-BFGS-B').x

df['s'] = np.dot(df['rate_of_change'].values, df['predictions'].values)

print(f"\n全期間の平均s: {df['s'].mean():.4e}")

df.to_csv("optimized_transformer_predictions.csv", index=False)