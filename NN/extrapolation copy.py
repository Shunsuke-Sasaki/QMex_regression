import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

# 1. データの読み込み
def load_csv_data(file_path, feature_columns, target_column):
    data = pd.read_csv(file_path)
    X = data[feature_columns].values  # 指定された特徴量カラム
    y = data[target_column].values   # 指定されたターゲットカラム
    return X, y

# 学習用データと評価用データのファイルパス
train_file_path = '/home/mdxuser/sasaki/example_data/Tb_QMex_inter.csv'
test_file_path = '/home/mdxuser/sasaki/example_data/Tb_QMex_extra.csv'

feature_columns = [
    "mw",
"Ne/Na",
"homo",
"lumo",
"Eiv",
"Eav",
"Vm",
"alpha",
"alpha/Vm",
"log10(R2)",
 "G",
 "log10(Q)",
 "log10(RA)",
 "mu",
 "min(esp)",
 "max(esp)",
 "min(mbo)",
 "max(mbo)",
 "dEw",
 "dEo",
 "dEacid",
 "dEbase"
 ]  # 学習データの特徴量
target_column = 'Tb'  # 学習データのターゲット


# データを読み込む
X_train_full, y_train_full = load_csv_data(train_file_path, feature_columns, target_column)
X_test, y_test = load_csv_data(test_file_path, feature_columns, target_column)

# 2. モデルの設計
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 20)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(20, 16)
        self.fc3 = nn.Linear(16, 1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 3. クロスバリデーションの設定
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_losses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
    # トレーニング用と検証用のデータに分割
    X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
    y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

    # PyTorchテンソルに変換
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # モデルの再初期化
    model = RegressionModel(input_size=X_train.shape[1])
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()  # MAE（平均絶対誤差）を使用
    #optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Optimizerの設定時にweight_decayを追加
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0)  # weight_decayでL2正則化を適用

    # トレーニングループ
    epochs = 50000  # 学習回数を調整
    for epoch in range(epochs):
        # 順伝播
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # 逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 検証用データで評価
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = torch.sqrt(torch.mean((val_predictions - y_val) ** 2))  # RMSE
        fold_losses.append(val_loss.item())
        print(f"Fold {fold + 1}, Validation Loss: {val_loss.item():.4f}")

# 平均損失を計算
mean_loss = np.mean(fold_losses)
print(f"Average Validation Loss: {mean_loss:.4f}")

# 4. テスト用データでモデルを最終評価
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = torch.sqrt(torch.mean((test_predictions - y_test) ** 2))  # RMSE
    print(f"Test Loss: {test_loss.item():.4f}")
    #print("Test Predictions:", test_predictions.view(-1).numpy())
    #print("Ground Truth:", y_test.view(-1).numpy())