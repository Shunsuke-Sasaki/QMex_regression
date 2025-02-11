import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import ast

# 1. データの読み込み
def load_csv_data(file_path, feature_columns, target_column):
    data = pd.read_csv(file_path)
    X = data[feature_columns].values  # 指定された特徴量カラム
    y = data[target_column].values   # 指定されたターゲットカラム
    return X, y

# 学習用データと評価用データのファイルパス
data_file = "/home/mdxuser/sasaki/example_data/Tb_QMex2.csv"
eval_set_file = "/home/mdxuser/sasaki/example_data/Tb_group_id.csv"

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
data = pd.read_csv(data_file)
eval_sets = pd.read_csv(eval_set_file)

# 内挿データと外挿データをeval_set_fileのインデックスに基づいて分ける
all_data = []

for _, eval_row in eval_sets.iterrows():
    eval_set_name = eval_row["task"]
    idx_inter = ast.literal_eval(eval_row["idx_inter"])
    idx_extra = ast.literal_eval(eval_row["idx_extra"])
    inter_data = data.iloc[idx_inter]
    extra_data = data.iloc[idx_extra]

    all_data.append({"inter_data": inter_data, "extra_data": extra_data})

# 特徴量とターゲットカラムを分割
X_inter = []
y_inter = []
X_extra = []
y_extra = []

# 内挿データと外挿データをリストに格納
for data_split in all_data:
    X_inter.append(data_split["inter_data"][feature_columns].values)
    y_inter.append(data_split["inter_data"][target_column].values)
    X_extra.append(data_split["extra_data"][feature_columns].values)
    y_extra.append(data_split["extra_data"][target_column].values)

# numpy arrayに変換
X_inter = np.vstack(X_inter)
y_inter = np.hstack(y_inter)
X_extra = np.vstack(X_extra)
y_extra = np.hstack(y_extra)

# データを標準化
scaler = StandardScaler()
X_inter = scaler.fit_transform(X_inter)
X_extra = scaler.transform(X_extra)

# 2. 線形回帰モデルの設計
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)  # 単一の全結合層

    def forward(self, x):
        return self.fc(x)

# 3. クロスバリデーションの設定
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_losses = []
test_losses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_inter)):
    
    # トレーニング用と検証用のデータに分割
    X_train, X_val = X_inter[train_idx], X_inter[val_idx]
    y_train, y_val = y_inter[train_idx], y_inter[val_idx]

    # PyTorchテンソルに変換
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # モデルの再初期化
    model = LinearRegressionModel(input_size=X_train_tensor.shape[1])
    criterion = nn.L1Loss()  # L1Loss (絶対誤差)
    optimizer = optim.SGD(model.parameters(), lr=0.0000001, weight_decay=0.1)  # L2正則化付き

    # トレーニングループ
    epochs = 10000  # 学習回数を調整
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 検証用データで評価
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_tensor)
        val_loss = torch.sqrt(torch.mean((val_predictions - y_val_tensor) ** 2))  # RMSE
        fold_losses.append(val_loss.item())
        print(f"Fold {fold + 1}, Validation Loss: {val_loss.item():.4f}")

    # 外挿データで評価 (テストデータ)
    X_extra_tensor = torch.tensor(X_extra, dtype=torch.float32)
    y_extra_tensor = torch.tensor(y_extra, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        test_predictions = model(X_extra_tensor)
        test_loss = torch.sqrt(torch.mean((test_predictions - y_extra_tensor) ** 2))  # RMSE
        test_losses.append(test_loss.item())
        print(f"Fold {fold + 1}, Test Loss (Extrapolation): {test_loss.item():.4f}")

# 平均損失を計算
mean_loss = np.mean(fold_losses)
mean_test_loss = np.mean(test_losses)
print(f"Average Validation Loss: {mean_loss:.4f}")
print(f"Average Test Loss (Extrapolation): {mean_test_loss:.4f}")