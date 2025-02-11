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
    "mw", "Ne/Na", "homo", "lumo", "Eiv", "Eav", "Vm", "alpha", "alpha/Vm", "log10(R2)",
    "G", "log10(Q)", "log10(RA)", "mu", "min(esp)", "max(esp)", "min(mbo)", "max(mbo)",
    "dEw", "dEo", "dEacid", "dEbase"
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

# R2の計算
def calculate_r2(y_true, y_pred):
    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# クロスバリデーションを20回繰り返す
num_repeats = 10
all_fold_losses = []
all_fold_mae = []
all_fold_r2 = []
all_test_losses = []
all_test_mae = []
all_test_r2 = []

for repeat in range(num_repeats):
    print(f"Repeat {repeat + 1}/{num_repeats}...")

    kf = KFold(n_splits=5, shuffle=True, random_state=42 + repeat)  # クロスバリデーションをリピートごとにシャッフル
    fold_losses = []
    fold_mae = []
    fold_r2 = []

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
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)  # weight_decayでL2正則化を適用

        # トレーニングループ
        epochs = 5000  # 学習回数を調整
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
            val_mae = torch.mean(torch.abs(val_predictions - y_val))  # MAE
            val_r2 = calculate_r2(y_val, val_predictions)  # R²
            fold_losses.append(val_loss.item())
            fold_mae.append(val_mae.item())
            fold_r2.append(val_r2.item())
            print(f"Fold {fold + 1}, Validation RMSE: {val_loss.item():.4f}, Validation MAE: {val_mae.item():.4f}, Validation R²: {val_r2.item():.4f}")

    # 各リピートごとにRMSE, MAE, R²の平均と分散を記録
    all_fold_losses.append(fold_losses)
    all_fold_mae.append(fold_mae)
    all_fold_r2.append(fold_r2)

    # テスト用データでの外挿評価（外挿評価をリピートごとに行う）
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_loss = torch.sqrt(torch.mean((test_predictions - y_test_tensor) ** 2))  # RMSE
        test_mae = torch.mean(torch.abs(test_predictions - y_test_tensor))  # MAE
        test_r2 = calculate_r2(y_test_tensor, test_predictions)  # R²
        all_test_losses.append(test_loss.item())
        all_test_mae.append(test_mae.item())
        all_test_r2.append(test_r2.item())
        print(f"Test RMSE: {test_loss.item():.4f}, Test MAE: {test_mae.item():.4f}, Test R²: {test_r2.item():.4f}")

# 20回のクロスバリデーションの平均と分散を計算
mean_loss_per_repeat = np.mean(all_fold_losses, axis=1)
mean_mae_per_repeat = np.mean(all_fold_mae, axis=1)
mean_r2_per_repeat = np.mean(all_fold_r2, axis=1)
var_loss_per_repeat = np.var(all_fold_losses, axis=1)
var_mae_per_repeat = np.var(all_fold_mae, axis=1)
var_r2_per_repeat = np.var(all_fold_r2, axis=1)

# 各評価指標の平均と分散
mean_loss = np.mean(mean_loss_per_repeat)
mean_mae = np.mean(mean_mae_per_repeat)
mean_r2 = np.mean(mean_r2_per_repeat)
var_loss = np.mean(var_loss_per_repeat)
var_mae = np.mean(var_mae_per_repeat)
var_r2 = np.mean(var_r2_per_repeat)

print(f"Average RMSE over 10 repeats: {mean_loss:.4f}, Variance of RMSE: {var_loss:.4f}")
print(f"Average MAE over 10 repeats: {mean_mae:.4f}, Variance of MAE: {var_mae:.4f}")
print(f"Average R² over 10 repeats: {mean_r2:.4f}, Variance of R²: {var_r2:.4f}")

# 外挿（テストデータ）の評価指標の平均と分散を計算
mean_test_loss = np.mean(all_test_losses)
mean_test_mae = np.mean(all_test_mae)
mean_test_r2 = np.mean(all_test_r2)
var_test_loss = np.var(all_test_losses)
var_test_mae = np.var(all_test_mae)
var_test_r2 = np.var(all_test_r2)
print(f"Test RMSE (average over 10 repeats): {mean_test_loss:.4f}, Variance of Test RMSE: {var_test_loss:.4f}")
print(f"Test MAE (average over 10 repeats): {mean_test_mae:.4f}, Variance of Test MAE: {var_test_mae:.4f}")
print(f"Test R² (average over 10 repeats): {mean_test_r2:.4f}, Variance of Test R²: {var_test_r2:.4f}")