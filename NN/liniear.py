import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
    #"mw", 
    #"Ne/Na", 
    #"homo", 
    #"lumo", 
    #"Eiv", 
    #"Eav", 
    #"Vm", 
    "alpha", 
    "alpha/Vm", 
    #"log10(R2)",
    #"G", 
    #"log10(Q)", 
    #"log10(RA)", 
    #"mu", 
    #"min(esp)", 
    #"max(esp)", 
    #"min(mbo)", 
    #"max(mbo)",
    #"dEw", 
    "dEo", 
    #"dEacid", 
    #"dEbase"
]  # 学習データの特徴量
target_column = 'Tb'  # 学習データのターゲット

# データを読み込む
X_train_full, y_train_full = load_csv_data(train_file_path, feature_columns, target_column)
X_test, y_test = load_csv_data(test_file_path, feature_columns, target_column)

scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)

# 2. 線形回帰モデルの設計
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)  # 単一の全結合層

    def forward(self, x):
        return self.fc(x)

# R²を計算する関数 (float64型)
def calculate_r2_float64(y_true, y_pred):
    ss_total = torch.sum((y_true - torch.mean(y_true, dtype=torch.float64)) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# 3. クロスバリデーションの設定
num_repeats = 5
all_fold_losses = []
all_test_losses = []
all_fold_r2 = []
all_test_r2 = []

all_fold_weights = []  # 各foldごとの重みを記録するリスト

for repeat in range(num_repeats):
    print(f"Repeat {repeat + 1}/{num_repeats}...")

    fold_losses = []
    fold_r2 = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42 + repeat)  # クロスバリデーションをリピートごとにシャッフル

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
        # トレーニング用と検証用のデータに分割
        X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

        # PyTorchテンソルに変換（float64型に変更）
        X_train = torch.tensor(X_train, dtype=torch.float64)
        y_train = torch.tensor(y_train, dtype=torch.float64).view(-1, 1)
        X_val = torch.tensor(X_val, dtype=torch.float64)
        y_val = torch.tensor(y_val, dtype=torch.float64).view(-1, 1)

        # モデルの再初期化
        model = LinearRegressionModel(input_size=X_train.shape[1]).double()  # doubleはfloat64を意味します
        criterion = nn.MSELoss()  # MSEを使用（線形回帰で一般的）
        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0)  # L2正則化付き

        # トレーニングループ
        epochs = 500  # 学習回数を調整
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
            val_r2 = calculate_r2_float64(y_val, val_predictions)  # R²
            fold_losses.append(val_loss.item())
            fold_r2.append(val_r2.item())
            print(f"Fold {fold + 1}, Validation Loss (RMSE): {val_loss.item():.6f}, Validation R²: {val_r2.item():.6f}")

        # 各foldの重みを記録
        fold_weights = model.fc.weight.detach().cpu().numpy()  # 重みをNumPy配列に変換
        all_fold_weights.append(fold_weights)

    # 各リピートごとにRMSEとR²の平均と分散を記録
    all_fold_losses.append(fold_losses)
    all_fold_r2.append(fold_r2)

    # テスト用データでの最終評価
    X_test_tensor = torch.tensor(X_test, dtype=torch.float64)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float64).view(-1, 1)

    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_loss = torch.sqrt(torch.mean((test_predictions - y_test_tensor) ** 2))  # RMSE
        test_r2 = calculate_r2_float64(y_test_tensor, test_predictions)  # R²
        all_test_losses.append(test_loss.item())
        all_test_r2.append(test_r2.item())
        print(f"Test Loss for Repeat {repeat + 1}: {test_loss.item():.6f}, Test R²: {test_r2.item():.6f}")

# クロスバリデーションの平均と分散を計算
mean_loss_per_repeat = np.mean(all_fold_losses, axis=1)
mean_r2_per_repeat = np.mean(all_fold_r2, axis=1)
var_loss_per_repeat = np.var(all_fold_losses, axis=1)
var_r2_per_repeat = np.var(all_fold_r2, axis=1)

# 各評価指標の平均と分散
mean_loss = np.mean(mean_loss_per_repeat)
mean_r2 = np.mean(mean_r2_per_repeat)
var_loss = np.mean(var_loss_per_repeat)
var_r2 = np.mean(var_r2_per_repeat)

print(f"Average Validation Loss (RMSE) over 5 repeats: {mean_loss:.6f}, Variance of Validation Loss: {var_loss:.6f}")
print(f"Average Validation R² over 5 repeats: {mean_r2:.6f}, Variance of Validation R²: {var_r2:.6f}")

# 外挿（テストデータ）の評価指標の平均と分散を計算
mean_test_loss = np.mean(all_test_losses)
mean_test_r2 = np.mean(all_test_r2)
var_test_loss = np.var(all_test_losses)
var_test_r2 = np.var(all_test_r2)

print(f"Test RMSE (average over 5 repeats): {mean_test_loss:.6f}, Variance of Test RMSE: {var_test_loss:.6f}")
print(f"Test R² (average over 5 repeats): {mean_test_r2:.6f}, Variance of Test R²: {var_test_r2:.6f}")

# 最終的な重みの平均を計算
mean_weights = np.mean(np.array(all_fold_weights), axis=0)

print("\nAverage Coefficients (weights) over all folds and repeats:")
print(mean_weights)