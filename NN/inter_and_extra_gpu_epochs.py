import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from tqdm import tqdm
# デバイスの設定 (GPUが使用可能かを確認)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# データの読み込みと分割
def load_and_split_data(file_path, feature_columns, target_column, top=0.05, bottom=0.05):
    data = pd.read_csv(file_path)
    data_size = len(data)
    top_indices = data.nlargest(int(data_size * top), target_column).index
    bottom_indices = data.nsmallest(int(data_size * bottom), target_column).index
    test_indices = top_indices.union(bottom_indices)
    train_indices = data.drop(test_indices).index

    X_train = data.loc[train_indices, feature_columns].values
    y_train = data.loc[train_indices, target_column].values
    X_test = data.loc[test_indices, feature_columns].values
    y_test = data.loc[test_indices, target_column].values
    return X_train, y_train, X_test, y_test

# 特徴量とターゲット列
feature_columns = [
    "mw", "Ne/Na", "homo", "lumo", "Eiv", "Eav", "Vm", "alpha", "alpha/Vm", "log10(R2)",
    "G", "log10(Q)", "log10(RA)", "mu", "min(esp)", "max(esp)", "min(mbo)", "max(mbo)",
    "dEw", "dEo", "dEacid", "dEbase"
]
target_column = ['dGs', 'Ebd', 'log10(lifetime)', 'logD', 'logP', 'logS', 'pKaA', 'pKaB', 'RI', 'Tb', 'Tm']

# モデル定義
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

# R²スコアの計算
def calculate_r2(y_true, y_pred):
    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# 結果を保存するリスト
results = []
# 訓練に使用するエポック数のリスト
epochs_list = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000,1000000]

# 結果を保存するリスト
all_results = []

for target in tqdm(target_column, desc="Processing Targets"):
    data_file_path = f'/home/mdxuser/sasaki/data/{target}_with_qmex_predict.csv'
    X_train_full, y_train_full, X_test, y_test = load_and_split_data(data_file_path, feature_columns, target)

    n=0
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_train_full), desc="Folds", leave=False)):
        X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

        # データをテンソル化し、デバイスに移動
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

        for epochs in epochs_list:
            # モデルの初期化
            model = RegressionModel(input_size=X_train.shape[1]).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # モデルの学習
            for epoch in range(epochs):
                model.train()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 検証データでの評価
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val)
                val_loss = torch.sqrt(torch.mean((val_predictions - y_val) ** 2)).item()
                val_mae = torch.mean(torch.abs(val_predictions - y_val)).item()
                val_r2 = calculate_r2(y_val, val_predictions).item()

                # テストデータでの評価
                test_predictions = model(X_test_tensor)
                test_loss = torch.sqrt(torch.mean((test_predictions - y_test_tensor) ** 2)).item()
                test_mae = torch.mean(torch.abs(test_predictions - y_test_tensor)).item()
                test_r2 = calculate_r2(y_test_tensor, test_predictions).item()

                # 結果を追加
                all_results.append({
                    'target': target,
                    'fold': fold + 1,
                    'epochs': epochs,
                    'val_rmse': val_loss,
                    'val_mae': val_mae,
                    'val_r2': val_r2,
                    'test_rmse': test_loss,
                    'test_mae': test_mae,
                    'test_r2': test_r2
                })
        n+=1
        if n>2:
            pass
            #break

# DataFrameに変換してCSVに保存
results_df = pd.DataFrame(all_results)
results_df.to_csv('/home/mdxuser/sasaki/results_epochs_evaluation.csv', index=False)
print("Results saved to /home/mdxuser/sasaki/results_epochs_evaluation.csv")