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

# 1. データの読み込みと分割
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

# 外挿データを準備する関数
def prepare_extrapolation_data(X_train, y_train, top_percent=0.1, bottom_percent=0.1):
    data = pd.DataFrame(X_train)
    data['target'] = y_train

    top_data = data.nlargest(int(len(data) * top_percent), 'target')
    bottom_data = data.nsmallest(int(len(data) * bottom_percent), 'target')

    extrapolation_data = pd.concat([top_data, bottom_data])
    return extrapolation_data.iloc[:, :-1].values, extrapolation_data.iloc[:, -1].values

feature_columns = [
    "mw", "Ne/Na", "homo", "lumo", "Eiv", "Eav", "Vm", "alpha", "alpha/Vm", "log10(R2)",
    "G", "log10(Q)", "log10(RA)", "mu", "min(esp)", "max(esp)", "min(mbo)", "max(mbo)",
    "dEw", "dEo", "dEacid", "dEbase"
]
target_column = ['dGs', 'Ebd', 'log10(lifetime)', 'logD', 'logP', 'logS', 'pKaA', 'pKaB', 'RI', 'Tb', 'Tm']

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

def calculate_r2(y_true, y_pred):
    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# 結果を保存するリスト
results = []

# tqdmでターゲットごとに進捗バーを追加
for target in tqdm(target_column, desc="Processing Targets"):
    data_file_path = f'/home/mdxuser/sasaki/data/{target}_with_qmex_predict.csv'
    X_train_full, y_train_full, X_test, y_test = load_and_split_data(data_file_path, feature_columns, target)

    num_repeats = 10
    for repeat in tqdm(range(num_repeats), desc=f"Repeats for {target}", leave=False):
        kf = KFold(n_splits=5, shuffle=True, random_state=42 + repeat)
        fold_extrapolation_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_train_full), desc="Folds", leave=False)):
            X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
            y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

            # 外挿データを準備
            X_extrapolation, y_extrapolation = prepare_extrapolation_data(X_train, y_train)

            # テンソルをデバイスに移動
            X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
            X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
            X_extrapolation = torch.tensor(X_extrapolation, dtype=torch.float32).to(device)
            y_extrapolation = torch.tensor(y_extrapolation, dtype=torch.float32).view(-1, 1).to(device)

            # モデルをデバイスに移動
            model = RegressionModel(input_size=X_train.shape[1]).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            # 学習
            for epoch in range(5000):
                model.train()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 検証と外挿データでの評価
            model.eval()
            with torch.no_grad():
                # 検証データ
                val_predictions = model(X_val)
                val_loss = torch.sqrt(torch.mean((val_predictions - y_val) ** 2)).item()
                val_mae = torch.mean(torch.abs(val_predictions - y_val)).item()
                val_r2 = calculate_r2(y_val, val_predictions).item()

                # 外挿データ
                extrapolation_predictions = model(X_extrapolation)
                extrapolation_loss = torch.sqrt(torch.mean((extrapolation_predictions - y_extrapolation) ** 2)).item()
                extrapolation_mae = torch.mean(torch.abs(extrapolation_predictions - y_extrapolation)).item()
                extrapolation_r2 = calculate_r2(y_extrapolation, extrapolation_predictions).item()

                fold_extrapolation_metrics.append({
                    'extrapolation_loss': extrapolation_loss,
                    'extrapolation_mae': extrapolation_mae,
                    'extrapolation_r2': extrapolation_r2
                })

                results.append({
                    'target': target,
                    'repeat': repeat + 1,
                    'fold': fold + 1,
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'val_r2': val_r2,
                    'extrapolation_loss': extrapolation_loss,
                    'extrapolation_mae': extrapolation_mae,
                    'extrapolation_r2': extrapolation_r2
                })

        # 各繰り返しごとの外挿性能の平均を計算
        avg_extrapolation_metrics = pd.DataFrame(fold_extrapolation_metrics).mean().to_dict()
        results.append({
            'target': target,
            'repeat': repeat + 1,
            'fold': 'Average',
            'val_loss': None,
            'val_mae': None,
            'val_r2': None,
            'extrapolation_loss': avg_extrapolation_metrics['extrapolation_loss'],
            'extrapolation_mae': avg_extrapolation_metrics['extrapolation_mae'],
            'extrapolation_r2': avg_extrapolation_metrics['extrapolation_r2']
        })

# DataFrameに変換してCSVに保存
results_df = pd.DataFrame(results)
results_df.to_csv('/home/mdxuser/sasaki/results.csv', index=False)