import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from sklearn.model_selection import KFold

# 1. データの読み込み
def load_data(file_path):
    data = np.loadtxt(file_path)
    X = data[:, :-1]  # 特徴量
    y = data[:, -1]   # ターゲット
    return X, y

# ファイルパスを指定
file_path = '/home/mdxuser/sasaki/data/input.txt'
X, y = load_data(file_path)

# 2. モデルの設計
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_units):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 3. ダブルクロスバリデーションの設定
outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 外側の5分割
outer_losses = []

# 外側クロスバリデーション
for fold, (outer_train_idx, outer_val_idx) in enumerate(outer_kf.split(X)):
    X_outer_train, X_outer_val = X[outer_train_idx], X[outer_val_idx]
    y_outer_train, y_outer_val = y[outer_train_idx], y[outer_val_idx]

    # Optunaでハイパーパラメータ最適化
    def objective(trial):
        hidden_units = trial.suggest_int('hidden_units', 10, 50)
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
        batch_size = trial.suggest_int('batch_size', 16, 64)

        # 内側クロスバリデーション
        inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)
        inner_losses = []
        for inner_train_idx, inner_val_idx in inner_kf.split(X_outer_train):
            X_inner_train, X_inner_val = X_outer_train[inner_train_idx], X_outer_train[inner_val_idx]
            y_inner_train, y_inner_val = y_outer_train[inner_train_idx], y_outer_train[inner_val_idx]

            # PyTorchテンソルに変換
            X_inner_train = torch.tensor(X_inner_train, dtype=torch.float32)
            y_inner_train = torch.tensor(y_inner_train, dtype=torch.float32).view(-1, 1)
            X_inner_val = torch.tensor(X_inner_val, dtype=torch.float32)
            y_inner_val = torch.tensor(y_inner_val, dtype=torch.float32).view(-1, 1)

            # モデルの初期化
            model = RegressionModel(input_size=X.shape[1], hidden_units=hidden_units)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # トレーニングループ
            epochs = 1000
            for epoch in range(epochs):
                model.train()
                for i in range(0, len(X_inner_train), batch_size):
                    X_batch = X_inner_train[i:i+batch_size]
                    y_batch = y_inner_train[i:i+batch_size]
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

            # 検証損失を計算
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_inner_val)
                val_loss = criterion(val_predictions, y_inner_val).item()
                inner_losses.append(val_loss)

        return np.mean(inner_losses)

    # Optuna Study の作成
    study = optuna.create_study(direction='minimize')  # 損失を最小化
    study.optimize(objective, n_trials=1)  # 試行回数を指定

    # 最適ハイパーパラメータでモデルを再訓練
    best_params = study.best_params
    hidden_units = best_params['hidden_units']
    lr = best_params['lr']
    batch_size = best_params['batch_size']

    X_outer_train_tensor = torch.tensor(X_outer_train, dtype=torch.float32)
    y_outer_train_tensor = torch.tensor(y_outer_train, dtype=torch.float32).view(-1, 1)
    X_outer_val_tensor = torch.tensor(X_outer_val, dtype=torch.float32)
    y_outer_val_tensor = torch.tensor(y_outer_val, dtype=torch.float32).view(-1, 1)

    model = RegressionModel(input_size=X.shape[1], hidden_units=hidden_units)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 50000
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_outer_train_tensor), batch_size):
            X_batch = X_outer_train_tensor[i:i+batch_size]
            y_batch = y_outer_train_tensor[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # 外側検証損失を計算
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_outer_val_tensor)
        val_loss = criterion(val_predictions, y_outer_val_tensor).item()
        outer_losses.append(val_loss)
        print(f"Outer Fold {fold + 1}, Validation Loss: {val_loss:.4f}")

# 平均損失を計算
mean_outer_loss = np.mean(outer_losses)
print(f"Average Outer Validation Loss: {mean_outer_loss:.4f}")