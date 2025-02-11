import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np

# 1. データの読み込み
def load_data(file_path):
    # 空白区切りのデータを読み込む
    data = np.loadtxt(file_path)
    X = data[:, :-1]  # 最後の列を除いた部分が特徴量
    y = data[:, -1]   # 最後の列がターゲット
    return X, y

# ファイルパスを指定（例: 'data.txt'）
file_path = '/home/mdxuser/sasaki/data/input.txt'
X, y = load_data(file_path)

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
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5分割
fold_train_losses = []
fold_val_losses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    # トレーニング用と検証用のデータに分割
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # PyTorchテンソルに変換
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # モデルの再初期化
    model = RegressionModel(input_size=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # トレーニングループ
    epochs = 5000
    train_losses = []
    for epoch in range(epochs):
        # 順伝播
        model.train()  # モデルをトレーニングモードに切り替え
        outputs = model(X_train)
        train_loss = criterion(outputs, y_train)
        
        # 逆伝播
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # トレーニング損失を記録
        train_losses.append(train_loss.item())

    # 検証用データで評価
    model.eval()  # モデルを評価モードに切り替え
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = torch.sqrt(torch.mean((val_predictions - y_val) ** 2))
        #val_loss = criterion(val_predictions, y_val)

    # 平均トレーニング損失を計算
    avg_train_loss = np.mean(train_losses)
    fold_train_losses.append(avg_train_loss)
    fold_val_losses.append(val_loss.item())

    # 損失を表示
    print(f"Fold {fold + 1}: Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss.item():.4f}")

# 平均損失を計算
mean_train_loss = np.mean(fold_train_losses)
mean_val_loss = np.mean(fold_val_losses)
print(f"Average Train Loss: {mean_train_loss:.4f}")
print(f"Average Validation Loss: {mean_val_loss:.4f}")

# 4. テスト用データでモデルを最終評価（オプション）
# 最後のfoldのモデルでX_valとy_valを使った性能確認
with torch.no_grad():
    test_predictions = model(X_val)
    print("Test Predictions:", test_predictions.view(-1).numpy())
    print("Ground Truth:", y_val.view(-1).numpy())