from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler  # 追加
import numpy as np
import pandas as pd
from tqdm import tqdm

# データの読み込みと分割
def load_and_split_data(file_path, feature_columns, target_column, top=0.2, bottom=0.2):
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

# 結果を保存するリスト
results = []

# tqdmでターゲットごとに進捗バーを追加
for target in tqdm(target_column, desc="Processing Targets"):
    data_file_path = f'/home/mdxuser/sasaki/data/{target}_with_qmex_predict.csv'
    X_train_full, y_train_full, X_test, y_test = load_and_split_data(data_file_path, feature_columns, target)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_train_full), desc="Folds", leave=False)):
        X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

        # ★ 各foldのトレーニングデータから標準化パラメータを算出し、
        #   検証データとテストデータに同じパラメータで変換する
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # 線形回帰モデルのトレーニング
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 検証データ（内挿評価）での評価
        val_predictions = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        val_mae = mean_absolute_error(y_val, val_predictions)
        val_r2 = r2_score(y_val, val_predictions)

        # テストデータ（外挿評価）での評価
        test_predictions = model.predict(X_test_scaled)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        test_mae = mean_absolute_error(y_test, test_predictions)
        test_r2 = r2_score(y_test, test_predictions)

        # 結果を保存（各フォールドの内挿と外挿評価を1行にまとめる）
        results.append({
            'target': target,
            'fold': fold + 1,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        })

# DataFrameに変換してCSVに保存
results_df = pd.DataFrame(results)
results_df.to_csv('/home/mdxuser/sasaki/results_LR_normalized.csv', index=False)
print("Results saved to /home/mdxuser/sasaki/results_LR_normalized.csv")