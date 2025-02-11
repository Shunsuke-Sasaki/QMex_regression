from ax.service.managed_loop import optimize
import numpy as np

# 目的関数を定義 (例: 改良版)
def objective_function(parameters):
    return sum([parameters[f"x{i}"]**2 + 2 * parameters[f"x{i}"] + 1 for i in range(1, 7)])

# 探索空間を定義
parameters = [{"name": f"x{i}", "type": "range", "bounds": [-5.0, 5.0]} for i in range(1, 7)]

# Axの最適化ループを開始
best_parameters, values, experiment, model = optimize(
    parameters=parameters,                 # 探索空間の設定
    evaluation_function=objective_function, # 目的関数
    total_trials=30,                        # 評価回数
    minimize=True                           # 最小化問題
)

# 結果の表示
print("最適なパラメータ:", best_parameters)
print("最適値:", values)