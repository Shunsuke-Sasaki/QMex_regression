#!/usr/bin/env python3
import csv
import sympy as sp
from tqdm import tqdm  # 進捗表示用

# --- 入力ファイルパス ---
DIM_CSV_PATH = '/home/mdxuser/sasaki/example_data/units_all.csv'  # 次元定義CSV

# --- 対象とする独立変数（固定） ---
CSV_INDEPENDENT = [
    "mw", "Ne/Na", "homo", "lumo", "Eiv", "Eav", "Vm", "alpha", "alpha/Vm", 
    "log10(R2)", "G", "log10(Q)", "log10(RA)", "mu", 
    "min(esp)", "max(esp)", "min(mbo)", "max(mbo)", 
    "dEw", "dEo", "dEacid", "dEbase"
]
# --- 定数 ---
DIMENSIONED_CONSTANTS = {
    "g": 9.81,
    "kb": 1.38e-23
}

# --- 対象となる目的変数リスト ---
TARGET_COLUMNS = ['dGs', 'Ebd', 'log10(lifetime)', 'logD', 'logP', 'logS', 'pKaA', 'pKaB', 'RI', 'Tb', 'Tm']

# --- 次元定義 CSV の読み込み ---
def read_dimension_csv(path):
    """
    次元定義 CSV を読み込み、各変数名について
      {"unit": 単位, "vector": [各基本次元の指数]} の辞書と、
      基本次元名（CSVヘッダの3列目以降）のリストを返す。
    ヘッダ例: "Variable,Units,m,s,kg,T,V,cd"
    """
    dims = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or len(reader.fieldnames) < 3:
            raise ValueError("次元定義CSVのヘッダが不正です。")
        # ヘッダの3列目以降を基本次元の名前とする
        fundamental_dims = reader.fieldnames[2:]
        print("【次元定義ファイル】基本次元:", fundamental_dims)
        for row in reader:
            var_name = row['Variable'].strip()
            unit = row['Units'].strip()
            try:
                dim_vector = [sp.Rational(row[dim].strip()) for dim in fundamental_dims]
            except Exception as e:
                raise ValueError(f"変数 '{var_name}' の次元指数変換エラー: {e}")
            dims[var_name] = {"unit": unit, "vector": dim_vector}
    return dims, fundamental_dims

# --- 与えられたシンボリック式 expr の次元ベクトルを計算する ---
def compute_dimension_vector(expr, dims, fundamental_dims):
    """
    expr.as_powers_dict() を用いて因子ごとの冪指数を抽出し、
    各シンボルの次元定義（dims）を加重和して次元ベクトルを求める。
    数値因子は無視する。
    """
    num_fund = len(fundamental_dims)
    dim_vec = [0] * num_fund
    factors = expr.as_powers_dict()
    for factor, exp in factors.items():
        if factor.is_Number:
            continue
        var_name = str(factor)
        if var_name not in dims:
            print(f"注意: 変数 '{var_name}' の次元定義がありません。")
            continue
        vec = dims[var_name]["vector"]
        for i in range(num_fund):
            dim_vec[i] += exp * vec[i]
    return [sp.nsimplify(x) for x in dim_vec]

# --- １つの目的変数について次元解析を行い、New_Pi の式を出力する ---
def process_target(target, dims, fundamental_dims):
    # 依存変数（目的変数）は target とする
    csv_dependent = target
    # 解析対象変数：依存変数 + 独立変数 + 定数
    variable_names = [csv_dependent] + CSV_INDEPENDENT + list(DIMENSIONED_CONSTANTS.keys())
    print("\n【対象変数】", variable_names)
    
    # 次元定義が全て存在するか確認
    for var in variable_names:
        if var not in dims:
            print(f"エラー: 次元定義 CSV に変数 '{var}' の定義が見つかりません。")
            return None

    # 次元行列 D の作成（行：基本次元, 列：各変数の次元指数）
    var_dim_vectors = [dims[var]["vector"] for var in variable_names]
    num_fundamental = len(fundamental_dims)
    n = len(variable_names)
    D = sp.Matrix(num_fundamental, n, lambda i, j: var_dim_vectors[j][i])
    print("\n【次元行列 D for target", target, "】")
    sp.pprint(D)
    
    # Buckingham 次元定理により、D の右零空間（nullspace）から無次元群の指数ベクトルを算出
    nullspace = D.nullspace()
    num_pi = len(nullspace)
    print(f"無次元群の個数（nullspace 次元）: {num_pi}")
    if num_pi == 0:
        print("次元行列の nullspace が 0 次元のため、無次元群は得られません。")
        return None

    # 各変数のシンボリック変数を作成（変数名と同じ名前）
    sym_vars = {var: sp.symbols(var) for var in variable_names}
    
    # Π群のシンボリック式作成
    pi_list = []  # (π名, シンボリック式, 指数ベクトル)
    for idx, vec in enumerate(nullspace, start=1):
        expr = 1
        vec_list = [vec[i] for i in range(vec.rows)]
        term_list = []
        for j, exponent in enumerate(vec_list):
            if exponent != 0:
                expr *= sym_vars[variable_names[j]] ** exponent
                term_list.append(f"{variable_names[j]}^({sp.nsimplify(exponent)})")
        pi_str = " * ".join(term_list) if term_list else "1"
        print(f"Π{idx} = {pi_str}")
        pi_list.append((f"Pi{idx}", expr, vec_list))
    
    # 各Π群に対して (homo/kb) を乗じた新変数を生成
    new_variables = []  # (変数名, シンボリック式)
    factor_expr = sp.simplify(sym_vars["homo"] / sym_vars["kb"])
    for pi_name, pi_expr, _ in pi_list:
        new_var_name = "New_" + pi_name
        new_expr = sp.simplify(factor_expr * pi_expr)
        print(f"{new_var_name} = {new_expr}")
        new_variables.append((new_var_name, new_expr))
    
    # 生成した各新変数のシンボリック式のみを１行にまとめたリストを作成
    expressions_row = [str(expr) for _, expr in new_variables]
    return expressions_row

def main():
    print("=== Buckingham Π-Theorem 次元出力プログラム（複数ターゲット版） ===")
    
    # 次元定義 CSV の読み込み（全ターゲット共通）
    try:
        dims, fundamental_dims = read_dimension_csv(DIM_CSV_PATH)
    except Exception as e:
        print("次元定義 CSV 読み込みエラー:", e)
        return

    # 各ターゲットに対して処理を繰り返す
    for target in tqdm(TARGET_COLUMNS, desc="Processing Targets"):
        # 対応するデータファイルパス（必要に応じて後続処理で利用可能）
        data_file_path = f'/home/mdxuser/sasaki/data/{target}_with_qmex_predict.csv'
        print("\n========================================")
        print("Processing target:", target)
        print("Data file path:", data_file_path)
        
        # 各ターゲットについて次元解析を実施し、New_Pi の式を取得
        expressions_row = process_target(target, dims, fundamental_dims)
        if expressions_row is None:
            continue  # 次元定義に問題がある場合はスキップ
        
        # 出力ファイル名はターゲットごとに変更
        output_csv_path = f'new_variable_expression_{target}.csv'
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(expressions_row)
        print(f"生成した各変数の式を一行で '{output_csv_path}' に出力しました。")

if __name__ == "__main__":
    main()