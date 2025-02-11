#!/usr/bin/env python3
import csv
import sympy as sp

# --- 入力・出力ファイルのパス ---
DIM_CSV_PATH    = '/home/mdxuser/sasaki/example_data/units_all.csv'  # 次元定義CSV
OUTPUT_CSV_PATH = 'new_variable_dimension.csv'                     # 出力先CSV（生成した変数の次元情報）

# --- 対象とする変数の設定 ---
CSV_DEPENDENT   = "Tb"  # 目的変数
CSV_INDEPENDENT = [
    "mw", "Ne/Na", "homo", "lumo", "Eiv", "Eav", "Vm", "alpha", "alpha/Vm", 
    "log10(R2)", "G", "log10(Q)", "log10(RA)", "mu", 
    "min(esp)", "max(esp)", "min(mbo)", "max(mbo)", 
    "dEw", "dEo", "dEacid", "dEbase"
]
# 定数（次元定義 CSV にも同じ変数名で定義しておくこと）
DIMENSIONED_CONSTANTS = {
    "g": 9.81,
    "kb": 1.38e-23
}

# --- 次元定義 CSV の読み込み ---
def read_dimension_csv(path):
    """
    次元定義 CSV を読み込み、各変数名について
      {"unit": 単位, "vector": [各基本次元の指数]} の辞書と、
      基本次元名（CSVヘッダ3列目以降）のリストを返す。
    CSV ヘッダは "Variable,Units,m,s,kg,T,V,cd" を想定。
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
    各シンボルの次元定義 dims を加重和して次元ベクトルを求める。
    数値因子は無視する。
    """
    num_fund = len(fundamental_dims)
    dim_vec = [0] * num_fund
    factors = expr.as_powers_dict()
    for factor, exp in factors.items():
        if factor.is_Number:
            continue
        # factor が Symbol の場合、その文字列表現で次元定義を探す
        var_name = str(factor)
        if var_name not in dims:
            print(f"注意: 変数 '{var_name}' の次元定義がありません。")
            continue
        vec = dims[var_name]["vector"]
        # 各基本次元について指数を加算
        for i in range(num_fund):
            dim_vec[i] += exp * vec[i]
    # 各要素を簡単な形にする
    dim_vec = [sp.nsimplify(x) for x in dim_vec]
    return dim_vec

def main():
    print("=== Buckingham Π-Theorem 次元出力プログラム ===")
    
    # ① 次元定義 CSV の読み込み
    try:
        dims, fundamental_dims = read_dimension_csv(DIM_CSV_PATH)
    except Exception as e:
        print("次元定義 CSV 読み込みエラー:", e)
        return

    # ② 解析対象の全変数（CSV変数＋定数）の順序リストを作成
    specified_csv_vars = [CSV_DEPENDENT] + CSV_INDEPENDENT
    variable_names = specified_csv_vars + list(DIMENSIONED_CONSTANTS.keys())
    print("\n【解析対象の変数】")
    print("CSV変数:", specified_csv_vars)
    print("定数:", list(DIMENSIONED_CONSTANTS.keys()))
    print("合計:", variable_names)

    # ③ 各変数の次元情報を次元定義 CSV から取得
    var_dim_vectors = []
    for var in variable_names:
        if var not in dims:
            print(f"エラー: 次元定義 CSV に変数 '{var}' の定義が見つかりません。")
            return
        var_dim_vectors.append(dims[var]["vector"])

    # ④ 次元行列 D の作成（行：基本次元, 列：各変数の次元指数）
    num_fundamental = len(fundamental_dims)
    n = len(variable_names)
    D = sp.Matrix(num_fundamental, n, lambda i, j: var_dim_vectors[j][i])
    print("\n【次元行列 D】")
    sp.pprint(D)

    # ⑤ Buckingham の次元定理に基づき、D の右零空間（nullspace）からΠ群の指数ベクトルを算出
    nullspace = D.nullspace()
    num_pi = len(nullspace)
    print(f"\n無次元群の個数（nullspace 次元）: {num_pi}")
    if num_pi == 0:
        print("次元行列の nullspace が 0 次元のため、無次元群は得られません。")
        return

    # ⑥ 各Π群のシンボリック式を作成
    print("\n【無次元群のシンボリック式】")
    # 各変数のシンボルを作成（変数名と同じ名前）
    sym_vars = {var: sp.symbols(var) for var in variable_names}
    pi_list = []  # (pi名, シンボリック式, 指数ベクトル)
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

    # ⑦ 新たな変数を各Π群に対して「homo/kb」を乗じた形で定義する
    #     すなわち、New_Pi{i} = (homo/kb) * Πi
    #     （なお、Πi は無次元なので、New_Pi{i} の次元は homo/kb の次元となり T になるはず）
    new_variables = []  # (変数名, シンボリック式)
    factor_expr = sp.simplify(sym_vars["homo"] / sym_vars["kb"])
    print("\n【生成する新たな変数（各Π群に homo/kb を乗じたもの）】")
    for pi_name, pi_expr, _ in pi_list:
        new_var_name = "New_" + pi_name
        new_expr = sp.simplify(factor_expr * pi_expr)
        print(f"{new_var_name} = {new_expr}")
        new_variables.append((new_var_name, new_expr))
    
    # ⑧ 各生成変数の次元ベクトルを計算する
    # ここでは、π群は無次元なので、各生成変数の次元は homo/kb の次元となるはず
    new_var_dims = []  # list of (variable name, dimension vector)
    for var_name, expr in new_variables:
        dim_vec = compute_dimension_vector(expr, dims, fundamental_dims)
        new_var_dims.append((var_name, dim_vec))
    
    print("\n【生成した各新変数の次元ベクトル】")
    for var_name, vec in new_var_dims:
        vec_str = ", ".join(f"{fd}:{vec_i}" for fd, vec_i in zip(fundamental_dims, vec))
        print(f"{var_name}: {vec_str}")
    
    # ⑨ CSV に生成した各変数の次元情報を出力する
    # 出力形式例:
    #   1行目: Variable, m, s, kg, T, V, cd
    #   以下: New_Pi1, <m指数>, <s指数>, ...　etc.
    out_rows = []
    header_out = ["Variable"] + fundamental_dims
    out_rows.append(header_out)
    for var_name, vec in new_var_dims:
        row = [var_name] + [str(exp) for exp in vec]
        out_rows.append(row)
    
    with open(OUTPUT_CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)
    print(f"\n生成した各変数の次元情報を '{OUTPUT_CSV_PATH}' に出力しました。")

if __name__ == "__main__":
    main()