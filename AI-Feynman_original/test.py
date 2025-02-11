#!/usr/bin/env python3
import csv
import sympy as sp

# --- ファイルパスの定義 ---
DIM_CSV_PATH    = '/home/mdxuser/sasaki/example_data/units_all.csv'    # 次元定義用 CSV のパス
DATA_CSV_PATH   = '/home/mdxuser/sasaki/data/Tb_with_qmex_predict.csv'  # データ CSV のパス
OUTPUT_CSV_PATH = 'data_with_pi.csv'                                   # 出力先 CSV のパス

# --- プログラム中で指定する CSV からの変数（CSV ヘッダに一致している必要があります） ---
CSV_DEPENDENT   = "Tb"  # 依存変数
CSV_INDEPENDENT = [
    "mw", "Ne/Na", "homo", "lumo", "Eiv", "Eav", "Vm", "alpha", "alpha/Vm", "log10(R2)",
    "G", "log10(Q)", "log10(RA)", "mu", "min(esp)", "max(esp)", "min(mbo)", "max(mbo)",
    "dEw", "dEo", "dEacid", "dEbase"
]

# --- プログラム中で定義する次元付き定数（次元定義 CSV にも同じ変数名で定義しておくこと） ---
DIMENSIONED_CONSTANTS = {
    "g": 9.81,    # 例：重力加速度 [m/s^2]（次元は、m s^-2）
    "kb": 1,
}

# --- 次元定義 CSV の読み込み ---
def read_dimension_csv(path):
    """
    次元定義 CSV を読み込み、変数名 → {"unit": 単位, "vector": [各基本次元の指数]} の辞書を返す。
    CSV ヘッダは "Variable,Units,m,s,kg,T,V,cd" とする。
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
    return dims

# --- データ CSV の読み込み ---
def read_data_csv(path):
    """
    データ CSV を読み込み、ヘッダ（リスト）と全行（リストのリスト）を返す。
    ※ 1列目は識別子（例：分子名）として利用する。
    """
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows or len(rows) < 2:
        raise ValueError("データCSVに十分な行がありません。")
    header = rows[0]
    data = rows[1:]
    return header, data

def main():
    print("=== Buckingham Π-Theorem（CSV 入力・出力版） ===")
    
    # ① 次元定義 CSV の読み込み
    try:
        dims = read_dimension_csv(DIM_CSV_PATH)
    except Exception as e:
        print("次元定義 CSV 読み込みエラー:", e)
        return

    # ② データ CSV の読み込み
    try:
        header, data_rows = read_data_csv(DATA_CSV_PATH)
    except Exception as e:
        print("データ CSV 読み込みエラー:", e)
        return

    # CSV ヘッダから各列名（余分な空白除去）を取得し、列名→列番号の辞書を作成
    header = [h.strip() for h in header]
    header_map = {col: idx for idx, col in enumerate(header)}
    
    # ③ プログラム指定の CSV 変数について、CSVに存在するかチェック
    specified_csv_vars = [CSV_DEPENDENT] + CSV_INDEPENDENT
    for var in specified_csv_vars:
        if var not in header_map:
            print(f"エラー: CSV に指定された変数 '{var}' が存在しません。")
            return

    # ④ 解析対象の全変数は「CSVからの変数」と「定数」とする
    variable_names = specified_csv_vars + list(DIMENSIONED_CONSTANTS.keys())
    print("\n【解析対象の変数】")
    print("CSV変数:", specified_csv_vars)
    print("定数:", list(DIMENSIONED_CONSTANTS.keys()))
    print("合計:", variable_names)

    # ⑤ 各変数の次元情報を次元定義 CSV から取得（すべての変数が次元定義 CSV に存在している必要があります）
    var_dim_vectors = []
    for var in variable_names:
        if var not in dims:
            print(f"エラー: 次元定義 CSV に変数 '{var}' の定義が見つかりません。")
            return
        var_dim_vectors.append(dims[var]["vector"])

    # ⑥ 次元行列 D の作成（行：基本次元, 列：各変数の次元指数）
    num_fundamental = len(list(dims.values())[0]["vector"])
    n = len(variable_names)
    D = sp.Matrix(num_fundamental, n, lambda i, j: var_dim_vectors[j][i])
    print("\n【次元行列 D】")
    sp.pprint(D)

    # ⑦ Buckingham の次元定理に基づき、D の右零空間（nullspace）から Π群の指数ベクトルを算出
    nullspace = D.nullspace()
    num_pi = len(nullspace)
    print(f"\n無次元群の個数（nullspace 次元）: {num_pi}")
    if num_pi == 0:
        print("次元行列の nullspace が 0 次元のため、無次元群は得られません。")
        return

    # ⑧ 各無次元群のシンボリック式（元の変数の関数形）を作成
    print("\n【無次元群のシンボリック式】")
    sym_vars = {var: sp.symbols(var) for var in variable_names}
    # 各Π群について (シンボリック文字列, シンボリック式, 指数ベクトルのリスト) のタプルをリストに保持
    pi_list = []
    for idx, vec in enumerate(nullspace, start=1):
        expr = 1
        # vec は sympy の Matrix なのでリストに変換
        vec_list = [vec[i] for i in range(vec.rows)]
        term_list = []
        for j, exponent in enumerate(vec_list):
            if exponent != 0:
                expr *= sym_vars[variable_names[j]] ** exponent
                term_list.append(f"{variable_names[j]}^({sp.nsimplify(exponent)})")
        pi_str = " * ".join(term_list) if term_list else "1"
        print(f"Π{idx} = {pi_str}")
        pi_list.append( (pi_str, expr, vec_list) )

    # ⑧.1 目的変数を含む無次元群が存在する場合、【目的変数以外の部分】と【目的変数を含まない他の無次元群】を組み合わせて、
    #      目的変数と同じ次元を持つ新たな変数（New_Tb）を生成する
    new_variable_expr = None
    dep_idx = variable_names.index(CSV_DEPENDENT)
    dep_candidates = []      # 目的変数を含む π 群
    non_dep_candidates = []  # 目的変数を含まない π 群
    for (pi_str, expr, vec_list) in pi_list:
        if vec_list[dep_idx] != 0:
            dep_candidates.append((pi_str, expr, vec_list))
        else:
            non_dep_candidates.append((pi_str, expr, vec_list))
    
    if dep_candidates and non_dep_candidates:
        # ここでは、最初に見つかった目的変数を含む π 群を使用
        cand_pi_str, cand_expr, cand_vec = dep_candidates[0]
        a = cand_vec[dep_idx]  # 目的変数 Tb の指数
        # 目的変数部分を除いた項を others_expr とする
        others_expr = cand_expr / (sym_vars[CSV_DEPENDENT] ** a)
        # 目的変数を含まない π 群から乗数候補をひとつ選ぶ（ここでは最初のもの）
        mult_pi_str, mult_expr, mult_vec = non_dep_candidates[0]
        # 新たな変数 New_Tb を以下のように定義：
        #     New_Tb = (mult_expr * others_expr)^{-1/a}
        # となると、次元は
        #    ( (mult_expr * others_expr) )^{-1/a} ~ Tb
        new_variable_expr = (mult_expr * others_expr) ** (-1/a)
        new_variable_expr = sp.simplify(new_variable_expr)
        print("\n【新たな変数のシンボリック式】")
        print(f"New_{CSV_DEPENDENT} = {new_variable_expr}")
    else:
        print("\n目的変数を含む π 群および目的変数を含まない π 群が揃わなかったため、新たな変数の生成は行いません。")

    # ⑨ 各分子について、Π群の数値評価を計算する
    #     出力CSVは以下の形式とする：
    #       ヘッダ行: ["Molecule", "Pi1", "Pi2", …, (必要なら "New_Tb")]
    #       2行目: ["Symbolic", <Π1のシンボリック表現>, <Π2のシンボリック表現>, …, (必要なら New_Tb の表現)]
    #       以降: 各分子の識別子と各Π群の数値評価（および New_Tb の評価）
    out_rows = []
    header_out = ["Molecule"] + [f"Pi{i}" for i in range(1, num_pi+1)]
    if new_variable_expr is not None:
        header_out.append(f"New_{CSV_DEPENDENT}")
    out_rows.append(header_out)
    symbolic_row = ["Symbolic"] + [pi_str for (pi_str, _, _) in pi_list]
    if new_variable_expr is not None:
        symbolic_row.append(f"New_{CSV_DEPENDENT} = {sp.pretty(new_variable_expr)}")
    out_rows.append(symbolic_row)

    # ⑩ 各データ行（各分子）について、CSV から指定した変数の値と定数の値を使って数値評価
    for row_index, row in enumerate(data_rows, start=1):
        # row[0] : 識別子（分子名など）
        subs_dict = {}
        # CSVからの変数の値（各変数は header_map から取り出す）
        for var in specified_csv_vars:
            try:
                subs_dict[var] = float(row[ header_map[var] ])
            except Exception as e:
                print(f"行 {row_index} で変数 '{var}' の数値変換エラー: {e}")
                continue
        # 定数の値を追加
        for const, val in DIMENSIONED_CONSTANTS.items():
            subs_dict[const] = float(val)
        
        # 各Π群の数値評価
        pi_values = []
        for (_, expr, _) in pi_list:
            try:
                val = sp.N(expr.subs(subs_dict))
            except Exception as e:
                print(f"行 {row_index} で Π群評価エラー: {e}")
                val = "Error"
            pi_values.append(val)
        out_row = [row[0]] + [str(pi_val) for pi_val in pi_values]
        
        # 新たな変数 New_Tb の評価（生成できている場合）
        if new_variable_expr is not None:
            try:
                new_val = sp.N(new_variable_expr.subs(subs_dict))
            except Exception as e:
                print(f"行 {row_index} で New_{CSV_DEPENDENT} の評価エラー: {e}")
                new_val = "Error"
            out_row.append(str(new_val))
        out_rows.append(out_row)

    # ⑪ 結果を書き込む（出力CSVファイルを生成）
    with open(OUTPUT_CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)
    print(f"\n各分子のΠ群の数値評価（および生成した場合 New_{CSV_DEPENDENT} の評価）を '{OUTPUT_CSV_PATH}' に出力しました。")

if __name__ == "__main__":
    main()