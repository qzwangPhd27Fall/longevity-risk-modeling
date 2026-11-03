# -*- coding: utf-8 -*-
"""
一键运行版：构建2010-2021年度分年龄死亡率（生命表）并转换为中心死力
================================================================
使用方法：
- 直接双击运行 / 在 Python 中运行本文件即可。
- 默认会读取下方 INPUT_FILE 指定的路径；若找不到，会自动弹出文件选择框。
- 运行完成后，会在输入文件同目录生成一个 Excel 输出（各年份一个工作表 + 汇总）。

依赖：pandas、openpyxl（常见的 Anaconda/科学计算环境中已自带）。

作者：ChatGPT
版本：v1.1（Standalone）
"""

import os
import math
from typing import Dict, List, Optional

# 仅标准库；数据处理依赖 pandas / openpyxl（写 Excel 用）
try:
    import pandas as pd
except Exception as e:
    raise SystemExit("缺少依赖：需要安装 pandas（以及 openpyxl）。请先安装后再运行。原始错误：" + str(e))

# -------------------- 可按需修改的参数（不需要命令行） --------------------
INPUT_FILE = r"C:\Users\user\Desktop\死亡率数据\生命表数据\【选题1数据】某保险产品死亡率（附件1）.xlsx"
YEAR_FROM = 2010
YEAR_TO = 2021
METHOD = "kendall"   # 可选："kendall" 或 "log"
# ------------------------------------------------------------------------

# 列名同义词（不区分大小写和空白）
COL_SYNONYMS: Dict[str, List[str]] = {
    "year": ["year", "年份", "年度", "出险年度", "统计年度", "保单年度"],
    "age": ["age", "年龄", "age_x", "投保年龄", "被保人年龄"],
    "claims": ["claims", "理赔数", "赔付数", "死亡数", "身故数", "出险人数", "死亡人数", "身故人数", "理赔人数"],
    "exposure": ["exposure", "暴露数", "在险人数", "风险暴露", "有效暴露", "保单暴露", "保额暴露", "中心暴露"],
}


def _normalize_col(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower().replace(" ", "")
    return s


def _pick_column(df: pd.DataFrame, logical_name: str) -> str:
    cand = COL_SYNONYMS.get(logical_name, [])
    colmap = {_normalize_col(c): c for c in df.columns}
    for alias in cand:
        key = _normalize_col(alias)
        if key in colmap:
            return colmap[key]
    # 容错的包含匹配
    for c in df.columns:
        c_norm = _normalize_col(c)
        for alias in cand:
            if _normalize_col(alias) in c_norm:
                return c
    raise KeyError(f"未在数据中识别到[{logical_name}]所需列。同义词尝试={cand}；实际列={list(df.columns)}")


def _to_int_safe(x: object) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        xi = int(float(x))
        return None if xi < 0 else xi
    except Exception:
        return None


def _to_float_nonneg(x: object) -> float:
    try:
        if pd.isna(x):
            return 0.0
        v = float(x)
        return 0.0 if v < 0 else v
    except Exception:
        return 0.0


def convert_q_to_mu(q: float, method: str = "kendall") -> float:
    """q(一年期死亡率) -> 中心死力 mu
    - kendall：Greville/Kendall 近似 mu = q / (1 - q/2)
    - log：    连续假设 mu = -ln(1 - q)
    """
    if q is None or (isinstance(q, float) and math.isnan(q)):
        return float("nan")
    q = min(max(float(q), 0.0), 0.999999)
    if method.lower() == "log":
        return -math.log(1.0 - q)
    denom = (1.0 - q / 2.0)
    if denom <= 0:
        return float("inf")
    return q / denom


def _build_default_outpath(infile: str, year_from: int, year_to: int, method: str) -> str:
    base = os.path.splitext(os.path.basename(infile))[0]
    outname = f"输出_生命表_{year_from}_{year_to}_method-{method}.xlsx"
    outdir = os.path.dirname(infile) or os.getcwd()
    return os.path.join(outdir, outname)


def build_life_tables(
    infile: str,
    year_from: int = 2010,
    year_to: int = 2021,
    method: str = "kendall",
) -> str:
    if not os.path.exists(infile):
        raise FileNotFoundError(f"找不到输入文件：{infile}")

    # 读入
    df = pd.read_excel(infile, header=0, dtype="object")
    if df.empty:
        raise ValueError("输入文件为空或未读取到数据。")

    # 识别列
    col_year = _pick_column(df, "year")
    col_age = _pick_column(df, "age")
    col_claims = _pick_column(df, "claims")
    col_expo = _pick_column(df, "exposure")

    # 精简并清洗
    use = df[[col_year, col_age, col_claims, col_expo]].copy()
    use.columns = ["Year", "Age", "Claims", "Exposure"]
    use["Year"] = use["Year"].apply(_to_int_safe)
    use["Age"] = use["Age"].apply(_to_int_safe)
    use["Claims"] = use["Claims"].apply(_to_float_nonneg)
    use["Exposure"] = use["Exposure"].apply(_to_float_nonneg)
    use = use.dropna(subset=["Year", "Age"]).reset_index(drop=True)

    # 过滤年份
    use = use[(use["Year"] >= year_from) & (use["Year"] <= year_to)].copy()
    if use.empty:
        raise ValueError(f"筛选 {year_from}-{year_to} 后无数据。")

    # 同一年份-年龄的重复记录聚合
    agg = (
        use.groupby(["Year", "Age"], as_index=False)[["Claims", "Exposure"]]
        .sum()
        .sort_values(["Year", "Age"])
    )

    # 计算 qx 与 mu
    eps = 1e-12
    agg["qx"] = agg.apply(
        lambda r: min(max((r["Claims"] / max(r["Exposure"], eps)) if r["Exposure"] > 0 else 0.0, 0.0), 0.999999),
        axis=1,
    )
    agg["mu"] = agg["qx"].apply(lambda q: convert_q_to_mu(q, method=method))

    # 写出
    out_path = _build_default_outpath(infile, year_from, year_to, method)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # 每年一个 Sheet
        for y in range(year_from, year_to + 1):
            sub = agg[agg["Year"] == y][["Age", "Exposure", "Claims", "qx", "mu"]].copy()
            if sub.empty:
                sub = pd.DataFrame(columns=["Age", "Exposure", "Claims", "qx", "mu"])
            sub.to_excel(writer, sheet_name=str(y), index=False)

        # 汇总：长表和宽表
        agg.sort_values(["Year", "Age"]).to_excel(writer, sheet_name="Summary_long", index=False)
        qx_wide = agg.pivot(index="Age", columns="Year", values="qx").sort_index()
        mu_wide = agg.pivot(index="Age", columns="Year", values="mu").sort_index()
        qx_wide.to_excel(writer, sheet_name="Summary_qx_wide")
        mu_wide.to_excel(writer, sheet_name="Summary_mu_wide")

    print(f"✅ 已输出：{out_path}")
    return out_path


def _fallback_file_dialog() -> Optional[str]:
    """如果默认路径不存在，弹出文件选择对话框（tkinter）。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="请选择包含 年份/年龄/理赔数/暴露数 的 Excel 文件",
            filetypes=[("Excel files", "*.xlsx *.xls")],
        )
        return path if path else None
    except Exception:
        # 若 tkinter 不可用，返回 None
        return None


def run_once():
    infile = INPUT_FILE
    if not os.path.exists(infile):
        print(f"⚠️ 未找到默认文件：{infile}")
        picked = _fallback_file_dialog()
        if not picked:
            raise SystemExit("未选择文件，程序结束。")
        infile = picked
        print(f"已选择：{infile}")

    try:
        out_path = build_life_tables(
            infile=infile,
            year_from=YEAR_FROM,
            year_to=YEAR_TO,
            method=METHOD,
        )
        print("🎉 处理完成。")
        print("输出文件位置：", out_path)
    except Exception as e:
        print("❌ 运行失败：", e)


if __name__ == "__main__":
    run_once()
