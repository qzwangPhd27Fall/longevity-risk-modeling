# -*- coding: utf-8 -*-

"""
构建2010-2021年度分年龄死亡率（生命表）并用“寇尔德曼/Greville近似”转换为中心死力
----------------------------------------------------------------------------------
用法（Windows 示例）：
    python build_life_tables.py "C:\Users\user\Desktop\死亡率数据\生命表数据\【选题1数据】某保险产品死亡率（附件1）.xlsx" --year_from 2010 --year_to 2021 --method kendall --out "C:\Users\user\Desktop\死亡率数据\生命表数据\输出_生命表_2010_2021.xlsx"

参数说明：
- 输入 Excel 需包含（或可识别为同义中文列名）以下列：
  年份/Year、年龄/Age、理赔数/赔付数/死亡数/身故数/Claims、暴露数/Exposure/在险人数。
- qx = 理赔数 / 暴露数（自动限制在[0, 1) 区间）
- 中心死力（mu）的转换方法（--method）提供两种：
    1) kendall  -> Greville/Kendall 近似：mu = qx / (1 - qx/2) ；常见中文教材有时称为“寇尔德曼公式”近似。
    2) log      -> 常用连续假设：mu = -ln(1 - qx)
- 输出：单个 Excel 工作簿，每个年份一个 Sheet（例如“2010”），包含 Age, Exposure, Claims, qx, mu。
- 同时生成一个“汇总表 Summary”Sheet：按 年份-年龄 的整齐透视表，便于后续分析。

作者：ChatGPT
版本：v1.0
"""

import argparse
import math
import os
import sys
from typing import Dict, List

import pandas as pd


# 允许识别的列名同义词（不区分大小写与空白差异）
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
    """根据同义词在 df 中找到对应列名；若失败则抛错。"""
    cand = COL_SYNONYMS.get(logical_name, [])
    colmap = {_normalize_col(c): c for c in df.columns}
    for alias in cand:
        key = _normalize_col(alias)
        if key in colmap:
            return colmap[key]
    # 还可尝试包含匹配（容错一点）
    for c in df.columns:
        c_norm = _normalize_col(c)
        for alias in cand:
            if _normalize_col(alias) in c_norm:
                return c
    raise KeyError(f"未在数据中识别到[{logical_name}]所需列。同义词尝试={cand}；实际列={list(df.columns)}")


def _to_int_safe(x):
    try:
        if pd.isna(x):
            return None
        # 年龄通常是整数；保留非负
        xi = int(float(x))
        return None if xi < 0 else xi
    except Exception:
        return None


def _to_float_nonneg(x):
    try:
        if pd.isna(x):
            return 0.0
        v = float(x)
        return 0.0 if v < 0 else v
    except Exception:
        return 0.0


def convert_q_to_mu(q: float, method: str = "kendall") -> float:
    """将一年期死亡率 q 转换为中心死力 mu。
    - kendall：Greville/Kendall 近似 -> mu = q / (1 - q/2)（当 q 较小，近似良好）
    - log：连续假设 -> mu = -ln(1 - q)
    """
    if q is None or math.isnan(q):
        return float("nan")
    q = min(max(q, 0.0), 0.999999)  # 数值安全
    if method.lower() == "log":
        return -math.log(1.0 - q)
    # 默认 kendall
    denom = (1.0 - q / 2.0)
    if denom <= 0:
        return float("inf")
    return q / denom


def build_life_tables(
    infile: str,
    year_from: int = 2010,
    year_to: int = 2021,
    method: str = "kendall",
) -> pd.ExcelWriter:
    if not os.path.exists(infile):
        raise FileNotFoundError(f"找不到输入文件：{infile}")

    # 读入：兼容多表头、自动列名推断
    df = pd.read_excel(infile, header=0, dtype="object")
    if df.empty:
        raise ValueError("输入文件为空或未读取到数据。")

    # 识别列
    col_year = _pick_column(df, "year")
    col_age = _pick_column(df, "age")
    col_claims = _pick_column(df, "claims")
    col_expo = _pick_column(df, "exposure")

    # 选取必要列并清洗
    use = df[[col_year, col_age, col_claims, col_expo]].copy()
    use.columns = ["Year", "Age", "Claims", "Exposure"]
    use["Year"] = use["Year"].apply(_to_int_safe)
    use["Age"] = use["Age"].apply(_to_int_safe)
    use["Claims"] = use["Claims"].apply(_to_float_nonneg)
    use["Exposure"] = use["Exposure"].apply(_to_float_nonneg)
    use = use.dropna(subset=["Year", "Age"]).reset_index(drop=True)

    # 仅保留目标年份
    use = use[(use["Year"] >= year_from) & (use["Year"] <= year_to)].copy()
    if use.empty:
        raise ValueError(f"筛选 {year_from}-{year_to} 后无数据。")

    # 对重复 年份-年龄 的记录进行聚合（求和）
    agg = (
        use.groupby(["Year", "Age"], as_index=False)[["Claims", "Exposure"]]
        .sum()
        .sort_values(["Year", "Age"])
    )

    # 计算 qx 与中心死力 mu
    eps = 1e-12
    agg["qx"] = agg.apply(
        lambda r: min(max((r["Claims"] / max(r["Exposure"], eps)) if r["Exposure"] > 0 else 0.0, 0.0), 0.999999),
        axis=1,
    )
    agg["mu"] = agg["qx"].apply(lambda q: convert_q_to_mu(q, method=method))

    # 输出到单一工作簿：每年一张表 + 汇总
    out_path = _build_default_outpath(infile, year_from, year_to, method)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for y in range(year_from, year_to + 1):
            sub = agg[agg["Year"] == y][["Age", "Exposure", "Claims", "qx", "mu"]].copy()
            if sub.empty:
                # 仍写一个空表头，方便核对
                sub = pd.DataFrame(columns=["Age", "Exposure", "Claims", "qx", "mu"])
            sub.to_excel(writer, sheet_name=str(y), index=False)

        # 汇总面板：透视成宽表（每年一个块）
        # 这里给出一个 tidy 的长表（更通用），以及一个宽表（更适合直观浏览）。
        agg.sort_values(["Year", "Age"]).to_excel(writer, sheet_name="Summary_long", index=False)

        # 宽表：以 Age 为行，多层列（qx, mu 分别透视每年）
        qx_wide = agg.pivot(index="Age", columns="Year", values="qx").sort_index()
        mu_wide = agg.pivot(index="Age", columns="Year", values="mu").sort_index()
        qx_wide.to_excel(writer, sheet_name="Summary_qx_wide")
        mu_wide.to_excel(writer, sheet_name="Summary_mu_wide")

    print(f"已输出：{out_path}")
    return out_path


def _build_default_outpath(infile: str, year_from: int, year_to: int, method: str) -> str:
    base = os.path.splitext(os.path.basename(infile))[0]
    outname = f"输出_生命表_{year_from}_{year_to}_method-{method}.xlsx"
    outdir = os.path.dirname(infile) or os.getcwd()
    return os.path.join(outdir, outname)


def main():
    parser = argparse.ArgumentParser(description="按年度-年龄构建死亡率表并转换中心死力")
    parser.add_argument("infile", help="输入 Excel 文件路径（包含 年份/年龄/理赔数/暴露数）")
    parser.add_argument("--year_from", type=int, default=2010, help="起始年份（含）")
    parser.add_argument("--year_to", type=int, default=2021, help="结束年份（含）")
    parser.add_argument("--method", choices=["kendall", "log"], default="kendall",
                        help="q→mu 转换方式：kendall (mu = q/(1 - q/2)) 或 log (mu = -ln(1-q))")
    parser.add_argument("--out", default=None, help="输出 Excel 路径（默认与输入同目录自动命名）")

    args = parser.parse_args()

    try:
        out_path = build_life_tables(
            infile=args.infile,
            year_from=args.year_from,
            year_to=args.year_to,
            method=args.method,
        )
        if args.out:
            # 如果用户提供 --out，则复制/重命名
            dst = args.out
            # 用 pandas 重新保存到指定路径（避免依赖系统 copy 行为）
            # 重新加载刚才的结果并另存
            xls = pd.ExcelFile(out_path)
            with pd.ExcelWriter(dst, engine="openpyxl") as writer:
                for sheet in xls.sheet_names:
                    pd.read_excel(xls, sheet_name=sheet).to_excel(writer, sheet_name=sheet, index=False)
            print(f"另存为：{dst}")
    except Exception as e:
        print("运行失败：", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
