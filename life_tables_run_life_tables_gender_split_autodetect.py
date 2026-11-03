# -*- coding: utf-8 -*-
"""
一键运行（自动识别表头与工作表）— 按男女分别输出 2010-2021 生命表 + 中心死力
=================================================================================
- 自动遍历 Excel 的所有 Sheet；
- 自动定位表头行（即使前面有空行/说明/合并单元格导致读入为 Unnamed）；
- 识别列：性别、到达年龄（Age）、观察年度（Year）、理赔数（Claims）、暴露数（Exposure），支持同义词；
- 按性别分别导出：F 与 M 两个工作簿；
- qx = 理赔数 / 暴露数（裁剪 [0,1)）；mu = q/(1 - q/2)（Greville/Kendall，“寇尔德曼公式”近似）。

使用：直接运行本文件；默认路径找不到会弹出选择框。
依赖：pandas, openpyxl
"""

import os
import math
from typing import Optional, Dict, List, Tuple

try:
    import pandas as pd
except Exception as e:
    raise SystemExit("缺少依赖：需要安装 pandas（以及 openpyxl）。请先安装后再运行。原始错误：" + str(e))

# -------- 参数 --------
INPUT_FILE = r"C:\Users\user\Desktop\死亡率数据\生命表数据\【选题1数据】某保险产品死亡率（附件1）.xlsx"
YEAR_FROM = 2010
YEAR_TO = 2021
METHOD = "kendall"  # 'kendall' 或 'log'
MAX_SCAN_ROWS = 20   # 每个 Sheet 前多少行用于识别表头
# ----------------------

# 同义词表（大小写/空白不敏感）
SYN: Dict[str, List[str]] = {
    "性别": ["性别", "gender", "sex"],
    "到达年龄": ["到达年龄", "年龄", "age", "达到年龄", "到达年齢"],
    "观察年度": ["观察年度", "年份", "年度", "year", "出险年度", "统计年度", "保单年度"],
    "理赔数": ["理赔数", "赔付数", "死亡数", "身故数", "出险人数", "理赔人数", "死亡人数", "身故人数", "claims"],
    "暴露数": ["暴露数", "在险人数", "风险暴露", "有效暴露", "exposure"],
}

NEEDED = list(SYN.keys())  # 目标逻辑列名


def norm(s: object) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip().lower().replace(" ", "")


def match_header_row(df: pd.DataFrame) -> Optional[int]:
    """在 df（header=None）前 MAX_SCAN_ROWS 行中寻找最像表头的一行，返回行号索引（0-based）"""
    for r in range(min(MAX_SCAN_ROWS, len(df))):
        vals = [norm(v) for v in df.iloc[r, :].tolist()]
        if all(v == "" for v in vals):
            continue
        # 统计本行命中多少个目标列（同义词匹配）
        hit = 0
        for key in NEEDED:
            aliases = [norm(a) for a in SYN[key]]
            if any(a for a in vals if any(alias in a and alias != "" for alias in aliases)):
                hit += 1
        if hit >= 3:  # 命中>=3 认为是表头
            return r
    return None


def rename_to_logical(columns: List[object]) -> Dict[str, str]:
    """从原始列名列表生成 逻辑列 -> 实际列 的映射"""
    actual = [str(c) for c in columns]
    mapping: Dict[str, str] = {}
    for key in NEEDED:
        aliases = [norm(a) for a in SYN[key]]
        # 先精确匹配
        for c in actual:
            if norm(c) in aliases:
                mapping[key] = c
                break
        if key not in mapping:
            # 再包含匹配
            for c in actual:
                c0 = norm(c)
                if any(alias and alias in c0 for alias in aliases):
                    mapping[key] = c
                    break
    return mapping


def read_excel_smart(path: str) -> Tuple[pd.DataFrame, str]:
    """遍历所有 Sheet，返回第一个成功识别到所需列的数据框和 sheet 名"""
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        df0 = pd.read_excel(path, sheet_name=sheet, header=None, dtype="object")
        hdr = match_header_row(df0)
        if hdr is None:
            continue
        df = pd.read_excel(path, sheet_name=sheet, header=hdr, dtype="object")
        mapping = rename_to_logical(list(df.columns))
        if all(k in mapping for k in NEEDED):
            # 只保留必要列，并标准化列名
            use = df[[mapping[k] for k in NEEDED]].copy()
            use.columns = NEEDED  # 标准化列名为逻辑列名
            # 去掉全空行
            use = use.dropna(how="all").reset_index(drop=True)
            if not use.empty:
                return use, sheet
    raise SystemExit("未能在任何 Sheet 的前 {0} 行识别到完整表头（需要列：{1}）。"
                     "请检查是否有合并单元格或列名异常。"
                     .format(MAX_SCAN_ROWS, NEEDED))


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
    if q is None or (isinstance(q, float) and pd.isna(q)):
        return float("nan")
    q = min(max(float(q), 0.0), 0.999999)
    if method.lower() == "log":
        return -math.log(1.0 - q)
    denom = (1.0 - q / 2.0)
    if denom <= 0:
        return float("inf")
    return q / denom


def _fallback_file_dialog() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        p = filedialog.askopenfilename(title="请选择包含：性别、到达年龄、观察年度、理赔数、暴露数 的 Excel 文件",
                                       filetypes=[("Excel files", "*.xlsx *.xls")])
        return p if p else None
    except Exception:
        return None


def process_gender(df: pd.DataFrame, gender_label: str, outdir: str) -> Optional[str]:
    sub = df.copy()
    sub["性别"] = sub["性别"].astype(str).str.strip().str.upper()
    sub = sub[sub["性别"] == gender_label.upper()]
    if sub.empty:
        return None

    sub["Year"] = sub["观察年度"].apply(_to_int_safe)
    sub["Age"] = sub["到达年龄"].apply(_to_int_safe)
    sub["Claims"] = sub["理赔数"].apply(_to_float_nonneg)
    sub["Exposure"] = sub["暴露数"].apply(_to_float_nonneg)
    sub = sub.dropna(subset=["Year", "Age"]).reset_index(drop=True)

    sub = sub[(sub["Year"] >= YEAR_FROM) & (sub["Year"] <= YEAR_TO)].copy()
    if sub.empty:
        return None

    agg = (sub.groupby(["Year", "Age"], as_index=False)[["Claims", "Exposure"]]
              .sum().sort_values(["Year", "Age"]))

    eps = 1e-12
    agg["qx"] = agg.apply(
        lambda r: min(max((r["Claims"] / max(r["Exposure"], eps)) if r["Exposure"] > 0 else 0.0, 0.0), 0.999999),
        axis=1,
    )
    agg["mu"] = agg["qx"].apply(lambda q: convert_q_to_mu(q, method=METHOD))

    out_path = os.path.join(outdir, f"输出_生命表_{gender_label}_{YEAR_FROM}_{YEAR_TO}_method-{METHOD}.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for y in range(YEAR_FROM, YEAR_TO + 1):
            tab = agg[agg["Year"] == y][["Age", "Exposure", "Claims", "qx", "mu"]].copy()
            if tab.empty:
                tab = pd.DataFrame(columns=["Age", "Exposure", "Claims", "qx", "mu"])
            tab.to_excel(writer, sheet_name=str(y), index=False)

        agg.sort_values(["Year", "Age"]).to_excel(writer, sheet_name="Summary_long", index=False)
        qx_wide = agg.pivot(index="Age", columns="Year", values="qx").sort_index()
        mu_wide = agg.pivot(index="Age", columns="Year", values="mu").sort_index()
        qx_wide.to_excel(writer, sheet_name="Summary_qx_wide")
        mu_wide.to_excel(writer, sheet_name="Summary_mu_wide")

    print(f"✅ {gender_label} 输出：{out_path}")
    return out_path


def run():
    infile = INPUT_FILE
    if not os.path.exists(infile):
        print(f"⚠️ 未找到默认文件：{infile}")
        picked = _fallback_file_dialog()
        if not picked:
            raise SystemExit("未选择文件，程序结束。")
        infile = picked
        print(f"已选择：{infile}")

    try:
        df, sheet = read_excel_smart(infile)
        print(f"已识别工作表：{sheet}；列：{list(df.columns)}")

        outdir = os.path.dirname(infile) or os.getcwd()
        produced = []

        for g in ["F", "M"]:
            p = process_gender(df, g, outdir)
            if p:
                produced.append(p)

        if not produced:
            raise SystemExit("没有在所选数据中（2010-2021）找到有效的 F 或 M 记录。")

        print("🎉 完成。输出文件：")
        for p in produced:
            print(" -", p)

    except Exception as e:
        print("❌ 运行失败：", e)


if __name__ == "__main__":
    run()
