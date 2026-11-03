# -*- coding: utf-8 -*-
"""
一键运行版（按男女分别输出）：2010-2021 年度生命表 + 中心死力
================================================================
- 读取列：性别、到达年龄（Age）、观察年度（Year）、理赔数（Claims）、暴露数（Exposure）
- 自动按 性别 分开计算并导出两个 Excel：F、M（若文件中只存在一种性别，就只输出那一种）
- qx = 理赔数 / 暴露数（裁剪到 [0, 1)）；mu = q/(1 - q/2)（Greville/Kendall；也常被叫作“寇尔德曼公式”近似）
- 每个输出包含：2010..2021 各年 Sheet + Summary_long + Summary_qx_wide + Summary_mu_wide

使用方法：直接运行本文件即可；默认路径找不到会弹出文件选择框。
依赖：pandas、openpyxl

作者：ChatGPT
版本：v1.0-gender-split
"""

import os
import math
from typing import Optional

try:
    import pandas as pd
except Exception as e:
    raise SystemExit("缺少依赖：需要安装 pandas（以及 openpyxl）。请先安装后再运行。原始错误：" + str(e))

# ---------- 可按需修改（无需命令行） ----------
INPUT_FILE = r"C:\Users\user\Desktop\死亡率数据\生命表数据\【选题1数据】某保险产品死亡率（附件1）.xlsx"
YEAR_FROM = 2010
YEAR_TO = 2021
METHOD = "kendall"  # 固定使用 Greville/Kendall 近似（'kendall'）；如需对数法可改为 'log'
# ---------------------------------------------


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
    """q -> 中心死力 mu"""
    if q is None or (isinstance(q, float) and math.isnan(q)):
        return float("nan")
    q = min(max(float(q), 0.0), 0.999999)
    if method.lower() == "log":
        return -math.log(1.0 - q)
    denom = (1.0 - q / 2.0)
    if denom <= 0:
        return float("inf")
    return q / denom


def _fallback_file_dialog() -> Optional[str]:
    """默认路径不存在时，弹出选择框"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="请选择包含：性别、到达年龄、观察年度、理赔数、暴露数 的 Excel 文件",
            filetypes=[("Excel files", "*.xlsx *.xls")],
        )
        return path if path else None
    except Exception:
        return None


def process_gender(df: pd.DataFrame, gender_label: str, outdir: str) -> Optional[str]:
    """对某一性别的数据生成输出 Excel，返回路径；若该性别无数据则返回 None"""
    sub = df[df["性别"].astype(str).str.upper() == gender_label.upper()].copy()
    if sub.empty:
        return None

    # 清洗列并转换类型
    sub["Year"] = sub["观察年度"].apply(_to_int_safe)
    sub["Age"] = sub["到达年龄"].apply(_to_int_safe)
    sub["Claims"] = sub["理赔数"].apply(_to_float_nonneg)
    sub["Exposure"] = sub["暴露数"].apply(_to_float_nonneg)
    sub = sub.dropna(subset=["Year", "Age"]).reset_index(drop=True)

    # 过滤年份
    sub = sub[(sub["Year"] >= YEAR_FROM) & (sub["Year"] <= YEAR_TO)].copy()
    if sub.empty:
        return None

    # 同一年份-年龄聚合
    agg = (
        sub.groupby(["Year", "Age"], as_index=False)[["Claims", "Exposure"]]
        .sum()
        .sort_values(["Year", "Age"])
    )

    # 计算 qx / mu
    eps = 1e-12
    agg["qx"] = agg.apply(
        lambda r: min(max((r["Claims"] / max(r["Exposure"], eps)) if r["Exposure"] > 0 else 0.0, 0.0), 0.999999),
        axis=1,
    )
    agg["mu"] = agg["qx"].apply(lambda q: convert_q_to_mu(q, method=METHOD))

    # 输出
    out_path = os.path.join(outdir, f"输出_生命表_{gender_label}_{YEAR_FROM}_{YEAR_TO}_method-{METHOD}.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # 每年一个 Sheet
        for y in range(YEAR_FROM, YEAR_TO + 1):
            tab = agg[agg["Year"] == y][["Age", "Exposure", "Claims", "qx", "mu"]].copy()
            if tab.empty:
                tab = pd.DataFrame(columns=["Age", "Exposure", "Claims", "qx", "mu"])
            tab.to_excel(writer, sheet_name=str(y), index=False)

        # 汇总
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
        # 读取数据（首行为列名）
        df = pd.read_excel(infile, header=0, dtype="object")
        cols = list(df.columns.astype(str))
        needed = ["性别", "到达年龄", "观察年度", "理赔数", "暴露数"]
        missing = [c for c in needed if c not in cols]
        if missing:
            raise SystemExit(f"Excel 缺少必要列：{missing}；实际列={cols}")

        outdir = os.path.dirname(infile) or os.getcwd()
        produced = []

        for g in ["F", "M"]:
            p = process_gender(df, g, outdir)
            if p:
                produced.append(p)

        if not produced:
            raise SystemExit("没有在 2010-2021 年间找到有效的 F 或 M 数据，未生成文件。")

        print("🎉 全部完成。输出文件：")
        for p in produced:
            print(" -", p)

    except Exception as e:
        print("❌ 运行失败：", e)


if __name__ == "__main__":
    run()
