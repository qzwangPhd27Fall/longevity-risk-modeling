# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 06:43:52 2025

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
三来源死亡率曲面对比（鲁棒版）
- 自动识别 Age×Year→qx 宽表；第三本找不到时自动跳过
- 中文字体（微软雅黑）、更清晰的颜色与角度
- 叠加曲面 / 差值曲面 / 切片对比
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ========= 用户配置 =========
PATHS = [
    r"C:\Users\user\Desktop\死亡率数据\10.12版本\保险人口死亡率预测表格.xlsx",
    r"C:\Users\user\Desktop\死亡率数据\10.12版本\国民人口\国民人口死亡率预测表格.xlsx",
    r"C:\Users\user\Desktop\死亡率数据\China Re Life_AXA - longevity swap RI Proposal_20250917 sent to client.xlsx",
]

# 如知道某文件的具体工作表名，可在此指定：{文件名: ["sheet1", "sheet2", ...]}
SHEET_HINTS = {
    # "China Re Life_AXA - longevity swap RI Proposal_20250917 sent to client.xlsx": ["RI_prem_F", "RI_prem_M"]
}

OUT_DIR = os.path.dirname(PATHS[0]) or os.getcwd()
# ===========================

# ========= 中文字体与显示 =========
matplotlib.rcParams["axes.unicode_minus"] = False
try:
    from matplotlib import font_manager as fm
    yahei = r"C:\Windows\Fonts\msyh.ttc"
    if os.path.exists(yahei):
        fm.fontManager.addfont(yahei)
        matplotlib.rcParams["font.family"] = "Microsoft YaHei"
    else:
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
except Exception:
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]

def _to_int_years(cols):
    new = []
    for c in cols:
        s = str(c).strip()
        # 去掉常见的 "Year", "年份" 等前缀后尝试转 int
        s = s.replace("Year", "").replace("年份", "").replace("年", "").strip()
        try:
            new.append(int(s))
        except:
            new.append(c)
    return new

def _looks_like_year(v):
    try:
        iv = int(str(v).strip())
        return 1800 <= iv <= 2200
    except:
        return False

def _matrix_from_header_none(df0):
    """header=None 情况：首行/首列都是数字就推断为 年×龄/龄×年 矩阵"""
    df = df0.copy()
    # 尝试：第一列为年龄，第一行（除第一格）为年份
    first_col = pd.to_numeric(df.iloc[:,0], errors="coerce")
    first_row = pd.to_numeric(df.iloc[0,1:], errors="coerce")
    if first_col.notna().sum() >= 20 and first_row.notna().sum() >= 10:
        ages = first_col.astype(int)
        years = _to_int_years(df.columns[1:])
        try:
            years = [int(y) for y in years]
            mat = df.iloc[1:,1:].apply(pd.to_numeric, errors="coerce")
            mat.index = ages.iloc[1:].values  # 去掉第一行
            mat.columns = years
            if np.isfinite(mat.to_numpy()).all():
                return mat
        except:
            pass
    return None

def detect_qx_matrix(path):
    """
    返回一个候选字典 {sheet_name: DataFrame(Age×Year→qx)}。
    - 支持列名 Age/年龄
    - 支持首列为年龄、其余列为年份数字
    - 支持 header=None 的纯数表推断
    """
    out = {}
    xls = pd.ExcelFile(path)
    sheets = SHEET_HINTS.get(os.path.basename(path), xls.sheet_names)
    for s in sheets:
        try:
            # 1) 常规读取
            df = pd.read_excel(xls, sheet_name=s)
            # (a) 直接存在 Age/年龄 列
            if "Age" in df.columns:
                df = df.set_index("Age")
            elif "年龄" in df.columns:
                df = df.rename(columns={"年龄": "Age"}).set_index("Age")
            else:
                # (b) 把第一列当作 Age（若明显是数字）
                if df.shape[1] > 1 and pd.api.types.is_numeric_dtype(df.iloc[:,0]):
                    df = df.rename(columns={df.columns[0]: "Age"}).set_index("Age")
                else:
                    df = None

            if df is not None:
                df.columns = _to_int_years(df.columns)
                # 保留整数年份列
                df = df.loc[:, [c for c in df.columns if _looks_like_year(c)]]
                df = df.apply(pd.to_numeric, errors="coerce")
                if df.shape[0] >= 15 and df.shape[1] >= 10 and np.isfinite(df.to_numpy()).all():
                    out[s] = df.astype(float)
                    continue

            # 2) header=None 路线（矩阵型）
            df2 = pd.read_excel(xls, sheet_name=s, header=None)
            mat = _matrix_from_header_none(df2)
            if mat is not None and mat.shape[0] >= 15 and mat.shape[1] >= 10:
                out[s] = mat.astype(float)
        except Exception:
            continue
    return out

def choose_first_matrix(path):
    found = detect_qx_matrix(path)
    if not found:
        print(f"⚠ 未在 {os.path.basename(path)} 找到可用矩阵（已跳过）。")
        return None
    # 选列最多的那张
    sheet, df = max(found.items(), key=lambda kv: kv[1].shape[1])
    return (os.path.basename(path), sheet, df)

def reindex_interp(df, ages, years):
    df2 = df.copy().astype(float)
    df2 = df2.reindex(ages).interpolate(axis=0).bfill(axis=0).ffill(axis=0)
    df2 = df2.reindex(columns=years).interpolate(axis=1).bfill(axis=1).ffill(axis=1)
    return df2

# --- 读取三个来源，自动跳过缺失 ---
picked = []
for p in PATHS:
    pick = choose_first_matrix(p)
    if pick is not None:
        picked.append(pick)

if len(picked) < 2:
    raise RuntimeError("至少需要检测到 2 张 Age×Year→qx 的矩阵才能做对比，请检查文件或在 SHEET_HINTS 指定工作表名。")

# --- 对齐到共同网格 ---
age_sets  = [set(df.index.astype(int)) for _,_,df in picked]
year_sets = [set([int(c) for c in df.columns]) for _,_,df in picked]

common_ages  = sorted(list(set.intersection(*age_sets))) if len(age_sets) >= 2 else sorted(list(age_sets[0]))
common_years = sorted(list(set.intersection(*year_sets))) if len(year_sets) >= 2 else sorted(list(year_sets[0]))
# 交集太小，用并集（并用插值/前后填充）
if len(common_ages) < 30 or len(common_years) < 30:
    common_ages  = sorted(list(set.union(*age_sets)))
    common_years = sorted(list(set.union(*year_sets)))

aligned = []
labels  = []
for fname, sname, df in picked:
    a = reindex_interp(df, common_ages, common_years)
    aligned.append(a.to_numpy())
    labels.append(f"{fname} | {sname}")

A, Y = np.meshgrid(common_ages, common_years, indexing="ij")

# ========= 图1：叠加曲面 =========
fig = plt.figure(figsize=(10, 7), dpi=140)
ax = fig.add_subplot(111, projection="3d")

# 颜色/透明度配置（按数量自适应）
cmaps_all  = [plt.cm.viridis, plt.cm.plasma, plt.cm.cividis, plt.cm.inferno]
alphas_all = [0.85, 0.60, 0.40, 0.30]

for i, Z in enumerate(aligned):
    cmap  = cmaps_all[i % len(cmaps_all)]
    alpha = alphas_all[i % len(alphas_all)]
    norm = (Z - np.nanmin(Z)) / max(np.nanmax(Z) - np.nanmin(Z), 1e-12)
    facecolors = cmap(norm)
    ax.plot_surface(A, Y, Z, rstride=1, cstride=1, facecolors=facecolors,
                    linewidth=0, antialiased=True, shade=False, alpha=alpha)

# 地面投影等高线（帮助读取）
zbase = np.nanmin([np.nanmin(Z) for Z in aligned])
for i, Z in enumerate(aligned):
    ax.contour(A, Y, Z, zdir='z', offset=zbase, cmap=cmaps_all[i % len(cmaps_all)],
               levels=10, linewidths=0.8, alpha=0.7)

ax.set_xlabel("年龄 Age"); ax.set_ylabel("年份 Year"); ax.set_zlabel("死亡率 qx")
ax.set_title("死亡率曲面叠加对比", pad=14)
ax.view_init(elev=28, azim=235)

# 图例（手工代理）
from matplotlib.lines import Line2D
proxies = [Line2D([0],[0], color=cmaps_all[i % len(cmaps_all)](0.7), lw=6) for i in range(len(aligned))]
ax.legend(proxies, labels, loc="upper left", title="数据来源", frameon=True)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "曲面对比_叠加.png")
plt.savefig(out1, dpi=300, bbox_inches="tight")
plt.close(fig)

# ========= 图2：差值曲面（至少需要 2 个来源）=========
def plot_diff(Za, Zb, la, lb, outname):
    Z = Za - Zb
    vmax = np.nanpercentile(np.abs(Z), 98)
    vmin = -vmax
    fig = plt.figure(figsize=(10,7), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.cm.coolwarm
    norm = np.clip((Z - vmin) / (vmax - vmin + 1e-12), 0, 1)
    ax.plot_surface(A, Y, Z, facecolors=cmap(norm), linewidth=0, antialiased=True, shade=False, alpha=0.95)
    ax.contour(A, Y, Z, zdir='z', offset=np.nanmin(Z), cmap=cmap, levels=12, linewidths=0.8, alpha=0.8)
    ax.set_xlabel("年龄 Age"); ax.set_ylabel("年份 Year"); ax.set_zlabel("Δqx")
    ax.set_title(f"差值曲面：{la} − {lb}", pad=14)
    ax.view_init(elev=28, azim=235)
    mappable = matplotlib.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(Z); mappable.set_clim(vmin, vmax)
    cb = plt.colorbar(mappable, shrink=0.6, pad=0.1)
    cb.set_label("Δqx")
    plt.tight_layout()
    plt.savefig(outname, dpi=300, bbox_inches="tight")
    plt.close(fig)

if len(aligned) >= 2:
    plot_diff(aligned[0], aligned[1], labels[0], labels[1], os.path.join(OUT_DIR, "曲面差异_1减2.png"))
if len(aligned) >= 3:
    plot_diff(aligned[0], aligned[2], labels[0], labels[2], os.path.join(OUT_DIR, "曲面差异_1减3.png"))

# ========= 图3：切片对比 =========
sel_ages  = [50, 65, 80]
sel_years = [common_years[0], common_years[len(common_years)//2], common_years[-1]]

# 固定年龄 → 年度剖面
fig = plt.figure(figsize=(10,6), dpi=140)
for a in sel_ages:
    if a not in common_ages: continue
    ia = common_ages.index(a)
    for i, Z in enumerate(aligned):
        ls = ["-", "--", ":", "-."][i % 4]
        plt.plot(common_years, Z[ia, :], ls=ls, lw=2.2, label=f"{labels[i]} | Age {a}")
plt.xlabel("年份 Year"); plt.ylabel("死亡率 qx"); plt.title("切片：固定年龄的时间剖面")
plt.legend(ncol=2, frameon=True); plt.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "切片_固定年龄_时间剖面.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# 固定年份 → 年龄剖面
fig = plt.figure(figsize=(10,6), dpi=140)
for y in sel_years:
    if y not in common_years: continue
    jy = common_years.index(y)
    for i, Z in enumerate(aligned):
        ls = ["-", "--", ":", "-."][i % 4]
        plt.plot(common_ages, Z[:, jy], ls=ls, lw=2.2, label=f"{labels[i]} | Year {y}")
plt.xlabel("年龄 Age"); plt.ylabel("死亡率 qx"); plt.title("切片：固定年份的年龄剖面")
plt.legend(ncol=2, frameon=True); plt.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "切片_固定年份_年龄剖面.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print("完成。输出文件：")
print(" -", out1)
if len(aligned) >= 2:
    print(" -", os.path.join(OUT_DIR, "曲面差异_1减2.png"))
if len(aligned) >= 3:
    print(" -", os.path.join(OUT_DIR, "曲面差异_1减3.png"))
print(" -", os.path.join(OUT_DIR, "切片_固定年龄_时间剖面.png"))
print(" -", os.path.join(OUT_DIR, "切片_固定年份_年龄剖面.png"))
