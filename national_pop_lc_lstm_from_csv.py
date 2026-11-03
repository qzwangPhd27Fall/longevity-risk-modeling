# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 06:20:31 2025

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
LC–LSTM for National Population (CSV 1981–2021)
===============================================
- 输入：两张 CSV（男/女），包含 1981–2021 年的年龄别中心死亡率（mu）或死亡率（qx）
- 方法：Lee–Carter 分解 log(mu) = a_x + b_x k_t
      k_t = 线性趋势 + 残差；残差由 LSTM 预测（无 torch 时自动 AR(1)）
- 归一化：sum_x b_x = 1，mean_t k_t = 0（在训练段）
- 输出：模板格式宽表（RI_prem_M / RI_prem_F），另含长表与 kt_series
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd

# ====== 路径配置（按需修改）======
PATH_M = r"C:\Users\user\Desktop\死亡率数据\10.12版本\国民人口\中心死亡率_男1981-2021.csv"
PATH_F = r"C:\Users\user\Desktop\死亡率数据\10.12版本\国民人口\中心死亡率_女1981-2021.csv"

YEARS_FUTURE = 100   # 预测未来年数
AGE_MAX_OUT  = 105   # 输出年龄上限（可改成 130）
OUT_DIR = os.path.dirname(PATH_M) or os.getcwd()
# =================================

# 避免 OpenMP 冲突（PyTorch / MKL 等）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

def log(*a): print(*a, flush=True)

# ---------- 读取 CSV 工具 ----------
def read_csv_flex(path: str) -> pd.DataFrame:
    """
    尝试用 utf-8-sig / gbk 读取；自动识别 Year/Age/mu/qx 列名（中英文常用别名），
    输出列：Year, Age, mu（若只有 qx 则用 Greville/Kendall 近似转换）
    """
    def try_read(_enc):
        return pd.read_csv(path, encoding=_enc)
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            df = try_read(enc)
            break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError(f"读取失败：{path}")

    # 标准化列名：尽可能识别
    cols_low = {c.lower(): c for c in df.columns}
    # 可能的列别名
    year_alias = [c for c in df.columns if any(k in str(c).lower() for k in ["year","年份","年度","t","yr"])]
    age_alias  = [c for c in df.columns if any(k in str(c).lower() for k in ["age","年龄","到达年龄","x"])]
    mu_alias   = [c for c in df.columns if any(k in str(c).lower() for k in ["mu","中心死力","中心死亡率","mx","m(x)"])]
    qx_alias   = [c for c in df.columns if any(k in str(c).lower() for k in ["qx","死亡率","q(x)","死亡概率"])]

    if not year_alias: raise ValueError(f"{os.path.basename(path)} 找不到 Year 列")
    if not age_alias:  raise ValueError(f"{os.path.basename(path)} 找不到 Age 列")

    year_col = year_alias[0]
    age_col  = age_alias[0]

    df = df.rename(columns={year_col: "Year", age_col: "Age"})
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    df["Age"]  = pd.to_numeric(df["Age"], errors="coerce").astype(int)

    if mu_alias:
        mu_col = mu_alias[0]
        df["mu"] = pd.to_numeric(df[mu_col], errors="coerce")
    elif qx_alias:
        q_col = qx_alias[0]
        q = pd.to_numeric(df[q_col], errors="coerce")
        q = np.clip(q.values.astype(float), 0.0, 0.999999)  # 防止越界
        df["mu"] = q/(1 - q/2.0)           # Greville/Kendall 近似：将 qx 转中心死力
    else:
        raise ValueError(f"{os.path.basename(path)} 未找到 mu / qx 列")

    # 清洗
    df["mu"] = pd.to_numeric(df["mu"], errors="coerce").clip(lower=1e-12)
    df = df.dropna(subset=["Year","Age","mu"])
    # 保留 1981–2021
    df = df[(df["Year"] >= 1981) & (df["Year"] <= 2021)].copy()
    if df.empty:
        raise ValueError(f"{os.path.basename(path)} 在 1981–2021 区间没有数据。")
    return df[["Year","Age","mu"]]

# ---------- LC 分解 ----------
def lee_carter_from_mu(df: pd.DataFrame):
    """
    稳健版 LC on log(mu)
    - 先构造 Age×Year 的 mu 矩阵
    - 对 log(mu) 做二维插值与截尾，剔除完全空的年龄/年份
    - SVD 多重兜底：np.linalg.svd -> sklearn TruncatedSVD -> scipy.sparse.linalg.svds
    - 标准化：sum_x b_x = 1，mean_t k_t = 0
    """
    # 1) Pivot 并按年龄/年份排序
    M = df.pivot(index="Age", columns="Year", values="mu").sort_index().sort_index(axis=1)

    # 2) 先把 mu 做安全裁剪，防止 log(0) / 过大溢出
    M = M.astype(float).clip(lower=1e-12, upper=1e3)

    # 3) 补齐缺失（二维）：先沿年龄方向插值，再沿年份方向插值，最后用列/行中位数兜底
    M = (
        M.interpolate(axis=0, limit_direction="both")  # 按年龄插值
         .interpolate(axis=1, limit_direction="both")  # 按年份插值
    )
    if M.isna().any().any():
        # 列/行中位数兜底
        col_med = M.median(axis=0, numeric_only=True)
        M = M.fillna(col_med)
        if M.isna().any().any():
            row_med = M.median(axis=1, numeric_only=True)
            for r in M.index:
                M.loc[r] = M.loc[r].fillna(row_med.loc[r])

    # 4) 去掉仍然全空/全常数的年龄或年份（极端异常数据保护）
    #    （常数列/行会让中心化后几乎为0，造成数值不稳定）
    # 年份维度方差过小则剔除
    col_var = M.var(axis=0, numeric_only=True)
    keep_cols = col_var[col_var > 1e-16].index
    M = M[keep_cols]

    # 年龄维度方差过小则剔除
    row_var = M.var(axis=1, numeric_only=True)
    keep_rows = row_var[row_var > 1e-16].index
    M = M.loc[keep_rows]

    if M.shape[0] < 2 or M.shape[1] < 2:
        raise ValueError("清洗后有效的年龄/年份太少，无法做 LC 分解。")

    # 5) log(mu) 并对极端值做截尾（winsorize）
    logM = np.log(M)
    q_low, q_high = np.nanpercentile(logM.values, [0.1, 99.9])
    logM = logM.clip(lower=q_low, upper=q_high)

    # 6) 计算 a_x 与中心化
    a_x = logM.mean(axis=1)
    centered = logM.sub(a_x, axis=0)

    # 7) SVD 多重兜底
    A = centered.values.astype(np.float64, copy=False)

    def svd_primary(A):
        # 首选：NumPy LAPACK SVD
        return np.linalg.svd(A, full_matrices=False)

    def svd_fallback_sklearn(A):
        # 备选：TruncatedSVD(k=1)
        from sklearn.decomposition import TruncatedSVD
        tsvd = TruncatedSVD(n_components=1, algorithm="randomized", n_iter=20, random_state=42)
        comp = tsvd.fit_transform(A)      # (n_age, 1) ~ U*s
        Vt   = tsvd.components_          # (1, n_year)
        s    = np.array([tsvd.singular_values_[0]], dtype=np.float64)
        # 还原 U：comp = U * s
        U = comp / s[0]
        return U, s, Vt

    def svd_fallback_scipy(A):
        # 备选：SciPy 稀疏 SVD
        from scipy.sparse.linalg import svds
        u, s, vt = svds(A, k=1, which="LM")
        # svds 返回按升序，k=1 就是最大特征
        return u, s, vt

    try:
        U, s, Vt = svd_primary(A)
    except Exception:
        try:
            U, s, Vt = svd_fallback_sklearn(A)
        except Exception:
            U, s, Vt = svd_fallback_scipy(A)

    # 8) 取第一主成分并标准化
    b_raw = U[:, 0]
    k_raw = s[0] * Vt[0, :]

    denom = np.sum(b_raw)
    if abs(denom) < 1e-12:
        denom = 1.0
    b_x = b_raw / denom
    k_t = k_raw * denom
    k_t = k_t - np.mean(k_t)

    # 转回 Series，索引对齐清洗后的行列
    b_x = pd.Series(b_x, index=centered.index, name="b_x")
    k_t = pd.Series(k_t, index=centered.columns, name="k_t")
    a_x.name = "a_x"

    return a_x, b_x, k_t


# ---------- k_t 预测：线性趋势 + LSTM 残差（AR(1) 兜底）----------
def forecast_k(kt: pd.Series, steps: int) -> pd.Series:
    # 线性趋势拟合
    t = np.arange(len(kt), dtype=float)
    A = np.column_stack([np.ones_like(t), t])
    a, b = np.linalg.lstsq(A, kt.values.astype(float), rcond=None)[0]
    trend_in = a + b * t
    resid = kt.values.astype(float) - trend_in

    def lstm_forecast_resid(resid_arr, steps):
        import torch
        import torch.nn as nn

        class LSTMres(nn.Module):
            def __init__(self, hidden=8):
                super().__init__()
                self.lstm = nn.LSTM(1, hidden, batch_first=True)
                self.fc = nn.Linear(hidden, 1)
            def forward(self, x):
                y, _ = self.lstm(x)
                return self.fc(y[:, -1, :])

        x = resid_arr.astype("float32")
        mu = float(x.mean()); sd = float(x.std() if x.std() > 0 else 1.0)
        z = (x - mu) / sd
        window = 5 if len(z) >= 8 else (3 if len(z) >= 5 else max(1, len(z)-2))

        # 太短直接返回 0 残差
        if len(z) <= window:
            return np.zeros(steps, dtype=float)

        X, y = [], []
        for i in range(len(z) - window):
            X.append(z[i:i+window]); y.append(z[i+window])
        X = torch.tensor(np.array(X)).unsqueeze(-1)
        y = torch.tensor(np.array(y)).unsqueeze(-1)

        model = LSTMres(hidden=12)
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        loss = nn.MSELoss()
        model.train()
        for _ in range(600):
            opt.zero_grad(); pred = model(X)
            l = loss(pred, y); l.backward(); opt.step()

        # 递推预测
        model.eval()
        last = z[-window:].tolist(); preds = []
        for _ in range(steps):
            inp = torch.tensor(last[-window:]).view(1, window, 1)
            with torch.no_grad():
                zh = model(inp).item()
            preds.append(zh); last.append(zh)
        return (np.array(preds)*sd + mu).astype(float)

    try:
        resid_fore = lstm_forecast_resid(resid, steps)
    except Exception:
        # 无 torch 或失败，用 AR(1) 兜底
        from statsmodels.tsa.arima.model import ARIMA
        m = ARIMA(resid, order=(1,0,0)).fit()
        resid_fore = m.forecast(steps=steps).values.astype(float)

    # 合成未来 k_t
    last_t = len(kt) - 1
    t_future = np.arange(last_t+1, last_t+1+steps, dtype=float)
    trend_future = a + b * t_future
    k_future = trend_future + resid_fore
    years = list(range(int(kt.index.max())+1, int(kt.index.max())+1+steps))
    return pd.Series(k_future, index=years, name="k_t_forecast")

# ---------- 重构与导出 ----------
def reconstruct_qx(a_x: pd.Series, b_x: pd.Series, k_all: pd.Series) -> pd.DataFrame:
    ages = a_x.index.values
    years = k_all.index.values
    ax = a_x.values.reshape(-1,1)
    bx = b_x.values.reshape(-1,1)
    kt = k_all.values.reshape(1,-1)
    logM = ax + bx * kt
    mu = np.exp(logM).clip(1e-12, None)
    qx = 1.0 - np.exp(-mu)
    out = pd.DataFrame(qx, index=ages, columns=years)
    out.index.name = "Age"; out.columns.name = "Year"
    return out

def ensure_contiguous_ages(qx: pd.DataFrame, age_min=None, age_max=AGE_MAX_OUT) -> pd.DataFrame:
    qx = qx.copy().sort_index()
    if age_min is None:
        age_min = int(qx.index.min())
    full_idx = pd.Index(range(age_min, age_max+1), name="Age")
    qx = qx.reindex(full_idx)
    qx = qx.interpolate(axis=0).bfill(axis=0).ffill(axis=0)
    return qx

def lc_lstm_from_csv(path_csv: str, label: str) -> str:
    df = read_csv_flex(path_csv)
    a_x, b_x, k_t = lee_carter_from_mu(df)
    k_future = forecast_k(k_t, YEARS_FUTURE)
    k_all = pd.concat([k_t, k_future])

    qx = reconstruct_qx(a_x, b_x, k_all)
    qx = ensure_contiguous_ages(qx, age_min=int(a_x.index.min()), age_max=AGE_MAX_OUT)

    out = os.path.join(OUT_DIR, f"Forecast_{label}_LC_LSTM_fromCSV_100y.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        qx.to_excel(w, sheet_name=f"RI_prem_{label}")
        qx.stack().rename("qx").reset_index().to_excel(w, sheet_name=f"RI_prem_{label}_long", index=False)
        kt_df = pd.DataFrame({"Year": list(k_t.index) + list(k_future.index),
                              "kt":  list(k_t.values) + list(k_future.values),
                              "type": ["train"]*len(k_t) + ["forecast"]*len(k_future)})
        kt_df.to_excel(w, sheet_name="kt_series", index=False)
    log(f"[OK] {label} -> {out}")
    return out

def main():
    try:
        log("=== LC–LSTM 预测（国民人口 CSV 1981–2021）启动 ===")
        outM = lc_lstm_from_csv(PATH_M, "M")
        outF = lc_lstm_from_csv(PATH_F, "F")
        log("已生成：")
        log("  ", outM)
        log("  ", outF)
    except Exception as e:
        log("❌ 运行失败：", e)
        traceback.print_exc()
        # 若双击运行，避免窗口秒关
        try: input("按回车退出…")
        except Exception: pass

if __name__ == "__main__":
    main()
