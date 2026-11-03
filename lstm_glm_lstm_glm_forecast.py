# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 09:12:56 2025

@author: Administrator
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许重复加载 OpenMP 库
os.environ["OMP_NUM_THREADS"] = "1"           # 限制 OpenMP 单线程运行
os.environ["MKL_NUM_THREADS"] = "1"           # 限制 MKL 单线程运行

import os
import math
import numpy as np
import pandas as pd

# ========== 路径配置（按需修改） ==========
F_PATH = r"C:\Users\user\Desktop\死亡率数据\10.12版本\保险人口数据\输出_生命表_F_2010_2021_method-kendall.xlsx"
M_PATH = r"C:\Users\user\Desktop\死亡率数据\10.12版本\保险人口数据\输出_生命表_M_2010_2021_method-kendall.xlsx"
YEARS_AHEAD = 100                      # 未来预测年数
OUTDIR = None                          # None 表示与输入文件同目录；也可指定其他目录
# =========================================

import statsmodels.api as sm


def read_summary_long(path: str) -> pd.DataFrame:
    """读取 Summary_long；若缺失则拼接年度 sheet。返回列：Year, Age, Claims, Exposure, qx, mu"""
    xls = pd.ExcelFile(path)
    if "Summary_long" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="Summary_long")
    else:
        parts = []
        for s in xls.sheet_names:
            if str(s).isdigit():
                d = pd.read_excel(xls, sheet_name=s)
                d["Year"] = int(s)
                parts.append(d)
        if not parts:
            raise ValueError(f"{os.path.basename(path)} 未找到 Summary_long 或年份工作表。")
        df = pd.concat(parts, ignore_index=True)

        # 若只有 qx 无 mu，则用 Greville/Kendall 近似补 mu
        if "mu" not in df and "qx" in df:
            q = np.clip(pd.to_numeric(df["qx"], errors="coerce").fillna(0).values, 0, 0.999999)
            df["mu"] = q / (1 - q / 2)

        if "Claims" not in df:
            df["Claims"] = np.nan
        if "Exposure" not in df:
            df["Exposure"] = np.nan

    needed = ["Year", "Age", "Claims", "Exposure", "qx", "mu"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"{os.path.basename(path)} 缺少必要列：{c}")

    df = df[needed].copy()
    df["Year"] = df["Year"].astype(int)
    df["Age"] = df["Age"].astype(int)
    for c in ["Claims", "Exposure", "qx", "mu"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_kt_from_logmu(df: pd.DataFrame) -> pd.Series:
    """以 log(mu) 的年均值构造时间因子 k_t"""
    eps = 1e-12
    df = df.copy()
    df["mu"] = df["mu"].clip(lower=eps)
    df["log_mu"] = np.log(df["mu"])
    kt = df.groupby("Year")["log_mu"].mean().sort_index()
    return kt


def forecast_kt(kt_series: pd.Series, steps: int) -> pd.Series:
    """优先用 PyTorch LSTM 预测 k_t；若不可用则回退到 ARIMA(1,1,0)"""
    try:
        import torch
        import torch.nn as nn

        class TinyLSTM(nn.Module):
            def __init__(self, input_size=1, hidden=8, layers=1):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True)
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        arr = kt_series.values.astype("float32")
        mu = arr.mean()
        sd = arr.std() if arr.std() > 0 else 1.0
        z = (arr - mu) / sd

        window = 3 if len(z) >= 5 else max(1, len(z) - 2)
        X, y = [], []
        for i in range(len(z) - window):
            X.append(z[i:i + window])
            y.append(z[i + window])
        X = torch.tensor(np.array(X)).unsqueeze(-1)  # (n, window, 1)
        y = torch.tensor(np.array(y)).unsqueeze(-1)

        model = TinyLSTM(input_size=1, hidden=8, layers=1)
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        loss_fn = nn.MSELoss()
        model.train()
        epochs = 500
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

        # 递推预测
        model.eval()
        last = z[-window:].tolist()
        preds = []
        for _ in range(steps):
            inp = torch.tensor(last[-window:]).view(1, window, 1)
            with torch.no_grad():
                zhat = model(inp).item()
            preds.append(zhat)
            last.append(zhat)

        preds = np.array(preds) * sd + mu
        years = list(range(int(kt_series.index.max()) + 1, int(kt_series.index.max()) + 1 + steps))
        return pd.Series(preds, index=years)

    except Exception:
        # 回退：ARIMA(1,1,0)
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(kt_series, order=(1, 1, 0))
        res = model.fit()
        f = res.forecast(steps=steps)
        f.index = list(range(int(kt_series.index.max()) + 1, int(kt_series.index.max()) + 1 + steps))
        return f


def fit_glm_and_predict(df: pd.DataFrame, kt_train: pd.Series, kt_future: pd.Series) -> pd.DataFrame:
    """
    泊松 GLM（log 链接，offset=log Exposure），特征 = 年龄独热 + (年龄独热 × k_t) 交互：
      Claims_{x,t} ~ Poisson( Exposure_{x,t} * exp( α_x + β_x * k_t ) )
    预测阶段用 k_t 的未来路径生成 mu_hat，并换算 qx_hat = 1 - exp(-mu_hat)。
    """
    d = df.copy()

    # 如果 Exposure 全为 0 或缺失，用 1 兜底，并用 qx*Exposure 近似 Claims；否则补齐缺失 Claims
    if (d["Exposure"] <= 0).all() or d["Exposure"].isna().all():
        d["Exposure"] = 1.0
        d["Claims"] = np.clip(d["qx"] * d["Exposure"], 0, None)
    else:
        m = d["Claims"].isna()
        d.loc[m, "Claims"] = np.clip(d.loc[m, "qx"] * d.loc[m, "Exposure"], 0, None)

    # 合并 k_t
    d = d.merge(kt_train.rename("kt"), left_on="Year", right_index=True, how="left")
    d = d[(d["Exposure"] > 0) & (~d["kt"].isna())].copy()

    # 设计矩阵：年龄独热 & 与 kt 的交互
    ages = sorted(d["Age"].unique())
    age_d = pd.get_dummies(d["Age"].astype(int), prefix="age", drop_first=False)
    X = pd.concat([age_d, age_d.multiply(d["kt"], axis=0).add_prefix("b_")], axis=1)

    # 强制转换为 float64，彻底清除 object 类型
    X = X.apply(pd.to_numeric, errors="coerce").astype("float64")
    y = pd.to_numeric(d["Claims"], errors="coerce").astype("float64")
    offset = np.log(np.maximum(pd.to_numeric(d["Exposure"], errors="coerce"), 1e-9)).astype("float64")

    # 确保无 NaN
    X = X.fillna(0.0)
    y = y.fillna(0.0)
    offset = offset.fillna(0.0)

    # 确认矩阵数值
    assert X.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all(), "X 含非数值列"
    assert np.issubdtype(y.dtype, np.number), "y 非数值"
    assert np.issubdtype(offset.dtype, np.number), "offset 非数值"

    # 重新生成 numpy 数组，保证 dtype float64
    X = np.asarray(sm.add_constant(X, has_constant="add"), dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)

    # 建模
    glm = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    res = glm.fit(maxiter=200, disp=False)

    # 未来预测网格
    fut_years = sorted(kt_future.index.tolist())
    grid = pd.MultiIndex.from_product([fut_years, ages], names=["Year", "Age"]).to_frame(index=False)
    grid = grid.merge(kt_future.rename("kt_future"), left_on="Year", right_index=True, how="left")

    age_df = pd.get_dummies(grid["Age"].astype(int), prefix="age", drop_first=False)
    Xf = pd.concat([age_df, age_df.multiply(grid["kt_future"], axis=0).add_prefix("b_")], axis=1)
    Xf = Xf.reindex(columns=pd.Index([c for c in age_d.columns] + ["b_" + c for c in age_d.columns]), fill_value=0)
    Xf = Xf.apply(pd.to_numeric, errors="coerce").astype("float64")
    Xf = np.asarray(sm.add_constant(Xf, has_constant="add"), dtype=np.float64)

    mu_lin = res.predict(Xf, offset=np.log(np.ones(len(Xf))))
    mu_hat = np.clip(mu_lin.astype(float), 1e-12, 10.0)
    qx_hat = 1.0 - np.exp(-mu_hat)

    out = grid.copy()
    out["mu_hat"] = mu_hat
    out["qx_hat"] = qx_hat
    return out



def run_one(label: str, path: str) -> str:
    df = read_summary_long(path)
    kt = build_kt_from_logmu(df)
    kt_future = forecast_kt(kt, steps=YEARS_AHEAD)
    pred = fit_glm_and_predict(df, kt, kt_future)

    outdir = OUTDIR or os.path.dirname(path) or os.getcwd()
    outfile = os.path.join(outdir, f"Forecast_{label}_LSTM_GLM_100y.xlsx")
    with pd.ExcelWriter(outfile, engine="openpyxl") as w:
        pred.sort_values(["Year", "Age"]).to_excel(w, sheet_name="Forecast_long", index=False)
        pred.pivot(index="Age", columns="Year", values="qx_hat").to_excel(w, sheet_name="Forecast_qx_wide")
        pred.pivot(index="Age", columns="Year", values="mu_hat").to_excel(w, sheet_name="Forecast_mu_wide")
        kt_df = pd.DataFrame({
            "Year": list(kt.index) + list(kt_future.index),
            "kt": list(kt.values) + list(kt_future.values),
            "type": ["train"] * len(kt) + ["forecast"] * len(kt_future)
        })
        kt_df.to_excel(w, sheet_name="kt_series", index=False)

    print(f"[OK] {label} -> {outfile}")
    return outfile


def main():
    outs = {}
    for label, p in {"F": F_PATH, "M": M_PATH}.items():
        outs[label] = run_one(label, p)
    return outs


if __name__ == "__main__":
    main()
