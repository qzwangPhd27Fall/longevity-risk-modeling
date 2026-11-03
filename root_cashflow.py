# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 22:35:38 2025

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
寿险现金流预测模型（核心计算版）
不导出Excel，仅在Spyder中显示结果预览
"""

import pandas as pd
import numpy as np
import os

# =====================================================
# 1️⃣ 文件路径
# =====================================================
BASE_PATH = r"C:\Users\user\Desktop\死亡率数据\10.12版本"
ASSUMP_PATH = os.path.join(BASE_PATH, "模型点.xlsx")
MORT_PATH = os.path.join(BASE_PATH, "mort_smooth_diag.xlsx")
CV_PATH = os.path.join(BASE_PATH, "CV.xlsx")

# =====================================================
# 2️⃣ 读取基础表格
# =====================================================
print("正在读取输入文件...")

# Assump
assump_raw = pd.read_excel(ASSUMP_PATH, header=None)
title_row = assump_raw[
    assump_raw.apply(lambda x: x.astype(str).str.contains("Age", case=False)).any(axis=1)
].index[0]
assump = pd.read_excel(ASSUMP_PATH, skiprows=title_row)
assump = assump[assump["Age"] != "Age"]
cols_to_num = ["Age", "SA", "sex", "Prem", "PremTerm", "DurationIF_M", "BenefitTerm"]
assump[cols_to_num] = assump[cols_to_num].apply(pd.to_numeric, errors="coerce")
assump = assump.dropna(subset=["Age", "Prem", "BenefitTerm"]).reset_index(drop=True)
print(f"✅ 模型点数：{len(assump)}")

# Mort
mort = pd.read_excel(MORT_PATH)
mort.columns = ["Age", "Male", "Female"]

# CV
cv = pd.read_excel(CV_PATH)
cv.columns = ["Age", "sex", "NP", "PolicyYear", "CV_Prem", "CV_E", "SurvivalBenefit"]

print("文件读取完成\n")

# =====================================================
# 3️⃣ 查表函数
# =====================================================
def get_mort_rate(age, sex, mort_df):
    # 统一类型
    age = int(round(age))
    col = "Male" if str(sex).lower() in ["1", "m", "male", "男"] else "Female"
    
    max_age = mort_df["Age"].max()
    min_age = mort_df["Age"].min()
    
    if age >= max_age:
        return 1.0  # 超出年龄范围直接强制为1
    elif age < min_age:
        return float(mort_df.iloc[0][col])
    
    cond = mort_df["Age"].astype(int) == age
    vals = mort_df.loc[cond, col].values
    if len(vals) == 0:
        # 查不到：改为平滑插值但不重复打印
        return float(np.interp(age, mort_df["Age"], mort_df[col]))
    else:
        return float(vals[0])



def lookup_cv(age, sex, np_term, policy_year, col="CV_E"):
    """现金价值查表"""
    row = cv.query("Age==@age and sex==@sex and NP==@np_term and PolicyYear==@policy_year")
    if not row.empty and not np.isnan(row[col].values[0]):
        return float(row[col].values[0])
    return 0.0

# =====================================================
# 4️⃣ 单模型点现金流计算
# =====================================================
def simulate_cf(policy, mort_df, cv_df, lapse_rate_annual=0.04, ratio=1.0):
    age0 = int(policy["Age"])
    sex = int(policy["sex"])
    sa = float(policy["SA"])
    prem = float(policy["Prem"])
    prem_term = int(policy["PremTerm"])
    np_term = int(policy["DurationIF_M"])
    benefit_term = int(policy["BenefitTerm"])

    months = benefit_term * 12
    df = pd.DataFrame({"Month": np.arange(1, months + 1)})
    df["PolicyYear"] = np.ceil(df["Month"] / 12).astype(int)
    df["Age"] = age0 + np.floor((df["Month"] - 1) / 12)

    # 死亡率（月化）
    df["qx_year"] = df["Age"].apply(lambda x: get_mort_rate(x, sex, mort_df))
    df["qx_month"] = 1 - (1 - df["qx_year"]) ** (1 / 12)

    # 退保率（月化）
    df["ql"] = 1 - (1 - lapse_rate_annual) ** (1 / 12)

    # 在保人数
    df["policy_e"] = 0.0
    df.loc[0, "policy_e"] = 1.0
    for i in range(1, len(df)):
        df.loc[i, "policy_e"] = max(df.loc[i - 1, "policy_e"] *
                                    (1 - df.loc[i - 1, "qx_month"] - df.loc[i - 1, "ql"]), 0)

    # 死亡 & 退保人数
    df["no_death"] = df["policy_e"].shift(1, fill_value=1) * df["qx_month"]
    df["no_lapse"] = df["policy_e"].shift(1, fill_value=1) * df["ql"]

    # 现金价值查表 + 插值
    df["cv_e"] = df.apply(lambda x: lookup_cv(age0, sex, np_term, x["PolicyYear"], "CV_E") * prem / 1000, axis=1)
    df["cv_b"] = df.apply(lambda x: lookup_cv(age0, sex, np_term, x["PolicyYear"] - 1, "CV_E") * prem / 1000, axis=1)
    df["cv_prem"] = df.apply(lambda x: lookup_cv(age0, sex, np_term, x["PolicyYear"], "CV_Prem") * prem / 1000, axis=1)
    frac = ((df["Month"] - 1) % 12 + 1) / 12
    df["cv_i"] = (df["cv_b"] + df["cv_prem"]) * (1 - frac) + df["cv_e"] * frac

    # 收支计算
    df["death_benefit"] = np.maximum(np.minimum(np_term, df["PolicyYear"]) * prem, df["cv_i"])
    df["annuity_benefit"] = np.where((df["Age"] >= 60) & (df["Month"] % 12 == 1), sa, 0)
    df["surrender_outgo"] = df["cv_i"] * df["no_lapse"]
    df["prem_income"] = np.where(df["Month"] % 12 == 1, prem, 0)
    df["death_outgo"] = df["death_benefit"] * df["no_death"]
    df["ncf"] = -df["prem_income"] + df["death_outgo"] + df["annuity_benefit"] + df["surrender_outgo"]

    df["PolicyID"] = int(policy["No"])
    return df

# =====================================================
# 5️⃣ 批量计算 & 汇总
# =====================================================
print("开始批量计算现金流\n")
results = []
for i, row in assump.iterrows():
    print(f"→ 正在计算第 {i+1}/{len(assump)} 个模型点 (No={row['No']}) ...")
    res = simulate_cf(row, mort, cv)
    results.append(res)

cf_total = pd.concat(results, ignore_index=True)
print("\n全部模型点计算完成！\n")

# =====================================================
# 6️⃣ 输出结果预览
# =====================================================
pd.set_option("display.max_columns", 30)
print("结果预览：")
print(cf_total.head(20))
print("\n字段数量：", len(cf_total.columns))
print("总记录数：", len(cf_total))
