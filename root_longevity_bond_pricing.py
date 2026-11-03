# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 09:41:07 2025

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Longevity-Linked Bond Pricing (based on Mao et al. 2025 pandemic bond framework)
---------------------------------------------------
以长寿风险为触发事件：当实际死亡率低于参考水平时，
发行人需向投资者支付较高收益，或本金折损。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 参数设定 =========
r = 0.02          # 无风险利率
F = 100           # 面值
n = 5             # 债券期限（年）
KA = 0.9          # 附着点：死亡率降到基准的 90%
KD = 0.7          # 脱离点：死亡率降到基准的 70%
alpha, beta, gamma = 0.3237, 0.0680, 0.1092   # 来自 Mao(2025) 的参数集3
simN = 10000      # 模拟次数

# ========= 数据读取 =========
path_qx = r"C:\Users\user\Desktop\死亡率数据\10.12版本\保险人口数据\F_Mortality_Surface_Continuous.xlsx"
qx_df = pd.read_excel(path_qx, index_col=0)   # index=Age, columns=Year
ref_year = qx_df.columns.min()
qx_ref = qx_df.loc[:, ref_year].mean()        # 参考死亡率（例如首年平均）

# ========= Monte Carlo 模拟 =========
# 假设未来 n 年死亡率服从 lognormal 随机扰动
years = qx_df.columns[:n]
qx_mean = qx_df.loc[:, years].mean(axis=0).values
qx_vol = qx_df.loc[:, years].std(axis=0).values

sim_qx = np.random.lognormal(np.log(qx_mean), qx_vol/np.maximum(qx_mean,1e-9), (simN, n))
y_ref = qx_ref

# ========= 计算每次模拟的本金偿付 =========
def loss_ratio(y_t, y_ref, KA, KD):
    return np.maximum(0, np.minimum((y_t - KA*y_ref) / (KD*y_ref - KA*y_ref), 1))

B_all = []
for i in range(simN):
    y_t = sim_qx[i]
    L_t = loss_ratio(y_t, y_ref, KA, KD)
    L = np.sum(L_t)
    B = F * max(1 - L, 0)
    B_all.append(B)
B_all = np.array(B_all)

# ========= 计算定价要素 =========
EL = np.mean(F - B_all)
PFL = np.mean(B_all < F)
CEL = EL / max(PFL, 1e-9)
EER = gamma * (PFL**alpha) * (CEL**beta)
PS = EL/F + EER
C_star = F * (r + PS)

print("=== Longevity Bond Pricing Results ===")
print(f"Expected Loss (EL): {EL/F:.4%}")
print(f"Prob. First Loss (PFL): {PFL:.4%}")
print(f"Cond. Expected Loss (CEL): {CEL/F:.4%}")
print(f"Excess Return (EER): {EER:.4%}")
print(f"Premium Spread (PS): {PS:.4%}")
print(f"Fair Annual Coupon C*: {C_star:.4f}")

# ========= 绘图：本金偿付分布 =========
plt.hist(B_all, bins=50, color='steelblue', alpha=0.7)
plt.xlabel("Principal Repayment B")
plt.ylabel("Frequency")
plt.title("Distribution of Principal Repayment (Longevity Bond)")
plt.show()
