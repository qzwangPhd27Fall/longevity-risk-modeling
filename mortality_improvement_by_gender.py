import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# 全局配置
# --------------------------
DATA_PATH = r"D:\_云同步\_办公室\_研究中心\长寿风险实验室\死亡率改善率\学生竞赛"
LOCAL_DATA_FILE = "【选题1数据】某保险产品死亡率.xlsx"
YEAR_RANGE = (2010, 2021)
AGE_RANGE = (0, 105)
STR_YEAR = 2019
FORECAST_H = 40

# 修复中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (18, 15)


# --------------------------
# 1. 数据加载
# --------------------------
def load_data():
    print("加载数据...")
    df = pd.read_excel(f"{DATA_PATH}/{LOCAL_DATA_FILE}", sheet_name="data")
    df = df.rename(columns={
        "性别": "Gender",
        "到达年龄": "Age",
        "观察年度": "Year",
        "理赔数": "dx",
        "暴露数": "Ex"
    })
    df['Gender'] = df['Gender'].map({"F": 0, "M": 1})
    # 过滤无效数据和超出范围的年份/年龄
    df = df[(df["Ex"] > 0) & (df["dx"] >= 0) & 
            (df["Year"].between(*YEAR_RANGE)) & 
            (df["Age"].between(*AGE_RANGE))].reset_index(drop=True)
    print(f"有效数据形状: {df.shape}")
    print(f"年份范围: {df['Year'].min()}-{df['Year'].max()}")
    print(f"年龄范围: {df['Age'].min()}-{df['Age'].max()}")
    print(f"性别分布: {df['Gender'].value_counts().to_dict()} (0=女, 1=男)")

    df_female = df[df['Gender'] == 0].copy()
    df_male = df[df['Gender'] == 1].copy()
    return {"female": df_female, "male": df_male}


# --------------------------
# 2. 核心计算：修复数据长度匹配问题
# --------------------------
def calc_weighted_mortality(df_gender):
    # --------------------------
    # （1）分年龄-年份的死亡率矩阵（确保维度对齐）
    # --------------------------
    # 获取实际存在的年份和年龄
    actual_years = sorted(df_gender["Year"].unique())
    actual_ages = sorted(df_gender["Age"].unique())
    print(f"实际年份数: {len(actual_years)}, 实际年龄数: {len(actual_ages)}")

    # 分组聚合
    age_year_agg = df_gender.groupby(["Age", "Year"]).agg(
        total_dx=("dx", "sum"),
        total_Ex=("Ex", "sum")
    ).reset_index()
    age_year_agg = age_year_agg[age_year_agg["total_Ex"] > 0].reset_index(drop=True)
    age_year_agg["mx"] = age_year_agg["total_dx"] / age_year_agg["total_Ex"]

    # 构建矩阵（确保所有年份/年龄都被包含）
    mx_matrix = age_year_agg.pivot_table(
        index="Age",
        columns="Year",
        values="mx",
        fill_value=0
    ).reindex(
        index=actual_ages,
        columns=actual_years,
        fill_value=0
    )

    # --------------------------
    # （2）分年龄汇总
    # --------------------------
    age_agg = df_gender.groupby("Age").agg(
        total_dx_all=("dx", "sum"),
        total_Ex_all=("Ex", "sum")
    ).reset_index()
    age_agg = age_agg[age_agg["total_Ex_all"] > 0].reset_index(drop=True)
    age_agg["total_mx"] = age_agg["total_dx_all"] / age_agg["total_Ex_all"]

    # --------------------------
    # （3）分年龄汇总改善率（修复长度匹配）
    # --------------------------
    age_mi = []
    for age in actual_ages:
        # 确保该年龄的年份数据完整
        if age not in mx_matrix.index:
            continue
        age_mx = mx_matrix.loc[age].values
        # 确保暴露数数组长度与年份一致
        age_Ex = df_gender[df_gender["Age"] == age].groupby("Year")["Ex"].sum()
        age_Ex = age_Ex.reindex(actual_years, fill_value=0).values
        
        # 计算逐年改善率
        mi_series = [np.nan]  # 第一年无改善率
        for t in range(1, len(actual_years)):
            if age_mx[t-1] == 0:
                mi_series.append(0)
            else:
                mi_series.append(1 - age_mx[t] / age_mx[t-1])
        
        # 加权平均（确保权重长度匹配）
        weight_sum = age_Ex[1:].sum()
        weighted_mi = np.average(mi_series[1:], weights=age_Ex[1:]) if weight_sum > 0 else 0
        age_mi.append({"Age": age, "weighted_avg_mi": weighted_mi})
    age_mi_df = pd.DataFrame(age_mi)

    # --------------------------
    # （4）总体死亡率改善率（所有年龄汇总）- 修复长度问题
    # --------------------------
    total_yearly = df_gender.groupby("Year").agg(
        total_dx_year=("dx", "sum"),
        total_Ex_year=("Ex", "sum")
    ).reset_index()
    total_yearly["total_mx_year"] = total_yearly["total_dx_year"] / total_yearly["total_Ex_year"]

    # 修复：初始化与数据长度一致的数组
    total_yearly["total_MI_year"] = np.full(len(total_yearly), np.nan)
    
    # 计算总体改善率
    for t in range(1, len(total_yearly)):
        prev_mx = total_yearly.iloc[t-1]["total_mx_year"]
        curr_mx = total_yearly.iloc[t]["total_mx_year"]
        if prev_mx > 0:
            total_yearly.loc[t, "total_MI_year"] = 1 - curr_mx / prev_mx
        else:
            total_yearly.loc[t, "total_MI_year"] = 0

    return {
        "mx_matrix": mx_matrix.values,
        "ages": actual_ages,
        "years": actual_years,
        "age_agg": age_agg,
        "age_mi": age_mi_df,
        "total_yearly": total_yearly
    }


# --------------------------
# 3. 分性别建模（修复维度校验）
# --------------------------
class MortalityModel:
    def __init__(self, mx_matrix, ages, years, df_gender, gender_name):
        self.mx_matrix = mx_matrix
        self.ages = ages
        self.years = years
        self.df_gender = df_gender
        self.gender_name = gender_name
        self.mi_matrix = self._calc_mi_matrix()

    def _calc_mi_matrix(self):
        n_ages, n_years = self.mx_matrix.shape
        mi_matrix = np.zeros((n_ages, n_years - 1))
        for age_idx in range(n_ages):
            for year_idx in range(1, n_years):
                prev_mx = self.mx_matrix[age_idx, year_idx-1]
                curr_mx = self.mx_matrix[age_idx, year_idx]
                mi_matrix[age_idx, year_idx-1] = 0 if prev_mx == 0 else (1 - curr_mx / prev_mx)
        return mi_matrix

    def calc_str(self, model_type):
        # 修复年份索引匹配：使用更安全的方式查找年份索引
        valid_years = np.array(self.years)
        if STR_YEAR in valid_years:
            # 找到STR_YEAR的索引
            year_idx = np.flatnonzero(valid_years == STR_YEAR)[0]
        else:
            # 如果STR_YEAR不在数据中，使用最接近的年份
            closest_year = min(valid_years, key=lambda x: abs(x - STR_YEAR))
            year_idx = np.flatnonzero(valid_years == closest_year)[0]
            print(f"[{self.gender_name}] 警告：{STR_YEAR}年不在数据中，使用最接近的{closest_year}年")
        
        # 确保索引不越界（mi_matrix的列数是n_years-1）
        max_year_idx = len(valid_years) - 2
        year_idx = min(year_idx, max_year_idx)
        year_idx = max(year_idx, 0)  # 确保索引不为负
        
        str_mi = self.mi_matrix[:, year_idx].copy()

        # 暴露数权重（确保长度匹配）
        age_Ex = self.df_gender.groupby("Age")["Ex"].sum().reindex(self.ages, fill_value=1).values
        age_Ex = np.where(age_Ex == 0, 1, age_Ex)

        # 加权平滑
        if model_type == "CMI":
            from scipy.signal import savgol_filter
            # 确保窗口长度为奇数且不超过数据长度
            window_length = min(7, len(self.ages))
            if window_length % 2 == 0:
                window_length -= 1
            if window_length >= 3:
                str_mi = savgol_filter(str_mi, window_length=window_length, polyorder=min(3, window_length-1), mode="nearest")
        elif model_type == "RPEC":
            kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
            conv_Ex = np.convolve(age_Ex, kernel, mode="same")
            conv_Ex = np.where(conv_Ex == 0, 1, conv_Ex)
            str_mi = np.convolve(str_mi * age_Ex, kernel, mode="same") / conv_Ex
        elif model_type == "MI":
            conv_Ex1 = np.convolve(age_Ex, np.ones(5)/5, mode="same")
            conv_Ex1 = np.where(conv_Ex1 == 0, 1, conv_Ex1)
            str_mi = np.convolve(str_mi * age_Ex, np.ones(5)/5, mode="same") / conv_Ex1
            
            conv_Ex2 = np.convolve(age_Ex, np.ones(3)/3, mode="same")
            conv_Ex2 = np.where(conv_Ex2 == 0, 1, conv_Ex2)
            str_mi = np.convolve(str_mi * age_Ex, np.ones(3)/3, mode="same") / conv_Ex2
        
        str_mi = np.clip(str_mi, 0, 0.1)
        return pd.DataFrame({"Age": self.ages, f"STR_{model_type}": str_mi})

    def calc_ltr(self, model_type):
        ltr_data = []
        for age in self.ages:
            if model_type == "CMI":
                ltr = 0.015 if age <= 85 else (0.015 * (110 - age) / 25 if age <= 110 else 0)
            elif model_type == "RPEC":
                ltr = 0.0135 if age <= 62 else (0.0135 - (age-62)*0.0025/18 if age <= 80 else 0.011 - (age-80)*0.007/15 if age <= 95 else 0.004 - (age-95)*0.004/20 if age <= 115 else 0)
            elif model_type == "MI":
                ltr = 0.01 if age <= 90 else (0.01 - (age-90)*0.008/10 if age <= 100 else 0.002 - (age-100)*0.002/5 if age <= 105 else 0)
            ltr_data.append({"Age": age, f"LTR_{model_type}": ltr})
        return pd.DataFrame(ltr_data)

    def calc_converge(self, str_df, ltr_df, model_type):
        forecast_years = np.arange(STR_YEAR + 1, STR_YEAR + FORECAST_H + 1)
        converge_data = []
        
        # 确保str_df和ltr_df的年龄维度一致
        common_ages = sorted(list(set(str_df["Age"]) & set(ltr_df["Age"]) & set(self.ages)))
        
        for age in common_ages:
            str_val = str_df[str_df["Age"] == age][f"STR_{model_type}"].iloc[0]
            ltr_val = ltr_df[ltr_df["Age"] == age][f"LTR_{model_type}"].iloc[0]
            
            # 收敛期
            if model_type == "CMI":
                tau = 10 if 20<=age<=50 else (20 if 60<=age<=80 else 5 if age>=95 else (10+(age-50) if age>50 else 10-(50-age)))
            elif model_type == "RPEC":
                tau = 10
            elif model_type == "MI":
                tau = 10 if age <= 40 else (20 if age >= 60 else 10 + ((age-40)//2))
            
            # 逐年份计算
            for year in forecast_years:
                dt = year - STR_YEAR
                if dt <= 0:
                    mi = str_val
                elif dt <= tau:
                    next_age = min(age+1, max(common_ages)) if (age+1) in common_ages else age
                    m0 = str_df[str_df["Age"] == age][f"STR_{model_type}"].iloc[0] - str_df[str_df["Age"] == next_age][f"STR_{model_type}"].iloc[0]
                    a = (-2 * tau * m0 + 3*(ltr_val - str_val)) / (tau**2)
                    b = (tau * m0 - 2*(ltr_val - str_val)) / (tau**3)
                    mi = str_val + m0*dt + a*dt**2 + b*dt**3
                else:
                    mi = ltr_val
                converge_data.append({"Age": age, "Year": year, f"MI_{model_type}": np.clip(mi, 0, 0.1)})
        
        return pd.DataFrame(converge_data)


# --------------------------
# 4. 可视化（修复数据长度匹配）
# --------------------------
def plot_results(gender_name, age_agg, age_mi, total_yearly, str_df, ltr_df, converge_df):
    plt.figure(figsize=(20, 18))

    # 子图1：分年龄汇总死亡率
    plt.subplot(3, 2, 1)
    plt.bar(age_agg["Age"], age_agg["total_mx"]*100, alpha=0.7, color="#1f77b4")
    plt.title(f"{gender_name} - 分年龄汇总死亡率（暴露数加权）", fontsize=12)
    plt.xlabel("年龄", fontsize=10)
    plt.ylabel("死亡率（%）", fontsize=10)
    plt.grid(alpha=0.3, axis="y")

    # 子图2：分年龄汇总改善率
    plt.subplot(3, 2, 2)
    plt.bar(age_mi["Age"], age_mi["weighted_avg_mi"]*100, alpha=0.7, color="#ff7f0e")
    plt.title(f"{gender_name} - 分年龄汇总改善率（暴露数加权）", fontsize=12)
    plt.xlabel("年龄", fontsize=10)
    plt.ylabel("改善率（%）", fontsize=10)
    plt.grid(alpha=0.3, axis="y")

    # 子图3：总体年度死亡率+改善率趋势
    plt.subplot(3, 2, 3)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(total_yearly["Year"], total_yearly["total_mx_year"]*100, "o-", color="#1f77b4", linewidth=2, label="总体死亡率")
    ax1.set_xlabel("年份", fontsize=10)
    ax1.set_ylabel("总体死亡率（%）", fontsize=10, color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    # 只绘制有效改善率
    valid_mi = total_yearly.dropna(subset=["total_MI_year"])
    ax2.plot(valid_mi["Year"], valid_mi["total_MI_year"]*100, "s-", color="#ff7f0e", linewidth=2, label="总体改善率")
    ax2.set_ylabel("总体改善率（%）", fontsize=10, color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    # 双图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)
    plt.title(f"{gender_name} - 总体年度死亡率+改善率趋势", fontsize=12)

    # 子图4：短期改善率(STR)模型对比
    plt.subplot(3, 2, 4)
    for col in [c for c in str_df.columns if c.startswith("STR")]:
        plt.plot(str_df["Age"], str_df[col]*100, label=f"{col.split('_')[1]}模型", linewidth=2)
    plt.title(f"{gender_name} - 短期改善率(STR)模型对比", fontsize=12)
    plt.xlabel("年龄", fontsize=10)
    plt.ylabel("改善率（%）", fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # 子图5：长期改善率(LTR)模型对比
    plt.subplot(3, 2, 5)
    for col in [c for c in ltr_df.columns if c.startswith("LTR")]:
        plt.plot(ltr_df["Age"], ltr_df[col]*100, label=f"{col.split('_')[1]}模型", linewidth=2)
    plt.title(f"{gender_name} - 长期改善率(LTR)模型对比", fontsize=12)
    plt.xlabel("年龄", fontsize=10)
    plt.ylabel("改善率（%）", fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # 子图6：关键年龄收敛模型对比（20/50/80岁）
    plt.subplot(3, 2, 6)
    key_ages = [20, 50, 80]
    # 只选择存在的关键年龄
    key_ages = [age for age in key_ages if age in converge_df["Age"].values]
    for age in key_ages:
        age_data = converge_df[converge_df["Age"] == age]
        for col in [c for c in converge_df.columns if c.startswith("MI")]:
            plt.plot(age_data["Year"], age_data[col]*100, label=f"{col.split('_')[1]}模型-{age}岁", linewidth=1.5)
    plt.title(f"{gender_name} - 关键年龄收敛模型对比", fontsize=12)
    plt.xlabel("年份", fontsize=10)
    plt.ylabel("改善率（%）", fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{DATA_PATH}/{gender_name}_结果可视化_增强版.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[{gender_name}] 增强版图表已保存")


# --------------------------
# 5. 导出结果
# --------------------------
def export_results(gender_name, age_agg, age_mi, total_yearly, str_df, ltr_df, converge_df):
    with pd.ExcelWriter(f"{DATA_PATH}/{gender_name}_死亡率改善率结果_增强版.xlsx") as writer:
        age_agg.to_excel(writer, sheet_name="1_分年龄汇总死亡率", index=False)
        age_mi.to_excel(writer, sheet_name="2_分年龄汇总改善率", index=False)
        total_yearly.to_excel(writer, sheet_name="3_总体年度趋势", index=False)
        str_df.to_excel(writer, sheet_name="4_短期改善率模型", index=False)
        ltr_df.to_excel(writer, sheet_name="5_长期改善率模型", index=False)
        converge_df.to_excel(writer, sheet_name="6_收敛期改善率", index=False)
    print(f"[{gender_name}] 增强版Excel结果已导出")


# --------------------------
# 主流程
# --------------------------
def main():
    try:
        data = load_data()
        genders = [("female", "女性"), ("male", "男性")]

        for gender_key, gender_name in genders:
            print(f"\n{'='*50}\n处理【{gender_name}】数据")
            df_gender = data[gender_key]

            # 1. 计算（含总体汇总）
            mortality_res = calc_weighted_mortality(df_gender)
            print(f"[{gender_name}] 有效年龄数：{len(mortality_res['ages'])}")
            print(f"[{gender_name}] 有效年份数：{len(mortality_res['years'])}")
            print(f"[{gender_name}] 总体年度趋势（前5年）：\n{mortality_res['total_yearly'][['Year', 'total_mx_year', 'total_MI_year']].head()}")

            # 2. 模型计算
            model = MortalityModel(
                mx_matrix=mortality_res["mx_matrix"],
                ages=mortality_res["ages"],
                years=mortality_res["years"],
                df_gender=df_gender,
                gender_name=gender_name
            )

            models = ["CMI", "RPEC", "MI"]
            str_df = pd.DataFrame({"Age": mortality_res["ages"]})
            ltr_df = pd.DataFrame({"Age": mortality_res["ages"]})
            converge_df = pd.DataFrame({"Age": [], "Year": []})
            for model_type in models:
                print(f"[{gender_name}] 计算{model_type}模型...")
                str_temp = model.calc_str(model_type)
                str_df = str_df.merge(str_temp, on="Age", how="left")
                ltr_temp = model.calc_ltr(model_type)
                ltr_df = ltr_df.merge(ltr_temp, on="Age", how="left")
                conv_temp = model.calc_converge(str_temp, ltr_temp, model_type)
                if converge_df.empty:
                    converge_df = conv_temp
                else:
                    converge_df = converge_df.merge(conv_temp, on=["Age", "Year"], how="left")

            # 3. 可视化+导出
            plot_results(gender_name, 
                         mortality_res["age_agg"], 
                         mortality_res["age_mi"], 
                         mortality_res["total_yearly"], 
                         str_df, ltr_df, converge_df)
            export_results(gender_name, 
                           mortality_res["age_agg"], 
                           mortality_res["age_mi"], 
                           mortality_res["total_yearly"], 
                           str_df, ltr_df, converge_df)

        print(f"\n{'='*50}\n所有性别处理完成！结果保存至：{DATA_PATH}")
        print("输出内容：")
        print("1. 增强版图表（含6个子图：分年龄+总体趋势+模型对比）")
        print("2. 增强版Excel（含6个工作表，完整保存所有分析结果）")
    
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        import traceback
        print("\n详细错误信息:")
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
总体死亡率改善率（所有年龄汇总）：计算所有年龄的年度总死亡率（总理赔数 / 总暴露数），并基于此计算总体年度改善率，反映整体趋势。
扩展可视化（3×2 布局）：新增 2 个子图：
子图 3：总体年度死亡率 + 改善率双轴趋势图（左轴：死亡率，右轴：改善率）
子图 6：关键年龄（20/50/80 岁）收敛模型对比图（展示不同模型的预测趋势）
增强版 Excel 导出：包含 6 个工作表，覆盖所有分析结果：
分年龄汇总死亡率
分年龄汇总改善率
总体年度死亡率 + 改善率趋势
"""
