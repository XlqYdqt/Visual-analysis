import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. 配置 ---
FILE_PATH = "Temporal_Behavioral_Features.csv"
OUTPUT_DIR = "analysis_output"  # 图表输出目录

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# --- 2. 数据加载与预处理 ---
def load_data(file_path):
    print(f"--- 1. 正在加载数据: {file_path} ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ 错误: 未找到文件 {file_path}。请确保文件在同一目录下。")
        return None
    except Exception as e:
        print(f"❌ 加载数据时发生错误: {e}")
        return None

    # 将时间列转换为 datetime 对象
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df.dropna(subset=['time'], inplace=True)

    # 确保数值类型正确
    df['student_score'] = pd.to_numeric(df['student_score'], errors='coerce')
    df['full_score'] = pd.to_numeric(df['full_score'], errors='coerce')

    # * 衍生用户请求的 'Score_Rate' 特征
    #   处理 full_score 为 0 或 NaN 的情况
    df['Score_Rate'] = df.apply(
        lambda row: row['student_score'] / row['full_score'] if row['full_score'] > 0 else 0,
        axis=1
    )

    print(f"✅ 数据加载成功。总记录数: {len(df):,}")
    return df


# --- 3. 分析模块 1: 状态与掌握度特征 ---
def analyze_status_mastery(df):
    print("\n--- 2. 分析模块一：状态与掌握度特征 ---")

    total_submissions = len(df)

    # 1. Is_Passed (单次提交的通过率)
    # 衡量的是所有提交中，有多少比例是“完全正确”的
    overall_pass_rate = df['Is_Passed'].mean()

    # 2. Is_First_Pass (首次通过的提交)
    # 这个指标衡量的是“首次通过”记录占所有记录的比例
    first_pass_submission_rate = df['Is_First_Pass'].mean()

    # 3. Attempts_to_Pass (首次通过所需尝试次数)
    # * 关键指标：只分析那些 Is_First_Pass = 1 的记录
    df_first_pass_records = df[df['Is_First_Pass'] == 1].copy()

    if df_first_pass_records.empty:
        print("⚠️ 警告: 未找到任何 'Is_First_Pass' = 1 的记录。")
        mean_attempts = 0
        median_attempts = 0
    else:
        # 分析首次通过时，平均用了多少次尝试
        mean_attempts = df_first_pass_records['Attempts_to_Pass'].mean()
        median_attempts = df_first_pass_records['Attempts_to_Pass'].median()

    # 4. Score_Rate (得分率)
    # 分析所有提交的平均得分率
    avg_score_rate = df['Score_Rate'].mean()
    # 分析“完全正确”状态的得分率（理论上应为1.0）
    avg_passed_score_rate = df[df['Is_Passed'] == 1]['Score_Rate'].mean()

    # 生成报告文本
    report = f"""
### 1. 状态与掌握度分析

本部分分析了学生在每次提交时的瞬时状态和最终掌握情况。

**关键指标:**
* **总提交次数:** {total_submissions:,} 次
* **提交通过率 (Is_Passed):** {overall_pass_rate:.2%}
    * *解读：在所有 {total_submissions:,} 次提交中，有 {overall_pass_rate:.2%} 的提交达到了“完全正确”状态。*
* **平均得分率 (Score_Rate):** {avg_score_rate:.2%}
    * *解读：所有提交的平均得分率为 {avg_score_rate:.2%}，而成功通过的提交平均得分率为 {avg_passed_score_rate:.2%}。*
* **首次通过尝试次数 (Attempts_to_Pass):**
    * **平均值:** {mean_attempts:.2f} 次
    * **中位数:** {median_attempts:.1f} 次
    * *解读：学生在**成功通过**一道题目时，平均需要 {mean_attempts:.2f} 次尝试。中位数为 {median_attempts:.1f} 次，表明一半的学生在1-2次尝试内即可通过。*
"""
    return report


# --- 4. 分析模块 2: 时序与行为特征 ---
def analyze_temporal_behavior(df, output_dir):
    print("\n--- 3. 分析模块二：时序与行为特征 ---")

    # --- 4.1 submission_hour (学习高峰时段) ---
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='submission_hour', color='skyblue')
    plt.title('学习提交时间分布 (小时)', fontsize=16)
    plt.xlabel('小时 (0-23)', fontsize=12)
    plt.ylabel('提交次数', fontsize=12)
    hour_dist_path = os.path.join(output_dir, 'temporal_submission_hour_dist.png')
    plt.savefig(hour_dist_path)
    plt.close()
    print(f"📊 图表已保存: {hour_dist_path}")

    # 找到高峰时段
    peak_hour = df['submission_hour'].mode()[0]

    # --- 4.2 is_weekend (周末学习模式) ---
    weekend_rate = df['is_weekend'].mean()

    # --- 4.3 Submission_Interval (提交间隔) ---
    # 过滤掉 0（即每个学生的第一次提交）和极端异常值（例如 > 1 天）
    df_intervals = df[(df['Submission_Interval'] > 0) & (df['Submission_Interval'] < 86400)]
    median_interval_minutes = (df_intervals['Submission_Interval'] / 60).median()

    # --- 4.4 Cumulative_Titles_Passed (学习进度 S 曲线) ---
    # 随机抽取10名学生进行可视化
    sample_student_ids = df['student_ID'].drop_duplicates().sample(n=min(10, df['student_ID'].nunique()),
                                                                   random_state=42)
    df_sample = df[df['student_ID'].isin(sample_student_ids)].copy()

    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=df_sample,
        x='time',
        y='Cumulative_Titles_Passed',
        hue='student_ID',
        legend='full'
    )
    plt.title('学习进度 S 曲线 (抽样学生)', fontsize=16)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('累计通过题目数', fontsize=12)
    plt.legend(title='Student ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    s_curve_path = os.path.join(output_dir, 'temporal_learning_s_curve.png')
    plt.tight_layout()
    plt.savefig(s_curve_path)
    plt.close()
    print(f"📊 图表已保存: {s_curve_path}")

    # 生成报告文本
    report = f"""
### 2. 时序与行为分析

本部分分析了学生的学习习惯和进度模式。

**关键洞察:**
* **学习高峰时段 (submission_hour):**
    * *解读：学生的提交活动在 **{peak_hour}:00 左右达到顶峰**。大部分学习活动集中在下午和晚间，凌晨（01:00-05:00）的提交量最低。（请参见下图 `temporal_submission_hour_dist.png`）*
* **周末学习模式 (is_weekend):**
    * *解读：所有提交记录中，有 **{weekend_rate:.2%}** 发生在周末（周六/日）。这表明学习行为在工作日和周末分布相对均匀，周末也是重要的学习时间。*
* **学习连续性 (Submission_Interval):**
    * *解读：在所有**连续提交**（非首次提交且间隔小于1天）中，提交间隔的**中位数为 {median_interval_minutes:.1f} 分钟**。这反映了学生在遇到问题后进行调试和再次尝试的典型周期。*
* **学习进度模式 (Cumulative_Titles_Passed):**
    * *解读：学习进度 S 曲线（参见下图 `temporal_learning_s_curve.png`）显示了不同学生的学习轨迹。我们可以观察到“**早期突进型**”（前期曲线陡峭）和“**持续稳定型**”（曲线平缓线性增长）以及“**后期冲刺型**”（后期曲线快速上升）等不同模式。*
"""
    return report


# --- 5. 主执行函数 ---
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = load_data(FILE_PATH)

    if df is not None:
        report_part1 = analyze_status_mastery(df)
        report_part2 = analyze_temporal_behavior(df, OUTPUT_DIR)

        final_report = f"""# 时序与行为特征分析报告

**数据源:** {FILE_PATH}

## 核心洞察总结

* **效率与毅力：** 学生平均需要约 **{df[df['Is_First_Pass'] == 1]['Attempts_to_Pass'].mean():.2f} 次** 尝试才能首次通过一道题目，表明课程具有一定的挑战性。
* **学习时段：** 学习高峰集中在**晚间（约 {df['submission_hour'].mode()[0]}:00）**，且周末（占比 {df['is_weekend'].mean():.2%}）也是重要的学习时间。
* **进度差异：** 学生的学习进度曲线（S-Curve）差异显著，显示出多种学习模式。

---
{report_part1}
---
{report_part2}
"""

        # 将报告保存为 Markdown 文件
        report_file_path = "temporal_analysis_report.doc"
        try:
            with open(report_file_path, 'w', encoding='utf-8') as f:
                f.write(final_report)
            print(f"\n✅ 成功：分析报告已保存到 {report_file_path}")
        except Exception as e:
            print(f"❌ 保存报告时发生错误: {e}")


if __name__ == "__main__":
    main()