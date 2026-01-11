import pandas as pd
import os
import io

# --- 1. 定义命名规范 ---
# 原始文件: SubmitRecord-ClassX.csv
# 定义原始数据所在的子文件夹
data_folder = "Data_SubmitRecord/"
# 最终合并和处理后的输出文件
final_output_file = "Final_Processed_Data.csv"

print("--- 开始批量处理15个班级的数据并进行特征工程 ---")
print(f"将从 {data_folder} 文件夹读取数据。")


# --- 2. 加载“维表”（只需加载一次） ---
def load_csv_safe(file_name):
    """Safely loads CSV with gb18030 encoding, replacing errors."""
    try:
        file_content = ""
        with open(file_name, 'r', encoding='gb18030', errors='replace') as f:
            file_content = f.read()
        df = pd.read_csv(io.StringIO(file_content))
        # 移除 index 列
        return df.drop(columns=['index'], errors='ignore')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file not found: {file_name}") from e


try:
    df_student = load_csv_safe("Data_StudentInfo.csv")
    # 立即重命名题目表中的列，避免合并后出现歧义 (score_x, score_y)
    df_title = load_csv_safe("Data_TitleInfo.csv").rename(columns={
        'score': 'full_score',
        'knowledge': 'knowledge_point',
        'sub_knowledge': 'sub_knowledge_point'
    })

except FileNotFoundError as e:
    print(f"错误: 必需的维表文件加载失败: {e}。无法继续处理。")
    exit()

print("学生信息和题目信息维表加载成功。")

processed_data_list = []
failed_files = []

# --- 3. 循环处理、合并与初步特征工程 ---
for i in range(1, 16):  # 循环 Class1 到 Class15
    class_file_name = f"SubmitRecord-Class{i}.csv"
    full_file_path = os.path.join(data_folder, class_file_name)

    try:
        if not os.path.exists(full_file_path):
            raise FileNotFoundError(f"File not found: {full_file_path}")

        # 3.1 & 3.2 安全读取文件
        file_content = ""
        with open(full_file_path, 'r', encoding='gb18030', errors='replace') as f:
            file_content = f.read()

        # 重命名提交记录中的列，避免冲突
        df_submit = pd.read_csv(io.StringIO(file_content)).rename(columns={
            'score': 'student_score',
            'time': 'submit_time',
            'timeconsume': 'time_duration'
        })

        print(f"--- 正在处理: {full_file_path} (原始 {df_submit.shape[0]} 行) ---")

        # 3.3 数据清洗
        df_submit['time_duration'] = pd.to_numeric(df_submit['time_duration'], errors='coerce')
        df_submit = df_submit.drop(columns=['index'], errors='ignore')

        # 3.4 数据合并
        df_merged = pd.merge(df_submit, df_student, on='student_ID', how='left')
        df_merged = pd.merge(df_merged, df_title, on='title_ID', how='left')

        # ----------------------------------------------------------------------
        # --- A. 掌握度/状态特征 (必须先于时序行为特征) ---
        # ----------------------------------------------------------------------

        # 清理分数
        df_merged['full_score'] = pd.to_numeric(df_merged['full_score'], errors='coerce')
        df_merged['student_score'] = pd.to_numeric(df_merged['student_score'], errors='coerce')
        df_merged['full_score'].fillna(0, inplace=True)

        # 1. 计算得分率 (Score_Rate)
        df_merged['Score_Rate'] = df_merged.apply(
            lambda row: row['student_score'] / row['full_score'] if row['full_score'] > 0 else 0,
            axis=1
        )

        # 2. 标记是否满分/通过 (Is_Passed)
        df_merged['Is_Passed'] = df_merged.apply(
            lambda row: 1 if (row['state'] == '完全正确') or (row['Score_Rate'] == 1.0) else 0,
            axis=1
        )

        # ----------------------------------------------------------------------
        # --- B. 时序与行为特征 ---
        # ----------------------------------------------------------------------

        # 1. 确保 'submit_time' 是 datetime 类型 (数据说明是时间戳，精确到秒)
        df_merged['submit_time'] = pd.to_datetime(df_merged['submit_time'], unit='s', errors='coerce')

        # 2. 提取时间维度特征 (高峰时段/周内周末)
        df_merged['submission_date'] = df_merged['submit_time'].dt.date
        df_merged['submission_hour'] = df_merged['submit_time'].dt.hour
        df_merged['submission_weekday'] = df_merged['submit_time'].dt.dayofweek
        df_merged['is_weekend'] = df_merged['submission_weekday'].apply(lambda x: 1 if x >= 5 else 0)

        # 3. 排序：计算时序行为特征的关键
        df_merged = df_merged.sort_values(by=['student_ID', 'title_ID', 'submit_time']).reset_index(drop=True)

        # 4. 计算尝试次数 (Submission_Count) 和是否为首次尝试 (Is_First_Attempt)
        df_merged['Submission_Count'] = df_merged.groupby(['student_ID', 'title_ID']).cumcount() + 1
        df_merged['Is_First_Attempt'] = df_merged['Submission_Count'].apply(lambda x: 1 if x == 1 else 0)

        # 5. 标记该提交是否是该题目的“首次通过” (Is_First_Pass)
        # a. 找出每个学生第一次通过某题的提交记录
        first_pass_records = df_merged[df_merged['Is_Passed'] == 1].groupby(['student_ID', 'title_ID'])[
            'submit_time'].min().reset_index()
        first_pass_records.rename(columns={'submit_time': 'First_Pass_Time'}, inplace=True)

        df_merged = pd.merge(df_merged, first_pass_records, on=['student_ID', 'title_ID'], how='left')

        # b. 标记 Is_First_Pass：该提交时间 == 首次通过时间 且 Is_Passed = 1
        df_merged['Is_First_Pass'] = df_merged.apply(
            lambda row: 1 if (row['Is_Passed'] == 1 and row['submit_time'] == row['First_Pass_Time']) else 0,
            axis=1
        )
        df_merged.drop(columns=['First_Pass_Time'], inplace=True)

        # 6. 计算首次通过所需尝试次数 (Attempts_to_Pass)
        df_attempts = df_merged[df_merged['Is_First_Pass'] == 1][['student_ID', 'title_ID', 'Submission_Count']].copy()
        df_attempts.rename(columns={'Submission_Count': 'Attempts_to_Pass'}, inplace=True)
        df_merged = pd.merge(df_merged, df_attempts, on=['student_ID', 'title_ID'], how='left')

        # 7. 计算学习间隔特征 (Submission_Interval)
        df_merged = df_merged.sort_values(by=['student_ID', 'submit_time'])
        df_merged['Prev_Submit_Time'] = df_merged.groupby('student_ID')['submit_time'].shift(1)

        df_merged['Submission_Interval'] = (df_merged['submit_time'] - df_merged['Prev_Submit_Time']).dt.total_seconds()
        df_merged.drop(columns=['Prev_Submit_Time'], inplace=True)

        processed_data_list.append(df_merged)
        print(f"成功: {full_file_path} 处理完成并添加到列表。")

    except FileNotFoundError:
        print(f"跳过: 未找到文件 {full_file_path}。")
        failed_files.append(class_file_name)
    except Exception as e:
        print(f"错误: 处理 {class_file_name} 时发生意外错误: {e}")
        failed_files.append(class_file_name)

    print("\n")

# --- 4. 最终整合所有班级数据 ---
if not processed_data_list:
    print("没有数据文件被成功处理。")
    exit()

df_final = pd.concat(processed_data_list, ignore_index=True)
print(f"所有班级数据合并完成，总记录数: {df_final.shape[0]}")

# ----------------------------------------------------------------------
# --- C. 掌握度与薄弱环节特征 (聚合特征) ---
# ----------------------------------------------------------------------

# **使用 Is_First_Attempt=1 的记录进行群体/知识点掌握度计算**
df_first_attempt = df_final[df_final['Is_First_Attempt'] == 1].copy()  # 使用副本避免 SettingWithCopyWarning

# 1. 题目难度聚合
title_agg = df_first_attempt.groupby('title_ID').agg(
    Title_Pass_Rate_Group=('Is_Passed', 'mean'),  # 群体首次通过率 (难度指标)
    Title_Avg_Score_Group=('student_score', 'mean'),
    Title_Avg_Attempts_to_Pass=('Attempts_to_Pass', 'mean')
).reset_index()

# 2. 知识点难度聚合
kp_agg = df_first_attempt.groupby('knowledge_point').agg(
    KP_Mastery_Rate_Group=('Is_Passed', 'mean'),  # 知识点群体掌握率
    KP_Total_Unique_Attempts=('student_score', 'count')
).reset_index()

# 3. 计算首次尝试的时间 (用于计算 Time_to_First_Pass)
first_attempt_time = df_first_attempt.groupby(['student_ID', 'title_ID'])['submit_time'].min().reset_index().rename(
    columns={'submit_time': 'First_Attempt_Time'})

# 合并首次尝试和首次通过的时间
time_diff_df = pd.merge(
    first_attempt_time,
    df_final[df_final['Is_First_Pass'] == 1][['student_ID', 'title_ID', 'submit_time']],
    on=['student_ID', 'title_ID'],
    how='inner'
).rename(columns={'submit_time': 'First_Pass_Time'})

# 4. 【新增可视化特征 2：首次通过耗时】
# 计算时间差并转换为小时
time_diff_df['Time_to_First_Pass_Hours'] = (time_diff_df['First_Pass_Time'] - time_diff_df[
    'First_Attempt_Time']).dt.total_seconds() / 3600
time_to_pass_agg = time_diff_df.drop(columns=['First_Attempt_Time', 'First_Pass_Time'])

# 5. 合并回主表
df_final = pd.merge(df_final, title_agg, on='title_ID', how='left')
df_final = pd.merge(df_final, kp_agg, on='knowledge_point', how='left')
# 注意：Time_to_First_Pass_Hours 只有在学生首次通过该题目的记录上才会有值
df_final = pd.merge(df_final, time_to_pass_agg, on=['student_ID', 'title_ID'], how='left')

# 6. 【新增可视化特征 3：相对难度差】
# Difficulty_Gap: 群体首次尝试的平均通过率 - 学生首次尝试的得分率
# 用于衡量学生在该题目上的相对表现，负值表示表现优于群体
# 仅对首次尝试的记录计算，以反映初始难度感
df_final['Difficulty_Gap'] = df_final.apply(
    lambda row: row['Title_Pass_Rate_Group'] - row['Score_Rate'] if row['Is_First_Attempt'] == 1 else None,
    axis=1
)

# ----------------------------------------------------------------------
# --- D. 学习模式建模特征 (学生维度聚合) ---
# ----------------------------------------------------------------------

# 1. 【新增可视化特征 1：累计通过题目数】
# 排序：必须按学生和时间全局排序，才能正确计算累计值
df_final = df_final.sort_values(by=['student_ID', 'submit_time']).reset_index(drop=True)
# Cumulative_Titles_Passed 只对 Is_First_Pass 计数，避免重复计算
df_final['Cumulative_Titles_Passed'] = df_final.groupby('student_ID')['Is_First_Pass'].cumsum()

# 2. 聚合所有记录到学生维度 (StuID)
student_agg = df_final.groupby('student_ID').agg(
    Total_Submissions=('student_score', 'count'),
    Unique_Titles_Attempted=('title_ID', 'nunique'),
    Avg_Attempts_Per_Title=('Attempts_to_Pass', 'mean'),
    Avg_Time_Duration=('time_duration', 'mean'),
    Overall_First_Pass_Count=('Is_First_Pass', 'sum'),
    Overall_Time_to_Pass_Avg=('Time_to_First_Pass_Hours', 'mean'),
    Learning_Days=('submission_date', 'nunique')
).reset_index()

# 3. 计算学生的答题时间分布特征 (夜猫子指数)
night_owl = df_final[(df_final['submission_hour'] >= 23) | (df_final['submission_hour'] <= 5)].groupby('student_ID')[
    'student_score'].count().reset_index().rename(columns={'student_score': 'Night_Submissions'})
total_submissions = df_final.groupby('student_ID')['student_score'].count().reset_index().rename(
    columns={'student_score': 'Total'})

night_owl_rate = pd.merge(total_submissions, night_owl, on='student_ID', how='left').fillna(0)
night_owl_rate['Night_Owl_Rate'] = night_owl_rate['Night_Submissions'] / night_owl_rate['Total']
night_owl_rate.drop(columns=['Night_Submissions', 'Total'], inplace=True)

# 4. 最终合并学生聚合特征
df_final = pd.merge(df_final, student_agg, on='student_ID', how='left')
df_final = pd.merge(df_final, night_owl_rate, on='student_ID', how='left')

# --- 5. 最终保存 ---
print(f"所有特征工程完成，正在保存最终文件: {final_output_file}")
df_final.to_csv(final_output_file, index=False, encoding='utf-8')
print(f"✅ 成功: 最终处理文件已保存为 {final_output_file} (共 {df_final.shape[0]} 行, {df_final.shape[1]} 列)")
if failed_files:
    print(f"警告: 以下文件未能成功处理或被跳过: {failed_files}")