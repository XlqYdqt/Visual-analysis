import pandas as pd
import os
import io

# --- 1. CONFIGURATION ---
data_folder = "Data_SubmitRecord/"
# 更改输出文件，用于保存按题目/知识点聚合的特征
output_features_file = "Title_Knowledge_Features_For_Analysis.csv"
print("--- Starting Data Processing and Title/Knowledge Feature Extraction ---")


# --- 2. SAFE CSV LOADING FUNCTION ---
def load_csv_safe(file_name):
    """Safely loads CSV with gb18030 encoding, replacing errors, and drops 'index'."""
    try:
        file_content = ""
        # Using 'gb18030' for robust reading, replacing errors
        with open(file_name, 'r', encoding='gb18030', errors='replace') as f:
            file_content = f.read()
        df = pd.read_csv(io.StringIO(file_content))
        # Remove redundant 'index' column
        return df.drop(columns=['index'], errors='ignore')
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {file_name}")
        raise  # Re-raise the exception to stop the script


# --- 3. LOAD DIMENSION TABLES ---
try:
    df_student = load_csv_safe("Data_StudentInfo.csv")
    df_title = load_csv_safe("Data_TitleInfo.csv")

    # Standardize column types and names
    df_student['student_ID'] = df_student['student_ID'].astype(str)
    # df_title 有可能一个 title_ID 对应多个 knowledge，这里我们保留多行
    df_title['title_ID'] = df_title['title_ID'].astype(str)
    df_title.rename(columns={'score': 'full_score', 'knowledge': 'knowledge_point'}, inplace=True)
    # 移除 knowledge/sub_knowledge 中的重复项，确保每个 title_ID 都有一个基础属性
    df_title = df_title.drop_duplicates(subset=['title_ID', 'knowledge_point', 'sub_knowledge']).reset_index(drop=True)
    print("✅ Dimension tables loaded successfully.")

except Exception as e:
    print(f"❌ Fatal Error during dimension table loading: {e}")
    exit()

# --- 4. LOAD & CLEAN ALL SUBMIT RECORDS ---
all_submit_dfs = []
for i in range(1, 16):
    file_name = f"SubmitRecord-Class{i}.csv"
    full_file_path = os.path.join(data_folder, file_name)

    if not os.path.exists(full_file_path):
        continue

    try:
        df_submit = load_csv_safe(full_file_path)

        # --- NOISE FILTERING & CLEANING ---
        df_submit = df_submit[df_submit['class'].astype(str).str.lower() != 'class']
        df_submit.dropna(subset=['student_ID'], inplace=True)

        # 3. Type Conversion & Anomaly Handling
        df_submit['student_ID'] = df_submit['student_ID'].astype(str)
        df_submit['title_ID'] = df_submit['title_ID'].astype(str)
        df_submit['submit_time'] = pd.to_datetime(df_submit['time'], unit='s', errors='coerce')
        df_submit.drop(columns=['time'], inplace=True)

        df_submit['student_score'] = pd.to_numeric(df_submit['score'], errors='coerce')
        df_submit.drop(columns=['score'], inplace=True)

        df_submit['time_duration'] = pd.to_numeric(df_submit['timeconsume'], errors='coerce')
        df_submit.drop(columns=['timeconsume'], inplace=True)

        # --- MERGE ---
        # 仅合并题目信息，不合并学生信息，以避免数据量过度膨胀
        df_merged = pd.merge(df_submit, df_title, on='title_ID', how='left')

        all_submit_dfs.append(df_merged)

    except Exception as e:
        print(f"❌ Error processing file {file_name}: {e}")

if not all_submit_dfs:
    print("❌ No valid submit record files were processed. Exiting.")
    exit()

df_final = pd.concat(all_submit_dfs, ignore_index=True)
print(f"✅ All raw data merged and cleaned. Total records: {len(df_final)}")

# --- 5. FEATURE ENGINEERING (Per Submission Status) ---

# CRITICAL FIX 1: Sort by time for correct cumulative calculations
df_final.sort_values(by=['student_ID', 'title_ID', 'submit_time'], inplace=True)

df_final['Is_Passed'] = (df_final['student_score'] >= df_final['full_score']).astype(int)  # 假设得分等于满分为通过
df_final['Submission_Count'] = df_final.groupby(['student_ID', 'title_ID']).cumcount() + 1
df_final['Is_First_Attempt'] = df_final['Submission_Count'].apply(lambda x: 1 if x == 1 else 0)

# Stable calculation of Is_First_Pass
df_final['Cumulative_Passed_Count'] = df_final.groupby(['student_ID', 'title_ID'])['Is_Passed'].cumsum()
df_final['Is_First_Pass'] = (
        (df_final['Cumulative_Passed_Count'] == 1) & (df_final['Is_Passed'] == 1)
).astype(int)

# --- 6. AGGREGATE BY TITLE/KNOWLEDGE POINT (难度特征提取) ---

# 6.1 聚合到题目层面
title_agg = df_final.groupby('title_ID').agg(
    # 题目难度特征 1: 群体最终通过率 (Title_Pass_Rate_Group)
    # 统计所有提交记录中，有多少独立学生最终通过了这道题 (Is_First_Pass=1)
    Unique_Students_Passed=('Is_First_Pass', 'sum'),
    # 统计所有提交记录中，有多少独立学生尝试了这道题
    Unique_Students_Attempted=('student_ID', 'nunique'),
    # 题目难度特征 2: 首次尝试成功率 (Title_First_Attempt_Pass_Rate)
    First_Attempts=('Is_First_Attempt', 'sum'),
    First_Attempt_Success=('Is_First_Attempt',
                           lambda x: df_final.loc[x.index][df_final.loc[x.index]['Is_First_Attempt'] == 1][
                               'Is_Passed'].sum()),
    # 题目效率特征: 平均通过尝试次数 (Title_Avg_Attempts_to_Pass)
    # 仅统计那些最终通过的学生（即 Is_First_Pass=1 的记录行），提取其 Submission_Count 的平均值
    Total_Attempts_When_Passed=('Is_First_Pass',
                                lambda x: df_final.loc[x.index][df_final.loc[x.index]['Is_First_Pass'] == 1][
                                    'Submission_Count'].sum()),
    Total_Passed_Students=('Is_First_Pass', 'sum'),
    # 题目时间特征: 群体平均耗时
    Avg_Time_Duration_Overall=('time_duration', 'mean')
).reset_index()

# 6.2 计算难度指标
title_agg['Title_Pass_Rate_Group'] = title_agg['Unique_Students_Passed'] / title_agg['Unique_Students_Attempted']
title_agg['Title_First_Attempt_Pass_Rate'] = title_agg['First_Attempt_Success'] / title_agg['First_Attempts']
title_agg['Title_Avg_Attempts_to_Pass'] = title_agg['Total_Attempts_When_Passed'] / title_agg['Total_Passed_Students']

# 清理中间列
title_agg.drop(
    columns=['Unique_Students_Passed', 'First_Attempts', 'First_Attempt_Success', 'Total_Attempts_When_Passed',
             'Total_Passed_Students'], inplace=True)

# 6.3 合并知识点信息 (可能存在多行，但我们只需要聚合后的难度信息)
# 使用 title_agg 作为事实表，左连接 df_title 的知识点信息
df_title_info = df_title[['title_ID', 'knowledge_point', 'sub_knowledge', 'full_score']].drop_duplicates(
    subset=['title_ID', 'knowledge_point', 'sub_knowledge'])
df_difficulty_features = pd.merge(title_agg, df_title_info, on='title_ID', how='left')

# 6.4 聚合到知识点层面 (KP_Mastery_Rate_Group)
kp_agg = df_difficulty_features.groupby('knowledge_point').agg(
    # 知识点难度特征: 知识点平均通过率
    KP_Avg_Pass_Rate=('Title_Pass_Rate_Group', 'mean'),
    # 知识点效率特征: 知识点平均尝试次数
    KP_Avg_Attempts_to_Pass=('Title_Avg_Attempts_to_Pass', 'mean')
).reset_index()

# 6.5 最终合并并导出 (保留题目和知识点信息)
df_difficulty_features = pd.merge(df_difficulty_features, kp_agg, on='knowledge_point', how='left')

# --- 7. EXPORT THE TITLE/KNOWLEDGE FEATURES ---
df_difficulty_features.to_csv(output_features_file, index=False, encoding='utf-8')
print(f"\n🎉 成功导出题目与知识点难度特征！文件已保存为: {output_features_file}")
print(f"该文件包含 {len(df_difficulty_features)} 条记录，涵盖所有题目及其所属知识点的难度特征。")

