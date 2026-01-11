import pandas as pd
import os
import io
import numpy as np

# --- 1. 配置和路径定义 ---
data_folder = "Data_SubmitRecord/"
NUM_CLASSES = 15

# 输出文件命名规范
# 脚本的目标是提取包含时序/行为特征的“交易级”日志
FINAL_OUTPUT_FILE = "Temporal_Behavioral_Features.csv"

print("--- 开始提取时序与行为特征 ---")


# --- 2. 辅助函数：安全加载CSV ---
def load_csv_safe(file_name, data_path=None):
    """Safely loads CSV with gb18030 encoding, replacing errors."""
    try:
        if data_path:
            full_path = os.path.join(data_path, file_name)
        else:
            full_path = file_name

        file_content = ""
        # 尝试使用 gb18030 编码读取，并替换无法识别的字符
        with open(full_path, 'r', encoding='gb18030', errors='replace') as f:
            file_content = f.read()

        df = pd.read_csv(io.StringIO(file_content))

        if 'index' in df.columns:
            df = df.drop(columns=['index'], errors='ignore')

        print(f"✅ 成功加载文件: {file_name}")
        return df
    except FileNotFoundError as e:
        if 'SubmitRecord' not in file_name:
            raise FileNotFoundError(f"❌ 错误: 必需的维表文件未找到: {file_name}") from e
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ 错误: 加载文件 {file_name} 时发生异常: {e}")
        return pd.DataFrame()


# --- 3. 基础数据加载 ---
try:
    df_student = load_csv_safe("Data_StudentInfo.csv")

    df_title = load_csv_safe("Data_TitleInfo.csv")
    df_title = df_title.rename(columns={'score': 'full_score',
                                        'knowledge': 'knowledge_point',
                                        'sub_knowledge': 'sub_knowledge_point'})

    # 循环加载并合并所有班级的提交记录
    all_submissions = []
    for i in range(1, NUM_CLASSES + 1):
        file_name = f"SubmitRecord-Class{i}.csv"
        df_sub = load_csv_safe(file_name, data_folder)
        if not df_sub.empty:
            all_submissions.append(df_sub)

    if not all_submissions:
        raise ValueError("❌ 错误: 未找到任何提交记录文件 (SubmitRecord-ClassX.csv)。")

    df_submit_all = pd.concat(all_submissions, ignore_index=True)
    print(f"✅ 所有提交记录合并完成，总记录数: {len(df_submit_all):,}")

except (FileNotFoundError, ValueError) as e:
    print(e)
    exit()
except Exception as e:
    print(f"❌ 发生致命错误: {e}")
    exit()

# --- 4. 数据清洗和合并 (第一阶段：预处理) ---

print("\n--- 4. 数据清洗与合并 ---")

df_submit_all['time'] = pd.to_datetime(df_submit_all['time'], unit='s', errors='coerce')
df_submit_all.dropna(subset=['time', 'student_ID', 'title_ID'], inplace=True)
df_submit_all.rename(columns={'score': 'student_score'}, inplace=True)
df_submit_all['student_ID'] = df_submit_all['student_ID'].astype(str)
df_submit_all['title_ID'] = df_submit_all['title_ID'].astype(str)

df_submit_all['timeconsume'] = pd.to_numeric(df_submit_all['timeconsume'], errors='coerce')
df_submit_all['timeconsume'] = df_submit_all['timeconsume'].replace(0, np.nan)

df_merged = pd.merge(df_submit_all, df_student, on='student_ID', how='left')
df_merged = pd.merge(df_merged, df_title, on='title_ID', how='left')

df_merged.sort_values(by=['student_ID', 'time'], inplace=True, ignore_index=True)

df_merged['Is_Passed'] = (df_merged['state'] == 'Absolutely_Correct').astype(int)

# --- 5. 交易级特征工程 (第二阶段) ---

print("\n--- 5. 交易级特征工程 (时序与行为特征) ---")

# 5.1. 时序特征
df_merged['submission_hour'] = df_merged['time'].dt.hour
df_merged['submission_date'] = df_merged['time'].dt.date
df_merged['is_weekend'] = df_merged['time'].dt.dayofweek.isin([5, 6]).astype(int)
df_merged['Submission_Interval'] = df_merged.groupby('student_ID')['time'].diff().dt.total_seconds().fillna(0)

# 5.2. 序列行为特征 (尝试次数、首次通过)
df_merged['cumulative_passes_by_title'] = df_merged.groupby(['student_ID', 'title_ID'])['Is_Passed'].cumsum()
df_merged['Is_First_Pass'] = (df_merged['cumulative_passes_by_title'] == 1).astype(int)
df_merged.drop(columns=['cumulative_passes_by_title'], inplace=True)

df_merged['Cumulative_Titles_Passed'] = df_merged.groupby('student_ID')['Is_First_Pass'].cumsum()

df_merged['Attempts_Count'] = df_merged.groupby(['student_ID', 'title_ID']).cumcount() + 1
df_merged['Attempts_to_Pass'] = df_merged['Attempts_Count'].where(df_merged['Is_First_Pass'] == 1)

df_merged['First_Submission_Time'] = df_merged.groupby(['student_ID', 'title_ID'])['time'].transform('min')
df_merged['Time_to_First_Pass_Hours'] = (
        df_merged['time'] - df_merged['First_Submission_Time']
).dt.total_seconds().div(3600).where(df_merged['Is_First_Pass'] == 1)

# 清理辅助列
df_merged.drop(columns=['Attempts_Count', 'First_Submission_Time'], inplace=True)

print("✅ 时序与行为特征计算完成。")

# --- 6. 数据保存 ---

print("\n--- 6. 数据保存 ---")

# 选择要保留的交易级日志列
log_features_to_keep = [
    'student_ID', 'title_ID', 'time', 'state', 'student_score', 'full_score',
    'knowledge_point', 'sub_knowledge_point', 'class', 'sex', 'age', 'major',
    'timeconsume',

    # 新增的时序与行为特征
    'Is_Passed', 'Is_First_Pass', 'Attempts_to_Pass',
    'Cumulative_Titles_Passed', 'Time_to_First_Pass_Hours',
    'submission_hour', 'is_weekend', 'Submission_Interval',
]
final_df = df_merged[[col for col in log_features_to_keep if col in df_merged.columns]]

# 最终保存
final_df.to_csv(FINAL_OUTPUT_FILE, index=False, encoding='utf-8')
print(f"✅ 完整的交易日志 (含时序与行为特征) 已保存到: {FINAL_OUTPUT_FILE}")
print(f"总记录数: {len(final_df)}")

print("\n--- 脚本执行完毕 ---")