import pandas as pd
import os
import io

# --- 1. CONFIGURATION ---
data_folder = "Data_SubmitRecord/"
# The name of the output file containing ONLY student-level aggregated features
output_features_file = "Student_Aggregated_Features_For_Analysis.csv"
print("--- Starting Data Processing and Student Feature Extraction ---")


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
        # Re-raise the exception to stop the script if critical files are missing
        raise


# --- 3. LOAD DIMENSION TABLES ---
try:
    df_student = load_csv_safe("Data_StudentInfo.csv")
    df_title = load_csv_safe("Data_TitleInfo.csv")

    # Standardize column types and names
    df_student['student_ID'] = df_student['student_ID'].astype(str)
    df_title['title_ID'] = df_title['title_ID'].astype(str)
    df_title.rename(columns={'score': 'full_score', 'knowledge': 'knowledge_point'}, inplace=True)
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
        # Load and clean current class file
        df_submit = load_csv_safe(full_file_path)

        # --- NOISE FILTERING & CLEANING ---
        # 1. Filter out 'class' header contamination noise
        df_submit = df_submit[df_submit['class'].astype(str).str.lower() != 'class']

        # 2. Drop records with missing student_ID (Invalid log records)
        df_submit.dropna(subset=['student_ID'], inplace=True)

        # 3. Type Conversion & Anomaly Handling
        df_submit['student_ID'] = df_submit['student_ID'].astype(str)
        df_submit['title_ID'] = df_submit['title_ID'].astype(str)
        # Convert time from Unix timestamp (unit='s')
        df_submit['submit_time'] = pd.to_datetime(df_submit['time'], unit='s', errors='coerce')
        df_submit.drop(columns=['time'], inplace=True)

        df_submit['student_score'] = pd.to_numeric(df_submit['score'], errors='coerce')
        df_submit.drop(columns=['score'], inplace=True)

        df_submit['time_duration'] = pd.to_numeric(df_submit['timeconsume'], errors='coerce')
        df_submit.drop(columns=['timeconsume'], inplace=True)

        # --- MERGE ---
        df_merged = pd.merge(df_submit, df_student, on='student_ID', how='left')
        df_merged = pd.merge(df_merged, df_title, on='title_ID', how='left')

        all_submit_dfs.append(df_merged)

    except Exception as e:
        print(f"❌ Error processing file {file_name}: {e}")

if not all_submit_dfs:
    print("❌ No valid submit record files were processed. Exiting.")
    exit()

df_final = pd.concat(all_submit_dfs, ignore_index=True)
print(f"✅ All raw data merged and cleaned. Total records: {len(df_final)}")

# --- 5. FEATURE ENGINEERING (FIXED FOR INDEX ALIGNMENT) ---

# CRITICAL FIX 1: Sort by time for correct cumulative calculations
df_final.sort_values(by=['student_ID', 'title_ID', 'submit_time'], inplace=True)

# 5.1 Temporal Features
df_final['submission_date'] = df_final['submit_time'].dt.date
df_final['submission_hour'] = df_final['submit_time'].dt.hour
df_final['is_weekend'] = df_final['submit_time'].dt.dayofweek.isin([5, 6]).astype(int)  # 5=Sat, 6=Sun

# 5.2 Status Features & Submission Count
df_final['Is_Passed'] = (df_final['student_score'] > 0).astype(int)
df_final['Submission_Count'] = df_final.groupby(['student_ID', 'title_ID']).cumcount() + 1
df_final['Is_First_Attempt'] = df_final['Submission_Count'].apply(lambda x: 1 if x == 1 else 0)

# CRITICAL FIX 2: Stable calculation of Is_First_Pass (The exact row of the first pass)
# 1. Calculate the cumulative number of passes for each student/title group
df_final['Cumulative_Passed_Count'] = df_final.groupby(['student_ID', 'title_ID'])['Is_Passed'].cumsum()

# 2. Mark a row as 'Is_First_Pass' if it is successful (Is_Passed=1) AND
#    it's the first successful submission in the sequence (Cumulative_Passed_Count=1)
df_final['Is_First_Pass'] = (
        (df_final['Cumulative_Passed_Count'] == 1) & (df_final['Is_Passed'] == 1)
).astype(int)

# 3. Calculate Attempts_to_Pass (The Submission_Count at the Is_First_Pass row)
attempts_map = df_final[df_final['Is_First_Pass'] == 1].set_index(
    ['student_ID', 'title_ID']
)['Submission_Count'].rename('Attempts_to_Pass')

# Merge attempts count back onto the main DataFrame
df_final = df_final.merge(attempts_map, on=['student_ID', 'title_ID'], how='left')

# --- Cleanup helper columns ---
df_final.drop(columns=['Cumulative_Passed_Count'], inplace=True)
df_final['Attempts_to_Pass'] = df_final['Attempts_to_Pass'].fillna(0)  # Fill NaN for students who never passed

# 5.3 Time-related features (Simplified/Fixed calculation)
# Calculate Time_to_First_Pass_Seconds (This is only relevant at the first pass row)
first_attempt_time = df_final[df_final['Is_First_Attempt'] == 1].set_index(['student_ID', 'title_ID'])[
    'submit_time'].rename('First_Attempt_Time')
first_pass_time = df_final[df_final['Is_First_Pass'] == 1].set_index(['student_ID', 'title_ID'])['submit_time'].rename(
    'First_Pass_Time')

time_to_pass = pd.merge(first_attempt_time.reset_index(), first_pass_time.reset_index(), on=['student_ID', 'title_ID'],
                        how='inner')
time_to_pass['Time_to_First_Pass_Seconds'] = (
            time_to_pass['First_Pass_Time'] - time_to_pass['First_Attempt_Time']).dt.total_seconds()

# Merge back the time difference, only relevant for aggregation later
df_final = df_final.merge(time_to_pass[['student_ID', 'title_ID', 'Time_to_First_Pass_Seconds']],
                          on=['student_ID', 'title_ID'], how='left')
df_final['Time_to_First_Pass_Hours'] = df_final['Time_to_First_Pass_Seconds'] / 3600  # Convert to hours

# --- 6. AGGREGATION & FEATURE ISOLATION ---

# 6.1 Aggregate by student_ID (using the correctly calculated features)
# We aggregate over all records for total submissions and mean scores
student_agg = df_final.groupby('student_ID').agg(
    Total_Submissions=('submit_time', 'count'),
    Average_Score=('student_score', 'mean'),
    Unique_Titles_Attempted=('title_ID', 'nunique'),
    # Average of Attempts_to_Pass, calculated ONLY on the successful passes (Is_First_Pass=1)
    Avg_Attempts_Per_Title=('Attempts_to_Pass', lambda x: x[x > 0].mean()),
    Avg_Time_Duration=('time_duration', 'mean'),
    Overall_First_Pass_Count=('Is_First_Pass', 'sum'),
    # Average of Time_to_First_Pass_Hours, calculated ONLY on the successful passes
    Overall_Time_to_Pass_Avg=('Time_to_First_Pass_Hours', 'mean'),
    Learning_Days=('submission_date', 'nunique')
).reset_index()

# 6.2 Calculate Night Owl Rate
night_owl = df_final[(df_final['submission_hour'] >= 23) | (df_final['submission_hour'] <= 5)].groupby('student_ID')[
    'student_score'].count().reset_index().rename(columns={'student_score': 'Night_Submissions'})

total_submissions = df_final.groupby('student_ID')['student_score'].count().reset_index().rename(
    columns={'student_score': 'Total'})

night_owl_rate = pd.merge(total_submissions, night_owl, on='student_ID', how='left').fillna(0)
night_owl_rate['Night_Owl_Rate'] = night_owl_rate['Night_Submissions'] / night_owl_rate['Total']
night_owl_rate.drop(columns=['Night_Submissions', 'Total'], inplace=True)

# 6.3 Final Student Feature Table
# Merge the main aggregation with the night owl rate
df_student_features = pd.merge(student_agg, night_owl_rate, on='student_ID', how='left')

# Add student static attributes back (sex, age, major)
df_student_static = df_student[['student_ID', 'sex', 'age', 'major']].drop_duplicates()
df_student_features = pd.merge(df_student_features, df_student_static, on='student_ID', how='left')

# --- 7. EXPORT THE STUDENT FEATURES ---
# This is the dedicated file the user requested.
df_student_features.to_csv(output_features_file, index=False, encoding='utf-8')
print(f"\n🎉 成功导出学生画像特征！文件已保存为: {output_features_file}")
print(f"该文件包含 {len(df_student_features)} 个学生的聚合特征数据。")