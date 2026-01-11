import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 配置与数据加载 ---
FILE_PATH = "Student_Aggregated_Features_For_Analysis.csv"
NUM_CLUSTERS = 4  # 预设的聚类数量
print(f"--- 1. 正在加载数据: {FILE_PATH} ---")

try:
    df = pd.read_csv(FILE_PATH)
    print(f"数据加载成功。总学生数: {len(df)}")
except FileNotFoundError:
    print(f"错误: 未找到文件 {FILE_PATH}。请确保文件与脚本在同一目录下。")
    exit()

# --- 2. 数据清洗与预处理 ---
# 填充缺失值：Avg_Attempts_Per_Title 和 Overall_Time_to_Pass_Avg 可能因从未通过题目而缺失。
# 填充策略：用 0 或该特征的平均值填充。这里我们用 0 来表示“没有通过记录/没有耗时”。
df['Avg_Attempts_Per_Title'] = df['Avg_Attempts_Per_Title'].fillna(0)
df['Overall_Time_to_Pass_Avg'] = df['Overall_Time_to_Pass_Avg'].fillna(0)
df['Night_Owl_Rate'] = df['Night_Owl_Rate'].fillna(0)

# 特征选择：选择用于聚类的数值特征
feature_cols = [
    'Total_Submissions', 'Average_Score', 'Unique_Titles_Attempted',
    'Avg_Attempts_Per_Title', 'Overall_First_Pass_Count', 'Learning_Days',
    'Night_Owl_Rate', 'Overall_Time_to_Pass_Avg'
]
X = df[feature_cols].copy()

# 异常值处理：对 Total_Submissions 等严重右偏的特征进行对数变换，以改善聚类效果
for col in ['Total_Submissions', 'Learning_Days']:
    # 使用 np.log1p(x) = log(1+x) 来处理可能为 0 的值
    X[col] = np.log1p(X[col])

# --- 3. 特征标准化 ---
# K-Means 对特征尺度敏感，必须进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
print("特征标准化完成。")

# --- 4. 寻找最佳聚类数 K (肘部法则 - Elbow Method) ---
# WCSS (Within-Cluster Sum of Squares)
wcss = []
k_range = range(2, 11)
print("\n--- 4. 正在计算肘部法则 (K=2 到 10) ---")

for i in k_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# 打印结果供用户查看肘部
print("WCSS 值 (inertia):")
for k, inertia in zip(k_range, wcss):
    print(f"K={k}: {inertia:.2f}")

# --- 5. 应用 K-Means 聚类 ---
print(f"\n--- 5. 应用 K-Means 聚类 (K={NUM_CLUSTERS}) ---")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, init='k-means++', random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# --- 6. 聚类结果分析 ---
# 将集群标签添加到原始数据中
df_clustered = df.copy()

# 计算每个集群的原始特征均值（这是生成画像的核心步骤）
cluster_profile = df_clustered.groupby('Cluster')[feature_cols].mean()

# 逆转对数变换，使结果更易读
for col in ['Total_Submissions', 'Learning_Days']:
    # 使用 np.expm1(x) = exp(x) - 1 来还原 log(1+x) 的变换
    cluster_profile[col] = np.expm1(cluster_profile[col])

# 添加每个集群的学生数量
cluster_profile['Student_Count'] = df_clustered.groupby('Cluster').size()
cluster_profile['Cluster_Size_Ratio'] = cluster_profile['Student_Count'] / len(df)

print("\n=======================================================")
print("✅ 聚类画像分析完成。请查看以下聚类中心点数据：")
print("=======================================================")

# *** 修复: 使用 to_string() 替代 to_markdown() 来避免 'tabulate' 依赖 ***
print(cluster_profile.to_string(float_format='%.2f'))

print("=======================================================")

# --- 7. 相关性分析 (为报告提供数据) ---
print("\n--- 7. 特征相关性矩阵 (用于报告) ---")
correlation_matrix = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Feature Correlation Matrix')
print("相关性矩阵已计算。")