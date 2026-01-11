import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
# --- 1. 配置和数据加载 ---
FILE_PATH = 'Title_Knowledge_Features_For_Analysis.csv'
OUTPUT_DIR = 'analysis_output'

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def identify_outliers(df):
    # 计算错配得分
    df['mismatch_score'] = df['KP_Avg_Pass_Rate'] - df['Title_Pass_Rate_Group']

    # 定义异常：掌握度高但正确率低的题目
    # 假设阈值为 0.35
    df['is_unreasonable'] = df['mismatch_score'] > 0.35

    # 绘制气泡图 (任务4要求的：题目难度-能力错配图)
    fig = px.scatter(
        df,
        x='KP_Avg_Pass_Rate',  # 横轴：学生在该知识点的整体掌握度
        y='Title_Pass_Rate_Group',  # 纵轴：该题的实际正确率
        size='Title_Avg_Attempts_to_Pass',  # 气泡大小：平均尝试次数
        color='is_unreasonable',
        hover_name='title_ID',
        labels={'KP_Avg_Pass_Rate': '知识点掌握度 (能力指标)',
                'Title_Pass_Rate_Group': '题目正确率 (难度指标)'},
        title='不合理题目识别：能力-难度错配图'
    )
    return fig


def get_drill_down_data(full_df, student_features_df, target_title_id):
    """
    下钻分析：获取特定题目在不同等级学生中的表现
    """
    # 1. 过滤出该题的所有作答记录
    title_records = full_df[full_df['title_ID'] == target_title_id].copy()

    # 2. 关联学生等级 (假设 student_features_df 包含 student_ID 和 Level)
    # Level 通常是通过聚类或正确率划分的 A, B, C, D
    detailed_df = pd.merge(
        title_records,
        student_features_df[['student_ID', 'Level']],
        on='student_ID',
        how='left'
    )

    return detailed_df


def plot_drill_down_box(detailed_df, title_id):
    """
    绘制下钻箱线图：展示不同等级学生对该异常题的尝试次数
    """
    import plotly.express as px

    fig = px.box(
        detailed_df,
        x='Level',
        y='attempts_count',  # 对应全量表中的尝试次数列
        color='Level',
        points="all",  # 显示所有点，便于观察极端值
        category_orders={"Level": ["A", "B", "C", "D"]},  # 强制按等级排序
        title=f"异常题目 {title_id} 的学生表现下钻",
        labels={'attempts_count': '尝试次数', 'Level': '学生能力等级'}
    )
    return fig
def load_data(file_path):
    """安全加载数据文件，并进行初步检查"""
    try:
        # 尝试使用 utf-8 读取，如果失败则尝试 gb18030
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='gb18030', errors='replace')

        print(f"✅ 成功加载文件: {file_path}")
        print(f"数据形状: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到，请确保 {file_path} 存在于当前目录下。")
        return None
    except Exception as e:
        print(f"❌ 错误: 加载文件时发生异常: {e}")
        return None


def preprocess_and_clean(df):
    """数据预处理和清洗"""
    print("\n--- 2. 数据预处理和清洗 ---")

    # 确保关键难度指标是数值类型
    numeric_cols = [
        'Avg_Time_Duration_Overall',
        'Title_Pass_Rate_Group',
        'Title_First_Attempt_Pass_Rate',
        'Title_Avg_Attempts_to_Pass',
        'KP_Avg_Pass_Rate',
        'KP_Avg_Attempts_to_Pass',
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 移除包含NaN的关键行
    df.dropna(subset=numeric_cols, inplace=True)

    # 由于一个 title_ID 可能对应多个 knowledge_point/sub_knowledge，
    # 我们保留所有行，因为这代表了题目与知识点的关联
    print(f"清洗后数据形状: {df.shape}")

    # 检查 knowledge_point 的唯一值，便于后续分组分析
    print(f"主要知识点 (knowledge_point) 数量: {df['knowledge_point'].nunique()}")

    return df


def analyze_and_visualize(df):
    """进行核心分析和可视化"""
    print("\n--- 3. 核心分析和可视化 ---")

    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 3.1 题目难度分布分析 (以 Title_Pass_Rate_Group 为例)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Title_Pass_Rate_Group'], kde=True, bins=15)
    # 修复 SyntaxWarning: 去掉 \_
    plt.title('题目群体通过率（Title_Pass_Rate_Group）分布', fontsize=16)
    plt.xlabel('通过率 (数值越低，难度越高)', fontsize=12)
    plt.ylabel('题目数量', fontsize=12)
    pass_rate_dist_path = os.path.join(OUTPUT_DIR, 'title_pass_rate_distribution.png')
    plt.savefig(pass_rate_dist_path)
    plt.close()
    print(f"📊 图表已保存: {pass_rate_dist_path}")
    print("分析：通过率分布可以直观看出题目库的整体难度趋势。")

    # 3.2 难度指标相关性分析
    difficulty_cols = ['Title_Pass_Rate_Group', 'Title_First_Attempt_Pass_Rate', 'Title_Avg_Attempts_to_Pass',
                       'Avg_Time_Duration_Overall']

    # 计算相关性矩阵
    corr_matrix = df[difficulty_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                cbar_kws={'label': '相关系数'})
    plt.title('题目难度特征相关性热力图', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    corr_heatmap_path = os.path.join(OUTPUT_DIR, 'difficulty_correlation_heatmap.png')
    plt.savefig(corr_heatmap_path)
    plt.close()
    print(f"📊 图表已保存: {corr_heatmap_path}")
    print("分析：检查各项难度指标之间的关联性，例如通过率与平均尝试次数是否高度负相关。")

    # 3.3 知识点难度对比 (Box Plot)
    # 使用 Title_Pass_Rate_Group 来比较不同 knowledge_point 的难度分散情况

    # 筛选出题目数量最多的前 N 个知识点
    top_n_kp = df['knowledge_point'].value_counts().nlargest(5).index
    df_top_kp = df[df['knowledge_point'].isin(top_n_kp)]

    plt.figure(figsize=(12, 7))
    sns.boxplot(x='knowledge_point', y='Title_Pass_Rate_Group', data=df_top_kp)
    # 修复 SyntaxWarning: 去掉 \_
    plt.title('不同主要知识点下题目的通过率（难度）分布', fontsize=16)
    plt.xlabel('主要知识点', fontsize=12)
    plt.ylabel('群体通过率 (Title_Pass_Rate_Group)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--')
    kp_pass_rate_box_path = os.path.join(OUTPUT_DIR, 'kp_difficulty_boxplot.png')
    plt.savefig(kp_pass_rate_box_path)
    plt.close()
    print(f"📊 图表已保存: {kp_pass_rate_box_path}")
    print("分析：箱线图展示了不同知识点难度范围和中位数，帮助识别最困难或难度最分散的知识点。")

    # 3.4 知识点聚合难度指标排名
    # 因为数据集中可能包含重复的 KP_Avg_Pass_Rate (因为每个题目都带上其知识点的聚合特征),
    # 我们需要先去重，以 knowledge_point 为唯一键。
    df_kp_agg = df[['knowledge_point', 'KP_Avg_Pass_Rate', 'KP_Avg_Attempts_to_Pass']].drop_duplicates().sort_values(
        'KP_Avg_Pass_Rate', ascending=True)  # 通过率越低，知识点越难

    print("\n--- 4. 知识点难度排名 (基于平均通过率) ---")
    df_kp_agg['KP_Avg_Pass_Rate'] = df_kp_agg['KP_Avg_Pass_Rate'].round(4)
    df_kp_agg['KP_Avg_Attempts_to_Pass'] = df_kp_agg['KP_Avg_Attempts_to_Pass'].round(2)

    # 修复依赖错误：使用 to_string() 代替 to_markdown() 来避免对 'tabulate' 的依赖
    print(df_kp_agg.head(10).to_string(index=False))
    print("\n分析：排名显示了哪个知识点整体上对学生最具挑战性。")


# --- 主执行流程 ---
if __name__ == "__main__":
    df = load_data(FILE_PATH)
    if df is not None:
        df_clean = preprocess_and_clean(df.copy())
        if not df_clean.empty:
            analyze_and_visualize(df_clean)
            print("\n--- 5. 分析完成 ---")
            print(f"所有生成的图表已保存到 '{OUTPUT_DIR}' 文件夹中。")
        else:
            print("数据清洗后为空，无法进行分析。")