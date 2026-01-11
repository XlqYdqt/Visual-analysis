import streamlit as st
import pandas as pd
import os
import plotly.express as px

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="学习行为交互式仪表板",
    page_icon="📊",
    layout="wide"
)

FINAL_DATA_FILE = "Final_Processed_Data.csv"


# --- 2. 数据加载 (缓存) ---
@st.cache_data
def load_final_data(file_name):
    """
    加载单个 Final_Processed_Data.csv 文件，并进行数据类型清洗，
    确保过滤字段（如 'major'）不会出现 float 和 str 混淆导致的排序错误。
    """
    if not os.path.exists(file_name):
        st.error(f"错误：未找到最终数据文件 '{file_name}'。请先运行 'process_data.py' 脚本。")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_name, encoding='utf-8')

        # 确保时间列被正确识别为 datetime 对象
        if 'submit_time' in df.columns:
            df['submit_time'] = pd.to_datetime(df['submit_time'], errors='coerce')

            # --- 关键修复：确保所有过滤字段为字符串类型，避免 TypeError ---
        # 遍历所有用于过滤的列，将缺失值(NaN/float)替换为字符串，并强制转换为字符串类型
        for col in ['class', 'major', 'knowledge_point', 'state']:
            if col in df.columns:
                # 填充 NaN 为 '未知/缺失'，然后转换为字符串
                df[col] = df[col].fillna('未知/缺失').astype(str)
        # -------------------------------------------------------------

        # 移除任何因 process_data.py 合并残留的无效行（例如表头污染）
        if 'class' in df.columns:
            # 过滤掉班级列不是以 'Class' 开头的行（非数据行或表头）
            df = df[df['class'].str.startswith('Class', na=False) | (df['class'] == '未知/缺失')]

        return df.dropna(subset=['student_ID'])

    except Exception as e:
        st.error(f"加载数据时发生错误: {e}")
        return pd.DataFrame()


# 加载数据
df_master = load_final_data(FINAL_DATA_FILE)

# 检查数据是否成功加载
if df_master.empty:
    st.stop()

st.title("学生学习行为特征分析")
st.markdown("通过侧边栏选择班级、专业和知识点，实时查看关键指标和可视化结果。")

# --- 3. 侧边栏过滤器 (Filtering) ---
st.sidebar.header("数据过滤选项")

# 1. 班级过滤
# 现在 unique() 确保只包含字符串，排序不会出错
unique_classes = sorted(df_master['class'].unique().tolist())
selected_classes = st.sidebar.multiselect(
    "选择班级 (Select Class)",
    options=unique_classes,
    default=unique_classes
)

# 2. 专业过滤
# 现在 unique() 确保只包含字符串，排序不会出错
unique_majors = sorted(df_master['major'].unique().tolist())
selected_majors = st.sidebar.multiselect(
    "选择专业 (Select Major)",
    options=unique_majors,
    default=unique_majors
)

# 3. 知识点过滤
unique_knowledge = sorted(df_master['knowledge_point'].unique().tolist())  # 已经处理了 NaN
selected_knowledge = st.sidebar.multiselect(
    "选择知识点 (Select Knowledge)",
    options=unique_knowledge,
    default=unique_knowledge
)

# 4. 答题状态过滤
unique_states = sorted(df_master['state'].unique().tolist())
selected_states = st.sidebar.multiselect(
    "选择答题状态 (Select State)",
    options=unique_states,
    default=unique_states
)

# --- 4. 应用过滤 ---
# 使用 isin() 方法进行过滤
try:
    df_filtered = df_master[
        (df_master['class'].isin(selected_classes)) &
        (df_master['major'].isin(selected_majors)) &
        (df_master['knowledge_point'].isin(selected_knowledge)) &
        (df_master['state'].isin(selected_states))
        ].copy()
except Exception as e:
    st.error(f"过滤条件应用错误: {e}")
    df_filtered = df_master.copy()

# 如果过滤后数据为空，显示提示并停止
if df_filtered.empty:
    st.warning("当前选择的过滤条件下没有数据记录。")
    st.stop()

# --- 5. 显示过滤后的结果 ---

# 5.1 关键指标 (KPIs)
st.header("关键指标 (KPIs)")

# 计算指标 (使用新的列名)
total_submissions = df_filtered.shape[0]
# 使用新的列名 'student_score'
avg_score = df_filtered['student_score'].mean()
# 仅计算 '完全正确' 的平均耗时，使用新列名 'time_duration'
avg_time = df_filtered.loc[
    df_filtered['state'] == '完全正确', 'time_duration'
].mean()

col1, col2, col3 = st.columns(3)
col1.metric("总提交次数 (Total Submissions)", f"{total_submissions:,}")
col2.metric("平均得分 (Average Score)", f"{avg_score:.2f}")
col3.metric("正确提交平均耗时 (Avg Correct Time)", f"{avg_time:.2f} s")

st.markdown("---")

# 5.2 核心特征可视化 (Visualization of Key Features)

st.header("核心特征可视化")

# --- 可视化 1: 累计通过题目数 (学习进度) ---
st.subheader("1. 学习进度 S 曲线 (Cumulative Titles Passed)")
st.markdown("该图表展示了 **每个学生** 随着时间推移，**累计**通过题目数量的变化。")

# 为了清晰，仅随机选择 10 个学生进行绘制
sample_students = df_filtered['student_ID'].unique()
if len(sample_students) > 10:
    sample_students = pd.Series(sample_students).sample(10).tolist()

df_plot_cumulative = df_filtered[df_filtered['student_ID'].isin(sample_students)].copy()

# 确保排序正确以便线条连贯
df_plot_cumulative = df_plot_cumulative.sort_values(by=['student_ID', 'submit_time'])

if 'Cumulative_Titles_Passed' in df_plot_cumulative.columns and not df_plot_cumulative.empty:
    fig_cumulative = px.line(
        df_plot_cumulative,
        x='submit_time',
        y='Cumulative_Titles_Passed',
        color='student_ID',
        title='学习进度积累曲线',
        labels={'submit_time': '时间', 'Cumulative_Titles_Passed': '累计通过题目数'},
        line_shape='linear',
        render_mode='svg'
    )
    fig_cumulative.update_layout(xaxis_title="提交时间", yaxis_title="累计通过题目数")
    st.plotly_chart(fig_cumulative, use_container_width=True)
else:
    st.warning("数据中未找到 'Cumulative_Titles_Passed' 列，请检查特征工程脚本。")

# --- 可视化 2: 题目难度与平均尝试次数 (卡点分析) ---
st.subheader("2. 题目难度 vs. 平均尝试次数 (Difficulty vs. Efforts)")
st.markdown(
    "每个点代表一道题目。通过率越低 (Title\_Pass\_Rate\_Group)，题目越难。平均尝试次数越高 (Title\_Avg\_Attempts\_to\_Pass)，说明学生在该题上**卡点越久**。")

# 聚合到题目维度（仅使用一次首次尝试的记录进行聚合）
df_title_agg = df_filtered[df_filtered['Is_First_Attempt'] == 1].groupby('title_ID').agg(
    Title_Pass_Rate_Group=('Is_Passed', 'mean'),
    Title_Avg_Attempts_to_Pass=('Attempts_to_Pass', 'mean')
).reset_index().dropna(subset=['Title_Pass_Rate_Group', 'Title_Avg_Attempts_to_Pass'])

if not df_title_agg.empty:
    fig_scatter = px.scatter(
        df_title_agg,
        x='Title_Pass_Rate_Group',
        y='Title_Avg_Attempts_to_Pass',
        hover_data=['title_ID'],
        size=df_title_agg['Title_Avg_Attempts_to_Pass'],  # 点的大小反映尝试次数
        title='题目难度与尝试次数散点图',
        labels={
            'Title_Pass_Rate_Group': '群体首次通过率 (难度)',
            'Title_Avg_Attempts_to_Pass': '平均首次通过尝试次数'
        }
    )
    fig_scatter.update_layout(xaxis_title="题目通过率 (0-1)", yaxis_title="平均尝试次数")
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("数据量过小，无法计算题目聚合指标。")

st.markdown("---")
# 5.3 原始数据预览
st.subheader("原始数据预览 (Filtered Data Preview)")
st.dataframe(df_filtered.head(100))  # 仅显示前 100 行