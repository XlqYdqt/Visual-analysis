import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import os

# --- 1. 基础配置 ---
st.set_page_config(
    page_title="学生学习画像分析系统 - 智能分类版",
    page_icon="🎓",
    layout="wide"
)

# 文件路径配置
STUDENT_FEATURES_FILE = "Student_Aggregated_Features_For_Analysis.csv"
TEMPORAL_LOG_FILE = "Temporal_Behavioral_Features.csv"

# --- 2. 字段中文映射 ---
AXIS_OPTIONS = {
    "提交总量 (投入)": "Total_Submissions",
    "活跃天数 (持久性)": "Learning_Days",
    "尝试题目数 (广度)": "Unique_Titles_Attempted",
    "平均试错次数 (挣扎度)": "Avg_Attempts_Per_Title",
    "平均分 (产出)": "Average_Score",
    "首次通过数 (能力)": "Overall_First_Pass_Count",
    "平均耗时 (效率)": "Overall_Time_to_Pass_Avg"
}

# --- 3. 数据加载 ---
@st.cache_data
def load_data():
    if not os.path.exists(STUDENT_FEATURES_FILE) or not os.path.exists(TEMPORAL_LOG_FILE):
        return None, None

    # 加载聚合特征
    df = pd.read_csv(STUDENT_FEATURES_FILE)
    # 填充缺失值
    for col in ['Avg_Attempts_Per_Title', 'Overall_Time_to_Pass_Avg', 'Night_Owl_Rate']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    df['student_ID'] = df['student_ID'].astype(str)

    # 加载时序日志
    df_temp = pd.read_csv(TEMPORAL_LOG_FILE)
    # 增强的时间解析，兼容 Unix 时间戳或字符串
    if 'time' in df_temp.columns:
         # 尝试自动推断，如果是 float/int 则是时间戳
        if pd.api.types.is_numeric_dtype(df_temp['time']):
             df_temp['time'] = pd.to_datetime(df_temp['time'], unit='s')
        else:
             df_temp['time'] = pd.to_datetime(df_temp['time'])
    
    df_temp['student_ID'] = df_temp['student_ID'].astype(str)
    # 提取月份用于桑基图
    df_temp['Month'] = df_temp['time'].dt.month

    return df, df_temp

# --- 4. 核心算法: 智能命名分类 (匹配文档) ---
def assign_cluster_names(centers_df):
    """
    [cite_start]根据文档 [cite: 200-226] 的定义，将聚类结果映射为具体名称。
    逻辑：
    1. 边缘夜猫子: 参与度极低 (Unique_Titles_Attempted 最小)
    2. 高效精英: 综合得分最高 (Average_Score 最大)
    3. 勤奋挣扎者: 试错次数极高 (Avg_Attempts_Per_Title 最大)
    4. 普通拖延型: 剩余的那一类 (通常特征平平)
    """
    mapping = {}
    remaining_indices = list(centers_df.index)
    
    # 1. 识别 "边缘夜猫子" (找覆盖题目数最少的)
    idx_owl = centers_df.loc[remaining_indices, 'Unique_Titles_Attempted'].idxmin()
    mapping[idx_owl] = "🦉 边缘夜猫子 (高风险)"
    remaining_indices.remove(idx_owl)
    
    # 2. 识别 "高效精英" (在剩下的里面找分最高的)
    idx_elite = centers_df.loc[remaining_indices, 'Average_Score'].idxmax()
    mapping[idx_elite] = "🌟 高效精英 (学霸)"
    remaining_indices.remove(idx_elite)
    
    # 3. 识别 "勤奋挣扎者" (在剩下的里面找试错次数最多的)
    if remaining_indices:
        idx_struggler = centers_df.loc[remaining_indices, 'Avg_Attempts_Per_Title'].idxmax()
        mapping[idx_struggler] = "💪 勤奋挣扎者 (需方法)"
        remaining_indices.remove(idx_struggler)
    
    # 4. 识别 "普通拖延型" (剩下的)
    if remaining_indices:
        idx_procrastinator = remaining_indices[0]
        mapping[idx_procrastinator] = "🐢 普通拖延型 (需督促)"
        
    return mapping

def run_kmeans_analysis(df, n_clusters):
    """执行聚类并打标签"""
    data = df.copy()
    
    features = [
        'Total_Submissions', 'Average_Score', 'Unique_Titles_Attempted',
        'Avg_Attempts_Per_Title', 'Overall_First_Pass_Count', 'Learning_Days',
        'Night_Owl_Rate', 'Overall_Time_to_Pass_Avg'
    ]
    
    # 1. 数据准备 (Log变换处理偏态分布)
    X = data[features].copy()
    for col in ['Total_Submissions', 'Learning_Days']:
        if col in X.columns:
            X[col] = np.log1p(X[col])
            
    # 2. 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. 聚类
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    data['Cluster_ID'] = kmeans.fit_predict(X_scaled)
    
    # 4. 计算原始数据的中心点 (用于业务含义判断)
    centers_raw = data.groupby('Cluster_ID')[features].mean()
    
    # 5. 智能命名
    if n_clusters == 4:
        # 只有4类时才能精确匹配文档定义的四种人
        name_mapping = assign_cluster_names(centers_raw)
        data['Cluster_Name'] = data['Cluster_ID'].map(name_mapping)
    else:
        # 如果用户选了别的数字，退化为按分数分级
        score_rank = centers_raw['Average_Score'].rank(ascending=False)
        data['Cluster_Name'] = data['Cluster_ID'].apply(lambda x: f"群体 {x} (能力Lv.{int(score_rank[x])})")
    
    # 6. 归一化中心点 (用于雷达图)
    minmax = MinMaxScaler()
    X_norm = pd.DataFrame(minmax.fit_transform(X), columns=features)
    X_norm['Cluster_Name'] = data['Cluster_Name']
    centers_norm = X_norm.groupby('Cluster_Name').mean()
    
    return data, centers_norm

# --- 5. 辅助算法: 月度模式判定 (用于桑基图) ---
def classify_monthly_pattern(group):
    """判断学生单月的学习模式"""
    submissions = len(group)
    unique_titles = group['title_ID'].nunique() if 'title_ID' in group.columns else 0
    
    if submissions < 5: return "😴 低活跃"
    
    # 广度 vs 深度
    if unique_titles > 8: return "🌐 广泛探索"
    if submissions / max(unique_titles, 1) > 3: return "🎯 集中攻坚" # 单题尝试多
    return "🔄 常规学习"

# --- 6. 页面主逻辑 ---

st.title("🎓 学生学习画像分析系统 - 智能分类版")
st.markdown("基于文档定义的四大典型群体：**高效精英、勤奋挣扎者、普通拖延型、边缘夜猫子**。")

df_features, df_temporal = load_data()

if df_features is None:
    st.error("❌ 未找到数据文件。")
    st.stop()

# --- 顶部控制栏 ---
with st.container():
    col1, col2 = st.columns([1, 3])
    with col1:
        st.info("⚙️ 参数设置")
        n_clusters = st.slider("聚类数量 (建议保持4类以匹配文档)", 2, 6, 4)
        if n_clusters != 4:
            st.warning("⚠️ 标准分类仅适用于4类，当前使用通用命名。")
            
    with col2:
        st.success("📊 班级概览")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("分析人数", len(df_features))
        k2.metric("平均分", f"{df_features['Average_Score'].mean():.2f}")
        pass_rate = (df_features['Average_Score'] >= 2.0).mean() * 100
        k3.metric("及格率 (Score>=2)", f"{pass_rate:.1f}%")
        k4.metric("人均提交", f"{df_features['Total_Submissions'].mean():.0f}")

st.divider()

# 运行聚类分析
df_analyzed, cluster_centers = run_kmeans_analysis(df_features, n_clusters)

# --- 核心 Tab 页 ---
tab1, tab2, tab3 = st.tabs(["🧩 群体画像 & 相关性", "🌊 学习模式演变 (桑基图)", "🔍 个体诊断"])

# === Tab 1: 群体画像 ===
with tab1:
    col_radar, col_corr = st.columns([1, 1.2])
    
    with col_radar:
        st.subheader("1. 群体特征雷达图")
        st.caption("面积越大代表该维度能力越强。注意观察'勤奋挣扎者'在试错次数上的突出表现。")
        
        fig_radar = go.Figure()
        categories = cluster_centers.columns.tolist()
        
        # 预设颜色以区分四类
        colors = px.colors.qualitative.Set1 
        
        for i, (name, row) in enumerate(cluster_centers.iterrows()):
            fig_radar.add_trace(go.Scatterpolar(
                r=row.values,
                theta=categories,
                fill='toself',
                name=name,
                line_color=colors[i % len(colors)]
            ))
            
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            margin=dict(t=30, b=30, l=40, r=40),
            height=400,
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with col_corr:
        st.subheader("2. 投入-产出相关性热力图")
        st.caption("红色=正相关，蓝色=负相关。验证：试错次数是否与成绩负相关？")
        
        # 计算相关性
        corr_cols = [
            'Average_Score', 'Total_Submissions', 'Learning_Days', 
            'Unique_Titles_Attempted', 'Avg_Attempts_Per_Title', 
            'Overall_Time_to_Pass_Avg'
        ]
        valid_cols = [c for c in corr_cols if c in df_analyzed.columns]
        corr_matrix = df_analyzed[valid_cols].corr(method='pearson')
        
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r", # 红蓝配色
            zmin=-1, zmax=1,
            aspect="auto"
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # 散点分布图
    st.subheader("3. 群体分布散点图")
    c1, c2 = st.columns(2)
    with c1: x_axis = st.selectbox("X轴", list(AXIS_OPTIONS.keys()), index=3) # 默认试错次数
    with c2: y_axis = st.selectbox("Y轴", list(AXIS_OPTIONS.keys()), index=4) # 默认平均分
    
    fig_scatter = px.scatter(
        df_analyzed,
        x=AXIS_OPTIONS[x_axis],
        y=AXIS_OPTIONS[y_axis],
        color="Cluster_Name", # 使用智能分类名称
        hover_name="student_ID",
        hover_data=["Total_Submissions", "Avg_Attempts_Per_Title"],
        title=f"{x_axis} vs {y_axis}"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# === Tab 2: 动态演变 (桑基图) ===
with tab2:
    st.subheader("🌊 月度学习模式流转 (11月 -> 12月)")
    st.markdown("追踪学生从 **期中 (11月)** 到 **期末 (12月)** 的行为模式变化及其对最终成绩的影响。")
    
    if df_temporal is not None:
        target_months = [11, 12]
        df_sankey_base = df_temporal[df_temporal['Month'].isin(target_months)].copy()
        
        # 1. 计算每个学生每个月的模式
        monthly_stats = df_sankey_base.groupby(['student_ID', 'Month']).apply(classify_monthly_pattern).reset_index(name='Pattern')
        monthly_pivot = monthly_stats.pivot(index='student_ID', columns='Month', values='Pattern').dropna()
        
        # 2. 加入最终成绩等级
        grade_map = df_features.set_index('student_ID')['Average_Score']
        def get_grade(s):
            if s >= 3.0: return "🏆 A (优秀)"
            if s >= 2.0: return "✅ B (良好)"
            if s >= 1.0: return "⚠️ C (及格)"
            return "❌ F (不及格)"
            
        monthly_pivot['Final_Grade'] = monthly_pivot.index.map(lambda x: get_grade(grade_map.get(x, 0)))
        
        if not monthly_pivot.empty:
            # 构建连接: 11月 -> 12月
            link1 = monthly_pivot.groupby([11, 12]).size().reset_index(name='Value')
            link1.columns = ['Source', 'Target', 'Value']
            link1['Source'] = link1['Source'].apply(lambda x: f"11月: {x}")
            link1['Target'] = link1['Target'].apply(lambda x: f"12月: {x}")
            
            # 构建连接: 12月 -> 最终成绩
            link2 = monthly_pivot.groupby([12, 'Final_Grade']).size().reset_index(name='Value')
            link2.columns = ['Source', 'Target', 'Value']
            link2['Source'] = link2['Source'].apply(lambda x: f"12月: {x}")
            
            all_links = pd.concat([link1, link2], ignore_index=True)
            all_nodes = list(pd.concat([all_links['Source'], all_links['Target']]).unique())
            node_map = {name: i for i, name in enumerate(all_nodes)}
            
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15, thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_nodes,
                    color="blue"
                ),
                link=dict(
                    source=all_links['Source'].map(node_map),
                    target=all_links['Target'].map(node_map),
                    value=all_links['Value'],
                    color='rgba(200, 200, 200, 0.5)'
                )
            )])
            fig_sankey.update_layout(height=600, title_text="学习模式流转路径")
            st.plotly_chart(fig_sankey, use_container_width=True)
        else:
            st.warning("⚠️ 选定月份的数据交集为空，无法生成流转图。")
    else:
        st.warning("暂无时序数据。")

# === Tab 3: 个体查阅 ===
with tab3:
    st.subheader("🔍 学生个体画像诊断")
    
    sel_student = st.selectbox("选择学生 ID", df_analyzed['student_ID'].unique())
    
    if sel_student:
        stu_data = df_analyzed[df_analyzed['student_ID'] == sel_student].iloc[0]
        cluster_name = stu_data['Cluster_Name']
        
        st.info(f"该学生属于: **{cluster_name}**")
        
        # 对比卡片
        cols = st.columns(4)
        metrics = [
            ("Average_Score", "平均分"), 
            ("Total_Submissions", "提交量 (Log)"),
            ("Avg_Attempts_Per_Title", "试错次数"),
            ("Overall_Time_to_Pass_Avg", "平均耗时")
        ]
        
        # 计算该群体的平均值
        cluster_avg = df_analyzed[df_analyzed['Cluster_Name'] == cluster_name].mean(numeric_only=True)
        
        for i, (col, label) in enumerate(metrics):
            val = stu_data[col]
            avg = cluster_avg[col] if col in cluster_avg else 0
            cols[i].metric(label, f"{val:.2f}", f"{val-avg:+.2f} vs 群体均值")
            
        st.divider()
        
        # 位置高亮图
        st.markdown("##### 📍 能力定位")
        fig_loc = px.scatter(
            df_analyzed, x="Avg_Attempts_Per_Title", y="Average_Score",
            color="Cluster_Name", opacity=0.3,
            title="试错次数 vs 平均分 (高亮当前学生)"
        )
        fig_loc.add_trace(go.Scatter(
            x=[stu_data['Avg_Attempts_Per_Title']], 
            y=[stu_data['Average_Score']],
            mode='markers', marker=dict(color='red', size=15, symbol='star'),
            name='当前学生'
        ))
        st.plotly_chart(fig_loc, use_container_width=True)