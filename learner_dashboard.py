# learner_dashboard_final_modified.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
import random

warnings.filterwarnings('ignore')

# 设置页面
st.set_page_config(
    page_title="学习者行为分析仪表板",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_profile_data():
    """加载学习者画像数据"""
    try:
        df = pd.read_csv('Learner_Profiles.csv', encoding='utf-8')
        # 确保student_ID是字符串类型
        df['student_ID'] = df['student_ID'].astype(str)
        st.success(f"✅ 成功加载画像数据: {len(df)} 名学习者")
        return df
    except Exception as e:
        st.error(f"加载画像数据时出错: {e}")
        return pd.DataFrame()


@st.cache_data
@st.cache_data
def load_submit_data():
    """加载原始提交数据"""
    try:
        submit_dfs = []
        for i in range(1, 16):
            try:
                df = pd.read_csv(f'Data_SubmitRecord/SubmitRecord-Class{i}.csv', encoding='gb18030')
                df['class'] = f'Class{i}'
                submit_dfs.append(df)
            except Exception as e:
                # st.warning(f"加载Class{i}数据失败: {e}")
                continue

        if not submit_dfs:
            st.error("无法加载任何提交数据")
            return pd.DataFrame()

        df_submit = pd.concat(submit_dfs, ignore_index=True)

        # 转换时间
        df_submit['submit_time'] = pd.to_datetime(df_submit['time'], unit='s')
        df_submit['date'] = df_submit['submit_time'].dt.date

        # 处理状态
        df_submit['is_passed'] = (df_submit['state'] == '完全正确').astype(int)

        # 确保ID是字符串
        df_submit['student_ID'] = df_submit['student_ID'].astype(str)
        df_submit['title_ID'] = df_submit['title_ID'].astype(str)

        # 处理数值列
        if 'score' in df_submit.columns:
            df_submit['score'] = pd.to_numeric(df_submit['score'], errors='coerce').fillna(0)

        if 'timeconsume' in df_submit.columns:
            df_submit['timeconsume'] = pd.to_numeric(df_submit['timeconsume'], errors='coerce')
            df_submit['timeconsume'] = df_submit['timeconsume'].fillna(0)
            df_submit['timeconsume'] = df_submit['timeconsume'].apply(lambda x: max(0, x) if pd.notnull(x) else 0)

        # 计算得分率
        df_submit['score_rate'] = df_submit['score'] / 4.0
        df_submit['is_passed'] = (df_submit['score_rate'] == 1.0).astype(int)

        try:
            # 加载题目知识点信息
            title_info_df = pd.read_csv('Data_TitleInfo.csv', encoding='utf-8')

            # 确保title_ID是字符串类型
            title_info_df['title_ID'] = title_info_df['title_ID'].astype(str)

            # 由于同一个题目可能有多个知识点，我们需要去重或处理重复
            # 首先按title_ID分组，取第一个知识点（大知识点）
            title_knowledge_map = title_info_df.groupby('title_ID')['knowledge'].first().to_dict()

            # 将知识点信息合并到提交数据
            df_submit['knowledge_point'] = df_submit['title_ID'].map(title_knowledge_map)

            # 处理没有知识点映射的记录
            missing_knowledge = df_submit['knowledge_point'].isnull().sum()
            if missing_knowledge > 0:
                st.warning(f"有 {missing_knowledge} 条记录没有对应的知识点信息")
                # 填充未知知识点
                df_submit['knowledge_point'] = df_submit['knowledge_point'].fillna('未知知识点')

            st.info(f"✅ 成功合并知识点信息，共 {df_submit['knowledge_point'].nunique()} 个知识点")

        except Exception as e:
            st.error(f"加载或合并知识点信息时出错: {e}")
            # 如果没有知识点信息，添加一个空列
            df_submit['knowledge_point'] = '未知知识点'

        st.success(f"✅ 成功加载提交数据: {len(df_submit)} 条记录")
        return df_submit
    except Exception as e:
        st.error(f"加载提交数据时出错: {e}")
        return pd.DataFrame()


@st.cache_data
def perform_clustering(df):
    """对学习者数据进行聚类分析，使用文档中的四类名称"""
    try:
        # 选择用于聚类的特征
        cluster_features = []
        possible_features = [
            'correct_rate', 'total_attempts', 'avg_time_per_attempt',
            'night_owl_ratio', 'weekend_ratio', 'knowledge_focus',
            'persistence_rate', 'learning_continuity', 'burst_behavior_ratio'
        ]

        for feature in possible_features:
            if feature in df.columns:
                cluster_features.append(feature)

        if len(cluster_features) >= 3:
            # 准备聚类数据
            X = df[cluster_features].fillna(0)

            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 固定为4类聚类（根据文档要求）
            optimal_k = 4
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            df['cluster'] = kmeans.fit_predict(X_scaled)

            # 使用文档中的固定名称分配
            default_names = ['高效精英', '勤奋挣扎者', '普通拖延型', '边缘夜猫子']
            cluster_names = {}

            for i, cluster_id in enumerate(sorted(df['cluster'].unique())):
                if i < len(default_names):
                    cluster_names[cluster_id] = default_names[i]
                else:
                    cluster_names[cluster_id] = f'聚类-{cluster_id}'

            # 添加聚类名称
            df['cluster_name'] = df['cluster'].map(cluster_names)

            # 确保所有学生都有聚类名称
            if df['cluster_name'].isnull().any():
                df.loc[df['cluster_name'].isnull(), 'cluster_name'] = '未分类'

            return df

        else:
            st.warning("特征不足，无法进行聚类分析")
            df['cluster'] = 0
            df['cluster_name'] = '未分类'
            return df

    except Exception as e:
        st.error(f"聚类分析时出错: {e}")
        df['cluster'] = 0
        df['cluster_name'] = '未分类'
        return df


def create_monthly_activity_heatmap(df_submit, student_id=None):
    """创建月内活跃度热力图，如果指定学生ID则只显示该学生的数据"""
    if df_submit.empty or 'date' not in df_submit.columns:
        return None

    try:
        # 如果指定了学生ID，只显示该学生的数据
        if student_id:
            df_submit = df_submit[df_submit['student_ID'] == student_id].copy()
            title_suffix = f"（学生 {student_id[:8]}...）"
        else:
            title_suffix = ""

        if df_submit.empty:
            return None

        # 提取小时信息
        df_submit['hour'] = df_submit['submit_time'].dt.hour

        # 创建日期-小时矩阵
        heatmap_data = df_submit.groupby(['date', 'hour']).size().reset_index(name='count')

        # 转换为透视表格式
        pivot_data = heatmap_data.pivot_table(
            index='date',
            columns='hour',
            values='count',
            aggfunc='sum',
            fill_value=0
        )

        # 按日期排序
        pivot_data = pivot_data.sort_index()

        # 创建热力图
        fig = px.imshow(
            pivot_data.T,
            labels=dict(x="日期", y="小时", color="提交次数"),
            title=f"月内每日活跃度热力图{title_suffix}",
            aspect="auto",
            color_continuous_scale="YlOrRd"
        )

        # 调整布局
        fig.update_layout(
            xaxis_title="日期",
            yaxis_title="小时 (0-23)",
            height=500
        )

        return fig
    except Exception as e:
        st.error(f"创建热力图时出错: {e}")
        return None


def create_enhanced_radar_chart(df_profiles):
    """创建增强版雷达图（对比4类学习者）"""
    if df_profiles.empty or 'cluster_name' not in df_profiles.columns:
        return None

    try:
        # 选择雷达图特征
        radar_features = []
        feature_options = [
            ('correct_rate', '掌握度'),
            ('total_attempts', '活跃度'),
            ('persistence_rate', '坚持度'),
            ('learning_continuity', '学习连续性'),
            ('knowledge_focus', '知识专注度'),
            ('night_owl_ratio', '夜间活跃度')
        ]

        for feature_id, feature_name in feature_options:
            if feature_id in df_profiles.columns:
                radar_features.append((feature_id, feature_name))

        if len(radar_features) < 3:
            return None

        # 计算每个聚类的特征均值并标准化
        cluster_names = ['高效精英', '勤奋挣扎者', '普通拖延型', '边缘夜猫子']
        radar_data = []

        for cluster_name in cluster_names:
            if cluster_name in df_profiles['cluster_name'].values:
                cluster_data = df_profiles[df_profiles['cluster_name'] == cluster_name]

                cluster_stats = {'cluster_name': cluster_name}
                for feature_id, feature_name in radar_features:
                    # 标准化到0-1范围
                    min_val = df_profiles[feature_id].min()
                    max_val = df_profiles[feature_id].max()
                    if max_val > min_val:
                        normalized = (cluster_data[feature_id].mean() - min_val) / (max_val - min_val)
                    else:
                        normalized = 0.5
                    cluster_stats[feature_name] = normalized

                radar_data.append(cluster_stats)

        if len(radar_data) < 2:
            return None

        radar_df = pd.DataFrame(radar_data)

        # 创建雷达图
        fig = go.Figure()

        # 颜色方案
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, row in radar_df.iterrows():
            values = row[[name for _, name in radar_features]].values.tolist()
            values += values[:1]  # 闭合图形

            theta = [name for _, name in radar_features] + [[name for _, name in radar_features][0]]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=theta,
                name=row['cluster_name'],
                fill='toself',
                opacity=0.7,
                line=dict(color=colors[i % len(colors)], width=2)
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title='四类学习者特征对比雷达图',
            height=600,
            legend=dict(
                x=1.05,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='gray',
                borderwidth=1
            )
        )

        return fig
    except Exception as e:
        st.error(f"创建雷达图时出错: {e}")
        return None


def create_cluster_timeline(df_submit, cluster_names, selected_clusters, sample_per_cluster=2):
    """创建选中聚类的提交时间轴分析"""
    if not df_submit.empty and cluster_names is not None:
        # 获取所有选中的聚类
        selected_cluster_names = selected_clusters

        if '所有学习者' in selected_cluster_names:
            # 如果是所有学习者，从每个聚类中取样
            all_clusters = ['高效精英', '勤奋挣扎者', '普通拖延型', '边缘夜猫子']
            all_student_ids = []
            for cluster in all_clusters:
                if cluster in cluster_names['cluster_name'].values:
                    cluster_ids = cluster_names[cluster_names['cluster_name'] == cluster]['student_ID'].tolist()
                    if cluster_ids:
                        # 从每个聚类中随机选择样本
                        sample_size = min(sample_per_cluster, len(cluster_ids))
                        sampled_ids = random.sample(cluster_ids, sample_size)
                        all_student_ids.extend(sampled_ids)
            student_ids = all_student_ids
        else:
            # 获取选中聚类的学生ID
            student_ids = []
            for cluster in selected_cluster_names:
                if cluster in cluster_names['cluster_name'].values:
                    cluster_ids = cluster_names[cluster_names['cluster_name'] == cluster]['student_ID'].tolist()
                    if cluster_ids:
                        # 从每个选中聚类中随机选择样本
                        sample_size = min(sample_per_cluster, len(cluster_ids))
                        sampled_ids = random.sample(cluster_ids, sample_size)
                        student_ids.extend(sampled_ids)

        if not student_ids:
            return None

        try:
            # 创建子图
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=len(student_ids),
                cols=1,
                subplot_titles=[
                    f'{cluster_names[cluster_names["student_ID"] == sid]["cluster_name"].iloc[0] if sid in cluster_names["student_ID"].values else "未知"} - {sid[:8]}...'
                    for sid in student_ids],
                vertical_spacing=0.12
            )

            for idx, student_id in enumerate(student_ids, 1):
                student_data = df_submit[df_submit['student_ID'] == student_id].copy()
                student_data = student_data.sort_values('submit_time')

                # 重置索引作为x轴
                student_data['submission_index'] = range(1, len(student_data) + 1)

                # 计算提交间隔
                student_data['time_diff'] = student_data['submit_time'].diff().dt.total_seconds().fillna(0)
                student_data['rapid_submission'] = (student_data['time_diff'] < 300) & (student_data['time_diff'] > 0)

                # 准备数据
                x_vals = student_data['submission_index']

                # 正确提交
                correct_data = student_data[student_data['is_passed'] == 1]
                if not correct_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=correct_data['submission_index'],
                            y=[0] * len(correct_data),
                            mode='markers',
                            marker=dict(color='green', symbol='circle', size=10),
                            name='正确提交',
                            legendgroup='正确提交',
                            showlegend=(idx == 1),
                            text=[f"时间: {t}<br>题目: {tid[:8]}..."
                                  for t, tid in zip(correct_data['submit_time'], correct_data['title_ID'])],
                            hoverinfo='text'
                        ),
                        row=idx, col=1
                    )

                # 错误提交
                error_data = student_data[student_data['is_passed'] == 0]
                if not error_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=error_data['submission_index'],
                            y=[0] * len(error_data),
                            mode='markers',
                            marker=dict(color='red', symbol='circle', size=10),
                            name='错误提交',
                            legendgroup='错误提交',
                            showlegend=(idx == 1),
                            text=[f"时间: {t}<br>题目: {tid[:8]}..."
                                  for t, tid in zip(error_data['submit_time'], error_data['title_ID'])],
                            hoverinfo='text'
                        ),
                        row=idx, col=1
                    )

                # 快速提交（5分钟内连续提交）
                rapid_data = student_data[student_data['rapid_submission'] == True]
                if not rapid_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=rapid_data['submission_index'],
                            y=[0.05] * len(rapid_data),
                            mode='markers',
                            marker=dict(color='orange', symbol='triangle-up', size=12),
                            name='快速连续提交',
                            legendgroup='快速连续提交',
                            showlegend=(idx == 1),
                            text=[f"时间: {t}<br>间隔: {diff:.0f}秒"
                                  for t, diff in zip(rapid_data['submit_time'], rapid_data['time_diff'])],
                            hoverinfo='text'
                        ),
                        row=idx, col=1
                    )

                # 获取学生聚类信息
                cluster_info = cluster_names[cluster_names['student_ID'] == student_id]
                cluster_name = cluster_info['cluster_name'].iloc[0] if not cluster_info.empty else '未知'

                # 添加统计信息
                total_submissions = len(student_data)
                correct_rate = student_data['is_passed'].mean() * 100
                rapid_rate = student_data['rapid_submission'].mean() * 100

                fig.add_annotation(
                    text=f"聚类: {cluster_name} | 总提交: {total_submissions} | 正确率: {correct_rate:.1f}% | 快速提交: {rapid_rate:.1f}%",
                    xref=f"x{idx}",
                    yref=f"y{idx}",
                    x=0.02,
                    y=1.08,
                    showarrow=False,
                    font=dict(size=10, color='black'),
                    bgcolor='rgba(255, 255, 255, 0.7)',
                    bordercolor='gray',
                    borderwidth=1,
                    borderpad=4,
                    row=idx,
                    col=1
                )

            # 更新布局
            fig.update_layout(
                height=250 * len(student_ids),
                title_text="学习者提交行为时间轴分析",
                showlegend=True,
                legend=dict(
                    x=1.02,
                    y=0.98,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='gray',
                    borderwidth=1
                )
            )

            # 隐藏y轴
            for i in range(1, len(student_ids) + 1):
                fig.update_yaxes(visible=False, row=i, col=1)
                fig.update_xaxes(title_text="提交序号", row=i, col=1)

            return fig
        except Exception as e:
            st.error(f"创建时间轴时出错: {e}")
            return None
    return None


def create_time_analysis_charts(df):
    """创建时间分析图表"""
    charts = []

    # 1. 高峰答题时段分布
    if 'peak_hour' in df.columns:
        hour_counts = df['peak_hour'].value_counts().sort_index()
        hour_df = pd.DataFrame({
            '小时': hour_counts.index,
            '人数': hour_counts.values
        })

        fig1 = px.bar(
            hour_df,
            x='小时',
            y='人数',
            title='高峰答题时段分布',
            color='人数',
            color_continuous_scale='sunset',
            labels={'小时': '小时 (24小时制)', '人数': '学习者数量'}
        )
        fig1.update_xaxes(tickmode='linear', dtick=1)
        charts.append(fig1)

    # 2. 时间段偏好分析
    time_periods = ['morning_sessions', 'afternoon_sessions', 'evening_sessions', 'night_sessions']
    period_labels = ['早晨 (6-12)', '下午 (12-18)', '晚上 (18-23)', '深夜 (23-6)']

    available_periods = []
    for period, label in zip(time_periods, period_labels):
        if period in df.columns:
            available_periods.append((period, label))

    if available_periods:
        period_data = []
        for period, label in available_periods:
            period_data.append({
                '时间段': label,
                '平均提交次数': df[period].mean()
            })

        period_df = pd.DataFrame(period_data)
        fig2 = px.bar(
            period_df,
            x='时间段',
            y='平均提交次数',
            title='时间段偏好分析',
            color='平均提交次数',
            color_continuous_scale='blues'
        )
        charts.append(fig2)

    return charts


def create_performance_charts(df):
    """创建学习表现图表"""
    charts = []

    # 1. 尝试次数 vs 正确率
    if 'total_attempts' in df.columns and 'correct_rate' in df.columns:
        fig1 = px.scatter(
            df,
            x='total_attempts',
            y='correct_rate',
            color='cluster_name' if 'cluster_name' in df.columns else None,
            size='total_attempts',
            hover_data=['student_ID'],
            title='学习活动与表现关系',
            labels={
                'total_attempts': '总尝试次数',
                'correct_rate': '正确率'
            }
        )
        charts.append(fig1)

    # 2. 知识专注度分析
    if 'top_knowledge' in df.columns:
        top_knowledge_counts = df['top_knowledge'].value_counts().head(15)
        knowledge_df = pd.DataFrame({
            '知识点': top_knowledge_counts.index,
            '学习者数量': top_knowledge_counts.values
        })

        fig2 = px.bar(
            knowledge_df,
            y='知识点',
            x='学习者数量',
            orientation='h',
            title='最受关注的知识点 (Top 15)',
            color='学习者数量',
            color_continuous_scale='greens'
        )
        fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
        charts.append(fig2)

    return charts


def make_subplots(*args, **kwargs):
    """Wrapper for plotly's make_subplots to handle import issues"""
    from plotly.subplots import make_subplots as ms
    return ms(*args, **kwargs)


def create_fireworks_plot(df_submit, student_id=None):
    """创建月内活跃度烟花图，如果指定学生ID则只显示该学生的数据"""
    if df_submit.empty or 'date' not in df_submit.columns:
        return None

    try:
        # 如果指定了学生ID，只显示该学生的数据
        if student_id:
            df_submit = df_submit[df_submit['student_ID'] == student_id].copy()
            title_suffix = f"（学生 {student_id[:8]}...）"
        else:
            title_suffix = ""

        if df_submit.empty:
            return None

        # 按日期统计活跃度
        daily_activity = df_submit.groupby('date').size()

        # 创建连续日期索引
        date_range = pd.date_range(
            start=min(daily_activity.index),
            end=max(daily_activity.index)
        )

        # 重新索引以填充缺失日期
        daily_activity = daily_activity.reindex(date_range, fill_value=0)

        # 创建烟花图
        fig = go.Figure()

        for i, (date, count) in enumerate(daily_activity.items()):
            if count > 0:
                # 计算点的大小和颜色
                size = min(30, count * 2) + 5
                color_intensity = min(1.0, count / max(daily_activity.max(), 1))
                color = f'rgb({int(255 * color_intensity)}, {int(100 * (1 - color_intensity))}, 0)'

                # 添加主点
                fig.add_trace(go.Scatter(
                    x=[date],
                    y=[count],
                    mode='markers',
                    marker=dict(
                        size=size,
                        color=color,
                        line=dict(width=1, color='darkred')
                    ),
                    name=f"日期: {date}",
                    text=f"日期: {date}<br>提交次数: {count}",
                    hoverinfo='text',
                    showlegend=False
                ))

                # 添加烟花效果（小点）
                if count > 10:
                    n_sparks = min(10, count // 5)
                    for _ in range(n_sparks):
                        spark_offset = np.random.uniform(-2, 2)
                        spark_size = np.random.uniform(2, 5)
                        fig.add_trace(go.Scatter(
                            x=[date + pd.Timedelta(days=spark_offset / 10)],
                            y=[count + np.random.uniform(-count / 10, count / 10)],
                            mode='markers',
                            marker=dict(
                                size=spark_size,
                                color='yellow',
                                opacity=0.6
                            ),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

        fig.update_layout(
            title=f'月内活跃度烟花图{title_suffix}',
            xaxis_title='日期',
            yaxis_title='提交次数',
            height=400,
            showlegend=False
        )

        return fig
    except Exception as e:
        st.error(f"创建烟花图时出错: {e}")
        return None


def create_student_analysis_timeline(df_submit, student_id):
    """创建特定学生的详细时间轴分析"""
    if df_submit.empty:
        return None

    try:
        # 获取该学生的所有提交记录
        student_data = df_submit[df_submit['student_ID'] == student_id].copy()

        if student_data.empty:
            return None

        student_data = student_data.sort_values('submit_time')

        # 重置索引作为x轴
        student_data['submission_index'] = range(1, len(student_data) + 1)

        # 计算提交间隔
        student_data['time_diff'] = student_data['submit_time'].diff().dt.total_seconds().fillna(0)
        student_data['rapid_submission'] = (student_data['time_diff'] < 300) & (student_data['time_diff'] > 0)

        # 创建时间轴图
        fig = go.Figure()

        # 正确提交
        correct_data = student_data[student_data['is_passed'] == 1]
        if not correct_data.empty:
            fig.add_trace(go.Scatter(
                x=correct_data['submission_index'],
                y=[0] * len(correct_data),
                mode='markers',
                marker=dict(color='green', symbol='circle', size=12),
                name='正确提交',
                text=[f"时间: {t}<br>题目: {tid[:8]}...<br>分数: {score}"
                      for t, tid, score in
                      zip(correct_data['submit_time'], correct_data['title_ID'], correct_data['score'])],
                hoverinfo='text'
            ))

        # 错误提交
        error_data = student_data[student_data['is_passed'] == 0]
        if not error_data.empty:
            fig.add_trace(go.Scatter(
                x=error_data['submission_index'],
                y=[0] * len(error_data),
                mode='markers',
                marker=dict(color='red', symbol='circle', size=12),
                name='错误提交',
                text=[f"时间: {t}<br>题目: {tid[:8]}...<br>分数: {score}"
                      for t, tid, score in zip(error_data['submit_time'], error_data['title_ID'], error_data['score'])],
                hoverinfo='text'
            ))

        # 快速提交（5分钟内连续提交）
        rapid_data = student_data[student_data['rapid_submission'] == True]
        if not rapid_data.empty:
            fig.add_trace(go.Scatter(
                x=rapid_data['submission_index'],
                y=[0.05] * len(rapid_data),
                mode='markers',
                marker=dict(color='orange', symbol='triangle-up', size=15),
                name='快速连续提交',
                text=[f"时间: {t}<br>间隔: {diff:.0f}秒"
                      for t, diff in zip(rapid_data['submit_time'], rapid_data['time_diff'])],
                hoverinfo='text'
            ))

        # 添加连接线
        fig.add_trace(go.Scatter(
            x=student_data['submission_index'],
            y=[0] * len(student_data),
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            name='提交序列',
            showlegend=False
        ))

        # 更新布局
        total_submissions = len(student_data)
        correct_rate = student_data['is_passed'].mean() * 100
        rapid_rate = student_data['rapid_submission'].mean() * 100
        avg_time_between = student_data['time_diff'].mean() / 60  # 转换为分钟
        total_time_spent = student_data['timeconsume'].sum() if 'timeconsume' in student_data.columns else 0

        fig.update_layout(
            title=f'学生 {student_id[:8]}... 的提交行为时间轴',
            height=400,
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='gray',
                borderwidth=1
            ),
            xaxis_title="提交序号",
            yaxis=dict(visible=False),
            margin=dict(l=20, r=200, t=60, b=40),
            annotations=[
                dict(
                    x=0.02,
                    y=1.08,
                    xref="paper",
                    yref="paper",
                    text=f"总提交: {total_submissions} | 正确率: {correct_rate:.1f}% | 快速提交: {rapid_rate:.1f}%",
                    showarrow=False,
                    font=dict(size=12, color='black'),
                    bgcolor='rgba(255, 255, 255, 0.7)',
                    bordercolor='gray',
                    borderwidth=1,
                    borderpad=4
                ),
                dict(
                    x=0.02,
                    y=1.02,
                    xref="paper",
                    yref="paper",
                    text=f"平均间隔: {avg_time_between:.1f}分钟 | 总耗时: {total_time_spent:.0f}秒",
                    showarrow=False,
                    font=dict(size=10, color='black'),
                    bgcolor='rgba(255, 255, 255, 0.5)',
                    bordercolor='lightgray',
                    borderwidth=1,
                    borderpad=4
                )
            ]
        )

        return fig
    except Exception as e:
        st.error(f"创建学生时间轴时出错: {e}")
        return None


def create_student_performance_summary(student_data, student_id, student_profile):
    """创建学生表现摘要"""
    if student_data.empty:
        return None

    # 计算关键指标
    total_submissions = len(student_data)

    # 使用学生画像数据中的正确率（更准确）
    correct_rate_from_profile = student_profile['correct_rate'] * 100 if 'correct_rate' in student_profile else 0

    # 也从提交数据中计算正确率，用于对比
    correct_rate_from_data = student_data['is_passed'].mean() * 100

    # 计算不同知识点的表现
    knowledge_performance = {}
    if 'knowledge_point' in student_data.columns:
        for knowledge in student_data['knowledge_point'].dropna().unique():
            knowledge_data = student_data[student_data['knowledge_point'] == knowledge]
            knowledge_correct_rate = knowledge_data['is_passed'].mean() * 100
            knowledge_performance[knowledge] = knowledge_correct_rate

    # 计算时间模式
    if 'submit_time' in student_data.columns:
        student_data['hour'] = student_data['submit_time'].dt.hour
        peak_hour = student_data['hour'].mode()[0] if not student_data['hour'].mode().empty else 'N/A'
        is_weekend = student_data['submit_time'].dt.dayofweek.isin([5, 6]).mean() * 100
        is_night = ((student_data['hour'] >= 23) | (student_data['hour'] <= 5)).mean() * 100
    else:
        peak_hour = 'N/A'
        is_weekend = 0
        is_night = 0

    # 计算效率指标 - 修复：确保timeconsume是数值类型
    if 'timeconsume' in student_data.columns:
        # 确保timeconsume是数值类型
        time_consume_numeric = pd.to_numeric(student_data['timeconsume'], errors='coerce')
        avg_time_per_submission = time_consume_numeric.mean()
        total_time_spent = time_consume_numeric.sum()
    else:
        avg_time_per_submission = 0
        total_time_spent = 0

    # 创建摘要文本
    summary = f"""
    ### 学生 {student_id[:8]}... 学习表现摘要

    #### 📊 基础指标
    - **总提交次数**: {total_submissions}
    - **平均正确率（画像数据）**: {correct_rate_from_profile:.1f}%
    - **平均正确率（提交数据）**: {correct_rate_from_data:.1f}%
    - **平均答题时间**: {avg_time_per_submission:.1f}秒
    - **总学习时长**: {total_time_spent:.0f}秒 ({total_time_spent / 3600:.1f}小时)

    #### ⏰ 时间模式
    - **最活跃时段**: {peak_hour}:00
    - **周末学习比例**: {is_weekend:.1f}%
    - **夜间学习比例**: {is_night:.1f}%

    """

    # 添加知识点表现
    if knowledge_performance:
        summary += "\n#### 📚 知识点掌握情况\n"
        sorted_knowledge = sorted(knowledge_performance.items(), key=lambda x: x[1], reverse=True)
        for knowledge, rate in sorted_knowledge[:5]:  # 只显示前5个
            summary += f"- **{knowledge}**: {rate:.1f}%\n"

        # 识别最强和最弱知识点
        if len(sorted_knowledge) >= 2:
            best_knowledge, best_rate = sorted_knowledge[0]
            worst_knowledge, worst_rate = sorted_knowledge[-1]
            summary += f"\n#### 💡 建议\n"
            summary += f"- **优势知识点**: {best_knowledge} ({best_rate:.1f}%)\n"
            summary += f"- **需加强知识点**: {worst_knowledge} ({worst_rate:.1f}%)\n"

    return summary


def extract_main_knowledge(knowledge_str):
    """从知识点字符串中提取大知识点（假设用'-'分割）"""
    if not isinstance(knowledge_str, str):
        return "未知知识点"

    # 如果知识点包含"-"，取前面的部分作为大知识点
    if '-' in knowledge_str:
        return knowledge_str.split('-')[0].strip()
    elif '/' in knowledge_str:
        return knowledge_str.split('/')[0].strip()
    elif ':' in knowledge_str:
        return knowledge_str.split(':')[0].strip()
    else:
        return knowledge_str.strip()


def create_student_knowledge_preference_analysis(student_data):
    """创建学生知识点偏好分析（只按大知识点）"""
    if student_data.empty or 'knowledge_point' not in student_data.columns:
        return None

    try:
        # 提取大知识点
        student_data = student_data.copy()
        student_data['main_knowledge'] = student_data['knowledge_point'].apply(extract_main_knowledge)

        # 统计不同大知识点的提交次数
        knowledge_counts = student_data['main_knowledge'].value_counts().head(10)
        knowledge_df = pd.DataFrame({
            '知识点': knowledge_counts.index,
            '提交次数': knowledge_counts.values
        })

        # 计算每个大知识点的正确率
        knowledge_correct_rates = []
        for knowledge in knowledge_df['知识点']:
            knowledge_data = student_data[student_data['main_knowledge'] == knowledge]
            correct_rate = knowledge_data['is_passed'].mean() * 100 if not knowledge_data.empty else 0
            knowledge_correct_rates.append(correct_rate)

        knowledge_df['正确率'] = knowledge_correct_rates

        # 创建知识点偏好条形图
        fig = px.bar(
            knowledge_df,
            y='知识点',
            x='提交次数',
            orientation='h',
            title='大知识点偏好分析 (Top 10)',
            color='正确率',
            color_continuous_scale='RdYlGn',
            labels={'知识点': '大知识点', '提交次数': '提交次数', '正确率': '正确率 (%)'},
            hover_data=['正确率']
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500
        )

        return fig, knowledge_df
    except Exception as e:
        st.error(f"创建知识点偏好分析时出错: {e}")
        return None, None


def create_student_daily_activity(df_submit, student_id):
    """创建学生个人每日活跃度趋势"""
    if df_submit.empty:
        return None

    try:
        # 筛选该学生的数据
        student_data = df_submit[df_submit['student_ID'] == student_id].copy()

        if student_data.empty:
            return None

        # 按日期统计
        daily_counts = student_data.groupby(student_data['submit_time'].dt.date).size().reset_index()
        daily_counts.columns = ['date', 'count']

        # 创建连续日期索引
        date_range = pd.date_range(
            start=daily_counts['date'].min(),
            end=daily_counts['date'].max()
        )

        # 重新索引以填充缺失日期
        daily_counts = daily_counts.set_index('date').reindex(date_range, fill_value=0).reset_index()
        daily_counts.columns = ['date', 'count']

        # 创建折线图
        fig = px.line(
            daily_counts,
            x='date',
            y='count',
            title=f"学生 {student_id[:8]}... 每日提交量趋势",
            labels={'date': '日期', 'count': '提交次数'},
            markers=True
        )

        # 添加平均线
        avg_count = daily_counts['count'].mean()
        fig.add_hline(y=avg_count, line_dash="dash", line_color="red",
                      annotation_text=f"日均提交: {avg_count:.1f}次",
                      annotation_position="bottom right")

        return fig
    except Exception as e:
        st.error(f"创建学生每日活跃度趋势图时出错: {e}")
        return None


def create_cluster_explanation_card(student_profile, df_profiles):
    """创建聚类原因解释卡片"""
    if student_profile.empty or df_profiles.empty:
        return ""

    cluster_name = student_profile['cluster_name'] if 'cluster_name' in student_profile else '未知'

    # 定义聚类特征阈值
    cluster_thresholds = {
        '高效精英': {
            'correct_rate': 0.7,
            'total_attempts': 50,
            'persistence_rate': 0.8,
            'learning_continuity': 0.7,
            'night_owl_ratio': 0.3
        },
        '勤奋挣扎者': {
            'correct_rate': 0.5,
            'total_attempts': 80,
            'persistence_rate': 0.9,
            'learning_continuity': 0.6,
            'night_owl_ratio': 0.4
        },
        '普通拖延型': {
            'correct_rate': 0.6,
            'total_attempts': 30,
            'persistence_rate': 0.6,
            'learning_continuity': 0.4,
            'night_owl_ratio': 0.5
        },
        '边缘夜猫子': {
            'correct_rate': 0.4,
            'total_attempts': 20,
            'persistence_rate': 0.4,
            'learning_continuity': 0.3,
            'night_owl_ratio': 0.7
        }
    }

    # 获取学生特征值
    student_features = {}
    feature_names = {
        'correct_rate': '正确率',
        'total_attempts': '尝试次数',
        'persistence_rate': '坚持率',
        'learning_continuity': '学习连续性',
        'night_owl_ratio': '夜间学习比例'
    }

    for feature in feature_names.keys():
        if feature in student_profile:
            student_features[feature] = student_profile[feature]
        else:
            student_features[feature] = 0

    # 获取聚类平均值
    cluster_avg = {}
    if cluster_name in df_profiles['cluster_name'].values:
        cluster_data = df_profiles[df_profiles['cluster_name'] == cluster_name]
        for feature in feature_names.keys():
            if feature in cluster_data.columns:
                cluster_avg[feature] = cluster_data[feature].mean()

    # 创建解释文本
    explanation = f"### 📍 聚类归属解释\n\n"
    explanation += f"该学生被归类为 **{cluster_name}**，主要基于以下特征：\n\n"

    for feature, display_name in feature_names.items():
        if feature in student_features:
            student_val = student_features[feature]
            cluster_val = cluster_avg.get(feature, 0)

            if feature in ['correct_rate', 'persistence_rate', 'learning_continuity', 'night_owl_ratio']:
                student_val_display = f"{student_val * 100:.1f}%"
                cluster_val_display = f"{cluster_val * 100:.1f}%"
            else:
                student_val_display = f"{student_val:.0f}"
                cluster_val_display = f"{cluster_val:.0f}"

            explanation += f"- **{display_name}**: {student_val_display} (该聚类平均: {cluster_val_display})\n"

    # 添加聚类特点描述
    cluster_descriptions = {
        '高效精英': '具有高正确率和高学习效率，学习习惯良好，通常能快速掌握知识',
        '勤奋挣扎者': '学习投入度高但效率偏低，可能需要更多方法指导和时间管理策略',
        '普通拖延型': '学习行为较为分散，存在一定拖延倾向，需要提高学习连续性',
        '边缘夜猫子': '夜间学习活跃度高，但整体学习参与度和表现偏低，需关注学习习惯和作息'
    }

    explanation += f"\n**聚类特点**: {cluster_descriptions.get(cluster_name, '暂无详细描述')}"

    return explanation


def main():
    st.title("📚 学习者行为模式分析仪表板")

    # 加载数据
    df_profiles = load_profile_data()
    df_submit = load_submit_data()

    if df_profiles.empty:
        st.error("无法加载学习者画像数据，请检查文件路径")
        st.stop()

    # 如果没有聚类列，执行聚类分析
    if 'cluster' not in df_profiles.columns:
        # st.info("正在进行聚类分析...")
        df_profiles = perform_clustering(df_profiles)

    # 显示基本信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("学习者总数", len(df_profiles))
    with col2:
        if 'correct_rate' in df_profiles.columns:
            avg_correct = df_profiles['correct_rate'].mean() * 100
            st.metric("平均正确率", f"{avg_correct:.1f}%")
    with col3:
        if 'night_owl_ratio' in df_profiles.columns:
            avg_night = df_profiles['night_owl_ratio'].mean() * 100
            st.metric("夜间学习比例", f"{avg_night:.1f}%")
    with col4:
        if 'weekend_ratio' in df_profiles.columns:
            avg_weekend = df_profiles['weekend_ratio'].mean() * 100
            st.metric("周末学习比例", f"{avg_weekend:.1f}%")

    st.divider()

    # 侧边栏过滤器
    st.sidebar.header("🔍 数据过滤器")

    # 1. 聚类筛选 - 单选下拉框
    if 'cluster_name' in df_profiles.columns:
        # 获取所有聚类类型
        clusters = ['所有学习者', '高效精英', '勤奋挣扎者', '普通拖延型', '边缘夜猫子']

        selected_cluster = st.sidebar.selectbox(
            "选择聚类类型",
            options=clusters,
            index=0,
            help="选择要分析的聚类类型"
        )

        # 根据选择筛选数据
        if selected_cluster == '所有学习者':
            df_filtered = df_profiles.copy()
            selected_clusters = ['所有学习者']
        else:
            df_filtered = df_profiles[df_profiles['cluster_name'] == selected_cluster].copy()
            selected_clusters = [selected_cluster]

    # 2. 特定学生选择 - 搜索框
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 特定学生分析")

    # 获取当前选择聚类内的学生ID列表
    if selected_cluster == '所有学习者':
        # 如果是所有学习者，搜索所有学生
        current_cluster_student_ids = df_profiles['student_ID'].tolist()
        cluster_student_count = len(df_profiles)
    else:
        # 如果是特定聚类，只搜索该聚类内的学生
        current_cluster_student_ids = df_profiles[df_profiles['cluster_name'] == selected_cluster][
            'student_ID'].tolist()
        cluster_student_count = len(current_cluster_student_ids)
        st.sidebar.info(f"当前聚类 **{selected_cluster}** 共有 {cluster_student_count} 名学生")

    # 创建学生搜索框
    student_search = st.sidebar.text_input(
        "搜索学生ID",
        placeholder=f"在当前聚类中输入学生ID搜索",
        help=f"在当前聚类({selected_cluster})中搜索学生ID"
    )

    selected_student_id = None

    # 如果输入了搜索词，显示匹配的学生
    if student_search:
        # 在当前聚类的学生中搜索匹配的学生ID
        matching_students = [sid for sid in current_cluster_student_ids if student_search.lower() in sid.lower()]

        if matching_students:
            # 显示匹配的学生列表
            st.sidebar.markdown(f"**找到 {len(matching_students)} 个匹配的学生**")

            # 如果是少量匹配，使用单选按钮
            if len(matching_students) <= 10:
                selected_student_id = st.sidebar.radio(
                    "选择学生",
                    options=matching_students,
                    format_func=lambda
                        x: f"{x[:12]}... ({df_profiles[df_profiles['student_ID'] == x]['cluster_name'].iloc[0] if not df_profiles[df_profiles['student_ID'] == x].empty else '未知'})"
                )
            else:
                # 如果匹配太多，使用下拉选择
                student_options = [
                    f"{sid[:12]}... ({df_profiles[df_profiles['student_ID'] == sid]['cluster_name'].iloc[0] if not df_profiles[df_profiles['student_ID'] == sid].empty else '未知'})"
                    for sid in matching_students[:50]]
                selected_student_option = st.sidebar.selectbox(
                    "选择学生 (显示前50个)",
                    options=student_options,
                    index=0
                )
                # 从选项字符串中提取学生ID
                if selected_student_option:
                    selected_student_id = selected_student_option.split(" (")[0]
                    # 找到完整的学生ID
                    for sid in matching_students:
                        if sid.startswith(selected_student_id.split("...")[0]):
                            selected_student_id = sid
                            break
        else:
            st.sidebar.warning(f"在当前聚类中未找到匹配的学生")

    # 如果选择了特定学生，更新筛选数据
    if selected_student_id:
        df_filtered = df_profiles[df_profiles['student_ID'] == selected_student_id].copy()
        st.sidebar.success(f"已选择学生: {selected_student_id[:12]}...")

    # 显示聚类分布统计
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 聚类分布统计")

    if 'cluster_name' in df_profiles.columns and not selected_student_id:
        cluster_counts = df_profiles['cluster_name'].value_counts()
        for cluster in ['高效精英', '勤奋挣扎者', '普通拖延型', '边缘夜猫子']:
            if cluster in cluster_counts:
                count = cluster_counts[cluster]
                percentage = (count / len(df_profiles)) * 100
                st.sidebar.markdown(f"**{cluster}**: {count}人 ({percentage:.1f}%)")

    st.sidebar.markdown(f"---")
    if selected_student_id:
        # 获取学生聚类信息
        student_cluster = df_filtered['cluster_name'].iloc[0] if not df_filtered.empty else '未知'
        st.sidebar.markdown(f"**当前分析**: 学生 {selected_student_id[:12]}...")
        st.sidebar.markdown(f"**所属聚类**: {student_cluster}")

        # 显示聚类原因解释
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 聚类归属原因")
        student_profile = df_filtered.iloc[0] if not df_filtered.empty else pd.Series()
        explanation = create_cluster_explanation_card(student_profile, df_profiles)
        if explanation:
            st.sidebar.markdown(explanation)
    else:
        st.sidebar.markdown(f"**当前分析**: {selected_cluster if 'selected_cluster' in locals() else '所有学习者'}")
        if selected_cluster != '所有学习者':
            st.sidebar.markdown(f"**聚类学生数**: {cluster_student_count} 名")
    st.sidebar.markdown(f"**筛选结果**: {len(df_filtered)} 名学习者")
    # 标签页布局
    if selected_student_id:
        # 如果是特定学生分析，显示不同的标签页
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 学生详细分析",
            "⏰ 时间模式分析",
            "📅 月内活跃度",
            "🎯 知识点偏好分析"  # 修改标签页名称
        ])
    else:
        # 如果是聚类分析，显示原来的标签页
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "⏰ 时间模式",
            "📈 学习表现",
            "📅 月内活跃度",
            "🧭 特征雷达图",
            "📊 行为时间轴分析"
        ])

    if selected_student_id:
        # 特定学生分析模式
        with tab1:
            st.header("学生详细分析")

            # 获取学生数据
            student_profile = df_filtered.iloc[0] if not df_filtered.empty else None
            student_submit_data = df_submit[
                df_submit['student_ID'] == selected_student_id] if not df_submit.empty else pd.DataFrame()

            if student_profile is not None and not student_submit_data.empty:
                # 显示学生基本信息
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("正确率", f"{student_profile['correct_rate'] * 100:.1f}%")
                with col2:
                    if 'total_attempts' in student_profile:
                        st.metric("总尝试次数", f"{student_profile['total_attempts']:.0f}")
                with col3:
                    if 'persistence_rate' in student_profile:
                        st.metric("坚持率", f"{student_profile['persistence_rate'] * 100:.1f}%")
                with col4:
                    if 'learning_continuity' in student_profile:
                        st.metric("学习连续性", f"{student_profile['learning_continuity'] * 100:.1f}%")

                # 解释正确率不一致问题
                st.info("""
                **注意**：上方正确率来自学生画像数据（基于所有提交的得分率计算），
                而下方摘要中的正确率可能基于不同的计算方法（如是否完全正确）。
                学生画像数据中的正确率通常更准确地反映了学生的整体表现。
                """)

                # 学生表现摘要
                st.markdown("---")
                summary = create_student_performance_summary(student_submit_data, selected_student_id, student_profile)
                if summary:
                    st.markdown(summary)

                # 学生时间轴分析
                st.markdown("---")
                st.subheader("📈 提交行为时间轴")
                timeline_fig = create_student_analysis_timeline(df_submit, selected_student_id)
                if timeline_fig:
                    st.plotly_chart(timeline_fig, use_container_width=True)

                # 知识点掌握情况
                if 'knowledge_point' in student_submit_data.columns:
                    st.markdown("---")
                    st.subheader("📚 知识点掌握情况")

                    # 计算各知识点表现
                    knowledge_stats = student_submit_data.groupby('knowledge_point').agg({
                        'is_passed': 'mean',
                        'title_ID': 'count'
                    }).reset_index()
                    knowledge_stats.columns = ['知识点', '正确率', '提交次数']
                    knowledge_stats['正确率'] = knowledge_stats['正确率'] * 100

                    # 创建知识点表现条形图
                    if not knowledge_stats.empty:
                        knowledge_fig = px.bar(
                            knowledge_stats.sort_values('正确率', ascending=False),
                            x='知识点',
                            y='正确率',
                            title='各知识点正确率',
                            color='正确率',
                            color_continuous_scale='RdYlGn',
                            labels={'正确率': '正确率 (%)', '知识点': '知识点'}
                        )
                        knowledge_fig.update_layout(xaxis_tickangle=45)
                        st.plotly_chart(knowledge_fig, use_container_width=True)
            else:
                st.warning("无法获取学生详细数据")

        with tab2:
            st.header("时间模式分析")

            time_charts = create_time_analysis_charts(df_filtered)

            if time_charts:
                if len(time_charts) >= 1:
                    st.plotly_chart(time_charts[0], use_container_width=True)

                if len(time_charts) >= 2:
                    st.plotly_chart(time_charts[1], use_container_width=True)

        with tab3:
            st.header("月内活跃度分析")

            if not df_submit.empty:
                # 热力图 - 只显示该学生的数据
                heatmap_fig = create_monthly_activity_heatmap(df_submit, selected_student_id)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)

                # 学生个人每日活跃度趋势
                daily_fig = create_student_daily_activity(df_submit, selected_student_id)
                if daily_fig:
                    st.plotly_chart(daily_fig, use_container_width=True)

                # 烟花图 - 只显示该学生的数据
                fireworks_fig = create_fireworks_plot(df_submit, selected_student_id)
                if fireworks_fig:
                    st.plotly_chart(fireworks_fig, use_container_width=True)
            else:
                st.warning("无法显示月内活跃度分析，提交数据加载失败")

        with tab4:
            st.header("知识点偏好分析")

            if not df_submit.empty:
                student_submit_data = df_submit[df_submit['student_ID'] == selected_student_id]

                if not student_submit_data.empty and 'knowledge_point' in student_submit_data.columns:
                    # 显示基础统计
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        unique_knowledge = student_submit_data['knowledge_point'].nunique()
                        st.metric("涉及知识点数", f"{unique_knowledge}")
                    with col2:
                        total_submissions = len(student_submit_data)
                        st.metric("总提交次数", f"{total_submissions}")
                    with col3:
                        if 'is_passed' in student_submit_data.columns:
                            correct_rate = student_submit_data['is_passed'].mean() * 100
                            st.metric("整体正确率", f"{correct_rate:.1f}%")

                    # 知识点偏好分析
                    st.subheader("📚 大知识点偏好分析")
                    knowledge_fig, knowledge_df = create_student_knowledge_preference_analysis(student_submit_data)

                    if knowledge_fig:
                        st.plotly_chart(knowledge_fig, use_container_width=True)
                    else:
                        st.warning("无法生成知识点偏好图表")

                    # 知识点偏好总结
                    st.subheader("🎯 知识点偏好总结")

                    if knowledge_df is not None and not knowledge_df.empty:
                        # 计算知识点覆盖度
                        # 所有大知识点
                        all_main_knowledge = df_submit.copy()
                        if 'knowledge_point' in all_main_knowledge.columns:
                            all_main_knowledge['main_knowledge'] = all_main_knowledge['knowledge_point'].apply(
                                extract_main_knowledge)
                            unique_student_knowledge = student_submit_data['knowledge_point'].apply(
                                extract_main_knowledge).nunique()
                            total_knowledge = all_main_knowledge['main_knowledge'].nunique()
                        else:
                            unique_student_knowledge = 0
                            total_knowledge = 0

                        # 获取最偏好和最不偏好的知识点
                        if len(knowledge_df) >= 2:
                            most_preferred = knowledge_df.iloc[0]
                            least_preferred = knowledge_df.iloc[-1]

                            st.markdown(f"""
                            #### 📊 知识点偏好统计
                            - **最偏好知识点**: {most_preferred['知识点']}
                              - 提交次数: {most_preferred['提交次数']}次
                              - 正确率: {most_preferred['正确率']:.1f}%
                            - **最少涉及知识点**: {least_preferred['知识点']}
                              - 提交次数: {least_preferred['提交次数']}次
                              - 正确率: {least_preferred['正确率']:.1f}%
                            """)

                        # 知识点覆盖率
                        if total_knowledge > 0:
                            coverage_rate = unique_student_knowledge / total_knowledge * 100
                            st.markdown(f"""
                            #### 📈 知识点覆盖情况
                            - **涉及大知识点数**: {unique_student_knowledge}个
                            - **总大知识点数**: {total_knowledge}个
                            - **大知识点覆盖率**: {coverage_rate:.1f}%
                            """)

                        # 学习建议
                        st.subheader("💡 学习建议")

                        # 根据偏好情况给出建议
                        if knowledge_df is not None and len(knowledge_df) >= 3:
                            # 获取正确率最高和最低的知识点
                            knowledge_df_sorted_by_correct = knowledge_df.sort_values('正确率', ascending=False)
                            best_knowledge = knowledge_df_sorted_by_correct.iloc[0]
                            worst_knowledge = knowledge_df_sorted_by_correct.iloc[-1]

                            if best_knowledge['正确率'] >= 70:
                                st.success(
                                    f"**优势领域**: {best_knowledge['知识点']}（正确率: {best_knowledge['正确率']:.1f}%）")
                            else:
                                st.info(
                                    f"**相对优势**: {best_knowledge['知识点']}（正确率: {best_knowledge['正确率']:.1f}%）")

                            if worst_knowledge['正确率'] <= 50:
                                st.warning(
                                    f"**需加强领域**: {worst_knowledge['知识点']}（正确率: {worst_knowledge['正确率']:.1f}%）")
                            else:
                                st.info(
                                    f"**可提升领域**: {worst_knowledge['知识点']}（正确率: {worst_knowledge['正确率']:.1f}%）")

                            # 学习平衡性建议
                            if coverage_rate < 50:
                                st.warning("**建议**: 知识点覆盖较窄，建议尝试更多不同类型的大知识点题目")
                            elif coverage_rate > 80:
                                st.success("**优势**: 知识点覆盖广泛，学习全面性较好")
                            else:
                                st.info("**建议**: 知识点覆盖适中，继续保持学习广度")

                        # 显示学生的top_knowledge（如果画像数据中有）
                        if student_profile is not None and 'top_knowledge' in student_profile:
                            top_knowledge = student_profile['top_knowledge']
                            if pd.notna(top_knowledge):
                                st.markdown("---")
                                st.subheader("📋 画像数据中的知识点信息")
                                st.info(f"**最常提交的知识点**: {top_knowledge}")
                    else:
                        st.warning("没有足够的知识点数据进行详细分析")

                    # 知识点分布统计表
                    if knowledge_df is not None and not knowledge_df.empty:
                        st.subheader("📋 知识点分布详情")
                        display_df = knowledge_df.copy()
                        display_df['正确率'] = display_df['正确率'].apply(lambda x: f"{x:.1f}%")
                        st.dataframe(display_df, use_container_width=True)

                else:
                    if student_submit_data.empty:
                        st.warning("该学生没有提交记录")
                    else:
                        st.warning("提交数据中缺少知识点信息。请检查Data_TitleInfo.csv文件是否包含正确的知识点映射。")
                        # 显示有哪些列可用
                        st.info(f"可用的列: {', '.join(student_submit_data.columns.tolist())}")
            else:
                st.warning("无法进行知识点偏好分析，提交数据加载失败")

    else:
        # 聚类分析模式
        with tab1:
            st.header("时间模式分析")

            time_charts = create_time_analysis_charts(df_filtered)

            if time_charts:
                if len(time_charts) >= 1:
                    st.plotly_chart(time_charts[0], use_container_width=True)

                if len(time_charts) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(time_charts[1], use_container_width=True)

                    with col2:
                        # 学习连续性分析
                        if 'learning_continuity' in df_filtered.columns:
                            continuity_fig = px.box(
                                df_filtered,
                                x='cluster_name' if 'cluster_name' in df_filtered.columns else None,
                                y='learning_continuity',
                                title='学习连续性分析',
                                labels={
                                    'learning_continuity': '学习连续性',
                                    'cluster_name': '聚类类型'
                                }
                            )
                            st.plotly_chart(continuity_fig, use_container_width=True)

        with tab2:
            st.header("学习表现分析")

            performance_charts = create_performance_charts(df_filtered)

            if performance_charts:
                col1, col2 = st.columns(2)
                with col1:
                    if len(performance_charts) >= 1:
                        st.plotly_chart(performance_charts[0], use_container_width=True)

                with col2:
                    if len(performance_charts) >= 2:
                        st.plotly_chart(performance_charts[1], use_container_width=True)

            # 行为模式分析
            st.subheader("行为模式分析")
            col1, col2 = st.columns(2)

            with col1:
                if 'persistence_rate' in df_filtered.columns and 'correct_rate' in df_filtered.columns:
                    behavior_fig1 = px.scatter(
                        df_filtered,
                        x='persistence_rate',
                        y='correct_rate',
                        color='cluster_name' if 'cluster_name' in df_filtered.columns else None,
                        size='total_attempts' if 'total_attempts' in df_filtered.columns else None,
                        title='坚持性与正确率关系',
                        labels={
                            'persistence_rate': '坚持率',
                            'correct_rate': '正确率'
                        }
                    )
                    st.plotly_chart(behavior_fig1, use_container_width=True)

            with col2:
                if 'avg_time_per_attempt' in df_filtered.columns and 'correct_rate' in df_filtered.columns:
                    behavior_fig2 = px.scatter(
                        df_filtered,
                        x='avg_time_per_attempt',
                        y='correct_rate',
                        color='cluster_name' if 'cluster_name' in df_filtered.columns else None,
                        title='答题效率分析',
                        labels={
                            'avg_time_per_attempt': '平均答题时间(秒)',
                            'correct_rate': '正确率'
                        }
                    )
                    st.plotly_chart(behavior_fig2, use_container_width=True)

        with tab3:
            st.header("月内活跃度分析")

            if not df_submit.empty:
                # 热力图
                heatmap_fig = create_monthly_activity_heatmap(df_submit)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)

                # 烟花图和趋势图
                col1, col2 = st.columns(2)
                with col1:
                    # 烟花图
                    fireworks_fig = create_fireworks_plot(df_submit)
                    if fireworks_fig:
                        st.plotly_chart(fireworks_fig, use_container_width=True)

                with col2:
                    # 每日提交量趋势
                    daily_counts = df_submit.groupby(df_submit['submit_time'].dt.date).size().reset_index()
                    daily_counts.columns = ['date', 'count']

                    daily_fig = px.line(
                        daily_counts,
                        x='date',
                        y='count',
                        title="每日提交量趋势",
                        labels={'date': '日期', 'count': '提交次数'},
                        markers=True
                    )
                    st.plotly_chart(daily_fig, use_container_width=True)

                # 活跃度分布箱线图
                if 'cluster_name' in df_profiles.columns:
                    st.subheader("各聚类活跃度对比")
                    activity_fig = px.box(
                        df_profiles,
                        x='cluster_name',
                        y='total_attempts' if 'total_attempts' in df_profiles.columns else 'total_sessions',
                        title="各聚类活跃度对比",
                        labels={
                            'cluster_name': '聚类类型',
                            'total_attempts': '总提交次数',
                            'total_sessions': '总学习次数'
                        }
                    )
                    st.plotly_chart(activity_fig, use_container_width=True)
            else:
                st.warning("无法显示月内活跃度分析，提交数据加载失败")

        with tab4:
            st.header("学习者特征雷达图对比")

            radar_fig = create_enhanced_radar_chart(df_profiles)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)

            # 添加特征解释
            with st.expander("特征解释"):
                st.markdown("""
                - **掌握度**: 基于正确率的综合评估
                - **活跃度**: 基于总提交次数的评估（已标准化）
                - **坚持度**: 失败后继续尝试的比例
                - **学习连续性**: 学习天数占总天数的比例，反映学习习惯
                - **知识专注度**: 在主要知识点上的集中程度
                - **夜间活跃度**: 深夜学习（23:00-05:00）的比例
                """)

                st.markdown("""
                **四类学习者特征对比**:

                1. **高效精英**: 掌握度高、效率高、学习习惯良好
                2. **勤奋挣扎者**: 活跃度高但掌握度低，需要方法指导
                3. **普通拖延型**: 各项指标中等，需要提高学习连续性
                4. **边缘夜猫子**: 夜间活跃度高但整体表现不佳
                """)

        with tab5:
            st.header("行为时间轴分析")

            if 'cluster_name' in df_profiles.columns and not df_submit.empty:
                # 显示当前选中的聚类信息
                if selected_cluster == '所有学习者':
                    st.info(f"正在分析所有学习者的行为模式（从每个聚类中随机取样）")
                else:
                    st.info(f"正在分析 {selected_cluster} 的行为模式")

                # 时间轴分析
                timeline_fig = create_cluster_timeline(df_submit, df_profiles, selected_clusters)
                if timeline_fig:
                    st.plotly_chart(timeline_fig, use_container_width=True)
                else:
                    st.warning("无法生成时间轴分析图")

                # 行为模式统计
                st.subheader(f"{selected_cluster} 行为模式统计")

                # 计算统计指标
                if selected_cluster == '所有学习者':
                    analysis_data = df_profiles
                    analysis_submit = df_submit
                else:
                    analysis_data = df_profiles[df_profiles['cluster_name'] == selected_cluster]
                    student_ids = analysis_data['student_ID'].tolist()
                    analysis_submit = df_submit[df_submit['student_ID'].isin(student_ids)]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if not analysis_submit.empty:
                        avg_correct = analysis_submit['is_passed'].mean() * 100
                        st.metric("平均正确率", f"{avg_correct:.1f}%")
                    else:
                        st.metric("平均正确率", "N/A")

                with col2:
                    if 'total_attempts' in analysis_data.columns and not analysis_data.empty:
                        avg_attempts = analysis_data['total_attempts'].mean()
                        st.metric("平均尝试次数", f"{avg_attempts:.0f}")
                    else:
                        st.metric("平均尝试次数", "N/A")

                with col3:
                    if 'night_owl_ratio' in analysis_data.columns and not analysis_data.empty:
                        night_ratio = analysis_data['night_owl_ratio'].mean() * 100
                        st.metric("夜间学习比例", f"{night_ratio:.1f}%")
                    else:
                        st.metric("夜间学习比例", "N/A")

                with col4:
                    if 'persistence_rate' in analysis_data.columns and not analysis_data.empty:
                        persistence = analysis_data['persistence_rate'].mean() * 100
                        st.metric("平均坚持率", f"{persistence:.1f}%")
                    else:
                        st.metric("平均坚持率", "N/A")

                # 核心发现总结
                st.subheader("🔍 核心发现与教学建议")

                if selected_cluster == '所有学习者':
                    st.markdown("""
                    **整体学习者行为模式分析**:

                    - 通过时间轴分析可以观察到不同聚类学习者的行为差异
                    - 高效精英型学习者通常提交间隔合理，正确率高
                    - 勤奋挣扎者尝试次数多但正确率偏低，可能存在试错模式
                    - 普通拖延型学习者提交时间分散，学习连续性有待提高
                    - 边缘夜猫子夜间活跃度高，需关注作息健康
                    """)
                elif selected_cluster == '勤奋挣扎者':
                    st.markdown("""
                    **勤奋挣扎者行为特点**:

                    1. **高投入低产出现象**
                    - 投入大量时间（高尝试次数）但正确率偏低
                    - 可能存在"机械刷题"而非深入理解

                    2. **学习行为模式问题**
                    - **频繁更换题目**: 答错后容易放弃当前题目，切换到其他题目
                    - **试错模式明显**: 依赖多次尝试而非深入思考
                    - **缺乏连续性**: 学习时间分散，难以形成知识体系

                    3. **教学干预建议**:
                    - 加强基础知识巩固，提供解题思路指导
                    - 培养深度思考习惯，减少盲目尝试
                    - 优化时间分配，提高学习效率
                    """)
                elif selected_cluster == '高效精英':
                    st.markdown("""
                    **高效精英行为特点**:

                    1. **高效率高产出**
                    - 正确率高，尝试次数合理
                    - 学习习惯良好，时间管理有效

                    2. **学习行为模式优势**
                    - 提交间隔合理，学习连续性高
                    - 知识掌握扎实，解题思路清晰

                    3. **教学建议**:
                    - 提供进阶挑战题目
                    - 鼓励参与竞赛和研究项目
                    - 培养创新思维和问题解决能力
                    """)
                elif selected_cluster == '普通拖延型':
                    st.markdown("""
                    **普通拖延型行为特点**:

                    1. **中等表现，时效问题**
                    - 正确率中等，但首次通过耗时长
                    - 学习时间分散，连续性不足

                    2. **学习行为模式问题**
                    - 可能在学习中遇到困难时选择拖延
                    - 缺乏及时的学习反馈和调整

                    3. **教学干预建议**:
                    - 设置短期学习目标和截止日期
                    - 提供及时的学习反馈和指导
                    - 培养学习计划和时间管理能力
                    """)
                elif selected_cluster == '边缘夜猫子':
                    st.markdown("""
                    **边缘夜猫子行为特点**:

                    1. **低参与度，作息异常**
                    - 学习参与度低，表现不佳
                    - 夜间学习比例高，作息不规律

                    2. **高风险群体**
                    - 学习投入不足，效率低下
                    - 可能存在学习动机或环境问题

                    3. **教学干预建议**:
                    - 关注学生身心健康和学习动机
                    - 提供个性化学习支持和辅导
                    - 建立健康的学习和作息习惯
                    """)
            else:
                st.warning("无法进行行为时间轴分析，请确保数据加载完整且包含聚类信息")

    # 页脚
    st.divider()
    st.caption("📊 学习者行为分析仪表板 | 基于时序多变量教育数据的可视分析")


if __name__ == "__main__":
    main()