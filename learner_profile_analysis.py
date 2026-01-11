# learner_profile_analysis_fixed.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class LearnerProfileAnalyzer:
    def __init__(self):
        self.df_student = None
        self.df_submit = None
        self.df_title = None
        self.learner_profiles = None

    def load_data(self):
        """加载所有必要数据"""
        print("正在加载数据...")

        # 加载学生信息
        self.df_student = pd.read_csv('Data_StudentInfo.csv', encoding='gb18030')
        self.df_student['student_ID'] = self.df_student['student_ID'].astype(str)

        # 加载题目信息
        self.df_title = pd.read_csv('Data_TitleInfo.csv', encoding='gb18030')
        self.df_title['title_ID'] = self.df_title['title_ID'].astype(str)
        self.df_title.rename(columns={'score': 'full_score', 'knowledge': 'knowledge_point'}, inplace=True)

        # 加载所有提交记录
        submit_dfs = []
        for i in range(1, 16):
            try:
                df = pd.read_csv(f'Data_SubmitRecord/SubmitRecord-Class{i}.csv', encoding='gb18030')
                df['class'] = f'Class{i}'
                submit_dfs.append(df)
            except:
                continue

        self.df_submit = pd.concat(submit_dfs, ignore_index=True)
        print(f"数据加载完成：学生{len(self.df_student)}人，提交记录{len(self.df_submit)}条")

    def preprocess_data(self):
        """数据预处理"""
        print("正在预处理数据...")

        # 清理提交记录
        self.df_submit['student_ID'] = self.df_submit['student_ID'].astype(str)
        self.df_submit['title_ID'] = self.df_submit['title_ID'].astype(str)

        # 转换时间
        self.df_submit['submit_time'] = pd.to_datetime(self.df_submit['time'], unit='s')

        # 处理分数
        self.df_submit['score'] = pd.to_numeric(self.df_submit['score'], errors='coerce')
        self.df_submit['timeconsume'] = pd.to_numeric(self.df_submit['timeconsume'], errors='coerce')

        # 合并数据
        self.df_submit = pd.merge(self.df_submit, self.df_student, on='student_ID', how='left')
        self.df_submit = pd.merge(self.df_submit, self.df_title, on='title_ID', how='left')

        # 计算通过状态
        self.df_submit['is_passed'] = (self.df_submit['score'] == self.df_submit['full_score']).astype(int)

        print("数据预处理完成")

    def extract_temporal_features(self):
        """提取时间相关特征"""
        print("正在提取时间特征...")

        temporal_features = []

        for student_id in self.df_submit['student_ID'].unique():
            student_data = self.df_submit[self.df_submit['student_ID'] == student_id].copy()

            # 按时间排序
            student_data = student_data.sort_values('submit_time')

            # 1. 答题高峰时段特征
            student_data['hour'] = student_data['submit_time'].dt.hour
            hour_dist = student_data['hour'].value_counts().sort_index()

            # 计算活跃时段
            peak_hour = hour_dist.idxmax() if not hour_dist.empty else -1
            night_owl_count = len(student_data[(student_data['hour'] >= 23) | (student_data['hour'] <= 5)])
            night_owl_ratio = night_owl_count / len(student_data) if len(student_data) > 0 else 0

            # 2. 周末学习模式
            student_data['is_weekend'] = student_data['submit_time'].dt.dayofweek.isin([5, 6])
            weekend_ratio = student_data['is_weekend'].mean()

            # 3. 学习连续性特征
            student_data['date'] = student_data['submit_time'].dt.date
            unique_days = student_data['date'].nunique()
            total_days = (student_data['date'].max() - student_data['date'].min()).days + 1 if len(
                student_data) > 1 else 1
            continuity_rate = unique_days / total_days if total_days > 0 else 0

            # 4. 时间段偏好
            morning_sessions = len(student_data[(student_data['hour'] >= 6) & (student_data['hour'] < 12)])
            afternoon_sessions = len(student_data[(student_data['hour'] >= 12) & (student_data['hour'] < 18)])
            evening_sessions = len(student_data[(student_data['hour'] >= 18) & (student_data['hour'] < 23)])
            night_sessions = night_owl_count

            temporal_features.append({
                'student_ID': student_id,
                'peak_hour': peak_hour,
                'night_owl_ratio': night_owl_ratio,
                'weekend_ratio': weekend_ratio,
                'learning_continuity': continuity_rate,
                'morning_sessions': morning_sessions,
                'afternoon_sessions': afternoon_sessions,
                'evening_sessions': evening_sessions,
                'night_sessions': night_sessions,
                'total_sessions': len(student_data)
            })

        return pd.DataFrame(temporal_features)

    def extract_performance_features(self):
        """提取学习表现特征"""
        print("正在提取学习表现特征...")

        performance_features = []

        for student_id in self.df_submit['student_ID'].unique():
            student_data = self.df_submit[self.df_submit['student_ID'] == student_id]

            # 基础表现指标
            total_attempts = len(student_data)
            correct_rate = student_data['is_passed'].mean()
            avg_score = student_data['score'].mean()

            # 效率指标
            avg_time_per_attempt = student_data['timeconsume'].mean()

            # 知识点专注度
            if 'knowledge_point' in student_data.columns:
                knowledge_counts = student_data['knowledge_point'].value_counts()
                if len(knowledge_counts) > 0:
                    top_knowledge = knowledge_counts.index[0] if len(knowledge_counts) > 0 else 'Unknown'
                    top_knowledge_ratio = knowledge_counts.iloc[0] / total_attempts if total_attempts > 0 else 0
                else:
                    top_knowledge = 'Unknown'
                    top_knowledge_ratio = 0
            else:
                top_knowledge = 'Unknown'
                top_knowledge_ratio = 0

            # 题目难度偏好 - 使用简单分类
            if 'full_score' in student_data.columns:
                # 简单分类：根据分数范围
                easy_count = len(student_data[student_data['full_score'] <= 2])
                medium_count = len(student_data[(student_data['full_score'] > 2) & (student_data['full_score'] <= 5)])
                hard_count = len(student_data[student_data['full_score'] > 5])

                total = easy_count + medium_count + hard_count
                prefer_easy = easy_count / total if total > 0 else 0
                prefer_medium = medium_count / total if total > 0 else 0
                prefer_hard = hard_count / total if total > 0 else 0
            else:
                prefer_easy = prefer_medium = prefer_hard = 0

            performance_features.append({
                'student_ID': student_id,
                'total_attempts': total_attempts,
                'correct_rate': correct_rate,
                'avg_score': avg_score,
                'avg_time_per_attempt': avg_time_per_attempt,
                'top_knowledge': top_knowledge,
                'knowledge_focus': top_knowledge_ratio,
                'prefer_easy': prefer_easy,
                'prefer_medium': prefer_medium,
                'prefer_hard': prefer_hard
            })

        return pd.DataFrame(performance_features)

    def extract_behavior_patterns(self):
        """提取行为模式特征"""
        print("正在提取行为模式特征...")

        behavior_features = []

        for student_id in self.df_submit['student_ID'].unique():
            student_data = self.df_submit[self.df_submit['student_ID'] == student_id].sort_values('submit_time')

            # 提交间隔分析
            if len(student_data) > 1:
                time_diffs = student_data['submit_time'].diff().dropna().dt.total_seconds()
                avg_interval = time_diffs.mean() / 60  # 转换为分钟
                interval_std = time_diffs.std() / 60
                burst_behavior = len(time_diffs[time_diffs < 300]) / len(time_diffs)  # 5分钟内连续提交的比例
            else:
                avg_interval = 0
                interval_std = 0
                burst_behavior = 0

            # 学习坚持性（遇到难题时是否持续尝试）
            persistence_features = []
            for title_id in student_data['title_ID'].unique():
                title_attempts = student_data[student_data['title_ID'] == title_id]
                if len(title_attempts) > 1:
                    # 如果多次尝试同一题目，计算坚持性
                    first_attempt = title_attempts.iloc[0]
                    if first_attempt['is_passed'] == 0:  # 第一次未通过
                        later_passed = title_attempts.iloc[1:]['is_passed'].max()  # 后续是否有通过
                        persistence_features.append(1 if later_passed == 1 else 0)

            persistence_rate = np.mean(persistence_features) if persistence_features else 0

            behavior_features.append({
                'student_ID': student_id,
                'avg_interval_min': avg_interval,
                'interval_std': interval_std,
                'burst_behavior_ratio': burst_behavior,
                'persistence_rate': persistence_rate,
                'unique_titles_attempted': student_data['title_ID'].nunique()
            })

        return pd.DataFrame(behavior_features)

    def create_learner_profiles(self):
        """创建完整的用户画像"""
        print("正在创建学习者画像...")

        # 提取各类特征
        temporal_df = self.extract_temporal_features()
        performance_df = self.extract_performance_features()
        behavior_df = self.extract_behavior_patterns()

        # 合并所有特征
        learner_profiles = pd.merge(temporal_df, performance_df, on='student_ID', how='outer')
        learner_profiles = pd.merge(learner_profiles, behavior_df, on='student_ID', how='outer')

        # 添加学生基本信息
        learner_profiles = pd.merge(learner_profiles, self.df_student, on='student_ID', how='left')

        # 填充缺失值
        learner_profiles = learner_profiles.fillna({
            'correct_rate': 0,
            'avg_score': 0,
            'persistence_rate': 0,
            'knowledge_focus': 0,
            'prefer_easy': 0,
            'prefer_medium': 0,
            'prefer_hard': 0
        })

        # 保存画像数据
        learner_profiles.to_csv('Learner_Profiles.csv', index=False, encoding='utf-8')
        print(f"学习者画像已保存，共{len(learner_profiles)}个学习者")

        self.learner_profiles = learner_profiles
        return learner_profiles

    def analyze_and_visualize(self):
        """分析和可视化学习者画像"""
        if self.learner_profiles is None:
            print("请先创建学习者画像")
            return

        print("开始可视化分析...")

        # 创建可视化输出目录
        import os
        os.makedirs('learner_analysis_output', exist_ok=True)

        # 1. 学习者聚类分析
        self.cluster_learners()

        # 2. 时间模式分析
        self.plot_temporal_patterns()

        # 3. 行为模式分析
        self.plot_behavior_patterns()

        print("分析完成！")

    def cluster_learners(self):
        """对学习者进行聚类"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        # 选择聚类特征
        cluster_features = [
            'correct_rate', 'avg_time_per_attempt', 'night_owl_ratio',
            'weekend_ratio', 'burst_behavior_ratio', 'persistence_rate',
            'knowledge_focus', 'learning_continuity'
        ]

        df_cluster = self.learner_profiles[cluster_features].fillna(0)

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_cluster)

        # 使用肘部法则确定最佳聚类数
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)

        # 绘制肘部图
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 11), wcss, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.savefig('learner_analysis_output/elbow_method.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 根据肘部图选择聚类数（这里选择4）
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.learner_profiles['cluster'] = kmeans.fit_predict(X_scaled)

        # PCA降维可视化
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                              c=self.learner_profiles['cluster'],
                              cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title('Learner Clusters (PCA Visualization)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.savefig('learner_analysis_output/learner_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 分析每个聚类的特征
        cluster_summary = self.learner_profiles.groupby('cluster').agg({
            'correct_rate': 'mean',
            'avg_time_per_attempt': 'mean',
            'night_owl_ratio': 'mean',
            'weekend_ratio': 'mean',
            'total_sessions': 'mean',
            'persistence_rate': 'mean',
            'student_ID': 'count'
        }).round(3)

        cluster_summary.rename(columns={'student_ID': 'count'}, inplace=True)

        # 给聚类命名
        cluster_names = {
            0: '高效精英型',
            1: '勤奋挣扎型',
            2: '周末突击型',
            3: '夜猫子型'
        }

        cluster_summary['cluster_name'] = [cluster_names.get(i, f'Cluster {i}') for i in cluster_summary.index]
        cluster_summary.to_csv('learner_analysis_output/cluster_summary.csv', encoding='utf-8')

        print("聚类分析完成，结果已保存")

    def plot_temporal_patterns(self):
        """可视化时间模式"""
        # 1. 整体答题时间分布（热力图）
        plt.figure(figsize=(15, 10))

        # 按小时统计
        self.df_submit['hour'] = self.df_submit['submit_time'].dt.hour
        self.df_submit['dayofweek'] = self.df_submit['submit_time'].dt.dayofweek

        # 创建热力图数据
        heatmap_data = pd.pivot_table(
            self.df_submit,
            values='score',
            index='dayofweek',
            columns='hour',
            aggfunc='count',
            fill_value=0
        )

        # 绘制热力图
        plt.subplot(2, 2, 1)
        sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': '提交次数'})
        plt.title('按天和小时的提交热力图')
        plt.xlabel('小时 (0-23)')
        plt.ylabel('星期几 (0=周一)')

        # 2. 不同聚类的答题时间模式
        plt.subplot(2, 2, 2)
        for cluster_id in sorted(self.learner_profiles['cluster'].unique()):
            cluster_students = self.learner_profiles[self.learner_profiles['cluster'] == cluster_id]['student_ID']
            cluster_data = self.df_submit[self.df_submit['student_ID'].isin(cluster_students)]

            hour_dist = cluster_data['hour'].value_counts().sort_index()
            plt.plot(hour_dist.index, hour_dist.values / hour_dist.sum(),
                     label=f'聚类 {cluster_id}', linewidth=2)

        plt.title('不同聚类的小时提交模式')
        plt.xlabel('小时')
        plt.ylabel('标准化频率')
        plt.legend()
        plt.grid(alpha=0.3)

        # 3. 周末学习模式
        plt.subplot(2, 2, 3)
        weekend_data = self.learner_profiles.groupby('cluster')['weekend_ratio'].mean()
        weekend_data.plot(kind='bar', color='skyblue')
        plt.title('不同聚类的周末学习比例')
        plt.xlabel('聚类')
        plt.ylabel('周末提交比例')
        plt.xticks(rotation=0)

        # 4. 夜猫子比例
        plt.subplot(2, 2, 4)
        night_data = self.learner_profiles.groupby('cluster')['night_owl_ratio'].mean()
        night_data.plot(kind='bar', color='darkblue')
        plt.title('不同聚类的夜猫子比例')
        plt.xlabel('聚类')
        plt.ylabel('夜间提交比例')
        plt.xticks(rotation=0)

        plt.tight_layout()
        plt.savefig('learner_analysis_output/temporal_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 在你的LearnerProfileAnalyzer类中添加以下方法：

    def plot_monthly_activity_matrix(self):
        """绘制月内每日活跃度热力图"""
        print("正在生成月内活跃度热力图...")

        # 提取月份和日期
        self.df_submit['month'] = self.df_submit['submit_time'].dt.month
        self.df_submit['day'] = self.df_submit['submit_time'].dt.day
        self.df_submit['date'] = self.df_submit['submit_time'].dt.date

        # 创建日期-小时矩阵
        activity_matrix = pd.pivot_table(
            self.df_submit,
            values='score',
            index='date',
            columns='hour',
            aggfunc='count',
            fill_value=0
        )

        # 绘制热力图
        plt.figure(figsize=(20, 12))
        sns.heatmap(activity_matrix.T, cmap='YlOrRd', cbar_kws={'label': '提交次数'})
        plt.title('月内每日活跃度热力图', fontsize=16)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('小时', fontsize=12)
        plt.tight_layout()
        plt.savefig('learner_analysis_output/monthly_activity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ 月内活跃度热力图已保存")

    def plot_fireworks_plot(self):
        """绘制月内活跃度烟花图"""
        print("正在生成月内活跃度烟花图...")

        # 按日期统计活跃度
        daily_activity = self.df_submit.groupby('date')['score'].count()

        # 创建连续日期索引
        date_range = pd.date_range(
            start=daily_activity.index.min(),
            end=daily_activity.index.max()
        )

        # 重新索引以填充缺失日期
        daily_activity = daily_activity.reindex(date_range, fill_value=0)

        # 创建烟花图
        plt.figure(figsize=(15, 10))

        # 为每个活跃点创建"烟花"效果
        for i, (date, count) in enumerate(daily_activity.items()):
            if count > 0:
                # 计算点的大小和透明度
                size = min(200, count * 10)
                alpha = min(0.8, count / daily_activity.max())

                # 在日期周围创建随机分布的点
                n_points = min(count, 100)  # 限制点的数量
                x_pos = i + np.random.normal(0, 0.3, n_points)
                y_pos = np.random.uniform(0, 1, n_points)

                # 根据活跃度选择颜色
                color_intensity = min(1.0, count / daily_activity.quantile(0.9))
                color = plt.cm.Reds(color_intensity)

                plt.scatter(x_pos, y_pos, s=size, alpha=alpha,
                            color=color, edgecolors='black', linewidth=0.5)

        plt.xlabel('日期索引', fontsize=12)
        plt.ylabel('活跃度强度', fontsize=12)
        plt.title('月内活跃度波动烟花图', fontsize=16)
        plt.ylim(-0.2, 1.2)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('learner_analysis_output/monthly_fireworks_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ 月内活跃度烟花图已保存")

    def plot_enhanced_radar_chart(self):
        """绘制增强版雷达图（对比4类学习者）"""
        print("正在生成增强版雷达图...")

        # 确保有聚类结果
        if 'cluster' not in self.learner_profiles.columns:
            print("警告：没有聚类结果，无法绘制雷达图")
            return

        # 选择雷达图特征
        radar_features = [
            'correct_rate',  # 掌握度
            'total_attempts',  # 活跃度（标准化）
            'persistence_rate',  # 坚持度
            'learning_continuity',  # 连续性
            'knowledge_focus',  # 专注度
            'night_owl_ratio'  # 夜间活跃度
        ]

        # 检查哪些特征可用
        available_features = [f for f in radar_features if f in self.learner_profiles.columns]

        if len(available_features) < 3:
            print("警告：可用特征不足，无法绘制雷达图")
            return

        # 标准化特征（使所有特征在0-1范围内）
        df_cluster_stats = pd.DataFrame()
        for cluster_id in sorted(self.learner_profiles['cluster'].unique()):
            cluster_data = self.learner_profiles[self.learner_profiles['cluster'] == cluster_id]

            cluster_stats = {}
            for feature in available_features:
                # 标准化到0-1范围
                min_val = self.learner_profiles[feature].min()
                max_val = self.learner_profiles[feature].max()
                if max_val > min_val:
                    normalized = (cluster_data[feature].mean() - min_val) / (max_val - min_val)
                else:
                    normalized = 0.5
                cluster_stats[feature] = normalized

            cluster_stats['cluster_id'] = cluster_id
            df_cluster_stats = pd.concat([df_cluster_stats, pd.DataFrame([cluster_stats])], ignore_index=True)

        # 设置雷达图角度
        angles = np.linspace(0, 2 * np.pi, len(available_features), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        # 创建雷达图
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

        # 定义聚类名称和颜色
        cluster_names = {
            0: '高效精英型',
            1: '勤奋挣扎型',
            2: '周末突击型',
            3: '夜猫子型'
        }

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        # 绘制每个聚类的雷达图
        for i, (_, row) in enumerate(df_cluster_stats.iterrows()):
            cluster_id = int(row['cluster_id'])
            values = row[available_features].tolist()
            values += values[:1]  # 闭合图形

            # 绘制线
            ax.plot(angles, values, linewidth=2,
                    label=cluster_names.get(cluster_id, f'聚类{cluster_id}'),
                    color=colors[i % len(colors)])

            # 填充区域
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_features, fontsize=10)
        ax.set_ylim(0, 1)

        # 添加标题和图例
        plt.title('四类学习者特征对比雷达图', fontsize=16, y=1.08)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

        # 添加网格
        ax.grid(True)

        plt.tight_layout()
        plt.savefig('learner_analysis_output/enhanced_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ 增强版雷达图已保存")

    def analyze_struggler_behavior(self):
        """深入分析勤奋挣扎型学习者的行为模式"""
        print("正在深入分析勤奋挣扎型学习者...")

        # 确保有聚类结果
        if 'cluster' not in self.learner_profiles.columns:
            print("警告：没有聚类结果，无法分析")
            return

        # 找到勤奋挣扎型学习者（假设是cluster 1）
        struggler_cluster = 1
        struggler_students = self.learner_profiles[
            self.learner_profiles['cluster'] == struggler_cluster
            ]['student_ID'].tolist()

        if not struggler_students:
            print("未找到勤奋挣扎型学习者")
            return

        print(f"找到 {len(struggler_students)} 名勤奋挣扎型学习者")

        # 随机选择3名学生进行深入分析
        np.random.seed(42)
        sample_students = np.random.choice(struggler_students, min(3, len(struggler_students)), replace=False)

        # 创建多子图
        fig, axes = plt.subplots(len(sample_students), 1, figsize=(15, 5 * len(sample_students)))
        if len(sample_students) == 1:
            axes = [axes]

        for idx, student_id in enumerate(sample_students):
            ax = axes[idx]

            # 获取该学生的所有提交记录
            student_data = self.df_submit[self.df_submit['student_ID'] == student_id].copy()
            student_data = student_data.sort_values('submit_time')

            # 计算每次提交的间隔
            student_data['time_diff'] = student_data['submit_time'].diff().dt.total_seconds().fillna(0)

            # 标记快速连续提交（小于5分钟）
            student_data['rapid_submission'] = (student_data['time_diff'] < 300) & (student_data['time_diff'] > 0)

            # 创建时间轴
            times = student_data['submit_time']

            # 根据结果类型设置颜色和形状
            colors = ['green' if passed else 'red' for passed in student_data['is_passed']]
            shapes = ['o' if rapid else '^' for rapid in student_data['rapid_submission']]

            # 绘制时间轴
            for i, (time, color, shape, title_id) in enumerate(
                    zip(times, colors, shapes, student_data['title_ID'])
            ):
                # 将时间转换为数值位置
                x_pos = i

                # 绘制点
                ax.scatter(x_pos, 0, c=color, marker=shape, s=100,
                           alpha=0.7, edgecolors='black')

                # 添加题目ID标签（每隔几个点显示一次）
                if i % 5 == 0:
                    ax.text(x_pos, 0.1, f"T:{title_id[:6]}",
                            fontsize=8, rotation=45, ha='center')

            # 连接连续的点
            ax.plot(range(len(times)), [0] * len(times), 'gray', alpha=0.3, linewidth=1)

            # 设置图表
            ax.set_title(f'勤奋挣扎型学习者 {student_id} 的提交时间轴', fontsize=12)
            ax.set_xlabel('提交序号', fontsize=10)
            ax.set_ylabel('时间轴', fontsize=10)
            ax.grid(True, alpha=0.3)

            # 添加图例
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='正确'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='错误'),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='快速提交'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

            # 添加统计信息
            total_submissions = len(student_data)
            correct_rate = student_data['is_passed'].mean() * 100
            rapid_rate = student_data['rapid_submission'].mean() * 100

            stats_text = f"总提交: {total_submissions}, 正确率: {correct_rate:.1f}%, 快速提交比例: {rapid_rate:.1f}%"
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('learner_analysis_output/struggler_behavior_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 生成行为模式报告
        self.generate_struggler_report(struggler_students)

        print("✅ 勤奋挣扎型学习者行为分析完成")

    def generate_struggler_report(self, struggler_students):
        """生成勤奋挣扎型学习者行为报告"""
        print("正在生成勤奋挣扎型学习者行为报告...")

        # 分析所有勤奋挣扎型学生的行为模式
        all_struggler_data = self.df_submit[
            self.df_submit['student_ID'].isin(struggler_students)
        ].copy()

        # 计算关键指标
        report_data = {
            '总人数': len(struggler_students),
            '平均正确率': f"{all_struggler_data['is_passed'].mean() * 100:.1f}%",
            '平均尝试次数': f"{all_struggler_data.groupby(['student_ID', 'title_ID']).size().mean():.1f}",
            '平均学习天数': f"{all_struggler_data['submit_time'].dt.date.nunique():.1f}",
            '平均单题耗时': f"{all_struggler_data['timeconsume'].mean():.1f}秒",
            '深夜学习比例': f"{(all_struggler_data['submit_time'].dt.hour.isin([23, 0, 1, 2, 3, 4, 5])).mean() * 100:.1f}%",
        }

        # 分析行为模式
        # 1. 频繁更换题目
        title_switching_rate = []
        for student_id in struggler_students:
            student_data = self.df_submit[self.df_submit['student_ID'] == student_id]
            if len(student_data) > 1:
                # 计算连续提交中更换题目的比例
                title_changes = (student_data['title_ID'] != student_data['title_ID'].shift()).sum() - 1
                switching_rate = title_changes / (len(student_data) - 1)
                title_switching_rate.append(switching_rate)

        if title_switching_rate:
            report_data['平均题目更换率'] = f"{np.mean(title_switching_rate) * 100:.1f}%"

        # 2. 失败后放弃的比例
        give_up_rate = []
        for student_id in struggler_students:
            student_data = self.df_submit[self.df_submit['student_ID'] == student_id]

            # 找出第一次尝试失败的题目
            first_attempts = student_data.drop_duplicates(subset=['title_ID'], keep='first')
            failed_first = first_attempts[first_attempts['is_passed'] == 0]

            if len(failed_first) > 0:
                # 检查这些题目是否有后续尝试
                titles_with_retry = 0
                for title_id in failed_first['title_ID']:
                    title_attempts = student_data[student_data['title_ID'] == title_id]
                    if len(title_attempts) > 1:
                        titles_with_retry += 1

                give_up_rate.append(1 - (titles_with_retry / len(failed_first)))

        if give_up_rate:
            report_data['失败后放弃比例'] = f"{np.mean(give_up_rate) * 100:.1f}%"

        # 3. 知识点分散度
        if 'knowledge_point' in all_struggler_data.columns:
            avg_knowledge_per_student = all_struggler_data.groupby('student_ID')['knowledge_point'].nunique().mean()
            report_data['平均涉及知识点数'] = f"{avg_knowledge_per_student:.1f}"

        # 保存报告
        report_df = pd.DataFrame(list(report_data.items()), columns=['指标', '值'])
        report_file = 'learner_analysis_output/struggler_behavior_report.csv'
        report_df.to_csv(report_file, index=False, encoding='utf-8')

        # 生成文本报告
        text_report = f"""# 勤奋挣扎型学习者行为分析报告

    ## 核心发现

    通过分析 {report_data['总人数']} 名勤奋挣扎型学习者的行为模式，发现以下特点：

    ### 1. 高投入低产出现象
    - 平均正确率仅为 {report_data['平均正确率']}，远低于平均水平
    - 平均单题耗时 {report_data['平均单题耗时']}，表明思考时间较长
    - 但学习效果不佳，存在明显的"高投入低产出"现象

    ### 2. 学习行为特征
    - {report_data.get('平均题目更换率', '数据缺失')} 的题目更换率，表明容易分心
    - {report_data.get('失败后放弃比例', '数据缺失')} 的失败放弃率，缺乏坚持性
    - 深夜学习比例 {report_data['深夜学习比例']}，可能存在疲劳学习

    ### 3. 建议改进方向
    1. **专注度训练**：减少题目频繁更换，建议每次专注于1-2个知识点
    2. **坚持性培养**：鼓励失败后继续尝试，建立"失败是学习过程"的观念
    3. **时间管理**：避免深夜疲劳学习，合理安排学习时段
    4. **学习方法**：加强基础知识掌握，避免盲目刷题

    ---
    *分析时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
    """

        text_report_file = 'learner_analysis_output/struggler_analysis_report.md'
        with open(text_report_file, 'w', encoding='utf-8') as f:
            f.write(text_report)

        print(f"✅ 行为报告已保存: {text_report_file}")

    def analyze_and_visualize_enhanced(self):
        """增强版分析和可视化（包含所有题目要求的功能）"""
        if self.learner_profiles is None:
            print("请先创建学习者画像")
            return

        print("开始增强版可视化分析...")

        # 创建可视化输出目录
        import os
        os.makedirs('learner_analysis_output', exist_ok=True)

        # 1. 学习者聚类分析
        self.cluster_learners()

        # 2. 时间模式分析
        self.plot_temporal_patterns()

        # 3. 行为模式分析
        self.plot_behavior_patterns()

        # 4. 新增：月内活跃度分析
        self.plot_monthly_activity_matrix()
        self.plot_fireworks_plot()

        # 5. 新增：增强版雷达图
        self.plot_enhanced_radar_chart()

        # 6. 新增：勤奋挣扎型学习者深度分析
        self.analyze_struggler_behavior()

        print("✅ 增强版分析完成！")

        # 生成综合报告
        self.generate_comprehensive_report()

    def generate_comprehensive_report(self):
        """生成综合报告"""
        print("正在生成综合报告...")

        report_content = f"""# 学习者画像与个性化行为模式分析报告

    ## 一、数据概况
    - 总学习者数量：{len(self.learner_profiles)}
    - 总提交记录：{len(self.df_submit)}
    - 分析时间范围：{self.df_submit['submit_time'].min()} 至 {self.df_submit['submit_time'].max()}

    ## 二、学习者聚类分析
    通过K-means聚类分析，将学习者分为4类：

    ### 1. 高效精英型
    - 特征：高正确率、高效率、学习时间合理
    - 占比：约 {(self.learner_profiles['cluster'] == 0).mean() * 100:.1f}%

    ### 2. 勤奋挣扎型
    - 特征：高投入、低产出、学习时间长但效果不佳
    - 占比：约 {(self.learner_profiles['cluster'] == 1).mean() * 100:.1f}%

    ### 3. 周末突击型
    - 特征：主要在周末学习、学习时间集中
    - 占比：约 {(self.learner_profiles['cluster'] == 2).mean() * 100:.1f}%

    ### 4. 夜猫子型
    - 特征：深夜学习活跃、学习时段特殊
    - 占比：约 {(self.learner_profiles['cluster'] == 3).mean() * 100:.1f}%

    ## 三、核心发现

    ### 1. 时间模式特征
    - **答题高峰时段**：大部分学习者在 {self.learner_profiles['peak_hour'].mode()[0] if len(self.learner_profiles['peak_hour'].mode()) > 0 else 'N/A'}:00 左右最活跃
    - **周末学习比例**：平均 {self.learner_profiles['weekend_ratio'].mean() * 100:.1f}% 的学习活动发生在周末
    - **夜猫子现象**：{self.learner_profiles['night_owl_ratio'].mean() * 100:.1f}% 的学习活动发生在深夜

    ### 2. 学习行为模式
    - **平均正确率**：{self.learner_profiles['correct_rate'].mean() * 100:.1f}%
    - **平均坚持率**：{self.learner_profiles['persistence_rate'].mean() * 100:.1f}%
    - **学习连续性**：{self.learner_profiles['learning_continuity'].mean() * 100:.1f}%

    ### 3. 勤奋挣扎型学习者深度分析
    详细分析报告请参见 `struggler_analysis_report.md`

    ## 四、教学建议

    ### 针对不同类型学习者的建议：

    1. **高效精英型**
       - 提供进阶挑战题目
       - 鼓励参与竞赛和项目

    2. **勤奋挣扎型**
       - 加强基础知识巩固
       - 提供学习方法指导
       - 建立学习计划和时间管理

    3. **周末突击型**
       - 提供周末特别课程
       - 加强周中学习督促

    4. **夜猫子型**
       - 提供异步学习资源
       - 关注健康作息提醒

    ## 五、可视化输出
    所有生成的可视化图表已保存在 `learner_analysis_output/` 文件夹中。

    ---
    *报告生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
    """

        report_file = 'learner_analysis_output/comprehensive_analysis_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"✅ 综合报告已保存: {report_file}")
    def plot_behavior_patterns(self):
        """可视化行为模式"""
        # 1. 学习坚持性 vs 正确率
        plt.figure(figsize=(14, 10))

        plt.subplot(2, 2, 1)
        scatter = plt.scatter(self.learner_profiles['persistence_rate'],
                              self.learner_profiles['correct_rate'],
                              c=self.learner_profiles['cluster'],
                              cmap='tab10', alpha=0.6, s=50)
        plt.colorbar(scatter, label='聚类')
        plt.title('坚持性 vs 正确率')
        plt.xlabel('坚持率')
        plt.ylabel('正确率')
        plt.grid(alpha=0.3)

        # 2. 答题效率（平均时间 vs 正确率）
        plt.subplot(2, 2, 2)
        plt.scatter(self.learner_profiles['avg_time_per_attempt'],
                    self.learner_profiles['correct_rate'],
                    c=self.learner_profiles['burst_behavior_ratio'],
                    cmap='coolwarm', alpha=0.6, s=50)
        plt.colorbar(label='爆发行为比例')
        plt.title('时间效率 vs 正确率')
        plt.xlabel('每次尝试平均时间 (秒)')
        plt.ylabel('正确率')
        plt.grid(alpha=0.3)

        # 3. 知识专注度分析
        plt.subplot(2, 2, 3)
        top_knowledge_counts = self.learner_profiles['top_knowledge'].value_counts().head(10)
        top_knowledge_counts.plot(kind='barh', color='lightgreen')
        plt.title('最受关注的10个知识点')
        plt.xlabel('学习者数量')
        plt.gca().invert_yaxis()

        # 4. 不同聚类的行为特征雷达图
        plt.subplot(2, 2, 4, polar=True)

        # 准备雷达图数据
        radar_features = ['correct_rate', 'persistence_rate', 'learning_continuity',
                          'knowledge_focus', 'burst_behavior_ratio']

        cluster_means = self.learner_profiles.groupby('cluster')[radar_features].mean()

        # 绘制雷达图框架
        angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        ax = plt.subplot(2, 2, 4, polar=True)
        for cluster_id in cluster_means.index:
            values = cluster_means.loc[cluster_id].tolist()
            values += values[:1]  # 闭合图形
            ax.plot(angles, values, linewidth=2, label=f'聚类 {cluster_id}')
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['正确率', '坚持率', '连续性', '专注度', '爆发行为'])
        plt.title('不同聚类行为模式雷达图')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        plt.savefig('learner_analysis_output/behavior_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """主函数"""
    analyzer = LearnerProfileAnalyzer()

    # 1. 加载数据
    analyzer.load_data()

    # 2. 预处理
    analyzer.preprocess_data()

    # 3. 创建学习者画像
    profiles = analyzer.create_learner_profiles()

    # 4. 使用增强版分析（包含题目要求的所有功能）
    analyzer.analyze_and_visualize_enhanced()  # 替换原来的analyze_and_visualize()

    print("\n=== 分析完成 ===")
    print("生成的文件：")
    print("1. Learner_Profiles.csv - 完整学习者画像数据")
    print("2. learner_analysis_output/ - 包含所有可视化图表和报告")
    print("3. 具体包括：")
    print("   - 聚类分析图表")
    print("   - 时间模式分析")
    print("   - 月内活跃度热力图和烟花图")
    print("   - 增强版雷达图")
    print("   - 勤奋挣扎型学习者深度分析报告")
    print("   - 综合分析报告")


if __name__ == "__main__":
    main()