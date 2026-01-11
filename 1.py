import pandas as pd
import numpy as np
import os
from dash import Dash, dcc, html, Input, Output, dash_table, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- 全局样式配置（紧凑风格） ----------------------
COLOR_PALETTE = {
    "primary": "#4CAF50",  # 主绿
    "secondary": "#2196F3",  # 辅助蓝
    "success": "#8BC34A",  # 浅绿
    "warning": "#FFC107",  # 黄色
    "danger": "#F44336",  # 红色
    "info": "#9C27B0",  # 紫色
    "light": "#F5F7FA",  # 浅灰背景
    "dark": "#424242",  # 深灰文字
    "border": "#E0E0E0",  # 边框色
    "card_bg": "#FFFFFF",  # 卡片背景白
}

# 紧凑样式模板
PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Microsoft YaHei, Arial, sans-serif", size=11, color=COLOR_PALETTE["dark"]),
        xaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE["border"], gridwidth=0.5, linecolor=COLOR_PALETTE["border"],
                   linewidth=1, tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE["border"], gridwidth=0.5, linecolor=COLOR_PALETTE["border"],
                   linewidth=1, tickfont=dict(size=9)),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor=COLOR_PALETTE["border"], borderwidth=1,
                    font=dict(size=9))
    )
)

FONT_FAMILY = "Microsoft YaHei, Arial, sans-serif"


# ---------------------- 工具函数（优先定义，确保调用前存在） ----------------------
def safe_division(numerator, denominator, default=0.0):
    """安全除法，避免除零错误"""
    if denominator == 0 or pd.isna(denominator) or denominator is None:
        return default
    return numerator / denominator


# ---------------------- 1. 数据加载与预处理 ----------------------
def load_data():
    """封装数据加载逻辑"""
    try:
        # 加载学生信息
        student_df = pd.read_csv("Data_StudentInfo.csv")
        logger.info(f"学生信息表加载完成，行数：{len(student_df)}")

        # 加载题目信息
        title_df = pd.read_csv("Data_TitleInfo.csv")
        logger.info(f"题目信息表加载完成，行数：{len(title_df)}")

        # 加载答题记录
        submit_records = []
        submit_folder = "Data_SubmitRecord"
        if not os.path.exists(submit_folder):
            raise FileNotFoundError(f"答题记录文件夹 {submit_folder} 不存在！")

        for file in os.listdir(submit_folder):
            if file.endswith(".csv"):
                file_path = os.path.join(submit_folder, file)
                df = pd.read_csv(file_path)
                submit_records.append(df)
                logger.info(f"加载答题记录 {file}，行数：{len(df)}")

        if not submit_records:
            raise ValueError("答题记录文件夹中无CSV文件！")

        submit_df = pd.concat(submit_records, ignore_index=True)
        logger.info(f"答题记录合并完成，总行数：{len(submit_df)}")

        return student_df, title_df, submit_df

    except Exception as e:
        logger.error(f"数据加载失败：{str(e)}")
        # 生成模拟数据（防止数据文件缺失导致程序崩溃）
        logger.warning("使用模拟数据运行程序")
        student_df = pd.DataFrame({
            "student_ID": [f"S{i}" for i in range(100)],
            "class": [f"Class_{i % 3}" for i in range(100)],
            "major": np.random.choice(["计算机", "数学", "物理", "化学"], 100),
            "age": np.random.choice(["18-20", "21-23", "24-26", "27+"], 100),
            "gender": np.random.choice(["男", "女"], 100)
        })

        title_df = pd.DataFrame({
            "title_ID": [f"T{i}" for i in range(20)],  # 至少生成20个题目
            "full_score": [100] * 20,
            "knowledge": [f"KP{i % 5}" for i in range(20)],
            "sub_knowledge": [f"SK{i % 10}" for i in range(20)]
        })

        submit_df = pd.DataFrame({
            "student_ID": np.random.choice([f"S{i}" for i in range(100)], 500),
            "title_ID": np.random.choice([f"T{i}" for i in range(20)], 500),
            "student_score": np.random.uniform(50, 95, 500),
            "time_consume": np.random.randint(0, 40, 500),
            "memory_usage": np.random.randint(160, 500, 500)
        })

        return student_df, title_df, submit_df


# 加载数据
student_df, title_df, submit_df = load_data()

# 数据预处理
title_df.rename(columns={"score": "full_score"}, inplace=True)
submit_df.rename(columns={"score": "student_score"}, inplace=True)

# 类型转换 + 空值填充
for df in [title_df, submit_df, student_df]:
    for col in df.columns:
        if "ID" in col or "class" in col or "knowledge" in col or "sub_knowledge" in col:
            df[col] = df[col].astype(str).fillna("未知")

# 补充知识点字段
title_df["Main_KP"] = title_df["knowledge"].apply(lambda x: f"{x[:5]}" if x != "未知" else "未知主知识点")
title_df["Sub_KP"] = title_df["sub_knowledge"].apply(lambda x: f"{x[:7]}" if x != "未知" else "未知子知识点")

# 合并数据
submit_title_df = pd.merge(submit_df, title_df[["title_ID", "full_score", "Main_KP", "Sub_KP"]],
                           on="title_ID", how="left")
all_df = pd.merge(submit_title_df, student_df, on="student_ID", how="left")

# 填充空值
all_df["Main_KP"] = all_df["Main_KP"].fillna("未知主知识点")
all_df["Sub_KP"] = all_df["Sub_KP"].fillna("未知子知识点")
all_df["class"] = all_df["class"].fillna("未知班级")
all_df["full_score"] = pd.to_numeric(all_df["full_score"], errors="coerce").fillna(0)
all_df["student_score"] = pd.to_numeric(all_df["student_score"], errors="coerce").fillna(0)

# 计算得分率（使用安全除法）
all_df["Score_Rate"] = all_df.apply(lambda row: safe_division(row["student_score"], row["full_score"]), axis=1)
all_df["Score_Rate"] = all_df["Score_Rate"].fillna(0).astype(float)

# 模拟扩展字段
all_df["title_code"] = all_df["title_ID"].apply(lambda x: f"Q_{x[:3]}")
all_df["Correct_Rate"] = np.random.uniform(0.4, 0.9, len(all_df))
all_df["Mastery_Level"] = np.random.uniform(0.3, 0.8, len(all_df))
all_df["multi_kp"] = np.random.choice([0, 1], len(all_df), p=[0.85, 0.15])
all_df["time_consume"] = np.random.randint(0, 40, len(all_df))
all_df["memory_usage"] = np.random.randint(160, 500, len(all_df))
all_df["major"] = np.random.choice(["计算机", "数学", "物理", "化学"], len(all_df))
all_df["age"] = np.random.choice(["18-20", "21-23", "24-26", "27+"], len(all_df))
all_df["gender"] = np.random.choice(["男", "女"], len(all_df))

# ---------------------- 2. 初始化Dash应用 ----------------------
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "知识点掌握度分析系统"

# ---------------------- 3. 重新设计的紧凑布局 ----------------------
app.layout = html.Div([
    # 1. 控制面板（紧凑版）- 置顶
    html.Div([
        html.H3("控制面板",
                style={"margin": 0, "fontSize": "14px", "fontWeight": "bold", "color": COLOR_PALETTE["dark"]}),
        html.Div([
            # 权重配置（紧凑排列）
            html.Div([html.Label("得分率"),
                      dcc.Input(id="score_weight", type="number", value=0.25, min=0, max=1, step=0.05,
                                style={"width": "50px", "height": "25px", "fontSize": "10px"})],
                     style={"marginRight": "8px"}),
            html.Div([html.Label("正确比"),
                      dcc.Input(id="comp_weight", type="number", value=0.25, min=0, max=1, step=0.05,
                                style={"width": "50px", "height": "25px", "fontSize": "10px"})],
                     style={"marginRight": "8px"}),
            html.Div([html.Label("用时"),
                      dcc.Input(id="time_weight", type="number", value=0.25, min=0, max=1, step=0.05,
                                style={"width": "50px", "height": "25px", "fontSize": "10px"})],
                     style={"marginRight": "8px"}),
            html.Div([html.Label("内存"), dcc.Input(id="mem_weight", type="number", value=0.25, min=0, max=1, step=0.05,
                                                    style={"width": "50px", "height": "25px", "fontSize": "10px"})],
                     style={"marginRight": "15px"}),

            # 筛选+按钮 - 新增班级下拉选择框
            dcc.Dropdown(id="user_filter",
                         options=[{"label": "所有学习者", "value": "all"}, {"label": "按班级", "value": "class"}],
                         value="all",
                         style={"width": "100px", "height": "25px", "fontSize": "10px", "marginRight": "8px"}),
            # 新增：班级选择下拉框（默认隐藏，选择按班级时显示）
            dcc.Dropdown(id="class_selector",
                         options=[{"label": cls, "value": cls} for cls in all_df["class"].unique()],
                         value=all_df["class"].unique()[0] if len(all_df["class"].unique()) > 0 else "",
                         style={"width": "100px", "height": "25px", "fontSize": "10px", "marginRight": "8px",
                                "display": "none"}),
            html.Button("重置", id="reset_btn", n_clicks=0,
                        style={"height": "25px", "padding": "0 8px", "fontSize": "10px", "marginRight": "8px"}),
            html.Button("初始化", id="init_btn", n_clicks=0,
                        style={"height": "25px", "padding": "0 8px", "fontSize": "10px",
                               "backgroundColor": COLOR_PALETTE["secondary"], "color": "white", "border": "none"}),
        ], style={"display": "flex", "alignItems": "center", "marginTop": "8px"}),
    ], style={"backgroundColor": COLOR_PALETTE["card_bg"], "padding": "10px",
              "borderBottom": f"1px solid {COLOR_PALETTE['border']}", "fontSize": "10px"}),

    # 2. 主内容区域（重新排版）
    html.Div([
        # 2.1 三个扇形图区域（纯饼图，横向三列）
        html.Div([
            html.H4("基础分布统计", style={"margin": "0 0 8px 0", "fontSize": "12px", "fontWeight": "bold"}),
            html.Div([
                html.Div([dcc.Graph(id="major_chart", style={"height": "180px", "width": "100%"})],
                         style={"width": "32%", "marginRight": "1.33%"}),
                html.Div([dcc.Graph(id="age_chart", style={"height": "180px", "width": "100%"})],
                         style={"width": "32%", "marginRight": "1.33%"}),
                html.Div([dcc.Graph(id="gender_chart", style={"height": "180px", "width": "100%"})],
                         style={"width": "32%"}),
            ], style={"display": "flex"}),
        ], style={"backgroundColor": COLOR_PALETTE["card_bg"], "padding": "10px", "borderRadius": "4px",
                  "boxShadow": "0 1px 2px rgba(0,0,0,0.05)", "marginBottom": "8px"}),

        # 2.2 班级排名柱状图（单独一行）
        html.Div([
            html.H4("班级得分率排名", style={"margin": "0 0 8px 0", "fontSize": "12px", "fontWeight": "bold"}),
            dcc.Graph(id="class_ranking_chart", style={"height": "200px", "width": "100%"}),
        ], style={"backgroundColor": COLOR_PALETTE["card_bg"], "padding": "10px", "borderRadius": "4px",
                  "boxShadow": "0 1px 2px rgba(0,0,0,0.05)", "marginBottom": "8px"}),

        # 2.3 题目掌握分析（修改为整体分析）
        html.Div([
            html.H4("题目整体掌握分析", style={"margin": "0 0 8px 0", "fontSize": "12px", "fontWeight": "bold"}),
            # 折线图（所有题目按知识点汇总的掌握程度）
            html.Div([dcc.Graph(id="mastery_line_chart", style={"height": "160px", "width": "100%"})],
                     style={"marginBottom": "8px"}),
            # 双柱状图（所有题的用时/内存整体分布）
            html.Div([
                html.Div([dcc.Graph(id="time_dist_chart", style={"height": "120px", "width": "100%"})],
                         style={"width": "48%", "marginRight": "4%"}),
                html.Div([dcc.Graph(id="mem_dist_chart", style={"height": "120px", "width": "100%"})],
                         style={"width": "48%"}),
            ], style={"display": "flex"}),
        ], style={"backgroundColor": COLOR_PALETTE["card_bg"], "padding": "10px", "borderRadius": "4px",
                  "boxShadow": "0 1px 2px rgba(0,0,0,0.05)", "marginBottom": "8px"}),

        # 2.4 桑基图（原有结构不变）
        html.Div([
            html.H4("知识点桑基图", style={"margin": "0 0 8px 0", "fontSize": "12px", "fontWeight": "bold"}),
            dcc.Graph(id="kp_sankey_chart", style={"height": "400px", "width": "100%"}),
        ], style={"backgroundColor": COLOR_PALETTE["card_bg"], "padding": "10px", "borderRadius": "4px",
                  "boxShadow": "0 1px 2px rgba(0,0,0,0.05)", "marginBottom": "8px"}),

        # 2.5 薄弱知识点表格（原有结构不变）
        html.Div([
            html.H4("薄弱知识点分析表", style={"margin": "0 0 8px 0", "fontSize": "12px", "fontWeight": "bold"}),
            dash_table.DataTable(
                id="weak_compare_table",
                columns=[
                    {"name": "主知识点", "id": "Main_KP"},
                    {"name": "子知识点", "id": "Sub_KP"},
                    {"name": "班级", "id": "class"},
                    {"name": "平均得分率", "id": "Avg_Score_Rate"},
                    {"name": "题目数量", "id": "Title_Count"},
                    {"name": "薄弱等级", "id": "Weak_Level"}
                ],
                style_table={"overflowX": "auto", "fontSize": "10px"},
                style_cell={"textAlign": "center", "padding": "4px 8px", "fontSize": "10px",
                            "borderBottom": f"1px solid {COLOR_PALETTE['border']}"},
                style_header={"backgroundColor": COLOR_PALETTE["light"], "fontWeight": "bold", "fontSize": "10px"},
                style_data_conditional=[
                    {"if": {"filter_query": "{Weak_Level} = '极弱'"}, "backgroundColor": "#ffebee", "color": "#d32f2f"},
                    {"if": {"filter_query": "{Weak_Level} = '较弱'"}, "backgroundColor": "#fff3e0", "color": "#f57c00"},
                    {"if": {"filter_query": "{Weak_Level} = '一般'"}, "backgroundColor": "#e8f5e9", "color": "#388e3c"},
                ],
                page_size=8,  # 紧凑分页
                sort_action="native",
                filter_action="native",
            ),
            html.Div(id="table_error", style={"color": COLOR_PALETTE["danger"], "marginTop": "5px", "fontSize": "10px",
                                              "textAlign": "center"}),
        ], style={"backgroundColor": COLOR_PALETTE["card_bg"], "padding": "10px", "borderRadius": "4px",
                  "boxShadow": "0 1px 2px rgba(0,0,0,0.05)"}),

    ], style={"padding": "10px", "backgroundColor": COLOR_PALETTE["light"], "minHeight": "100vh", "fontSize": "10px"}),

], style={"margin": 0, "padding": 0, "fontFamily": FONT_FAMILY})


# ---------------------- 新增回调：控制班级选择框的显示/隐藏 ----------------------
@app.callback(
    Output("class_selector", "style"),
    Input("user_filter", "value")
)
def toggle_class_selector(filter_value):
    if filter_value == "class":
        return {"width": "100px", "height": "25px", "fontSize": "10px", "marginRight": "8px", "display": "block"}
    else:
        return {"width": "100px", "height": "25px", "fontSize": "10px", "marginRight": "8px", "display": "none"}


# ---------------------- 4. 回调函数（适配新排版） ----------------------
## 回调1：专业分布扇形图（修复比例计算）
@app.callback(Output("major_chart", "figure"),
              [Input("score_weight", "value"),
               Input("user_filter", "value"),
               Input("class_selector", "value")])
def update_major_chart(score_w, user_filter, selected_class):
    df = all_df.copy()

    # 筛选逻辑修复：按用户选择的班级筛选，而非默认第一个
    if user_filter == "class" and selected_class:
        df = df[df["class"] == selected_class]

    # 仅保留唯一学生数据（避免答题记录重复导致人数统计错误）
    unique_students = df.drop_duplicates(subset=["student_ID"])

    # 专业分布统计（唯一学生数）
    major_stats = unique_students.groupby("major").agg(人数=("student_ID", "nunique")).reset_index()
    total_students = major_stats["人数"].sum()

    # 处理空数据
    if total_students == 0:
        fig = go.Figure(go.Pie(values=[1], labels=["无数据"], hole=0.7))
        fig.update_layout(
            title="专业分布（无数据）", title_font=dict(size=11, weight="bold"),
            height=180, template=PLOTLY_TEMPLATE,
            margin=dict(t=20, l=5, r=5, b=5)
        )
        return fig

    # 显式计算百分比（确保精度）
    major_stats["百分比"] = (major_stats["人数"] / total_students * 100).round(1)

    fig = go.Figure(
        go.Pie(
            labels=major_stats["major"],
            values=major_stats["人数"],
            hole=0.7,
            # 自定义文本：显示「标签 + 人数 + 百分比」
            text=[f"{row['major']}<br>{row['人数']}人 ({row['百分比']}%)" for _, row in major_stats.iterrows()],
            textinfo="text",
            marker=dict(colors=[COLOR_PALETTE["secondary"], COLOR_PALETTE["primary"], COLOR_PALETTE["success"],
                                COLOR_PALETTE["warning"]], line=dict(color="white", width=1)),
            textfont=dict(size=9)
        )
    )

    fig.update_layout(
        title=f"专业分布（总计：{total_students}人）",
        title_font=dict(size=11, weight="bold"),
        height=180, template=PLOTLY_TEMPLATE,
        margin=dict(t=20, l=5, r=5, b=5),
        legend=dict(font=dict(size=9), orientation="v", yanchor="middle", y=0.5)
    )
    return fig


## 回调2：年龄分布扇形图（修复比例计算）
@app.callback(Output("age_chart", "figure"),
              [Input("score_weight", "value"),
               Input("user_filter", "value"),
               Input("class_selector", "value")])
def update_age_chart(score_w, user_filter, selected_class):
    df = all_df.copy()

    # 筛选逻辑修复
    if user_filter == "class" and selected_class:
        df = df[df["class"] == selected_class]

    # 仅保留唯一学生
    unique_students = df.drop_duplicates(subset=["student_ID"])

    # 年龄分布统计
    age_stats = unique_students.groupby("age").agg(人数=("student_ID", "nunique")).reset_index()
    total_students = age_stats["人数"].sum()

    # 处理空数据
    if total_students == 0:
        fig = go.Figure(go.Pie(values=[1], labels=["无数据"], hole=0.7))
        fig.update_layout(
            title="年龄分布（无数据）", title_font=dict(size=11, weight="bold"),
            height=180, template=PLOTLY_TEMPLATE,
            margin=dict(t=20, l=5, r=5, b=5)
        )
        return fig

    # 显式计算百分比
    age_stats["百分比"] = (age_stats["人数"] / total_students * 100).round(1)

    fig = go.Figure(
        go.Pie(
            labels=age_stats["age"],
            values=age_stats["人数"],
            hole=0.7,
            text=[f"{row['age']}<br>{row['人数']}人 ({row['百分比']}%)" for _, row in age_stats.iterrows()],
            textinfo="text",
            marker=dict(colors=[COLOR_PALETTE["primary"], COLOR_PALETTE["secondary"], COLOR_PALETTE["success"],
                                COLOR_PALETTE["warning"]], line=dict(color="white", width=1)),
            textfont=dict(size=9)
        )
    )

    fig.update_layout(
        title=f"年龄分布（总计：{total_students}人）",
        title_font=dict(size=11, weight="bold"),
        height=180, template=PLOTLY_TEMPLATE,
        margin=dict(t=20, l=5, r=5, b=5),
        legend=dict(font=dict(size=9), orientation="v", yanchor="middle", y=0.5)
    )
    return fig


## 回调3：性别分布扇形图（修复比例计算）
@app.callback(Output("gender_chart", "figure"),
              [Input("score_weight", "value"),
               Input("user_filter", "value"),
               Input("class_selector", "value")])
def update_gender_chart(score_w, user_filter, selected_class):
    df = all_df.copy()

    # 筛选逻辑修复
    if user_filter == "class" and selected_class:
        df = df[df["class"] == selected_class]

    # 仅保留唯一学生
    unique_students = df.drop_duplicates(subset=["student_ID"])

    # 性别分布统计
    gender_stats = unique_students.groupby("gender").agg(人数=("student_ID", "nunique")).reset_index()
    total_students = gender_stats["人数"].sum()

    # 处理空数据
    if total_students == 0:
        fig = go.Figure(go.Pie(values=[1], labels=["无数据"], hole=0.7))
        fig.update_layout(
            title="性别分布（无数据）", title_font=dict(size=11, weight="bold"),
            height=180, template=PLOTLY_TEMPLATE,
            margin=dict(t=20, l=5, r=5, b=5)
        )
        return fig

    # 显式计算百分比
    gender_stats["百分比"] = (gender_stats["人数"] / total_students * 100).round(1)

    fig = go.Figure(
        go.Pie(
            labels=gender_stats["gender"],
            values=gender_stats["人数"],
            hole=0.7,
            text=[f"{row['gender']}<br>{row['人数']}人 ({row['百分比']}%)" for _, row in gender_stats.iterrows()],
            textinfo="text",
            marker=dict(colors=[COLOR_PALETTE["secondary"], COLOR_PALETTE["primary"]],
                        line=dict(color="white", width=1)),
            textfont=dict(size=9)
        )
    )

    fig.update_layout(
        title=f"性别分布（总计：{total_students}人）",
        title_font=dict(size=11, weight="bold"),
        height=180, template=PLOTLY_TEMPLATE,
        margin=dict(t=20, l=5, r=5, b=5),
        legend=dict(font=dict(size=9), orientation="v", yanchor="middle", y=0.5)
    )
    return fig


## 新增回调4：班级排名柱状图（单独展示）
@app.callback(Output("class_ranking_chart", "figure"),
              [Input("score_weight", "value"),
               Input("user_filter", "value"),
               Input("class_selector", "value")])
def update_class_ranking_chart(score_w, user_filter, selected_class):
    df = all_df.copy()
    if user_filter == "class" and selected_class:
        df = df[df["class"] == selected_class]

    # 班级得分率统计并排序
    class_stats = df.groupby("class").agg(
        平均得分率=("Score_Rate", "mean"),
        人数=("student_ID", "nunique")
    ).reset_index()
    class_stats_sorted = class_stats.sort_values("平均得分率", ascending=False)

    fig = go.Figure(
        go.Bar(
            x=class_stats_sorted["class"],
            y=class_stats_sorted["平均得分率"],
            marker_color=COLOR_PALETTE["secondary"],
            width=0.6,
            text=[f"{v:.2f}<br>人数:{class_stats_sorted.loc[i, '人数']}" for i, v in
                  enumerate(class_stats_sorted["平均得分率"])],
            textposition="outside",
            textfont=dict(size=9)
        )
    )

    fig.update_layout(
        height=200, template=PLOTLY_TEMPLATE,
        margin=dict(t=20, l=10, r=10, b=30),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(range=[0, 1], tickfont=dict(size=9), title="平均得分率", title_font=dict(size=10)),
        showlegend=False
    )
    return fig


## 回调5：题目掌握折线图（修改为支持班级筛选）
@app.callback(Output("mastery_line_chart", "figure"),
              [Input("score_weight", "value"),
               Input("user_filter", "value"),  # 新增：班级筛选类型
               Input("class_selector", "value")])  # 新增：选中的班级
def update_mastery_line_chart(score_w, user_filter, selected_class):
    # 复制数据并根据班级筛选
    df = all_df.copy()
    if user_filter == "class" and selected_class:
        df = df[df["class"] == selected_class]

    # 按主知识点汇总所有题目的核心指标
    kp_stats = df.groupby("Main_KP").agg(
        平均掌握程度=("Mastery_Level", "mean"),
        平均得分率=("Score_Rate", "mean"),
        平均正确占比=("Correct_Rate", "mean"),
        题目数量=("title_code", "nunique")  # 补充题目数量维度
    ).reset_index()

    # 按题目数量排序（便于观察重点知识点）
    kp_stats = kp_stats.sort_values("题目数量", ascending=False)

    fig = go.Figure()
    # 平均掌握程度
    fig.add_trace(go.Scatter(
        x=kp_stats["Main_KP"],
        y=kp_stats["平均掌握程度"],
        name="平均掌握程度",
        line=dict(color=COLOR_PALETTE["secondary"], width=1.5),
        marker=dict(size=5),
        text=[f"题目数：{v}" for v in kp_stats["题目数量"]],  # 悬浮显示题目数量
        hoverinfo="x+y+text"
    ))
    # 平均得分率
    fig.add_trace(go.Scatter(
        x=kp_stats["Main_KP"],
        y=kp_stats["平均得分率"],
        name="平均得分率",
        line=dict(color=COLOR_PALETTE["primary"], width=1.5),
        marker=dict(size=5),
        text=[f"题目数：{v}" for v in kp_stats["题目数量"]],
        hoverinfo="x+y+text"
    ))
    # 平均正确占比
    fig.add_trace(go.Scatter(
        x=kp_stats["Main_KP"],
        y=kp_stats["平均正确占比"],
        name="平均正确占比",
        line=dict(color=COLOR_PALETTE["warning"], width=1.5),
        marker=dict(size=5),
        text=[f"题目数：{v}" for v in kp_stats["题目数量"]],
        hoverinfo="x+y+text"
    ))

    # 标记掌握程度最低的知识点（整体薄弱点）
    if len(kp_stats) > 0:
        weak_idx = kp_stats["平均掌握程度"].idxmin()
        fig.add_trace(
            go.Scatter(
                x=[kp_stats.loc[weak_idx, "Main_KP"]],
                y=[kp_stats.loc[weak_idx, "平均掌握程度"]],
                mode="markers",
                marker=dict(size=10, color=COLOR_PALETTE["danger"], symbol="star"),
                showlegend=False,
                name="最弱知识点"
            )
        )

    # 动态修改标题，显示当前筛选的班级
    title = "各知识点整体掌握情况（按题目数量排序）"
    if user_filter == "class" and selected_class:
        title = f"{selected_class} - 各知识点掌握情况（按题目数量排序）"

    fig.update_layout(
        title=title,
        title_font=dict(size=11, weight="bold"),
        height=160,
        template=PLOTLY_TEMPLATE,
        legend=dict(x=0.01, y=0.99, orientation="h", font=dict(size=9)),
        margin=dict(t=30, l=5, r=5, b=20)
    )
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=8), title="主知识点")
    fig.update_yaxes(range=[0, 1], tickfont=dict(size=8), title="指标值")
    return fig


## 回调6：用时分布柱状图（修改为支持班级筛选）
@app.callback(Output("time_dist_chart", "figure"),
              [Input("score_weight", "value"),
               Input("user_filter", "value"),  # 新增：班级筛选类型
               Input("class_selector", "value")])  # 新增：选中的班级
def update_time_dist_chart(score_w, user_filter, selected_class):
    # 复制数据并根据班级筛选
    df = all_df.copy()
    if user_filter == "class" and selected_class:
        df = df[df["class"] == selected_class]

    # 所有答题记录的用时分布（整体视角）
    fig = go.Figure(go.Histogram(
        x=df["time_consume"],
        nbinsx=15,  # 增加分箱数，更细致展示分布
        marker_color=COLOR_PALETTE["secondary"],
        opacity=0.8,
        # 增加统计信息
        text="用时分布",
        hovertemplate="用时：%{x}s<br>答题次数：%{y}<extra></extra>"
    ))

    # 计算并标注平均值
    avg_time = df["time_consume"].mean()
    fig.add_vline(
        x=avg_time,
        line_dash="dash",
        line_color=COLOR_PALETTE["danger"],
        annotation_text=f"平均值：{avg_time:.1f}s",
        annotation_font=dict(size=8, color=COLOR_PALETTE["danger"])
    )

    # 动态修改标题，显示当前筛选的班级
    title = "所有题目答题用时整体分布"
    if user_filter == "class" and selected_class:
        title = f"{selected_class} - 答题用时分布"

    fig.update_layout(
        title=title,
        title_font=dict(size=10),
        height=120,
        template=PLOTLY_TEMPLATE,
        margin=dict(t=25, l=5, r=5, b=15),
        showlegend=False
    )
    fig.update_xaxes(title="用时（s）", title_font=dict(size=9), tickfont=dict(size=8))
    fig.update_yaxes(title="答题次数", title_font=dict(size=9), tickfont=dict(size=8))
    return fig


## 回调7：内存分布柱状图（修改为支持班级筛选）
@app.callback(Output("mem_dist_chart", "figure"),
              [Input("score_weight", "value"),
               Input("user_filter", "value"),  # 新增：班级筛选类型
               Input("class_selector", "value")])  # 新增：选中的班级
def update_mem_dist_chart(score_w, user_filter, selected_class):
    # 复制数据并根据班级筛选
    df = all_df.copy()
    if user_filter == "class" and selected_class:
        df = df[df["class"] == selected_class]

    # 所有答题记录的内存使用分布（整体视角）
    fig = go.Figure(go.Histogram(
        x=df["memory_usage"],
        nbinsx=15,
        marker_color=COLOR_PALETTE["primary"],
        opacity=0.8,
        hovertemplate="内存：%{x}MB<br>答题次数：%{y}<extra></extra>"
    ))

    # 计算并标注平均值
    avg_mem = df["memory_usage"].mean()
    fig.add_vline(
        x=avg_mem,
        line_dash="dash",
        line_color=COLOR_PALETTE["danger"],
        annotation_text=f"平均值：{avg_mem:.0f}MB",
        annotation_font=dict(size=8, color=COLOR_PALETTE["danger"])
    )

    # 动态修改标题，显示当前筛选的班级
    title = "所有题目答题内存整体分布"
    if user_filter == "class" and selected_class:
        title = f"{selected_class} - 答题内存分布"

    fig.update_layout(
        title=title,
        title_font=dict(size=10),
        height=120,
        template=PLOTLY_TEMPLATE,
        margin=dict(t=25, l=5, r=5, b=15),
        showlegend=False
    )
    fig.update_xaxes(title="内存（MB）", title_font=dict(size=9), tickfont=dict(size=8))
    fig.update_yaxes(title="答题次数", title_font=dict(size=9), tickfont=dict(size=8))
    return fig


## 回调8：桑基图（原有逻辑不变）
@app.callback(Output("kp_sankey_chart", "figure"),
              [Input("score_weight", "value"), Input("comp_weight", "value"), Input("time_weight", "value"),
               Input("mem_weight", "value")])
def update_sankey_chart(score_w, comp_w, time_w, mem_w):
    # 原有桑基图逻辑完全保留
    nodes = []
    links = []

    # 总览节点
    nodes.append({"id": 0, "label": "总览", "color": COLOR_PALETTE["primary"]})

    # 主知识点
    main_kps = all_df["Main_KP"].unique()
    main_id_map = {kp: i + 1 for i, kp in enumerate(main_kps)}
    for kp, idx in main_id_map.items():
        nodes.append({"id": idx, "label": kp, "color": COLOR_PALETTE["success"]})
        value = len(all_df[all_df["Main_KP"] == kp])
        links.append({"source": 0, "target": idx, "value": value})

    # 子知识点
    sub_kps = all_df[["Main_KP", "Sub_KP"]].drop_duplicates()
    sub_id_map = {}
    current_id = len(main_id_map) + 1
    for _, row in sub_kps.iterrows():
        main_kp = row["Main_KP"]
        sub_kp = row["Sub_KP"]
        if sub_kp not in sub_id_map:
            sub_id_map[sub_kp] = current_id
            nodes.append({"id": current_id, "label": sub_kp, "color": COLOR_PALETTE["secondary"]})
            current_id += 1

        value = len(all_df[(all_df["Main_KP"] == main_kp) & (all_df["Sub_KP"] == sub_kp)])
        links.append({"source": main_id_map[main_kp], "target": sub_id_map[sub_kp], "value": value})

    # 题目
    titles = all_df[["Main_KP", "Sub_KP", "title_code"]].drop_duplicates()
    title_id_map = {}
    for _, row in titles.iterrows():
        title = row["title_code"]
        if title not in title_id_map:
            title_id_map[title] = current_id
            nodes.append({"id": current_id, "label": title, "color": COLOR_PALETTE["border"]})
            current_id += 1

        sub_kp = row["Sub_KP"]
        value = len(all_df[(all_df["Sub_KP"] == sub_kp) & (all_df["title_code"] == title)])
        links.append({"source": sub_id_map[sub_kp], "target": title_id_map[title], "value": value})

    # 创建桑基图
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=10,  # 紧凑间距
            thickness=15,  # 紧凑厚度
            line=dict(color=COLOR_PALETTE["border"], width=0.5),
            label=[n["label"] for n in nodes],
            color=[n["color"] for n in nodes]
        ),
        link=dict(
            source=[l["source"] for l in links],
            target=[l["target"] for l in links],
            value=[l["value"] for l in links],
            color="rgba(150,150,150,0.2)"
        )
    )])

    # 紧凑样式
    fig.update_layout(height=400, template=PLOTLY_TEMPLATE, margin=dict(t=15, l=10, r=10, b=15), font=dict(size=9))
    return fig


## 回调9：薄弱知识点表格（原有逻辑不变，适配新筛选）
@app.callback([Output("weak_compare_table", "data"), Output("table_error", "children")],
              [Input("user_filter", "value"), Input("class_selector", "value")])
def update_weak_table(selected_filter, selected_class):
    try:
        df = all_df.copy()
        if selected_filter == "class" and selected_class:
            df = df[df["class"] == selected_class]

        stats_df = df.groupby(["class", "Main_KP", "Sub_KP"], as_index=False).agg(
            Avg_Score_Rate=("Score_Rate", "mean"),
            Title_Count=("title_code", "nunique")
        ).dropna()

        if len(stats_df) == 0:
            return [], "暂无数据"

        def get_weak_level(score):
            score = float(score)
            if score < 0.4:
                return "极弱"
            elif score < 0.6:
                return "较弱"
            else:
                return "一般"

        stats_df["Weak_Level"] = stats_df["Avg_Score_Rate"].apply(get_weak_level)
        stats_df["Avg_Score_Rate"] = stats_df["Avg_Score_Rate"].round(4)
        stats_df = stats_df.sort_values("Avg_Score_Rate")

        return stats_df.to_dict("records"), ""

    except Exception as e:
        logger.error(f"表格更新失败：{str(e)}")
        return [], f"数据加载失败：{str(e)[:50]}"


## 回调10：重置按钮（原有逻辑不变）
@app.callback([Output("score_weight", "value"), Output("comp_weight", "value"), Output("time_weight", "value"),
               Output("mem_weight", "value"), Output("user_filter", "value")],
              Input("reset_btn", "n_clicks"), prevent_initial_call=True)
def reset_values(n_clicks):
    return 0.25, 0.25, 0.25, 0.25, "all"


# ---------------------- 5. 运行应用 ----------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=8050)