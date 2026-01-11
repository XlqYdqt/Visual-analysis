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


# ---------------------- 1. 数据加载与预处理（适配真实数据字段） ----------------------
def load_data():
    """封装数据加载逻辑（适配真实CSV字段结构）"""
    try:
        # 加载学生信息（真实字段：index,student_ID,sex,age,major）
        student_df = pd.read_csv("Data_StudentInfo.csv")
        # 字段重命名：sex → gender（适配原代码逻辑）
        student_df.rename(columns={"sex": "gender"}, inplace=True)
        # 移除无用的index列（如果存在）
        if "index" in student_df.columns:
            student_df.drop(columns=["index"], inplace=True)
        logger.info(f"学生信息表加载完成，行数：{len(student_df)}，字段：{student_df.columns.tolist()}")

        # 加载题目信息（真实字段：index,title_ID,score,knowledge,sub_knowledge）
        title_df = pd.read_csv("Data_TitleInfo.csv")
        # 字段重命名：score → full_score（适配原代码逻辑）
        title_df.rename(columns={"score": "full_score"}, inplace=True)
        # 移除无用的index列
        if "index" in title_df.columns:
            title_df.drop(columns=["index"], inplace=True)
        logger.info(f"题目信息表加载完成，行数：{len(title_df)}，字段：{title_df.columns.tolist()}")

        # 加载答题记录（真实字段：index,class,time,state,score,title_ID,method,memory,timeconsume,student_ID）
        submit_records = []
        submit_folder = "Data_SubmitRecord"
        if not os.path.exists(submit_folder):
            raise FileNotFoundError(f"答题记录文件夹 {submit_folder} 不存在！")

        for file in os.listdir(submit_folder):
            if file.endswith(".csv"):
                file_path = os.path.join(submit_folder, file)
                df = pd.read_csv(file_path)
                # 字段重命名（适配原代码逻辑）
                df.rename(columns={
                    "score": "student_score",  # 学生得分
                    "timeconsume": "time_consume",  # 答题耗时
                    "memory": "memory_usage"  # 内存使用
                }, inplace=True)
                # 移除无用的index列
                if "index" in df.columns:
                    df.drop(columns=["index"], inplace=True)
                submit_records.append(df)
                logger.info(f"加载答题记录 {file}，行数：{len(df)}，字段：{df.columns.tolist()}")

        if not submit_records:
            raise ValueError("答题记录文件夹中无CSV文件！")

        submit_df = pd.concat(submit_records, ignore_index=True)
        logger.info(f"答题记录合并完成，总行数：{len(submit_df)}")

        # 验证核心字段是否存在
        required_student_fields = ["student_ID", "gender", "age", "major"]
        missing_student_fields = [f for f in required_student_fields if f not in student_df.columns]
        if missing_student_fields:
            raise ValueError(f"学生信息表缺失核心字段：{missing_student_fields}")

        required_title_fields = ["title_ID", "full_score", "knowledge", "sub_knowledge"]
        missing_title_fields = [f for f in required_title_fields if f not in title_df.columns]
        if missing_title_fields:
            raise ValueError(f"题目信息表缺失核心字段：{missing_title_fields}")

        required_submit_fields = ["class", "student_score", "title_ID", "time_consume", "memory_usage", "student_ID"]
        missing_submit_fields = [f for f in required_submit_fields if f not in submit_df.columns]
        if missing_submit_fields:
            raise ValueError(f"答题记录表缺失核心字段：{missing_submit_fields}")

        return student_df, title_df, submit_df

    except Exception as e:
        logger.error(f"真实数据加载失败：{str(e)}")
        # 生成模拟数据（仅作为备用，保持字段与真实数据一致）
        logger.warning("使用模拟数据运行程序（模拟真实字段结构）")

        # 模拟学生信息（匹配真实字段：student_ID,gender,age,major）
        student_df = pd.DataFrame({
            "student_ID": [f"S{i}" for i in range(100)],
            "gender": np.random.choice(["男", "女"], 100),
            "age": np.random.choice(["18-20", "21-23", "24-26", "27+"], 100),
            "major": np.random.choice(["计算机", "数学", "物理", "化学"], 100)
        })

        # 模拟题目信息（匹配真实字段：title_ID,full_score,knowledge,sub_knowledge）
        title_df = pd.DataFrame({
            "title_ID": [f"T{i}" for i in range(20)],
            "full_score": [100] * 20,
            "knowledge": [f"KP{i % 5}" for i in range(20)],
            "sub_knowledge": [f"SK{i % 10}" for i in range(20)]
        })

        # 模拟答题记录（匹配真实字段：class,time,state,student_score,title_ID,method,memory_usage,time_consume,student_ID）
        submit_df = pd.DataFrame({
            "class": [f"Class_{i % 3}" for i in range(500)],
            "time": pd.date_range(start="2024-01-01", periods=500, freq="H"),
            "state": np.random.choice(["完成", "未完成"], 500),
            "student_score": np.random.uniform(50, 95, 500),
            "title_ID": np.random.choice([f"T{i}" for i in range(20)], 500),
            "method": np.random.choice(["方法A", "方法B", "方法C"], 500),
            "memory_usage": np.random.randint(160, 500, 500),
            "time_consume": np.random.randint(0, 40, 500),
            "student_ID": np.random.choice([f"S{i}" for i in range(100)], 500)
        })

        return student_df, title_df, submit_df


# 加载数据
student_df, title_df, submit_df = load_data()

# 数据预处理（适配真实字段）
# 类型转换 + 空值填充
for df in [title_df, submit_df, student_df]:
    for col in df.columns:
        if "ID" in col or "class" in col or "knowledge" in col or "sub_knowledge" in col:
            df[col] = df[col].astype(str).fillna("未知")
        # 新增：强制清洗数值型字段（重点处理memory_usage）
        elif col == "memory_usage":
            # 转为数值类型，无法转换的填充为中位数（或0，根据业务调整）
            df[col] = pd.to_numeric(df[col], errors="coerce")  # 无法转换的设为NaN
            median_val = df[col].median() if not df[col].isnull().all() else 300  # 兜底默认值
            df[col] = df[col].fillna(median_val).astype(int)
        elif col in ["student_score", "full_score", "time_consume"]:
            # 顺带清洗其他数值字段，防止同类问题
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# 补充知识点字段
title_df["Main_KP"] = title_df["knowledge"].apply(lambda x: f"{x[:5]}" if x != "未知" else "未知主知识点")
title_df["Sub_KP"] = title_df["sub_knowledge"].apply(lambda x: f"{x[:7]}" if x != "未知" else "未知子知识点")

# 合并数据（class字段来自答题记录，而非学生信息）
submit_title_df = pd.merge(submit_df, title_df[["title_ID", "full_score", "Main_KP", "Sub_KP"]],
                           on="title_ID", how="left")
all_df = pd.merge(submit_title_df, student_df[["student_ID", "gender", "age", "major"]],
                  on="student_ID", how="left")

# 填充空值
all_df["Main_KP"] = all_df["Main_KP"].fillna("未知主知识点")
all_df["Sub_KP"] = all_df["Sub_KP"].fillna("未知子知识点")
all_df["class"] = all_df["class"].fillna("未知班级")
all_df["gender"] = all_df["gender"].fillna("未知性别")
all_df["age"] = all_df["age"].fillna("未知年龄")
all_df["major"] = all_df["major"].fillna("未知专业")
all_df["full_score"] = pd.to_numeric(all_df["full_score"], errors="coerce").fillna(0)
all_df["student_score"] = pd.to_numeric(all_df["student_score"], errors="coerce").fillna(0)

# 计算得分率（使用安全除法）
all_df["Score_Rate"] = all_df.apply(lambda row: safe_division(row["student_score"], row["full_score"]), axis=1)
all_df["Score_Rate"] = all_df["Score_Rate"].fillna(0).astype(float)

# 模拟扩展字段（保持原代码的分析维度）
all_df["title_code"] = all_df["title_ID"].apply(lambda x: f"Q_{x[:3]}")
all_df["Correct_Rate"] = np.random.uniform(0.4, 0.9, len(all_df))  # 可根据真实state字段计算真实正确率
all_df["Mastery_Level"] = all_df["Score_Rate"] * 0.7 + all_df["Correct_Rate"] * 0.3  # 基于真实得分率计算掌握度
all_df["multi_kp"] = np.random.choice([0, 1], len(all_df), p=[0.85, 0.15])

# ---------------------- 2. 初始化Dash应用 ----------------------
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "知识点掌握度分析系统"

# ---------------------- 3. 重新设计的紧凑布局（保持原布局不变） ----------------------
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
                    {"name": "主知识点", "id": "主知识点"},
                    {"name": "子知识点", "id": "子知识点"},
                    {"name": "班级", "id": "班级"},
                    {"name": "平均得分率", "id": "平均得分率"},
                    {"name": "题目数量", "id": "题目数量"},
                    {"name": "薄弱等级", "id": "薄弱等级"},
                    {"name": "问题类型", "id": "问题类型"}
                ],
                style_table={"overflowX": "auto", "fontSize": "10px"},
                style_cell={"textAlign": "center", "padding": "4px 8px", "fontSize": "10px",
                            "borderBottom": f"1px solid {COLOR_PALETTE['border']}"},
                style_header={"backgroundColor": COLOR_PALETTE["light"], "fontWeight": "bold", "fontSize": "10px"},
                style_data_conditional=[
                    {"if": {"filter_query": "{问题类型} = '全员难题'"}, "backgroundColor": "#ffebee",
                     "color": "#d32f2f"},
                    {"if": {"filter_query": "{问题类型} = '个别班级问题'"}, "backgroundColor": "#fff3e0",
                     "color": "#f57c00"},
                    {"if": {"filter_query": "{薄弱等级} = '极弱'"}, "backgroundColor": "#ffebee", "color": "#d32f2f"},
                    {"if": {"filter_query": "{薄弱等级} = '较弱'"}, "backgroundColor": "#fff3e0", "color": "#f57c00"},
                    {"if": {"filter_query": "{薄弱等级} = '一般'"}, "backgroundColor": "#e8f5e9", "color": "#388e3c"},
                ],
                page_size=8,
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


# ---------------------- 4. 回调函数（适配新排版，保持原逻辑） ----------------------
## 回调1：专业分布扇形图（修复比例计算 + 移除未知专业）
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
    # 移除「未知专业」的记录
    unique_students = unique_students[unique_students["major"] != "未知专业"]

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


## 回调2：年龄分布扇形图（修复比例计算 + 移除未知年龄）
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
    # 移除「未知年龄」的记录
    unique_students = unique_students[unique_students["age"] != "未知年龄"]

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


## 回调3：性别分布扇形图（修复比例计算 + 移除未知性别）
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
    # 移除「未知性别」的记录
    unique_students = unique_students[unique_students["gender"] != "未知性别"]

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
## 回调5：题目掌握折线图（修复题目数量统计 + 支持班级筛选）
@app.callback(Output("mastery_line_chart", "figure"),
              [Input("score_weight", "value"),
               Input("user_filter", "value"),  # 新增：班级筛选类型
               Input("class_selector", "value")])  # 新增：选中的班级
def update_mastery_line_chart(score_w, user_filter, selected_class):
    # 复制数据并根据班级筛选
    df = all_df.copy()
    if user_filter == "class" and selected_class:
        df = df[df["class"] == selected_class]

    # 数据校验：确保title_ID和Main_KP字段存在且非空
    if "title_ID" not in df.columns or "Main_KP" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="数据缺失：缺少title_ID或Main_KP字段",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=10, color=COLOR_PALETTE["danger"])
        )
        fig.update_layout(
            title="各知识点整体掌握情况（数据异常）",
            title_font=dict(size=11, weight="bold"),
            height=160, template=PLOTLY_TEMPLATE,
            margin=dict(t=30, l=5, r=5, b=20)
        )
        return fig

    # 按主知识点汇总所有题目的核心指标
    # 关键修改：用原始title_ID统计唯一题目数（替代截取的title_code）
    kp_stats = df.groupby("Main_KP").agg(
        平均掌握程度=("Mastery_Level", "mean"),
        平均得分率=("Score_Rate", "mean"),
        平均正确占比=("Correct_Rate", "mean"),
        题目数量=("title_ID", "nunique")  # 修正：使用title_ID统计真实题目数
    ).reset_index()

    # 过滤空知识点（避免干扰）
    kp_stats = kp_stats[kp_stats["Main_KP"] != "未知主知识点"]

    # 按题目数量排序（便于观察重点知识点）
    kp_stats = kp_stats.sort_values("题目数量", ascending=False)

    # 空数据处理
    if len(kp_stats) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="暂无有效知识点数据",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=10, color=COLOR_PALETTE["danger"])
        )
        fig.update_layout(
            title="各知识点整体掌握情况（无数据）",
            title_font=dict(size=11, weight="bold"),
            height=160, template=PLOTLY_TEMPLATE,
            margin=dict(t=30, l=5, r=5, b=20)
        )
        return fig

    fig = go.Figure()
    # 平均掌握程度
    fig.add_trace(go.Scatter(
        x=kp_stats["Main_KP"],
        y=kp_stats["平均掌握程度"],
        name="平均掌握程度",
        line=dict(color=COLOR_PALETTE["secondary"], width=1.5),
        marker=dict(size=5),
        text=[f"题目数：{v}" for v in kp_stats["题目数量"]],  # 悬浮显示真实题目数
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

## 回调6：用时分布柱状图（修复显示问题 + 异常值处理）
@app.callback(Output("time_dist_chart", "figure"),
              [Input("score_weight", "value"),
               Input("user_filter", "value"),
               Input("class_selector", "value")])
def update_time_dist_chart(score_w, user_filter, selected_class):
    # 复制数据并根据班级筛选
    df = all_df.copy()
    if user_filter == "class" and selected_class:
        df = df[df["class"] == selected_class]

    # 1. 过滤无效用时（<=0的数值）
    df = df[df["time_consume"] > 0]

    # 空数据处理
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="暂无有效用时数据",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=10, color=COLOR_PALETTE["danger"])
        )
        fig.update_layout(
            title="答题用时分布（无数据）",
            title_font=dict(size=10),
            height=120,
            template=PLOTLY_TEMPLATE,
            margin=dict(t=25, l=5, r=5, b=15),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    # 2. 处理异常值（四分位数法过滤离群值）
    q1 = df["time_consume"].quantile(0.25)
    q3 = df["time_consume"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # 只保留合理范围内的数据（避免极端值拉伸x轴）
    df_filtered = df[(df["time_consume"] >= lower_bound) & (df["time_consume"] <= upper_bound)]

    # 如果过滤后无数据，使用原始数据（兜底）
    if len(df_filtered) == 0:
        df_filtered = df

    # 3. 优化分箱数（基于过滤后的数据）
    time_min = df_filtered["time_consume"].min()
    time_max = df_filtered["time_consume"].max()
    # 分箱数控制在5-20之间，保证视觉效果
    nbinsx = min(20, max(5, int((time_max - time_min) / 2)))  # 每2秒一个区间

    # 4. 绘制直方图（使用过滤后的数据）
    fig = go.Figure(go.Histogram(
        x=df_filtered["time_consume"],
        nbinsx=nbinsx,
        marker_color=COLOR_PALETTE["secondary"],
        opacity=0.8,
        hovertemplate="用时：%{x}秒<br>答题次数：%{y}<extra></extra>",
        marker_line=dict(color=COLOR_PALETTE["border"], width=0.5)
    ))

    # 计算并标注平均值（基于过滤后的数据）
    avg_time = df_filtered["time_consume"].mean()
    fig.add_vline(
        x=avg_time,
        line_dash="dash",
        line_color=COLOR_PALETTE["danger"],
        annotation_text=f"平均值：{avg_time:.1f}秒",
        annotation_font=dict(size=8, color=COLOR_PALETTE["danger"]),
        annotation_xanchor="left",
        annotation_position="top"
    )

    # 动态标题
    title = "答题用时整体分布"
    if user_filter == "class" and selected_class:
        title = f"{selected_class} - 答题用时分布"

    # 5. 优化x轴范围（仅显示过滤后的数据范围，留少量余量）
    x_range_padding = (time_max - time_min) * 0.1  # 10%的余量
    fig.update_layout(
        title=title,
        title_font=dict(size=10),
        height=120,
        template=PLOTLY_TEMPLATE,
        margin=dict(t=25, l=5, r=5, b=15),
        showlegend=False,
        xaxis=dict(
            title="用时（秒）",
            title_font=dict(size=9),
            tickfont=dict(size=8),
            range=[max(0, time_min - x_range_padding), time_max + x_range_padding]
        ),
        yaxis=dict(
            title="答题次数",
            title_font=dict(size=9),
            tickfont=dict(size=8),
            range=[0, None]
        )
    )
    return fig


## 回调7：内存分布柱状图（修复显示问题 + 异常值处理）
@app.callback(Output("mem_dist_chart", "figure"),
              [Input("score_weight", "value"),
               Input("user_filter", "value"),
               Input("class_selector", "value")])
def update_mem_dist_chart(score_w, user_filter, selected_class):
    # 复制数据并根据班级筛选
    df = all_df.copy()
    if user_filter == "class" and selected_class:
        df = df[df["class"] == selected_class]

    # 1. 过滤无效内存（<=0的数值）
    df = df[df["memory_usage"] > 0]

    # 空数据处理
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="暂无有效内存数据",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=10, color=COLOR_PALETTE["danger"])
        )
        fig.update_layout(
            title="答题内存分布（无数据）",
            title_font=dict(size=10),
            height=120,
            template=PLOTLY_TEMPLATE,
            margin=dict(t=25, l=5, r=5, b=15),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    # 2. 处理异常值（四分位数法过滤离群值）
    q1 = df["memory_usage"].quantile(0.25)
    q3 = df["memory_usage"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_filtered = df[(df["memory_usage"] >= lower_bound) & (df["memory_usage"] <= upper_bound)]

    # 兜底逻辑
    if len(df_filtered) == 0:
        df_filtered = df

    # 3. 优化分箱数（基于过滤后的数据）
    mem_min = df_filtered["memory_usage"].min()
    mem_max = df_filtered["memory_usage"].max()
    # 每10MB一个区间，分箱数控制在5-20之间
    nbinsx = min(20, max(5, int((mem_max - mem_min) / 10)))

    # 4. 绘制直方图
    fig = go.Figure(go.Histogram(
        x=df_filtered["memory_usage"],
        nbinsx=nbinsx,
        marker_color=COLOR_PALETTE["primary"],
        opacity=0.8,
        hovertemplate="内存：%{x}MB<br>答题次数：%{y}<extra></extra>",
        marker_line=dict(color=COLOR_PALETTE["border"], width=0.5)
    ))

    # 计算并标注平均值
    avg_mem = df_filtered["memory_usage"].mean()
    fig.add_vline(
        x=avg_mem,
        line_dash="dash",
        line_color=COLOR_PALETTE["danger"],
        annotation_text=f"平均值：{avg_mem:.0f}MB",
        annotation_font=dict(size=8, color=COLOR_PALETTE["danger"]),
        annotation_xanchor="left",
        annotation_position="top"
    )

    # 动态标题
    title = "答题内存整体分布"
    if user_filter == "class" and selected_class:
        title = f"{selected_class} - 答题内存分布"

    # 5. 优化x轴范围
    x_range_padding = (mem_max - mem_min) * 0.1
    fig.update_layout(
        title=title,
        title_font=dict(size=10),
        height=120,
        template=PLOTLY_TEMPLATE,
        margin=dict(t=25, l=5, r=5, b=15),
        showlegend=False,
        xaxis=dict(
            title="内存（MB）",
            title_font=dict(size=9),
            tickfont=dict(size=8),
            range=[max(0, mem_min - x_range_padding), mem_max + x_range_padding]
        ),
        yaxis=dict(
            title="答题次数",
            title_font=dict(size=9),
            tickfont=dict(size=8),
            range=[0, None]
        )
    )
    return fig


## 回调8：桑基图（原有逻辑不变，补充完整）
@app.callback(Output("kp_sankey_chart", "figure"),
              [Input("score_weight", "value"), Input("comp_weight", "value"), Input("time_weight", "value"),
               Input("mem_weight", "value")])
def update_sankey_chart(score_w, comp_w, time_w, mem_w):
    # 计算主知识点的平均掌握度
    main_kp_mastery = all_df.groupby("Main_KP")["Mastery_Level"].mean().reset_index()
    main_kp_mastery.columns = ["Main_KP", "avg_mastery"]
    main_kp_mastery = main_kp_mastery[main_kp_mastery["avg_mastery"].notna()]

    # 计算子知识点的平均掌握度
    sub_kp_mastery = all_df.groupby(["Main_KP", "Sub_KP"])["Mastery_Level"].mean().reset_index()
    sub_kp_mastery.columns = ["Main_KP", "Sub_KP", "avg_mastery"]
    sub_kp_mastery = sub_kp_mastery[sub_kp_mastery["avg_mastery"].notna()]

    # 创建颜色映射函数，将掌握度映射到更明显的蓝色系（浅蓝->深蓝）
    def get_color(mastery):
        # 掌握度从0到1，颜色从浅天蓝色（0）到深蓝色（1）
        # 浅天蓝色: (180, 220, 255) - 更浅的蓝色
        # 深蓝色: (0, 0, 100) - 更深的蓝色
        r = max(0, min(255, int(180 * (1 - mastery*1.3))))
        g = max(0, min(255, int(220 * (1 - mastery*1.3))))
        b = max(0, min(255, int(255 - 155 * mastery * 1.4)))
        return f"rgb({r}, {g}, {b})"

    # 创建节点列表
    nodes = []
    links = []

    # 总览节点
    nodes.append({"id": 0, "label": "总览", "color": COLOR_PALETTE["primary"]})

    # 主知识点节点（排除"未知主知识点"）
    main_kps = all_df[all_df["Main_KP"] != "未知主知识点"]["Main_KP"].unique()
    main_id_map = {kp: i + 1 for i, kp in enumerate(main_kps)}

    for kp, idx in main_id_map.items():
        # 根据平均掌握度获取颜色
        if kp in main_kp_mastery["Main_KP"].values:
            avg_mastery = main_kp_mastery[main_kp_mastery["Main_KP"] == kp]["avg_mastery"].values[0]
            color = get_color(avg_mastery)
        else:
            color = COLOR_PALETTE["danger"]  # 如果没有掌握度数据，用红色表示
        nodes.append({"id": idx, "label": kp, "color": color})

        value = len(all_df[all_df["Main_KP"] == kp])
        links.append({"source": 0, "target": idx, "value": value})

    # 子知识点节点（排除"未知子知识点"）
    sub_kp_mastery = sub_kp_mastery[sub_kp_mastery["Sub_KP"] != "未知子知识点"]

    # 为每个子知识点分配唯一ID
    sub_id_map = {}
    current_id = len(main_id_map) + 1

    # 为每个子知识点分配ID
    for _, row in sub_kp_mastery.iterrows():
        sub_kp = row["Sub_KP"]
        if sub_kp not in sub_id_map:
            sub_id_map[sub_kp] = current_id
            current_id += 1

    # 为子知识点分配颜色
    for _, row in sub_kp_mastery.iterrows():
        sub_kp = row["Sub_KP"]
        avg_mastery = row["avg_mastery"]
        color = get_color(avg_mastery)
        # 为节点添加颜色
        nodes.append({"id": sub_id_map[sub_kp], "label": sub_kp, "color": color})

    # 主知识点→子知识点链接
    for _, row in sub_kp_mastery.iterrows():
        main_kp = row["Main_KP"]
        sub_kp = row["Sub_KP"]
        if main_kp in main_id_map and sub_kp in sub_id_map:
            main_id = main_id_map[main_kp]
            sub_id = sub_id_map[sub_kp]
            value = len(all_df[(all_df["Main_KP"] == main_kp) & (all_df["Sub_KP"] == sub_kp)])
            links.append({"source": main_id, "target": sub_id, "value": value})

    # 构建桑基图数据
    node_labels = [n["label"] for n in nodes]
    node_colors = [n["color"] for n in nodes]
    link_sources = [l["source"] for l in links]
    link_targets = [l["target"] for l in links]
    link_values = [l["value"] for l in links]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=link_sources,
            target=link_targets,
            value=link_values,
            color="rgba(76, 175, 80, 0.2)"
        )
    )])

    fig.update_layout(
        title_text="知识点层级关联桑基图（颜色越深表示掌握度越高）",
        title_font=dict(size=12, weight="bold"),
        height=400,
        template=PLOTLY_TEMPLATE,
        margin=dict(t=30, l=20, r=20, b=20)
    )

    return fig

## 回调9：薄弱知识点表格（补充完整逻辑）
## 回调9：薄弱知识点表格（补充完整逻辑）
@app.callback([Output("weak_compare_table", "data"), Output("table_error", "children")],
              [Input("score_weight", "value"), Input("user_filter", "value"), Input("class_selector", "value")])
def update_weak_table(score_w, user_filter, selected_class):
    df = all_df.copy()
    if user_filter == "class" and selected_class:
        df = df[df["class"] == selected_class]

    # 按主知识点、子知识点、班级分组计算
    weak_stats = df.groupby(["Main_KP", "Sub_KP", "class"]).agg(
        Avg_Score_Rate=("Score_Rate", "mean"),
        Title_Count=("title_ID", "nunique")
    ).reset_index()

    # 计算每个知识点在所有班级的平均得分率（用于判断问题类型）
    overall_avg = df.groupby(["Main_KP", "Sub_KP"]).agg(Overall_Avg_Score_Rate=("Score_Rate", "mean")).reset_index()

    # 将整体平均得分率合并到弱知识点统计表中
    weak_stats = pd.merge(weak_stats, overall_avg, on=["Main_KP", "Sub_KP"], how="left")

    # 定义薄弱等级
    def get_weak_level(rate):
        if rate < 0.6:
            return "极弱"
        elif rate < 0.75:
            return "较弱"
        else:
            return "一般"

    weak_stats["Weak_Level"] = weak_stats["Avg_Score_Rate"].apply(get_weak_level)

    # 判断问题类型（关键修改：添加了NaN处理）
    def get_problem_type(row):
        # 处理NaN值
        overall_avg_val = row["Overall_Avg_Score_Rate"]
        if pd.isna(overall_avg_val):
            overall_avg_val = 1.0  # 一个高于阈值的值，避免进入"全员难题"或"个别班级问题"

        # 如果该知识点在所有班级的得分率都低于0.6，就是"全员难题"
        if row["Avg_Score_Rate"] < 0.6 and overall_avg_val < 0.6:
            return "全员难题"
        # 如果该知识点在某个班级的得分率低于0.6，但在其他班级的得分率高于0.6，就是"个别班级问题"
        elif row["Avg_Score_Rate"] < 0.6 and overall_avg_val >= 0.6:
            return "个别班级问题"
        # 否则就是正常
        else:
            return "正常"

    weak_stats["Problem_Type"] = weak_stats.apply(get_problem_type, axis=1)

    # 按问题类型和得分率排序（使"全员难题"和"个别班级问题"排在前面）
    weak_stats = weak_stats.sort_values(["Problem_Type", "Avg_Score_Rate"], ascending=[True, True])

    # 保留4位小数
    weak_stats["Avg_Score_Rate"] = weak_stats["Avg_Score_Rate"].round(4)
    weak_stats["Overall_Avg_Score_Rate"] = weak_stats["Overall_Avg_Score_Rate"].round(4)

    if len(weak_stats) == 0:
        return [], "暂无薄弱知识点数据"

    # 重命名列，确保表格显示正确的列名
    weak_stats = weak_stats.rename(columns={
        "Main_KP": "主知识点",
        "Sub_KP": "子知识点",
        "class": "班级",
        "Avg_Score_Rate": "平均得分率",
        "Title_Count": "题目数量",
        "Weak_Level": "薄弱等级",
        "Overall_Avg_Score_Rate": "整体平均得分率",
        "Problem_Type": "问题类型"
    })

    # 只返回需要的列
    return weak_stats[["主知识点", "子知识点", "班级", "平均得分率", "题目数量", "薄弱等级", "问题类型"]].to_dict(
        "records"), ""

## 回调10：重置按钮逻辑
@app.callback(
    [Output("score_weight", "value"),
     Output("comp_weight", "value"),
     Output("time_weight", "value"),
     Output("mem_weight", "value"),
     Output("user_filter", "value")],
    Input("reset_btn", "n_clicks"),
    prevent_initial_call=True
)
def reset_controls(n_clicks):
    return 0.25, 0.25, 0.25, 0.25, "all"


## 回调11：初始化按钮逻辑
@app.callback(
    Output("weak_compare_table", "data", allow_duplicate=True),
    Input("init_btn", "n_clicks"),
    State("user_filter", "value"),
    State("class_selector", "value"),
    prevent_initial_call=True
)
def init_analysis(n_clicks, user_filter, selected_class):
    df = all_df.copy()
    if user_filter == "class" and selected_class:
        df = df[df["class"] == selected_class]

    weak_stats = df.groupby(["Main_KP", "Sub_KP", "class"]).agg(
        Avg_Score_Rate=("Score_Rate", "mean"),
        Title_Count=("title_ID", "nunique")
    ).reset_index()

    weak_stats["Weak_Level"] = weak_stats["Avg_Score_Rate"].apply(
        lambda x: "极弱" if x < 0.6 else "较弱" if x < 0.75 else "一般")
    weak_stats["Avg_Score_Rate"] = weak_stats["Avg_Score_Rate"].round(4)
    return weak_stats.to_dict("records")


# 运行应用
if __name__ == "__main__":
    app.run(debug=True)