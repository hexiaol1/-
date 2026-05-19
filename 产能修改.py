import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import io

# 设置页面配置
st.set_page_config(page_title="页岩气产能智能预测系统", layout="wide")

# 初始化 Session State 存储模型
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

st.title("📊 页岩气产能主控因素与智能预测平台")

tab1, tab2, tab3 = st.tabs(["1. 训练与主控分析", "2. 参数预测 (EUR)", "3. 部署说明"])

# --- TAB 1: 训练与主控分析 ---
with tab1:
    st.subheader("模型训练：揭示常压页岩气主控因素")
    train_file = st.file_uploader("上传已投产井历史数据 (CSV)", type=['csv'],
                                  help="需包含列: pressure_coeff, toc, sand_intensity, dist_to_fault, eur")

    if train_file:
        df = pd.read_csv(train_file)
        features = ['pressure_coeff', 'toc', 'sand_intensity', 'dist_to_fault']
        target = 'eur'

        if st.button("运行训练 & 因子分析"):
            # 数据预处理
            X = df[features]
            y = df[target]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 训练模型
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X_scaled, y)

            # 保存到 Session
            st.session_state.model = model
            st.session_state.scaler = scaler

            st.success("模型训练成功！")

            # 主控因素分析 (Feature Importance)
            importances = pd.DataFrame({'特征参数': features, '权重': model.feature_importances_})
            fig = px.bar(importances, x='权重', y='特征参数', orientation='h', title="主控因素权重分析 (基于随机森林)")
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: 参数预测 ---
with tab2:
    st.subheader("单井 EUR 预测")
    if st.session_state.model is None:
        st.warning("请先在 Tab 1 完成模型训练。")
    else:
        col1, col2 = st.columns(2)
        with col1:
            pc = st.number_input("压力系数", value=1.0)
            toc = st.number_input("TOC (%)", value=2.5)
        with col2:
            si = st.number_input("加砂强度 (t/m)", value=2.5)
            dtf = st.number_input("距断层距离 (m)", value=500)

        if st.button("预测 EUR"):
            input_data = np.array([[pc, toc, si, dtf]])
            scaled_input = st.session_state.scaler.transform(input_data)
            pred_eur = st.session_state.model.predict(scaled_input)
            st.metric("预测最终可采储量 (EUR)", f"{pred_eur[0]:.2f} 亿方")

# --- TAB 3: 部署与使用建议 ---
with tab3:
    st.markdown("""
    ### 平台功能说明
    1. **数据驱动**：本模型基于随机森林回归，自动处理非线性映射。
    2. **物理/统计结合**：通过 `StandardScaler` 处理量纲差异，确保地质参数与工程参数在同一维度分析。
    3. **主控因素揭示**：通过特征重要性排序，直接量化保存条件（断层）与储层品质的相对影响力。

    ### 下一步建议
    * 导入更丰富的样本库，引入 **SHAP 解释器**，以可视化展示每个参数对具体井位产能的影响方向（正向还是负向）。
    * 在生产环境部署时，建议使用 `joblib` 将训练好的 `model` 文件导出，实现线上推理。
    """)
