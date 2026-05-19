import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay
import joblib

# 设置页面配置
st.set_page_config(page_title="页岩气产能智能预测系统", layout="wide")

# 初始化 Session State
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'features' not in st.session_state:
    st.session_state.features = ['pressure_coeff', 'toc', 'sand_intensity', 'dist_to_fault']

st.title("📊 页岩气产能主控因素与智能预测平台")

tab1, tab2, tab3 = st.tabs(["1. 训练与主控分析", "2. 单井产能评估", "3. 部署说明"])

# --- TAB 1: 训练与主控分析 ---
with tab1:
    st.subheader("模型训练：揭示常压页岩气主控因素")
    train_file = st.file_uploader("上传历史井数据 (CSV)", type=['csv'])
    
    if train_file:
        df = pd.read_csv(train_file)
        if st.button("运行训练 & 科研分析"):
            # 数据预处理
            X = df[st.session_state.features]
            y = df['eur']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 训练模型
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X_scaled, y)
            
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.success("模型训练成功！")
            
            # A. 特征重要性可视化
            importances = pd.DataFrame({'特征参数': st.session_state.features, '权重': model.feature_importances_})
            fig1 = px.bar(importances, x='权重', y='特征参数', orientation='h', title="主控因素权重排序")
            st.plotly_chart(fig1, use_container_width=True)
            
            # B. 相关性热力图
            st.subheader("特征与产能的相关性矩阵")
            fig2 = px.imshow(df.corr(), text_auto=True, title="参数耦合关系")
            st.plotly_chart(fig2, use_container_width=True)

            # C. 部分依赖图 (PDP)
            st.subheader("关键因素对产能的非线性影响 (PDP)")
            fig_pdp, ax = plt.subplots(figsize=(10, 4))
            PartialDependenceDisplay.from_estimator(model, X_scaled, features=[0, 1], 
                                                   feature_names=st.session_state.features, ax=ax)
            st.pyplot(fig_pdp)

# --- TAB 2: 单井评估 ---
with tab2:
    st.subheader("新井参数输入预测")
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
            
            # 视觉反馈：雷达图展示匹配度
            st.metric("预测最终可采储量 (EUR)", f"{pred_eur[0]:.2f} 亿方")
            st.info("该预测已基于历史样本训练模型进行非线性修正。")

# --- TAB 3: 部署与使用建议 ---
with tab3:
    st.markdown("""
    ### 关键科研建议：
    1. **特征相关性**：重点观察热力图中的 `pressure_coeff` 与 `eur` 是否高度正相关，以此确立保存条件的主控地位。
    2. **非线性分析**：PDP 图中的曲线若出现转折，说明该参数对产能存在临界值或饱和效应，这通常是论文中的核心论点。
    3. **数据迭代**：随着新井资料补充，请通过重新上传数据集更新模型，看板将自动校准产能权重。
    """)
