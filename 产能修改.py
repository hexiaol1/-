import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay

# 设置页面配置
st.set_page_config(page_title="页岩气产能智能预测系统", layout="wide")

# 初始化 Session State
if 'model' not in st.session_state: st.session_state.model = None
if 'scaler' not in st.session_state: st.session_state.scaler = None
if 'features' not in st.session_state: st.session_state.features = ['pressure_coeff', 'toc', 'sand_intensity', 'dist_to_fault']

st.title("📊 页岩气产能主控因素与智能预测平台")

tab1, tab2, tab3 = st.tabs(["1. 训练与主控分析", "2. 单井产能评估", "3. 使用指南"])

# --- TAB 1: 训练与主控分析 ---
with tab1:
    st.subheader("模型训练：揭示常压页岩气主控因素")
    train_file = st.file_uploader("上传历史井数据 (CSV)", type=['csv'])
    
    if train_file:
        df = pd.read_csv(train_file)
        # 数据清洗：仅保留数值型特征
        df_numeric = df.select_dtypes(include=[np.number])
        
        if st.button("运行训练 & 科研分析"):
            X = df_numeric[st.session_state.features]
            y = df_numeric['eur']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X_scaled, y)
            
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.success("模型训练成功！")
            
            # 可视化布局
            col1, col2 = st.columns(2)
            with col1:
                # 特征重要性
                importances = pd.DataFrame({'特征参数': st.session_state.features, '权重': model.feature_importances_})
                fig1 = px.bar(importances, x='权重', y='特征参数', orientation='h', title="主控因素权重排序")
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                # 相关性热力图
                fig2 = px.imshow(df_numeric.corr(), text_auto=True, title="参数耦合关系热力图")
                st.plotly_chart(fig2, use_container_width=True)

            # PDP 部分依赖图
            st.subheader("关键因素的非线性响应 (PDP)")
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
        c1, c2, c3, c4 = st.columns(4)
        pc = c1.number_input("压力系数", value=1.0)
        toc = c2.number_input("TOC (%)", value=2.5)
        si = c3.number_input("加砂强度 (t/m)", value=2.5)
        dtf = c4.number_input("距断层距离 (m)", value=500)
            
        if st.button("开始预测"):
            input_data = np.array([[pc, toc, si, dtf]])
            scaled_input = st.session_state.scaler.transform(input_data)
            pred_eur = st.session_state.model.predict(scaled_input)
            st.metric("预测最终可采储量 (EUR)", f"{pred_eur[0]:.2f} 亿方")

# --- TAB 3: 使用指南 ---
with tab3:
    st.markdown("""
    ### 科研分析要点：
    1. **参数耦合**：观察“参数耦合关系热力图”。若某参数与 EUR 负相关（如距断层距离过大时产能反下降），需结合绿豆岩地质特征进行机理解释。
    2. **饱和效应**：在“非线性响应图”中，若曲线随压力系数增加而平缓，可定义为开发甜点的“临界阈值”。
    3. **数据要求**：训练集请确保包含 `well_id` 以外的纯数值字段，且无空值。
    """)
