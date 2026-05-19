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

# --- TAB 2: 单井评估 (静动耦合预测) ---
with tab2:
    st.subheader("新井产能动态剖面预测 (ML + Duong模型耦合)")
    if st.session_state.model is None:
        st.warning("请先在 Tab 1 完成静态特征模型训练。")
    else:
        # 输入区
        c1, c2, c3, c4 = st.columns(4)
        pc = c1.number_input("压力系数", value=1.0)
        toc = c2.number_input("TOC (%)", value=2.5)
        si = c3.number_input("加砂强度 (t/m)", value=2.5)
        dtf = c4.number_input("距断层距离 (m)", value=500)
        
        # 物理模型参数设置（默认按常压页岩气典型值）
        with st.expander("⚙️ 高级选项: Duong 模型衰减参数设置"):
            d_col1, d_col2 = st.columns(2)
            m_val = d_col1.slider("递减指数 m (控制早期衰减速度, >1)", 1.01, 1.50, 1.15)
            a_val = d_col2.slider("常数 a (控制晚期尾部产量)", 0.5, 2.0, 1.05)
            q_ab = st.number_input("经济极限/废弃产量 (万方/天)", value=0.5, step=0.1)
            
        if st.button("🚀 生成产能动态剖面"):
            # 1. 机器学习预测最终体量 (EUR)
            input_data = np.array([[pc, toc, si, dtf]])
            scaled_input = st.session_state.scaler.transform(input_data)
            pred_eur = st.session_state.model.predict(scaled_input)[0]  # 单位: 亿方
            
            # 2. 物理规律约束：反算初期产量 (qi)
            # 核心逻辑：预测出的 EUR 必须等于未来所有时间日产量的积分
            t_days = np.arange(1, 3651) # 模拟 10 年 (3650天)
            # Duong 模型的形状函数
            shape_func = (t_days**-m_val) * np.exp((a_val / (1 - m_val)) * ((t_days**(1 - m_val)) - 1))
            
            # 数值积分 (将日产量累加转换为亿方)
            shape_integral = np.sum(shape_func) / 10000.0 
            qi_calc = pred_eur / shape_integral # 反算出所需的初期产量 (万方/天)
            
            # 3. 生成真实的日产量曲线
            q_time = qi_calc * shape_func
            
            # 找到跌破废弃产量的时间点
            valid_idx = np.where(q_time >= q_ab)[0]
            life_days = valid_idx[-1] if len(valid_idx) > 0 else 0
            life_years = life_days / 365.0
            
            # 4. 结果展示
            st.markdown("### 📊 综合预测结果")
            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric("预测 EUR (机器学习)", f"{pred_eur:.2f} 亿方")
            res_c2.metric("推算初期日产 ($q_i$)", f"{qi_calc:.2f} 万方/天")
            res_c3.metric("经济开采寿命", f"{life_years:.1f} 年")
            
            # 5. 绘制传统的递减曲线
            fig_curve = go.Figure()
            # 产量曲线
            fig_curve.add_trace(go.Scatter(x=t_days, y=q_time, mode='lines', name='日产量 (Duong)', line=dict(color='#1f77b4', width=3)))
            # 经济极限废弃线
            fig_curve.add_hline(y=q_ab, line_dash="dash", line_color="red", annotation_text=f"废弃产量 ({q_ab} 万方/d)")
            
            fig_curve.update_layout(
                title="单井生命周期产能递减预测 (10年模拟)", 
                xaxis_title="生产时间 (天)", 
                yaxis_title="日产量 (万方/天)",
                hovermode="x unified"
            )
            st.plotly_chart(fig_curve, use_container_width=True)
            
            # 双对数诊断图 (油气行业标配)
            fig_log = go.Figure()
            fig_log.add_trace(go.Scatter(x=t_days, y=q_time, mode='lines', name='日产量 (Log-Log)'))
            fig_log.update_layout(
                title="流动阶段诊断图 (双对数坐标)", 
                xaxis_title="生产时间 (天)", 
                yaxis_title="日产量 (万方/天)",
                xaxis_type="log", yaxis_type="log"
            )
            with st.expander("查看高级双对数诊断图"):
                st.plotly_chart(fig_log, use_container_width=True)

# --- TAB 3: 使用指南 ---
with tab3:
    st.markdown("""
    ### 科研分析要点：
    1. **参数耦合**：观察“参数耦合关系热力图”。若某参数与 EUR 负相关（如距断层距离过大时产能反下降），需结合绿豆岩地质特征进行机理解释。
    2. **饱和效应**：在“非线性响应图”中，若曲线随压力系数增加而平缓，可定义为开发甜点的“临界阈值”。
    3. **数据要求**：训练集请确保包含 `well_id` 以外的纯数值字段，且无空值。
    """)
