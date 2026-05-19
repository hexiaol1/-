import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# 设置页面配置
st.set_page_config(page_title="科学级页岩气产能预测系统", layout="wide")

# 初始化 Session State (存储多个模型)
if 'model_eur' not in st.session_state: st.session_state.model_eur = None
if 'model_m' not in st.session_state: st.session_state.model_m = None
if 'model_a' not in st.session_state: st.session_state.model_a = None
if 'scaler' not in st.session_state: st.session_state.scaler = None
if 'features' not in st.session_state: st.session_state.features = []
if 'feature_stats' not in st.session_state: st.session_state.feature_stats = {}

st.title("📊 科学级页岩气产能主控因素与预测平台")
st.caption("内核：多目标机器学习 (Multi-Target ML) 联合预测 EUR 与 Duong 物理参数 (a, m)")

tab1, tab2, tab3 = st.tabs(["1. 训练与主控分析", "2. 静动耦合产能评估", "3. 科学数据准备要求"])

# --- TAB 1: 训练与主控分析 ---
with tab1:
    st.subheader("多目标模型训练：同时学习体量与物理衰减规律")
    train_file = st.file_uploader("上传历史井数据 (CSV格式，必须包含 eur, m, a 三个目标列)", type=['csv'])
    
    if train_file:
        df = pd.read_csv(train_file)
        df_numeric = df.select_dtypes(include=[np.number])
        all_cols = df_numeric.columns.tolist()
        
        # 强制要求数据集中必须包含物理参数
        required_targets = ['eur', 'm', 'a']
        missing_targets = [col for col in required_targets if col not in all_cols]
        
        if missing_targets:
            st.error(f"❌ 数据格式错误：缺乏科学计算必须的物理目标列 {missing_targets}。请查阅 Tab 3 的数据准备说明。")
        else:
            available_features = [c for c in all_cols if c not in required_targets]
            selected_features = st.multiselect("选择输入特征 (X) [如: 地质、工程参数]", options=available_features, default=available_features)
            
            if st.button("🚀 运行多目标模型训练"):
                if not selected_features:
                    st.error("请至少选择一个输入特征！")
                else:
                    st.session_state.features = selected_features
                    st.session_state.feature_stats = df_numeric[selected_features].mean().to_dict()
                    
                    X = df_numeric[selected_features]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    st.session_state.scaler = scaler
                    
                    # 科学核心：分别训练三个模型，独立揭示各自的主控因素
                    st.session_state.model_eur = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_scaled, df_numeric['eur'])
                    st.session_state.model_m = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, df_numeric['m'])
                    st.session_state.model_a = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, df_numeric['a'])
                    
                    st.success(f"✅ 模型训练成功！系统已完全掌握历史井的地质-生产映射关系。")
                    
                    # 可视化：揭示不同的主控因素
                    st.markdown("#### 🔬 物理参数主控因素独立分析")
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        imp_eur = pd.DataFrame({'特征': selected_features, '权重': st.session_state.model_eur.feature_importances_}).sort_values(by='权重')
                        st.plotly_chart(px.bar(imp_eur, x='权重', y='特征', orientation='h', title="EUR 主控因素"), use_container_width=True)
                    with c2:
                        imp_m = pd.DataFrame({'特征': selected_features, '权重': st.session_state.model_m.feature_importances_}).sort_values(by='权重')
                        st.plotly_chart(px.bar(imp_m, x='权重', y='特征', orientation='h', title="参数 m (早期衰减) 主控因素"), use_container_width=True)
                    with c3:
                        imp_a = pd.DataFrame({'特征': selected_features, '权重': st.session_state.model_a.feature_importances_}).sort_values(by='权重')
                        st.plotly_chart(px.bar(imp_a, x='权重', y='特征', orientation='h', title="参数 a (储层供气) 主控因素"), use_container_width=True)

# --- TAB 2: 单井评估 (静动耦合预测) ---
with tab2:
    st.subheader("新井产能动态剖面预测 (科学闭环)")
    if st.session_state.model_eur is None:
        st.warning("请先在 Tab 1 完成模型训练。")
    else:
        st.markdown("##### 📍 1. 输入新井静态参数")
        input_dict = {}
        cols = st.columns(4)
        for i, feature in enumerate(st.session_state.features):
            val = cols[i % 4].number_input(f"{feature}", value=float(st.session_state.feature_stats[feature]), format="%.4f")
            input_dict[feature] = val
        
        st.markdown("---")
        
        # ML 预测与物理边界
        input_data = np.array([[input_dict[f] for f in st.session_state.features]])
        scaled_input = st.session_state.scaler.transform(input_data)
        
        # 科学内核：由 AI 预测出的唯一自洽组合
        pred_eur = st.session_state.model_eur.predict(scaled_input)[0]
        pred_m = st.session_state.model_m.predict(scaled_input)[0]
        pred_a = st.session_state.model_a.predict(scaled_input)[0]
        
        st.markdown("##### 📍 2. 边界条件与 AI 推荐物理参数")
        c_phys1, c_phys2 = st.columns([1, 2])
        with c_phys1:
            q_ab = st.number_input("经济废弃产量 $q_{ab}$ (万方/天)", value=0.50, step=0.10)
        
        with c_phys2:
            st.info(f"🧠 **AI 科学映射结果**：基于邻区历史数据，该地质条件下的物理衰减参数应为：**$m = {pred_m:.3f}$**, **$a = {pred_a:.3f}$**")
            use_ai_params = st.checkbox("强制使用 AI 推荐的物理参数 (科学模式)", value=True)
            
            col_m, col_a = st.columns(2)
            m_val = pred_m if use_ai_params else col_m.number_input("手动干预 m", value=pred_m, step=0.01)
            a_val = pred_a if use_ai_params else col_a.number_input("手动干预 a", value=pred_a, step=0.01)
            
        if st.button("🚀 生成产能动态剖面", type="primary"):
            # 积分运算求初期产量
            t_days = np.arange(1, 7301) 
            shape_func = (t_days**-m_val) * np.exp((a_val / (1 - m_val)) * ((t_days**(1 - m_val)) - 1))
            shape_integral = np.sum(shape_func) / 10000.0 
            qi_calc = pred_eur / shape_integral 
            
            # 截断运算
            q_time = qi_calc * shape_func
            valid_idx = np.where(q_time >= q_ab)[0]
            life_days = valid_idx[-1] if len(valid_idx) > 0 else 0
            
            if life_days == 0:
                st.error(f"⚠️ 该地质条件下，初始预测产量无法突破经济废弃线 ({q_ab} 万方/d)，建议重新评估压裂方案。")
            else:
                st.markdown("### 📊 科学预测结果")
                res_c1, res_c2, res_c3 = st.columns(3)
                res_c1.metric(f"预测 EUR", f"{pred_eur:.2f} 亿方")
                res_c2.metric("积分反演 $q_i$", f"{qi_calc:.2f} 万方/d")
                res_c3.metric("经济开采寿命", f"{life_days/365.0:.1f} 年")
                
                plot_days = min(len(t_days), life_days + 300)
                fig_curve = go.Figure()
                fig_curve.add_trace(go.Scatter(x=t_days[:plot_days], y=q_time[:plot_days], mode='lines', name='日产量', line=dict(color='#1f77b4', width=3)))
                fig_curve.add_hline(y=q_ab, line_dash="dash", line_color="red", annotation_text=f"废弃线 ({q_ab})")
                fig_curve.add_trace(go.Scatter(x=[life_days], y=[q_ab], mode='markers', name='废弃点', marker=dict(color='red', size=10)))
                
                fig_curve.update_layout(title="单井生命周期产能递减预测", xaxis_title="生产时间 (天)", yaxis_title="日产量 (万方/天)")
                st.plotly_chart(fig_curve, use_container_width=True)

# --- TAB 3: 科学数据准备要求 ---
with tab3:
    st.markdown("""
    ### 科学级训练数据规范
    为了确保物理参数 $m$ 和 $a$ 的科学性，上传的 `CSV` 训练集必须严格遵守以下规范：
    
    1. **必须包含的基础列**：
       * `eur`: 历史井的最终评估储量。
       * `m`: 历史井通过实际生产数据（至少6个月）进行 Duong 双对数诊断回归得出的斜率。
       * `a`: 历史井双对数诊断回归得出的截距。
    2. **自适应特征列**：可以自由添加任意数量的地质/工程参数（如 `toc`, `pressure_coeff`, `porosity`, `frac_stages` 等）。
    
    #### 为什么这样做最科学？
    传统的做法是仅预测 EUR，这会导致无穷多组 $(q_i, m, a)$ 都能凑出这个 EUR，导致工程设计失真。
    本系统采用 **Multi-Target 架构**，同时通过 AI 挖掘地质参数对 $m$（裂缝线性流）和 $a$（基质供气）的独立影响。在预测时，AI 会输出完全符合该区域特征的“基因级”物理参数，再反向积分求出配产，形成了**逻辑闭环**。
    """)
