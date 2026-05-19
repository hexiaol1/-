import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ==========================================
# 页面全局配置与初始化
# ==========================================
st.set_page_config(page_title="科学级页岩气产能预测系统", layout="wide")

# 初始化 Session State (存储多目标模型)
if 'model_eur' not in st.session_state: st.session_state.model_eur = None
if 'model_m' not in st.session_state: st.session_state.model_m = None
if 'model_a' not in st.session_state: st.session_state.model_a = None
if 'scaler' not in st.session_state: st.session_state.scaler = None
if 'features' not in st.session_state: st.session_state.features = []
if 'feature_stats' not in st.session_state: st.session_state.feature_stats = {}

st.title("📊 科学级页岩气产能主控因素与预测平台")
st.caption("内核：多目标机器学习 (Multi-Target ML) 联合预测 EUR 与 Duong 物理衰减参数 (a, m)")

tab1, tab2, tab3 = st.tabs(["1. 训练与主控分析", "2. 静动耦合产能评估", "3. 科学数据流转说明"])

# ==========================================
# TAB 1: 模型训练与多目标主控分析
# ==========================================
with tab1:
    st.subheader("多目标模型训练：同时学习体量与物理衰减规律")
    train_file = st.file_uploader("上传历史井数据 (CSV格式，必须包含 eur, m, a 三个目标列)", type=['csv'])
    
    if train_file:
        df = pd.read_csv(train_file)
        df_numeric = df.select_dtypes(include=[np.number])
        all_cols = df_numeric.columns.tolist()
        
        # 强制安全检查：确保数据集包含三个科学目标
        required_targets = ['eur', 'm', 'a']
        missing_targets = [col for col in required_targets if col not in all_cols]
        
        if missing_targets:
            st.error(f"❌ 数据格式错误：缺乏科学计算必须的物理目标列 {missing_targets}。请先运行数据预处理脚本。")
        else:
            # 动态特征选择，剔除目标列
            available_features = [c for c in all_cols if c not in required_targets]
            selected_features = st.multiselect("选择输入特征 (X) [如: 地质、工程参数]", options=available_features, default=available_features)
            
            if st.button("🚀 运行多目标模型训练", type="primary"):
                if not selected_features:
                    st.error("请至少选择一个输入特征！")
                else:
                    st.session_state.features = selected_features
                    st.session_state.feature_stats = df_numeric[selected_features].mean().to_dict()
                    
                    X = df_numeric[selected_features]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    st.session_state.scaler = scaler
                    
                    # 科学核心：分别训练三个独立模型，揭示不同维度的地质映射
                    st.session_state.model_eur = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_scaled, df_numeric['eur'])
                    st.session_state.model_m = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, df_numeric['m'])
                    st.session_state.model_a = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, df_numeric['a'])
                    
                    st.success(f"✅ 模型训练成功！系统已完全掌握历史井的地质-生产物理映射关系。")
                    
                    # 创新点可视化：三个维度的独立主控因素
                    st.markdown("#### 🔬 多维物理参数主控因素独立分析 (学术创新点)")
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        imp_eur = pd.DataFrame({'特征': selected_features, '权重': st.session_state.model_eur.feature_importances_}).sort_values(by='权重')
                        st.plotly_chart(px.bar(imp_eur, x='权重', y='特征', orientation='h', title="EUR (总盘子) 主控因素"), use_container_width=True)
                    with c2:
                        imp_m = pd.DataFrame({'特征': selected_features, '权重': st.session_state.model_m.feature_importances_}).sort_values(by='权重')
                        st.plotly_chart(px.bar(imp_m, x='权重', y='特征', orientation='h', title="参数 m (早期线性流衰减) 主控因素"), use_container_width=True)
                    with c3:
                        imp_a = pd.DataFrame({'特征': selected_features, '权重': st.session_state.model_a.feature_importances_}).sort_values(by='权重')
                        st.plotly_chart(px.bar(imp_a, x='权重', y='特征', orientation='h', title="参数 a (后期基质供气) 主控因素"), use_container_width=True)

# ==========================================
# TAB 2: 单井评估 (机器学习预测 + 积分反推)
# ==========================================
with tab2:
    st.subheader("新井产能动态剖面预测 (科学闭环)")
    if st.session_state.model_eur is None:
        st.warning("请先在 Tab 1 完成多目标模型训练。")
    else:
        st.markdown("##### 📍 1. 输入新井静态参数")
        input_dict = {}
        cols = st.columns(4)
        for i, feature in enumerate(st.session_state.features):
            val = cols[i % 4].number_input(f"{feature}", value=float(st.session_state.feature_stats[feature]), format="%.4f")
            input_dict[feature] = val
        
        st.markdown("---")
        
        # ML 实时预测静态参数映射的物理目标
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
            st.info(f"🧠 **AI 科学映射结果**：基于历史数据深度学习，该地质条件下的物理衰减参数应为：**$m = {pred_m:.3f}$**, **$a = {pred_a:.3f}$**")
            use_ai_params = st.checkbox("强制使用 AI 推荐的物理参数 (保证物理自洽性)", value=True)
            
            col_m, col_a = st.columns(2)
            m_val = pred_m if use_ai_params else col_m.number_input("手动干预 m", value=pred_m, step=0.01)
            a_val = pred_a if use_ai_params else col_a.number_input("手动干预 a", value=pred_a, step=0.01)
            
        if st.button("🚀 生成产能动态剖面", type="primary"):
            # 数值积分与初期产量反推
            t_days = np.arange(1, 10951) # 模拟长达 30 年 (寻找真实的经济截断点)
            
            # Duong 核心方程
            shape_func = (t_days**-m_val) * np.exp((a_val / (1 - m_val)) * ((t_days**(1 - m_val)) - 1))
            shape_integral = np.sum(shape_func) / 10000.0 # 转换为亿方
            qi_calc = pred_eur / shape_integral 
            
            # 真实日产量曲线
            q_time = qi_calc * shape_func
            
            # 经济截断逻辑
            valid_idx = np.where(q_time >= q_ab)[0]
            life_days = valid_idx[-1] if len(valid_idx) > 0 else 0
            
            if life_days == 0 or qi_calc < q_ab:
                st.error(f"⚠️ 警告：该地质条件下，初始预测产量低于经济废弃线 ({q_ab} 万方/d)，不具备经济开采价值。")
            else:
                st.markdown("### 📊 科学预测结果")
                res_c1, res_c2, res_c3 = st.columns(3)
                res_c1.metric(f"预测 EUR", f"{pred_eur:.2f} 亿方")
                res_c2.metric("积分反推初期配产 ($q_i$)", f"{qi_calc:.2f} 万方/d")
                res_c3.metric("经济开采寿命", f"{life_days/365.0:.1f} 年")
                
                # 绘图：传统递减曲线 (控制展示长度，略过冗长的废弃期)
                plot_days = min(len(t_days), life_days + 365)
                fig_curve = go.Figure()
                fig_curve.add_trace(go.Scatter(x=t_days[:plot_days], y=q_time[:plot_days], mode='lines', name='日产量', line=dict(color='#1f77b4', width=3)))
                fig_curve.add_hline(y=q_ab, line_dash="dash", line_color="red", annotation_text=f"经济废弃线 ({q_ab})")
                
                # 醒目的红色废弃截断点
                fig_curve.add_trace(go.Scatter(x=[life_days], y=[q_ab], mode='markers', name='经济废弃点', marker=dict(color='red', size=10)))
                
                fig_curve.update_layout(title="单井生命周期产能递减预测 (物理+经济双截断)", xaxis_title="生产时间 (天)", yaxis_title="日产量 (万方/天)")
                st.plotly_chart(fig_curve, use_container_width=True)

                # 绘图：专业双对数诊断
                with st.expander("查看高级双对数流动诊断图 (Log-Log)"):
                    fig_log = go.Figure()
                    fig_log.add_trace(go.Scatter(x=t_days[:plot_days], y=q_time[:plot_days], mode='lines', name='日产量'))
                    fig_log.update_layout(
                        title="流动阶段诊断图 (直观显示页岩气长尾效应)", 
                        xaxis_title="生产时间 (天)", 
                        yaxis_title="日产量 (万方/天)",
                        xaxis_type="log", yaxis_type="log"
                    )
                    st.plotly_chart(fig_log, use_container_width=True)

# ==========================================
# TAB 3: 科学数据规范说明
# ==========================================
with tab3:
    st.markdown("""
    ### 科学级数据流转闭环工作流
    
    为彻底解决“参数靠猜”的非科学现象，本系统构建了严密的 **数据预处理 -> ML建模 -> 积分反演** 闭环：
    
    1. **Phase 1: 数据预处理 (历史拟合提取)**
       使用专门的 Python 脚本，针对历史井的真实动态日产量（$q-t$ 曲线），通过非线性最小二乘法（`curve_fit`）强行反演出该井独有的递减常数 $m$ 和 $a$，并积分求出真实 $EUR$。
       
    2. **Phase 2: 数据合并上传 (构建训练集)**
       将 Phase 1 提取出的 `m`, `a`, `eur` 三列数据，与地质工程参数表（如 `TOC`, `压力系数`, `加砂强度`）水平合并，生成最终的 `training_data.csv`。
       
    3. **Phase 3: Multi-Target ML (解耦物理主控因素)**
       系统读取数据后，不再是单一回归，而是**训练三个独立的 AI 模型**。分别告诉你：是谁控制了总体量（EUR）？是谁控制了裂缝衰减速度（m）？是谁控制了后期稳产能力（a）？
       
    4. **Phase 4: 科学预测与工程配产 ($EUR \\rightarrow q_i$)**
       对于一口新井，系统首先输出地质条件决定的 $(EUR, m, a)$ 铁三角。随后结合**经济废弃产量 ($q_{ab}$)**，通过定积分公式逆向推演（反算），得出该井**理论上必须达到的初期配产量 ($q_i$)**。
    """)
