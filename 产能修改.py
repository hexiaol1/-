import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay

# 设置页面配置
st.set_page_config(page_title="通用页岩气产能智能预测系统", layout="wide")

# 初始化 Session State
if 'model' not in st.session_state: st.session_state.model = None
if 'scaler' not in st.session_state: st.session_state.scaler = None
if 'features' not in st.session_state: st.session_state.features = []
if 'target' not in st.session_state: st.session_state.target = ""
if 'feature_stats' not in st.session_state: st.session_state.feature_stats = {}

st.title("📊 通用型页岩气产能主控因素与预测平台")
st.caption("内核：机器学习预测 EUR + 物理经验映射锁定衰减方程")

tab1, tab2, tab3 = st.tabs(["1. 训练与主控分析", "2. 静动耦合产能评估", "3. 算法释义"])

# --- TAB 1: 训练与主控分析 ---
with tab1:
    st.subheader("模型训练：动态特征引擎")
    train_file = st.file_uploader("上传历史井数据 (CSV格式，包含地质/工程及产能数据)", type=['csv'])
    
    if train_file:
        df = pd.read_csv(train_file)
        df_numeric = df.select_dtypes(include=[np.number])
        all_cols = df_numeric.columns.tolist()
        
        if len(all_cols) < 2:
            st.error("数据集中数值列不足，无法训练。请检查数据格式。")
        else:
            col_sel1, col_sel2 = st.columns([1, 3])
            with col_sel1:
                default_target_idx = all_cols.index('eur') if 'eur' in all_cols else len(all_cols) - 1
                target_col = st.selectbox("选择预测目标 (Y)", options=all_cols, index=default_target_idx)
            
            with col_sel2:
                available_features = [c for c in all_cols if c != target_col]
                selected_features = st.multiselect("选择输入特征 (X)", options=available_features, default=available_features)
            
            if st.button("🚀 运行训练 & 科研分析"):
                if not selected_features:
                    st.error("请至少选择一个输入特征！")
                else:
                    st.session_state.features = selected_features
                    st.session_state.target = target_col
                    st.session_state.feature_stats = df_numeric[selected_features].mean().to_dict()
                    
                    X = df_numeric[selected_features]
                    y = df_numeric[target_col]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    model = RandomForestRegressor(n_estimators=200, random_state=42)
                    model.fit(X_scaled, y)
                    
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.success(f"模型训练成功！共使用 {len(selected_features)} 个特征。")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        importances = pd.DataFrame({'特征参数': selected_features, '权重': model.feature_importances_})
                        importances = importances.sort_values(by='权重', ascending=True)
                        fig1 = px.bar(importances, x='权重', y='特征参数', orientation='h', title="主控因素权重排序")
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with c2:
                        corr_cols = selected_features + [target_col]
                        fig2 = px.imshow(df_numeric[corr_cols].corr(), text_auto=".2f", title="参数耦合关系热力图")
                        st.plotly_chart(fig2, use_container_width=True)

                    top_features = importances.sort_values(by='权重', ascending=False)['特征参数'].tolist()[:2]
                    top_idx = [selected_features.index(f) for f in top_features]
                    
                    st.subheader(f"关键因素非线性响应 (PDP) - {top_features[0]} & {top_features[1]}")
                    fig_pdp, ax = plt.subplots(figsize=(10, 4))
                    PartialDependenceDisplay.from_estimator(model, X_scaled, features=top_idx, 
                                                           feature_names=selected_features, ax=ax)
                    st.pyplot(fig_pdp)

# --- TAB 2: 单井评估 (静动耦合预测) ---
with tab2:
    st.subheader("新井产能动态剖面预测 (ML + 物理基准耦合)")
    if st.session_state.model is None or not st.session_state.features:
        st.warning("请先在 Tab 1 完成动态特征模型训练。")
    else:
        st.markdown("##### 📍 1. 输入新井静态参数")
        input_dict = {}
        cols = st.columns(4)
        for i, feature in enumerate(st.session_state.features):
            default_val = float(st.session_state.feature_stats[feature])
            with cols[i % 4]:
                val = st.number_input(f"{feature}", value=default_val, format="%.4f")
                input_dict[feature] = val
        
        st.markdown("---")
        st.markdown("##### 📍 2. 边界条件与物理约束")
        
        c_phys1, c_phys2 = st.columns([1, 2])
        with c_phys1:
            st.markdown("**经济边界设定**")
            # 自由设定废弃产量，但给出合规的默认值和提示
            q_ab = st.number_input(
                "经济废弃产量 $q_{ab}$ (万方/天)", 
                value=0.50, step=0.10, 
                help="四川盆地常压区通常因水处理成本较高，经济极限设定在 0.5~0.8 之间。"
            )
        
        with c_phys2:
            st.markdown("**Duong 模型参数 $a, m$ (物理经验映射)**")
            
            # 核心逻辑：自动寻找压力系数和加砂强度，如果找不到则用基础默认值
            pc_val = input_dict.get('pressure_coeff', 1.0)
            si_val = input_dict.get('sand_intensity', 2.5)
            
            # 经验法则计算基准值 (博士研究可以自行修改这里的线性系数)
            # 逻辑: 改造强度越大，初期递减越快(m越大)；压力系数越高，整体能量越足(a越大)
            m_base = round(max(1.01, min(1.35, 1.05 + 0.04 * si_val)), 3)
            a_base = round(max(0.5, min(2.0, 0.70 + 0.35 * pc_val)), 3)
            
            use_auto = st.checkbox("启用静态参数自动映射 (推荐)", value=True, help="系统将根据输入的压力系数和工程强度，自动锁定物理上自洽的 a 和 m 值。")
            
            col_m, col_a = st.columns(2)
            if use_auto:
                st.info(f"💡 系统已基于静态输入锁定物理参数: $m \approx {m_base}$, $a \approx {a_base}$")
                m_val = m_base
                a_val = a_base
            else:
                m_val = col_m.number_input("手动指定递减指数 m (>1)", value=m_base, step=0.01)
                a_val = col_a.number_input("手动指定常数 a", value=a_base, step=0.01)
            
        if st.button("🚀 生成产能动态剖面", type="primary"):
            input_data = np.array([[input_dict[f] for f in st.session_state.features]])
            scaled_input = st.session_state.scaler.transform(input_data)
            
            # ML 预测 EUR
            pred_target = st.session_state.model.predict(scaled_input)[0]
            
            # 物理规律积分与反算
            t_days = np.arange(1, 7301) # 扩大模拟时间到20年 (7300天)，寻找真正的截断点
            shape_func = (t_days**-m_val) * np.exp((a_val / (1 - m_val)) * ((t_days**(1 - m_val)) - 1))
            shape_integral = np.sum(shape_func) / 10000.0 
            qi_calc = pred_target / shape_integral 
            
            # 截断运算
            q_time = qi_calc * shape_func
            valid_idx = np.where(q_time >= q_ab)[0]
            life_days = valid_idx[-1] if len(valid_idx) > 0 else 0
            life_years = life_days / 365.0
            
            # 如果因为废弃产量定得太高，导致第一天就无法生产
            if life_days == 0:
                st.error(f"⚠️ 警告：预测初期产量低于经济废弃产量 ({q_ab} 万方/天)，该井不具备经济开采价值。")
            else:
                st.markdown("### 📊 综合预测结果")
                res_c1, res_c2, res_c3 = st.columns(3)
                res_c1.metric(f"预测 {st.session_state.target} (ML天花板)", f"{pred_target:.2f}")
                res_c2.metric("推算初期日产 ($q_i$)", f"{qi_calc:.2f} 万方/天")
                res_c3.metric("经济开采寿命", f"{life_years:.1f} 年")
                
                # 绘制递减曲线 (仅展示到废弃年限 + 100天)
                plot_days = min(len(t_days), life_days + 300)
                
                fig_curve = go.Figure()
                fig_curve.add_trace(go.Scatter(x=t_days[:plot_days], y=q_time[:plot_days], mode='lines', name='日产量', line=dict(color='#1f77b4', width=3)))
                fig_curve.add_hline(y=q_ab, line_dash="dash", line_color="red", annotation_text=f"废弃线 ({q_ab})")
                
                # 标出废弃点
                fig_curve.add_trace(go.Scatter(x=[life_days], y=[q_ab], mode='markers', name='废弃点', marker=dict(color='red', size=10)))
                
                fig_curve.update_layout(
                    title=f"单井生命周期产能递减预测 (物理截断)", 
                    xaxis_title="生产时间 (天)", 
                    yaxis_title="日产量 (万方/天)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_curve, use_container_width=True)
                
                with st.expander("查看高级双对数诊断图"):
                    fig_log = go.Figure()
                    fig_log.add_trace(go.Scatter(x=t_days[:plot_days], y=q_time[:plot_days], mode='lines', name='日产量'))
                    fig_log.update_layout(
                        title="流动阶段诊断图 (双对数坐标)", 
                        xaxis_title="生产时间 (天)", 
                        yaxis_title="产量",
                        xaxis_type="log", yaxis_type="log"
                    )
                    st.plotly_chart(fig_log, use_container_width=True)

# --- TAB 3: 算法释义 ---
with tab3:
    st.markdown("""
    ### 核心算法架构说明
    本系统摒弃了传统的“纯黑盒”模式，采用 **机器学习静态约束 + 油气藏物理衰减** 的静动耦合机制。
    
    1. **经济废弃产量 ($q_{ab}$)**：不再内置繁琐的（气价/OPEX/水气比）财务核算系统，提供自由输入口。常压气藏建议参考标准设为 0.5~0.8 万方/天。
    2. **Duong 参数防随机机制 ($a, m$)**：系统通过提取输入特征中的地质与工程参数，自动进行经验法则映射。例如：工程改造强度(`sand_intensity`)越大，诱发的初期裂缝线性流越明显，系统会自动推高递减指数 $m$；压力系数(`pressure_coeff`)越高，整体地层势能越强，系统会自动补偿截距常数 $a$。此机制确保了动态曲线的物理自洽性。
    """)
