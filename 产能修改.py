import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit

# ==========================================
# 物理模型定义 (Duong Model)
# ==========================================
def duong_model(t, qi, a, m):
    t = np.where(t == 0, 1e-5, t) 
    rate = qi * (t**-m) * np.exp((a / (1 - m)) * ((t**(1 - m)) - 1))
    return rate

def process_single_well(time_days, rate_data, q_ab=0.5):
    try:
        initial_guess = [rate_data[0] if len(rate_data)>0 else 20, 1.05, 1.15]
        lower_bounds = [0.1, 0.01, 1.01]
        upper_bounds = [np.inf, 5.0, 3.0]
        
        popt, _ = curve_fit(
            duong_model, time_days, rate_data, 
            p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=10000
        )
        fitted_qi, fitted_a, fitted_m = popt
        
        # 计算 EUR
        t_sim = np.arange(1, 10951) 
        q_sim = duong_model(t_sim, fitted_qi, fitted_a, fitted_m)
        valid_q = q_sim[q_sim >= q_ab]
        eur = np.sum(valid_q) / 10000.0  
        return fitted_qi, fitted_a, fitted_m, eur
    except Exception:
        return None, None, None, None

# ==========================================
# 页面全局配置与 Session 初始化
# ==========================================
st.set_page_config(page_title="全栈式页岩气产能预测", layout="wide")

if 'processed_data' not in st.session_state: st.session_state.processed_data = None
if 'model_eur' not in st.session_state: st.session_state.model_eur = None
if 'model_m' not in st.session_state: st.session_state.model_m = None
if 'model_a' not in st.session_state: st.session_state.model_a = None
if 'scaler' not in st.session_state: st.session_state.scaler = None
if 'features' not in st.session_state: st.session_state.features = []
if 'feature_stats' not in st.session_state: st.session_state.feature_stats = {}

st.title("📊 全栈式页岩气产能预测（成都理工大学）")
st.caption("端到端闭环：历史生产数据物理提取 ➔ 多目标 AI 建模 ➔ 物理经济双截断预测")

tab1, tab2, tab3, tab4 = st.tabs(["1. 数据预处理引擎", "2. 多目标模型训练", "3. 静动耦合产能评估", "4. 数据格式说明"])

# ==========================================
# TAB 1: 数据预处理 (历史拟合与特征提取)
# ==========================================
with tab1:
    st.subheader("🛠️ Phase 1: 历史井动态特征提取")
    st.markdown("自动读取原始日产曲线，通过最小二乘法反演 `a, m` 参数并计算真实的经济 `EUR`。")
    
    col_static, col_dynamic = st.columns(2)
    with col_static:
        static_file = st.file_uploader("1. 上传静态参数汇总表 (1个CSV)", type=['csv'], help="包含 well_id 以及 TOC, 压力系数等地质工程参数")
    with col_dynamic:
        dynamic_files = st.file_uploader("2. 批量上传单井日产动态数据 (支持拖拽多个CSV)", type=['csv'], accept_multiple_files=True, help="文件名应为井号(如 well_01.csv)，内含 Days 和 Rate 两列")
        
    q_ab_fit = st.number_input("拟合时采用的经济废弃产量界限 (万方/天)", value=0.5, step=0.1)

    if st.button("🚀 开始批量拟合与清洗数据", type="primary"):
        if not static_file or not dynamic_files:
            st.error("请同时上传静态汇总表和动态日产文件！")
        else:
            with st.spinner(f"正在拟合 {len(dynamic_files)} 口井的物理参数，请稍候..."):
                static_df = pd.read_csv(static_file)
                results = []
                
                # 遍历所有上传的单井动态文件
                progress_bar = st.progress(0)
                for idx, file in enumerate(dynamic_files):
                    well_id = file.name.replace(".csv", "")
                    dyn_df = pd.read_csv(file)
                    
                    if 'Days' in dyn_df.columns and 'Rate' in dyn_df.columns:
                        qi_f, a_f, m_f, eur_f = process_single_well(dyn_df['Days'].values, dyn_df['Rate'].values, q_ab_fit)
                        results.append({
                            'well_id': well_id, 'a': round(a_f, 4) if a_f else None,
                            'm': round(m_f, 4) if m_f else None, 'eur': round(eur_f, 4) if eur_f else None
                        })
                    progress_bar.progress((idx + 1) / len(dynamic_files))
                
                # 合并数据
                dyn_res_df = pd.DataFrame(results)
                final_df = pd.merge(static_df, dyn_res_df, on='well_id', how='inner')
                final_df = final_df.dropna() # 剔除拟合失败的井
                
                # 存入全局缓存，供 Tab 2 使用
                st.session_state.processed_data = final_df
                
            st.success(f"✅ 提取成功！共成功清洗并拟合 {len(final_df)} 口井的数据。已自动流转至模型训练模块。")
            st.dataframe(final_df.head(10))
            
            # 提供下载功能
            csv_export = final_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 下载带有物理参数的完整训练集", data=csv_export, file_name='training_data_ready.csv', mime='text/csv')

# ==========================================
# TAB 2: 模型训练与多目标主控分析
# ==========================================
with tab2:
    st.subheader("🧠 Phase 2: 多目标 AI 核心模型训练")
    
    # 支持从 Tab 1 直接继承数据，或者手动重新上传
    use_cached_data = st.checkbox("使用在 Tab 1 中刚刚预处理完成的数据", value=True)
    train_df = None
    
    if use_cached_data and st.session_state.processed_data is not None:
        train_df = st.session_state.processed_data
        st.info("已自动加载预处理缓存数据。")
    else:
        uploaded_train = st.file_uploader("或者上传已处理好的宽表 (需包含 eur, m, a)", type=['csv'])
        if uploaded_train:
            train_df = pd.read_csv(uploaded_train)

    if train_df is not None:
        df_numeric = train_df.select_dtypes(include=[np.number])
        all_cols = df_numeric.columns.tolist()
        required_targets = ['eur', 'm', 'a']
        
        if not all([t in all_cols for t in required_targets]):
            st.error("数据缺乏 eur, m, a 目标列！")
        else:
            available_features = [c for c in all_cols if c not in required_targets]
            selected_features = st.multiselect("选择输入特征 (X)", options=available_features, default=available_features)
            
            if st.button("🚀 运行多目标机器学习", type="primary"):
                st.session_state.features = selected_features
                st.session_state.feature_stats = df_numeric[selected_features].mean().to_dict()
                
                X = df_numeric[selected_features]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                st.session_state.scaler = scaler
                
                # 训练三个模型
                st.session_state.model_eur = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_scaled, df_numeric['eur'])
                st.session_state.model_m = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, df_numeric['m'])
                st.session_state.model_a = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, df_numeric['a'])
                
                st.success("✅ 模型训练成功！特征地质映射关系已建立。")
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    imp_eur = pd.DataFrame({'特征': selected_features, '权重': st.session_state.model_eur.feature_importances_}).sort_values(by='权重')
                    st.plotly_chart(px.bar(imp_eur, x='权重', y='特征', orientation='h', title="EUR 主控因素"), use_container_width=True)
                with c2:
                    imp_m = pd.DataFrame({'特征': selected_features, '权重': st.session_state.model_m.feature_importances_}).sort_values(by='权重')
                    st.plotly_chart(px.bar(imp_m, x='权重', y='特征', orientation='h', title="m (早期衰减) 主控因素"), use_container_width=True)
                with c3:
                    imp_a = pd.DataFrame({'特征': selected_features, '权重': st.session_state.model_a.feature_importances_}).sort_values(by='权重')
                    st.plotly_chart(px.bar(imp_a, x='权重', y='特征', orientation='h', title="a (晚期供气) 主控因素"), use_container_width=True)

# ==========================================
# TAB 3: 单井评估 (机器学习预测 + 积分反推)
# ==========================================
with tab3:
    st.subheader("🎯 Phase 3: 新井产能预测与动态推演")
    if st.session_state.model_eur is None:
        st.warning("请先在 Tab 2 完成模型训练。")
    else:
        st.markdown("##### 📍 1. 输入新井静态参数")
        input_dict = {}
        cols = st.columns(4)
        for i, feature in enumerate(st.session_state.features):
            val = cols[i % 4].number_input(f"{feature}", value=float(st.session_state.feature_stats[feature]), format="%.4f")
            input_dict[feature] = val
            
        input_data = np.array([[input_dict[f] for f in st.session_state.features]])
        scaled_input = st.session_state.scaler.transform(input_data)
        
        pred_eur = st.session_state.model_eur.predict(scaled_input)[0]
        pred_m = st.session_state.model_m.predict(scaled_input)[0]
        pred_a = st.session_state.model_a.predict(scaled_input)[0]
        
        st.markdown("---")
        st.markdown("##### 📍 2. 边界条件与参数锁定")
        c_phys1, c_phys2 = st.columns([1, 2])
        with c_phys1:
            q_ab = st.number_input("预测截断：经济废弃产量 $q_{ab}$", value=0.50, step=0.10)
        with c_phys2:
            st.info(f"🧠 **AI 物理推导**：该地质条件下，推荐衰减参数为 **$m = {pred_m:.3f}$**, **$a = {pred_a:.3f}$**")
            use_ai_params = st.checkbox("强制使用 AI 推荐值保障物理自洽", value=True)
            col_m, col_a = st.columns(2)
            m_val = pred_m if use_ai_params else col_m.number_input("微调 m", value=pred_m, step=0.01)
            a_val = pred_a if use_ai_params else col_a.number_input("微调 a", value=pred_a, step=0.01)
            
        if st.button("🚀 生成智能产能剖面", type="primary"):
            t_days = np.arange(1, 10951) 
            shape_func = (t_days**-m_val) * np.exp((a_val / (1 - m_val)) * ((t_days**(1 - m_val)) - 1))
            shape_integral = np.sum(shape_func) / 10000.0 
            qi_calc = pred_eur / shape_integral 
            
            q_time = qi_calc * shape_func
            valid_idx = np.where(q_time >= q_ab)[0]
            life_days = valid_idx[-1] if len(valid_idx) > 0 else 0
            
            if life_days == 0 or qi_calc < q_ab:
                st.error(f"⚠️ 初始产量低于废弃线 ({q_ab})，井况堪忧，建议修正压裂设计。")
            else:
                res_c1, res_c2, res_c3 = st.columns(3)
                res_c1.metric(f"AI 预测 EUR", f"{pred_eur:.2f} 亿方")
                res_c2.metric("反演算推荐配产 $q_i$", f"{qi_calc:.2f} 万方/d")
                res_c3.metric("经济开采寿命", f"{life_days/365.0:.1f} 年")
                
                plot_days = min(len(t_days), life_days + 365)
                fig_curve = go.Figure()
                fig_curve.add_trace(go.Scatter(x=t_days[:plot_days], y=q_time[:plot_days], mode='lines', name='日产量'))
                fig_curve.add_hline(y=q_ab, line_dash="dash", line_color="red", annotation_text=f"废弃线")
                fig_curve.add_trace(go.Scatter(x=[life_days], y=[q_ab], mode='markers', marker=dict(color='red', size=10)))
                fig_curve.update_layout(title="智能单井生命周期产能预测", xaxis_title="生产时间 (天)", yaxis_title="日产量 (万方/天)")
                st.plotly_chart(fig_curve, use_container_width=True)

# ==========================================
# TAB 4: 数据格式说明
# ==========================================
with tab4:
    st.markdown("""
    ### 📁 预处理所需的数据格式标准
    为了让 **Tab 1** 的代码能正确读取您的文件，请务必保证您的表格符合以下格式：

    **1. 静态参数表 (1个文件)：**
    * 必须有一列名为 `well_id`（井号）。
    * 其他列任意（比如 `toc`, `pressure_coeff` 等，全中文列名也可以）。
    
    **2. 单井动态表 (多口井对应多个CSV文件)：**
    * 文件名：必须以您的井号命名，例如 `well_01.csv`。系统会通过文件名与静态表里的 `well_id` 进行关联匹配。
    * 表格内容：必须包含两列，一列叫 `Days` (生产天数，如 1,2,3...)，另一列叫 `Rate` (当天的产气量，万方/天)。
    """)
