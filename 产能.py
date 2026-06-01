import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 物理模型定义 (Duong Model)
# ==========================================
def duong_model(t, qi, a, m):
    t = np.where(t == 0, 1e-5, t)
    rate = qi * (t**-m) * np.exp((a / (1 - m)) * ((t**(1 - m)) - 1))
    return rate

def fit_duong_with_quality(time_days, rate_data, bounds=None, min_points=5):
    """
    拟合 Duong 模型并返回参数及拟合质量指标 R²
    """
    if len(time_days) < min_points or len(rate_data) < min_points:
        return None, None, None, None, 0.0, "数据点不足"
    
    # 默认边界
    if bounds is None:
        bounds = ([0.1, 0.01, 1.01], [np.inf, 5.0, 3.0])
    
    try:
        initial_guess = [rate_data[0], 1.05, 1.15]
        popt, _ = curve_fit(duong_model, time_days, rate_data,
                            p0=initial_guess, bounds=bounds, maxfev=10000)
        fitted_qi, fitted_a, fitted_m = popt
        
        # 计算 R²
        pred_rates = duong_model(time_days, fitted_qi, fitted_a, fitted_m)
        ss_res = np.sum((rate_data - pred_rates) ** 2)
        ss_tot = np.sum((rate_data - np.mean(rate_data)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return fitted_qi, fitted_a, fitted_m, r2, "成功"
    except Exception as e:
        return None, None, None, None, 0.0, f"拟合失败: {str(e)[:50]}"

def clean_dynamic_data(df, days_col='Days', rate_col='Rate', max_rate=1000):
    """
    清洗动态数据：确保 Days 连续，插值缺失，剔除异常高值
    """
    df = df.copy()
    df = df.dropna(subset=[days_col, rate_col])
    df[rate_col] = df[rate_col].clip(upper=max_rate)  # 异常值截断
    
    # 确保 Days 为整数且从 1 开始
    df[days_col] = df[days_col].astype(float)
    min_day = int(df[days_col].min())
    if min_day > 1:
        # 插入第1天的数据（假设初始产量为第一条记录的Rate）
        first_row = df.iloc[0].copy()
        first_row[days_col] = 1
        df = pd.concat([pd.DataFrame([first_row]), df], ignore_index=True)
    
    # 重采样到连续整数天数
    full_days = np.arange(1, int(df[days_col].max()) + 1)
    interp_func = interp1d(df[days_col], df[rate_col], kind='linear', fill_value='extrapolate')
    full_rates = interp_func(full_days)
    return full_days, full_rates

def compute_eur_from_params(qi, a, m, t_max=10950, q_ab=0.5):
    t_sim = np.arange(1, t_max + 1)
    q_sim = duong_model(t_sim, qi, a, m)
    valid_q = q_sim[q_sim >= q_ab]
    eur = np.sum(valid_q) / 10000.0  # 万方 → 亿方
    return eur

def process_single_well_enhanced(file, static_well_id, q_ab, bounds, min_points=5, max_rate=1000):
    """
    处理单个井文件，返回特征参数及质量标记
    """
    dyn_df = pd.read_csv(file)
    if 'Days' not in dyn_df.columns or 'Rate' not in dyn_df.columns:
        return None, "缺少Days或Rate列"
    
    try:
        days, rates = clean_dynamic_data(dyn_df, max_rate=max_rate)
        qi_fit, a_fit, m_fit, r2, status = fit_duong_with_quality(days, rates, bounds=bounds, min_points=min_points)
        if qi_fit is None:
            return None, status
        # 计算基于拟合参数的 EUR (用于质量评估)
        eur_fit = compute_eur_from_params(qi_fit, a_fit, m_fit, q_ab=q_ab)
        return {
            'well_id': static_well_id,
            'a': round(a_fit, 4),
            'm': round(m_fit, 4),
            'r2': round(r2, 4),
            'qi_fit': round(qi_fit, 2),
            'eur_fit': round(eur_fit, 4),
            'fit_status': status
        }, "成功"
    except Exception as e:
        return None, f"异常: {str(e)[:50]}"

# ==========================================
# 页面全局配置
# ==========================================
st.set_page_config(page_title="全栈式页岩气产能预测（增强版）", layout="wide")

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model_eur' not in st.session_state:
    st.session_state.model_eur = None
if 'model_m' not in st.session_state:
    st.session_state.model_m = None
if 'model_a' not in st.session_state:
    st.session_state.model_a = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'features' not in st.session_state:
    st.session_state.features = []
if 'feature_stats' not in st.session_state:
    st.session_state.feature_stats = {}
if 'test_metrics' not in st.session_state:
    st.session_state.test_metrics = {}

st.title("📊 全栈式页岩气产能预测（成都理工大学·增强版）")
st.caption("集成数据清洗、拟合质量评估、自动验证与物理约束的端到端预测系统")

tab1, tab2, tab3, tab4 = st.tabs(["1. 数据预处理引擎", "2. 多目标模型训练", "3. 静动耦合产能评估", "4. 数据格式说明"])

# ==========================================
# TAB 1: 预处理（带质量控制）
# ==========================================
with tab1:
    st.subheader("🛠️ Phase 1: 历史井动态特征提取（带拟合质量筛选）")
    col_static, col_dynamic = st.columns(2)
    with col_static:
        static_file = st.file_uploader("1. 静态参数汇总表 (CSV)", type=['csv'])
    with col_dynamic:
        dynamic_files = st.file_uploader("2. 单井日产动态数据 (支持多CSV)", type=['csv'], accept_multiple_files=True)
    
    with st.expander("⚙️ 高级拟合参数"):
        q_ab_fit = st.number_input("经济废弃产量 (万方/天)", value=0.5, step=0.1)
        min_data_points = st.number_input("最少有效数据点数", value=5, min_value=3, step=1)
        max_rate_cap = st.number_input("最大合理日产 (万方/天)", value=1000, step=100)
        r2_threshold = st.slider("拟合质量 R² 阈值", 0.0, 1.0, 0.6, 0.05)
        col_a_low, col_a_up = st.columns(2)
        a_low = col_a_low.number_input("a 下限", value=0.01, format="%.3f")
        a_up = col_a_up.number_input("a 上限", value=5.0, format="%.2f")
        col_m_low, col_m_up = st.columns(2)
        m_low = col_m_low.number_input("m 下限", value=1.01, format="%.2f")
        m_up = col_m_up.number_input("m 上限", value=3.0, format="%.2f")
        fit_bounds = ([0.1, a_low, m_low], [np.inf, a_up, m_up])
    
    if st.button("🚀 开始批量拟合与清洗", type="primary"):
        if not static_file or not dynamic_files:
            st.error("请同时上传静态汇总表和动态日产文件！")
        else:
            static_df = pd.read_csv(static_file)
            results = []
            failed_logs = []
            progress_bar = st.progress(0)
            for idx, file in enumerate(dynamic_files):
                well_id = file.name.replace(".csv", "")
                # 检查静态表中是否有该井
                if well_id not in static_df['well_id'].values:
                    failed_logs.append(f"{well_id}: 静态表中无此井号")
                    continue
                res, msg = process_single_well_enhanced(
                    file, well_id, q_ab_fit, fit_bounds,
                    min_points=min_data_points, max_rate=max_rate_cap
                )
                if res is not None:
                    results.append(res)
                else:
                    failed_logs.append(f"{well_id}: {msg}")
                progress_bar.progress((idx + 1) / len(dynamic_files))
            
            if not results:
                st.error("没有一口井拟合成功！请检查数据格式或放宽拟合参数阈值。")
            else:
                dyn_res_df = pd.DataFrame(results)
                # 合并静态数据，筛选高质量井
                final_df = pd.merge(static_df, dyn_res_df, on='well_id', how='inner')
                final_df = final_df[final_df['r2'] >= r2_threshold].copy()
                final_df = final_df.dropna(subset=['a', 'm', 'eur_fit'])
                
                st.session_state.processed_data = final_df
                st.success(f"✅ 成功拟合 {len(final_df)} 口井（R² ≥ {r2_threshold}），失败/剔除 {len(failed_logs)} 口。")
                with st.expander("查看拟合质量详情"):
                    st.dataframe(final_df[['well_id', 'a', 'm', 'r2', 'eur_fit']].head(10))
                if failed_logs:
                    with st.expander("⚠️ 拟合失败/跳过井日志"):
                        st.write("\n".join(failed_logs[:20]))
                
                csv_export = final_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 下载清洗后的训练集", csv_export, "training_data_clean.csv", "text/csv")

# ==========================================
# TAB 2: 模型训练（带验证）
# ==========================================
with tab2:
    st.subheader("🧠 Phase 2: 多目标 AI 核心模型（自动验证）")
    use_cached = st.checkbox("使用Tab 1预处理数据", value=True)
    train_df = None
    if use_cached and st.session_state.processed_data is not None:
        train_df = st.session_state.processed_data
        st.info(f"已加载 {len(train_df)} 口井的预处理数据。")
    else:
        uploaded = st.file_uploader("上传已处理的宽表 (需含 well_id, eur_fit, m, a 及其他特征)", type=['csv'])
        if uploaded:
            train_df = pd.read_csv(uploaded)
    
    if train_df is not None:
        df_num = train_df.select_dtypes(include=[np.number])
        all_cols = df_num.columns.tolist()
        target_cols = ['eur_fit', 'm', 'a']  # 注意列名匹配
        missing_targets = [c for c in target_cols if c not in all_cols]
        if missing_targets:
            st.error(f"缺少目标列: {missing_targets}，请确保列名为 eur_fit, m, a")
        else:
            features = [c for c in all_cols if c not in target_cols + ['well_id']]
            selected_features = st.multiselect("选择输入特征 (X)", features, default=features[:min(8, len(features))])
            
            # 允许自动剔除低重要性特征（可选）
            auto_prune = st.checkbox("自动剔除重要性为0的特征（基于初步训练）")
            
            if st.button("🚀 训练并验证模型", type="primary"):
                X = train_df[selected_features].copy()
                # 处理缺失值
                X = X.fillna(X.median())
                y_eur = train_df['eur_fit']
                y_m = train_df['m']
                y_a = train_df['a']
                
                # 划分训练/测试集
                X_train, X_test, y_eur_train, y_eur_test, y_m_train, y_m_test, y_a_train, y_a_test = train_test_split(
                    X, y_eur, y_m, y_a, test_size=0.2, random_state=42
                )
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # 训练
                model_eur = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train_scaled, y_eur_train)
                model_m = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_scaled, y_m_train)
                model_a = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_scaled, y_a_train)
                
                # 评估
                metrics = {}
                for name, model, y_test in [('EUR', model_eur, y_eur_test), ('m', model_m, y_m_test), ('a', model_a, y_a_test)]:
                    pred = model.predict(X_test_scaled)
                    r2 = r2_score(y_test, pred)
                    mae = mean_absolute_error(y_test, pred)
                    metrics[name] = {'R²': r2, 'MAE': mae}
                
                st.session_state.model_eur = model_eur
                st.session_state.model_m = model_m
                st.session_state.model_a = model_a
                st.session_state.scaler = scaler
                st.session_state.features = selected_features
                st.session_state.feature_stats = X.median().to_dict()
                st.session_state.test_metrics = metrics
                
                st.success("✅ 模型训练完成！")
                st.metric("测试集样本数", len(X_test))
                col1, col2, col3 = st.columns(3)
                col1.metric("EUR 预测 R²", f"{metrics['EUR']['R²']:.3f}", help="越接近1越好")
                col2.metric("m 预测 R²", f"{metrics['m']['R²']:.3f}")
                col3.metric("a 预测 R²", f"{metrics['a']['R²']:.3f}")
                
                # 特征重要性
                for target, model in zip(['EUR', 'm', 'a'], [model_eur, model_m, model_a]):
                    imp = pd.DataFrame({'特征': selected_features, '重要性': model.feature_importances_}).sort_values('重要性', ascending=True)
                    fig = px.bar(imp, x='重要性', y='特征', orientation='h', title=f"{target} 特征重要性")
                    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 3: 预测（带物理约束）
# ==========================================
with tab3:
    st.subheader("🎯 Phase 3: 新井产能预测与动态推演")
    if st.session_state.model_eur is None:
        st.warning("请先在 Tab 2 完成模型训练。")
    else:
        st.markdown("##### 📍 1. 输入新井静态参数")
        input_dict = {}
        cols = st.columns(4)
        for i, feat in enumerate(st.session_state.features):
            default_val = st.session_state.feature_stats.get(feat, 0.0)
            val = cols[i % 4].number_input(f"{feat}", value=float(default_val), format="%.4f")
            input_dict[feat] = val
        
        input_array = np.array([[input_dict[f] for f in st.session_state.features]])
        scaled_input = st.session_state.scaler.transform(input_array)
        
        pred_eur = st.session_state.model_eur.predict(scaled_input)[0]
        pred_m = st.session_state.model_m.predict(scaled_input)[0]
        pred_a = st.session_state.model_a.predict(scaled_input)[0]
        
        st.markdown("---")
        st.markdown("##### 📍 2. 边界条件与参数锁定")
        col_phys1, col_phys2 = st.columns([1, 2])
        with col_phys1:
            q_ab = st.number_input("经济废弃产量 q_ab (万方/天)", value=0.5, step=0.1)
            max_qi_factor = st.number_input("最大初始产量倍数 (相对于历史均值)", value=3.0, step=0.5,
                                            help="防止反算 qi 超出合理范围，例如设置为3表示不超过区块平均初始产量的3倍")
        with col_phys2:
            st.info(f"🧠 AI 推荐参数： **m = {pred_m:.3f}**, **a = {pred_a:.3f}**, **EUR = {pred_eur:.2f} 亿方**")
            use_ai = st.checkbox("使用AI推荐参数（强制物理自洽）", value=True)
            if not use_ai:
                col_m, col_a = st.columns(2)
                m_val = col_m.number_input("手动设置 m", value=pred_m, step=0.01)
                a_val = col_a.number_input("手动设置 a", value=pred_a, step=0.01)
            else:
                m_val, a_val = pred_m, pred_a
        
        if st.button("🚀 生成智能产能剖面", type="primary"):
            t_days = np.arange(1, 10951)
            shape_func = (t_days**-m_val) * np.exp((a_val / (1 - m_val)) * ((t_days**(1 - m_val)) - 1))
            shape_integral = np.sum(shape_func) / 10000.0
            qi_calc = pred_eur / shape_integral
            
            # 合理性检查（基于特征统计中是否有历史 qi？我们使用训练集中所有井的 eur_fit/形状积分估算历史平均qi）
            # 简单起见，设置一个绝对上限（例如500万方/天）或倍数
            with st.spinner("计算产量剖面..."):
                # 获取训练集中所有拟合井的初始产量范围（作为参考）
                if st.session_state.processed_data is not None and 'qi_fit' in st.session_state.processed_data.columns:
                    hist_qi = st.session_state.processed_data['qi_fit'].dropna()
                    if len(hist_qi) > 0:
                        max_reasonable_qi = hist_qi.quantile(0.95) * max_qi_factor
                        if qi_calc > max_reasonable_qi:
                            st.warning(f"⚠️ 反算初始产量 ({qi_calc:.2f} 万方/天) 超过历史合理上限 ({max_reasonable_qi:.2f})，将自动截断。")
                            qi_calc = max_reasonable_qi
                            # 重新计算 EUR 以保持一致（可选）
                            pred_eur = compute_eur_from_params(qi_calc, a_val, m_val, q_ab=q_ab)
                
                # 模拟产量曲线
                q_time = qi_calc * shape_func
                valid_idx = np.where(q_time >= q_ab)[0]
                if len(valid_idx) == 0:
                    st.error("⚠️ 初始产量低于废弃线，无法经济开采！请调整参数。")
                else:
                    life_days = valid_idx[-1]
                    res_c1, res_c2, res_c3 = st.columns(3)
                    res_c1.metric("最终预测 EUR", f"{pred_eur:.2f} 亿方")
                    res_c2.metric("推荐初始配产 qi", f"{qi_calc:.2f} 万方/天")
                    res_c3.metric("经济开采寿命", f"{life_days/365:.1f} 年")
                    
                    plot_days = min(len(t_days), life_days + 365)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=t_days[:plot_days], y=q_time[:plot_days], mode='lines', name='日产量'))
                    fig.add_hline(y=q_ab, line_dash="dash", line_color="red", annotation_text="废弃线")
                    fig.add_trace(go.Scatter(x=[life_days], y=[q_ab], mode='markers', marker=dict(color='red', size=10), name='经济寿命终点'))
                    fig.update_layout(title="智能单井生命周期产能预测", xaxis_title="生产天数", yaxis_title="日产量 (万方/天)")
                    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 4: 数据格式说明（更新）
# ==========================================
with tab4:
    st.markdown("""
    ### 📁 数据格式要求 (增强版)
    **静态参数表**:
    - 必须包含列 `well_id` (井号，与动态文件名主体一致)
    - 其他特征列均为数值型 (如 TOC, 压力系数, 埋深等)
    
    **单井动态文件**:
    - 文件名: `井号.csv` (例如 `W001.csv`)
    - 内部必须包含列 `Days` (生产天数，整数) 和 `Rate` (日产气量，万方/天)
    - 程序会自动插值补齐缺失天数和截断异常高值
    
    **拟合质量控制**:
    - 系统会计算 Duong 模型拟合的决定系数 R²，只有高于设定阈值的井才会进入模型训练
    - 可在高级选项中调整拟合参数边界和 R² 阈值
    
    **模型验证**:
    - 自动按 80/20 划分训练/测试集，输出测试集 R² 和 MAE
    - 特征重要性图可辅助筛选主控因素
    """)
