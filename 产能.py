import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# 设置页面布局宽度为宽屏模式
st.set_page_config(page_title="页岩气单井产能预测看板", layout="wide")


# ==========================================
# 1. 数学模型定义区
# ==========================================
def duong_model(t, qi, a, m):
    """标准的 Duong 递减模型"""
    # 避免除以 0 或无穷大
    t = np.where(t == 0, 1e-5, t)
    rate = qi * (t ** -m) * np.exp((a / (1 - m)) * ((t ** (1 - m)) - 1))
    return rate


# ==========================================
# 2. 侧边栏：数据导入与参数配置
# ==========================================
st.sidebar.header("⚙️ 数据与参数设置")

# 文件上传模块
uploaded_file = st.sidebar.file_uploader("1. 上传测试井/邻井生产数据 (CSV)", type=['csv'])

# 经验参数滑块
st.sidebar.subheader("2. 蒙特卡洛模拟参数")
mc_runs = st.sidebar.number_input("模拟次数", min_value=100, max_value=5000, value=1000, step=100)
q_ab = st.sidebar.number_input("废弃产量 (万方/天)", value=1.0, step=0.1)
qi_variance = st.sidebar.slider("初期产能不确定性波动 (%)", 5, 50, 20)

# ==========================================
# 3. 主界面：核心工作流执行
# ==========================================
st.title("📊 页岩气单井产能及 EUR 预测评估系统")

if uploaded_file is not None:
    # 读取数据（假设包含 'Days' 和 'Rate_万方' 两列）
    df = pd.read_csv(uploaded_file)
    t_data = df['Days'].values
    q_data = df['Rate_万方'].values

    st.subheader("步骤一：生产动态数据与 Duong 模型拟合")

    # 使用 SciPy 进行非线性曲线拟合
    try:
        # 提供初始猜测值 [qi, a, m]
        popt, pcov = curve_fit(duong_model, t_data, q_data, p0=[20, 1.1, 1.2], bounds=(0, [100, 5, 5]))
        fitted_qi, fitted_a, fitted_m = popt

        # 布局：分两列展示拟合参数和图表
        col1, col2 = st.columns([1, 3])
        with col1:
            st.success("模型拟合成功！")
            st.metric("初期产量 (qi)", f"{fitted_qi:.2f} 万方/d")
            st.metric("参数 a", f"{fitted_a:.4f}")
            st.metric("参数 m", f"{fitted_m:.4f}")

        with col2:
            # 使用 Plotly 绘制高可交互图表
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_data, y=q_data, mode='markers', name='实际生产数据'))
            # 生成 10 年预测时间轴
            t_pred = np.linspace(1, 3650, 500)
            q_pred = duong_model(t_pred, fitted_qi, fitted_a, fitted_m)
            fig.add_trace(go.Scatter(x=t_pred, y=q_pred, mode='lines', name='Duong 模型预测', line=dict(color='red')))
            fig.update_layout(title="单井产量递减历史拟合与预测", xaxis_title="生产时间 (天)",
                              yaxis_title="日产量 (万方/天)")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"拟合失败: {e}")

    # ------------------------------------------
    # 步骤二：蒙特卡洛 EUR 评估
    # ------------------------------------------
    st.subheader("步骤二：EUR 不确定性分析 (P10 / P50 / P90)")

    if st.button("🚀 运行蒙特卡洛模拟"):
        with st.spinner("正在执行随机采样与积分计算..."):
            # 以拟合出的 qi 为均值，生成正态分布的随机 qi 样本
            std_dev = fitted_qi * (qi_variance / 100.0)
            qi_samples = np.random.normal(loc=fitted_qi, scale=std_dev, size=mc_runs)

            eur_results = []
            for qi_sim in qi_samples:
                if qi_sim <= 0: continue
                # 简单积分逻辑：从第 1 天积分到 10 年，或者当产量跌破废弃产量时截断
                t_sim = np.arange(1, 3651)
                q_sim = duong_model(t_sim, qi_sim, fitted_a, fitted_m)
                valid_q = q_sim[q_sim >= q_ab]
                eur = np.sum(valid_q) / 10000.0  # 假设累加转换为亿方
                eur_results.append(eur)

            eur_array = np.array(eur_results)
            p90 = np.percentile(eur_array, 10)
            p50 = np.percentile(eur_array, 50)
            p10 = np.percentile(eur_array, 90)

            # 展示最终结果
            c1, c2, c3 = st.columns(3)
            c1.metric("P90 EUR (保守)", f"{p90:.2f} 亿方")
            c2.metric("P50 EUR (中值)", f"{p50:.2f} 亿方")
            c3.metric("P10 EUR (乐观)", f"{p10:.2f} 亿方")

else:
    st.info("请在左侧栏上传数据文件以启动预测流程。")