"""
app.py — SPA Python 交互界面（Streamlit）

对应 SPA Manual Step 3 的完整演示：
  - 加载数据 → 调整参数 → 运行 SPA → 查看前后对比

启动：
  source .venv/bin/activate
  streamlit run app.py
"""

import os
import sys
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mne

mne.set_log_level("WARNING")

sys.path.insert(0, os.path.dirname(__file__))
from spa.core import spa_eeg, compute_pc_amplitudes

SAMPLE_SET = os.path.join(os.path.dirname(__file__), "sample_data", "sample_data.set")

# ─────────────────────────────────────────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPA — EEG 伪迹去除",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 SPA — Segment-by-segment PCA-based Artifact Removal")
st.caption(
    "Ouyang, G., Dien, J., & Lorenz, R. (2021). *Journal of Neural Engineering.*  "
    "Python 实现，对应 Manual Step 3"
)

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="正在加载 EEG 数据...")
def load_data(path: str):
    raw = mne.io.read_raw_eeglab(path, preload=True)
    data = raw.get_data()
    srate = raw.info["sfreq"]
    ch_names = raw.ch_names
    return data, srate, ch_names


def make_eeg_figure(
    data_v: np.ndarray,
    ch_names: list,
    srate: float,
    title: str,
    scale_uv: float,
    t_start: float,
    t_end: float,
    highlight_chs: list | None = None,
) -> go.Figure:
    """绘制多通道 EEG 堆叠波形图（Plotly）"""
    data_uv = data_v * 1e6
    n_ch = data_uv.shape[0]
    i_start = int(t_start * srate)
    i_end = int(t_end * srate)
    times = np.arange(i_start, i_end) / srate

    fig = go.Figure()
    spacing = scale_uv * 2.5

    for i in range(n_ch):
        seg = data_uv[i, i_start:i_end] + i * spacing
        color = "crimson" if (highlight_chs and ch_names[i] in highlight_chs) else "steelblue"
        width = 1.5 if (highlight_chs and ch_names[i] in highlight_chs) else 0.8
        fig.add_trace(
            go.Scatter(
                x=times,
                y=seg,
                mode="lines",
                name=ch_names[i],
                line=dict(color=color, width=width),
                hovertemplate=f"<b>{ch_names[i]}</b><br>时间: %{{x:.3f}} s<br>幅度: %{{customdata:.1f}} μV<extra></extra>",
                customdata=data_uv[i, i_start:i_end],
            )
        )

    # Y 轴刻度显示通道名
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(title="时间 (s)", showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(
            tickvals=[i * spacing for i in range(n_ch)],
            ticktext=ch_names,
            showgrid=False,
        ),
        height=600,
        margin=dict(l=60, r=20, t=40, b=40),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def make_pca_dist_figure(
    amps_before: np.ndarray,
    amps_after: np.ndarray,
    threshold_uv: float,
) -> go.Figure:
    """PC 幅度分布双直方图（对应论文 Fig 1）"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("SPA 前 — 双峰分布（含伪迹）", "SPA 后 — 大方差峰消失"),
        shared_yaxes=True,
    )
    bins = dict(start=0, end=200, size=2)

    fig.add_trace(
        go.Histogram(x=amps_before * 1e6, xbins=bins, name="SPA 前",
                     marker_color="steelblue", opacity=0.75),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(x=amps_after * 1e6, xbins=bins, name="SPA 后",
                     marker_color="darkorange", opacity=0.75),
        row=1, col=2,
    )

    for col in [1, 2]:
        fig.add_vline(
            x=threshold_uv, line_dash="dash", line_color="red", line_width=2,
            annotation_text=f"阈值 {threshold_uv} μV",
            annotation_position="top right",
            row=1, col=col,
        )

    fig.update_xaxes(title_text="PC 幅度 (μV)", range=[0, 200])
    fig.update_yaxes(title_text="频次", col=1)
    fig.update_layout(
        height=400,
        showlegend=False,
        title="PC 幅度分布（对应论文 Fig 1）",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return fig


def make_variance_figure(
    data_before: np.ndarray,
    data_after: np.ndarray,
    ch_names: list,
) -> go.Figure:
    """各通道方差减少柱状图"""
    var_before = np.var(data_before * 1e6, axis=1)
    var_after = np.var(data_after * 1e6, axis=1)
    reduction = (1 - var_after / np.maximum(var_before, 1e-20)) * 100

    colors = ["crimson" if r > 20 else "steelblue" for r in reduction]

    fig = go.Figure(
        go.Bar(
            x=ch_names,
            y=reduction,
            marker_color=colors,
            text=[f"{r:.1f}%" for r in reduction],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="各通道方差减少百分比（红色：减少 >20%，为主要伪迹通道）",
        xaxis_title="通道",
        yaxis_title="方差减少 (%)",
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 侧边栏 — 数据加载 & 参数
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("数据加载")

    use_sample = st.button("📂 使用示例数据 (sample_data.set)", use_container_width=True)
    st.markdown("**或上传自己的数据**")
    st.caption("EEGLAB 格式需同时上传 .set 和 .fdt 两个文件")
    uploaded_set = st.file_uploader("上传 .set 文件", type=["set"])
    uploaded_fdt = st.file_uploader("上传 .fdt 文件", type=["fdt"])

    st.divider()
    st.header("SPA 参数")
    st.caption("对应 `SPA_EEG(EEG, threshold, win_size, smooth_para)`")

    threshold_uv = st.slider(
        "Threshold — 幅度阈值 (μV)", min_value=5, max_value=100, value=30, step=1,
        help="PC 幅度超过此值视为伪迹，默认 30 μV"
    )
    win_size = st.slider(
        "Window Size — 窗口大小 (s)", min_value=0.5, max_value=5.0, value=2.0, step=0.5,
        help="每段 EEG 的时间长度，默认 2 秒"
    )
    smooth_para = st.slider(
        "Smoothing — 平滑参数", min_value=1.0, max_value=5.0, value=2.0, step=0.5,
        help="相邻段平滑强度，越大过渡越陡，默认 2"
    )

    st.divider()
    st.header("显示设置")
    scale_uv = st.slider(
        "纵轴幅度缩放 (μV)", min_value=10, max_value=200, value=50, step=10,
        help="Manual 建议设置为 50"
    )
    t_start = st.number_input("查看起始时间 (s)", min_value=0.0, value=0.0, step=1.0)
    t_end = st.number_input("查看结束时间 (s)", min_value=1.0, value=30.0, step=1.0)

    highlight_ocular = st.checkbox("高亮眼电通道 (Fp1, Fp2)", value=True)
    highlight_chs = ["Fp1", "Fp2"] if highlight_ocular else []

# ─────────────────────────────────────────────────────────────────────────────
# Session state 管理
# ─────────────────────────────────────────────────────────────────────────────
if "data_raw" not in st.session_state:
    st.session_state.data_raw = None
    st.session_state.data_spa = None
    st.session_state.srate = None
    st.session_state.ch_names = None
    st.session_state.amps_before = None
    st.session_state.amps_after = None
    st.session_state.upload_tmpdir = None

# ── 加载数据 ─────────────────────────────────────────────────────────────────
if use_sample:
    if os.path.exists(SAMPLE_SET):
        data, srate, ch_names = load_data(SAMPLE_SET)
        st.session_state.data_raw = data
        st.session_state.srate = srate
        st.session_state.ch_names = ch_names
        st.session_state.data_spa = None
        st.session_state.amps_before = None
        st.session_state.amps_after = None
        st.success(f"已加载示例数据：{data.shape[0]} 通道，{data.shape[1]/srate:.1f} 秒，{srate} Hz")
    else:
        st.error(f"未找到示例数据，请确认路径：{SAMPLE_SET}")

elif uploaded_set is not None:
    if uploaded_fdt is None:
        st.warning("请同时上传对应的 .fdt 文件（与 .set 配套的二进制数据文件）")
    else:
        import tempfile
        # 持久化临时目录到 session state，避免 MNE 读取时目录已被清理
        if "upload_tmpdir" not in st.session_state or st.session_state.upload_tmpdir is None:
            tmpdir_obj = tempfile.mkdtemp()
            st.session_state.upload_tmpdir = tmpdir_obj
        else:
            tmpdir_obj = st.session_state.upload_tmpdir

        set_path = os.path.join(tmpdir_obj, uploaded_set.name)
        fdt_name = uploaded_set.name.replace(".set", ".fdt")
        fdt_path = os.path.join(tmpdir_obj, fdt_name)

        with open(set_path, "wb") as f:
            f.write(uploaded_set.read())
        with open(fdt_path, "wb") as f:
            f.write(uploaded_fdt.read())

        try:
            data, srate, ch_names = load_data(set_path)
            st.session_state.data_raw = data
            st.session_state.srate = srate
            st.session_state.ch_names = ch_names
            st.session_state.data_spa = None
            st.session_state.amps_before = None
            st.session_state.amps_after = None
            st.success(f"已加载：{data.shape[0]} 通道，{data.shape[1]/srate:.1f} 秒")
        except Exception as e:
            st.error(f"加载失败：{e}")

# ─────────────────────────────────────────────────────────────────────────────
# 主区域
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.data_raw is None:
    st.info("👈 请先在左侧点击「使用示例数据」，或同时上传 .set 和 .fdt 文件")
    st.stop()

data_raw = st.session_state.data_raw
srate = st.session_state.srate
ch_names = st.session_state.ch_names
t_end = min(t_end, data_raw.shape[1] / srate)

# ── 数据信息卡片 ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("通道数", data_raw.shape[0])
col2.metric("采样率", f"{srate:.0f} Hz")
col3.metric("时长", f"{data_raw.shape[1]/srate:.1f} s")
col4.metric("幅度范围", f"{data_raw.min()*1e6:.0f} ~ {data_raw.max()*1e6:.0f} μV")

st.divider()

# ── 运行 SPA 按钮 ─────────────────────────────────────────────────────────────
run_col, _ = st.columns([2, 5])
with run_col:
    run_btn = st.button("▶ 运行 SPA", type="primary", use_container_width=True)

if run_btn:
    progress_bar = st.progress(0, text="SPA 处理中...")

    def update_progress(cur, total):
        progress_bar.progress(cur / total, text=f"SPA 处理中... {cur*100//total}%")

    data_spa = spa_eeg(
        data_raw, srate,
        threshold=threshold_uv * 1e-6,
        win_size=win_size,
        smooth_para=smooth_para,
        progress_callback=update_progress,
    )
    progress_bar.progress(1.0, text="完成！")

    st.session_state.data_spa = data_spa
    st.session_state.amps_before = compute_pc_amplitudes(data_raw, srate, win_size)
    st.session_state.amps_after = compute_pc_amplitudes(data_spa, srate, win_size)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Step 3：EEG 波形对比",
    "📈 PC 幅度分布",
    "📋 通道统计",
])

# ── Tab 1：EEG 波形对比 ───────────────────────────────────────────────────────
with tab1:
    st.subheader("SPA 前 — 原始 EEG（对应 Manual Fig 1）")
    fig_raw = make_eeg_figure(
        data_raw, ch_names, srate,
        title="原始 EEG（含眼电伪迹）",
        scale_uv=scale_uv,
        t_start=t_start, t_end=t_end,
        highlight_chs=highlight_chs,
    )
    st.plotly_chart(fig_raw, use_container_width=True)

    if st.session_state.data_spa is not None:
        st.subheader("SPA 后 — 去伪迹 EEG（对应 Manual Fig 2）")
        fig_spa = make_eeg_figure(
            st.session_state.data_spa, ch_names, srate,
            title=f"SPA 后（threshold={threshold_uv}μV, win={win_size}s, smooth={smooth_para}）",
            scale_uv=scale_uv,
            t_start=t_start, t_end=t_end,
            highlight_chs=highlight_chs,
        )
        st.plotly_chart(fig_spa, use_container_width=True)
    else:
        st.info("点击「▶ 运行 SPA」查看处理后的波形")

# ── Tab 2：PC 幅度分布 ─────────────────────────────────────────────────────────
with tab2:
    if st.session_state.amps_before is not None:
        fig_dist = make_pca_dist_figure(
            st.session_state.amps_before,
            st.session_state.amps_after,
            threshold_uv,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**SPA 前**")
            amps_uv = st.session_state.amps_before * 1e6
            above = np.sum(amps_uv > threshold_uv)
            st.metric("超阈值 PC 数", above, help=f"幅度 > {threshold_uv} μV 的 PC 数量")
            st.metric("最大 PC 幅度", f"{amps_uv.max():.1f} μV")
        with col_b:
            st.markdown("**SPA 后**")
            amps_after_uv = st.session_state.amps_after * 1e6
            above_after = np.sum(amps_after_uv > threshold_uv)
            st.metric("超阈值 PC 数", above_after)
            st.metric("最大 PC 幅度", f"{amps_after_uv.max():.1f} μV")
    else:
        st.info("点击「▶ 运行 SPA」后查看 PC 幅度分布")

# ── Tab 3：通道统计 ───────────────────────────────────────────────────────────
with tab3:
    if st.session_state.data_spa is not None:
        fig_var = make_variance_figure(data_raw, st.session_state.data_spa, ch_names)
        st.plotly_chart(fig_var, use_container_width=True)

        var_before = np.var(data_raw * 1e6, axis=1)
        var_after = np.var(st.session_state.data_spa * 1e6, axis=1)
        reduction = (1 - var_after / np.maximum(var_before, 1e-20)) * 100
        total_reduction = (1 - var_after.sum() / var_before.sum()) * 100

        st.metric("总体方差减少", f"{total_reduction:.1f}%")

        import pandas as pd
        df = pd.DataFrame({
            "通道": ch_names,
            "SPA 前方差 (μV²)": np.round(var_before, 2),
            "SPA 后方差 (μV²)": np.round(var_after, 2),
            "方差减少 (%)": np.round(reduction, 1),
        }).sort_values("方差减少 (%)", ascending=False).reset_index(drop=True)

        st.dataframe(df, use_container_width=True, height=400)
    else:
        st.info("点击「▶ 运行 SPA」后查看通道统计")
