"""
validate.py — SPA 命令行验证脚本

复现 SPA Manual Step 3 的完整流程：
  1. 用 MNE 加载 sample_data.set
  2. 运行 SPA_EEG(EEG, 30, 2, 2)
  3. 输出统计指标
  4. 保存 raw_eeg.png / spa_eeg.png / pca_dist.png

运行：
  source .venv/bin/activate
  python validate.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

mne.set_log_level("WARNING")

# 自动定位 sample_data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_SET = os.path.join(SCRIPT_DIR, "sample_data", "sample_data.set")
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 导入 SPA ─────────────────────────────────────────────────────────────────
sys.path.insert(0, SCRIPT_DIR)
from spa.core import spa_eeg, compute_pc_amplitudes


def plot_eeg_butterfly(data_v: np.ndarray, ch_names: list, srate: float,
                       title: str, out_path: str, scale_uv: float = 50.0):
    """绘制多通道 EEG 叠加图（butterfly plot），对应 Manual Fig 1 / Fig 2"""
    data_uv = data_v * 1e6                    # V → μV
    n_ch, n_t = data_uv.shape
    times = np.arange(n_t) / srate

    fig, ax = plt.subplots(figsize=(14, 6))
    offset = np.arange(n_ch) * scale_uv * 2  # 通道间距
    for i in range(n_ch):
        ax.plot(times, data_uv[i] + offset[i], lw=0.5, color="steelblue", alpha=0.7)
        ax.text(-0.3, offset[i], ch_names[i], fontsize=6, va="center", ha="right")

    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("幅度 (μV)")
    ax.set_title(title)
    ax.set_xlim(times[0], min(times[-1], 30))   # 只显示前 30 秒
    ax.axhline(0, color="k", lw=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  已保存：{out_path}")


def plot_pca_dist(amps_before: np.ndarray, amps_after: np.ndarray,
                  threshold_uv: float, out_path: str):
    """绘制 PC 幅度分布直方图（对应论文 Fig 1）"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, amps, label, color in zip(
        axes,
        [amps_before, amps_after],
        ["SPA 前（双峰分布）", "SPA 后（大方差峰消失）"],
        ["steelblue", "darkorange"],
    ):
        ax.hist(amps * 1e6, bins=100, range=(0, 200), color=color, alpha=0.75, edgecolor="none")
        ax.axvline(threshold_uv, color="red", linestyle="--", lw=1.5,
                   label=f"阈值 {threshold_uv} μV")
        ax.set_xlabel("PC 幅度 (μV)")
        ax.set_ylabel("频次")
        ax.set_title(label)
        ax.legend(fontsize=9)

    fig.suptitle("SPA 前后 PC 幅度分布（对应论文 Fig 1）", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  已保存：{out_path}")


def main():
    threshold_uv = 30.0
    threshold_v = threshold_uv * 1e-6
    win_size = 2.0
    smooth_para = 2.0

    # ── 1. 加载数据 ──────────────────────────────────────────────────────────
    print(f"\n[1/4] 用 MNE 加载数据：{SAMPLE_SET}")
    raw = mne.io.read_raw_eeglab(SAMPLE_SET, preload=True)
    srate = raw.info["sfreq"]
    ch_names = raw.ch_names
    data_raw = raw.get_data()                 # (n_ch, n_times)，单位 V

    print(f"      通道数:   {data_raw.shape[0]}")
    print(f"      采样点数: {data_raw.shape[1]}")
    print(f"      采样率:   {srate} Hz")
    print(f"      时长:     {data_raw.shape[1]/srate:.1f} s")
    print(f"      幅度范围: {data_raw.min()*1e6:.1f} ~ {data_raw.max()*1e6:.1f} μV")

    # ── 2. 计算 SPA 前 PC 幅度 ───────────────────────────────────────────────
    print("\n[2/4] 计算 SPA 前 PC 幅度分布...")
    amps_before = compute_pc_amplitudes(data_raw, srate, win_size)

    # ── 3. 运行 SPA ──────────────────────────────────────────────────────────
    print(f"\n[3/4] 运行 SPA_EEG(threshold={threshold_uv}μV, win={win_size}s, smooth={smooth_para})...")

    progress_marks = set()
    def progress_cb(cur, total):
        pct = int(cur * 100 / total)
        mark = pct // 10 * 10
        if mark not in progress_marks and pct >= mark:
            progress_marks.add(mark)
            print(f"      进度: {mark}%")

    data_spa = spa_eeg(data_raw, srate, threshold_v, win_size, smooth_para, progress_cb)

    amps_after = compute_pc_amplitudes(data_spa, srate, win_size)

    # ── 4. 统计指标 ──────────────────────────────────────────────────────────
    var_before = np.var(data_raw, axis=1)
    var_after = np.var(data_spa, axis=1)
    reduction_pct = (1 - var_after.sum() / var_before.sum()) * 100

    print("\n[4/4] 统计结果：")
    print(f"      总方差减少: {reduction_pct:.1f}%")
    print(f"      SPA 前幅度 RMS: {np.sqrt(np.mean(data_raw**2))*1e6:.2f} μV")
    print(f"      SPA 后幅度 RMS: {np.sqrt(np.mean(data_spa**2))*1e6:.2f} μV")
    print(f"\n      各通道方差减少 Top 5（伪迹通道）:")
    ch_reduction = (1 - var_after / var_before) * 100
    top5 = np.argsort(ch_reduction)[::-1][:5]
    for idx in top5:
        print(f"        {ch_names[idx]:8s}  减少 {ch_reduction[idx]:.1f}%")

    # ── 5. 保存图像 ──────────────────────────────────────────────────────────
    print("\n保存图像...")
    plot_eeg_butterfly(data_raw, ch_names, srate,
                       "SPA 前 — 原始 EEG（含眼电伪迹）\n对应 Manual Fig 1",
                       os.path.join(OUT_DIR, "raw_eeg.png"))
    plot_eeg_butterfly(data_spa, ch_names, srate,
                       "SPA 后 — 清理后 EEG\n对应 Manual Fig 2",
                       os.path.join(OUT_DIR, "spa_eeg.png"))
    plot_pca_dist(amps_before, amps_after, threshold_uv,
                  os.path.join(OUT_DIR, "pca_dist.png"))

    print("\n✓ 验证完成！输出文件目录：", OUT_DIR)


if __name__ == "__main__":
    main()
