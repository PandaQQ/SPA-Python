"""
SPA (Segment-by-segment PCA-based Artifact removal) — Python 实现
对应 MATLAB 版本：SPA_EEG.m / smooth_fusing_epochs.m

参考文献：
  Ouyang, G., Dien, J., & Lorenz, R. (2021).
  Handling EEG artifacts and searching individually optimal experimental
  parameter in real time. Journal of Neural Engineering.
"""

import numpy as np
from typing import Callable


def spa_segment(segment: np.ndarray, threshold: float) -> np.ndarray:
    """
    对单个 EEG 片段做 PCA，清零幅度超过阈值的主成分，重建去伪迹后的信号。

    对应 MATLAB：
        [a,b,c] = pca(temp');
        b(:, c > threshold^2) = 0;
        temp = (b * a')';

    参数
    ----
    segment   : ndarray, shape (n_channels, n_times)
    threshold : float，幅度阈值，单位与数据一致（μV 或 V）

    返回
    ----
    cleaned   : ndarray, shape (n_channels, n_times)
    """
    X = segment.T                              # (n_times, n_channels)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    n = X.shape[0]
    latent = S ** 2 / (n - 1)                 # 各 PC 方差（对应 MATLAB 的 c）
    scores = U * S                             # 得分矩阵（对应 MATLAB 的 b）

    # 幅度 = sqrt(latent)，清零超过阈值的 PC
    scores[:, latent > threshold ** 2] = 0

    # 重建：(b * a')' + 均值，对应 MATLAB (b*a')'
    cleaned = (scores @ Vt + mu).T            # (n_channels, n_times)
    return cleaned


def smooth_fusing_epochs(
    sig1: np.ndarray, sig2: np.ndarray, smooth_para: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    平滑拼接两个相邻 EEG 片段的边界，消除跳变。

    对应 MATLAB：smooth_fusing_epochs.m

    参数
    ----
    sig1        : ndarray, shape (n_times,)，前一段某通道信号
    sig2        : ndarray, shape (n_times,)，后一段某通道信号
    smooth_para : float，平滑强度（>1），越大过渡越陡

    返回
    ----
    sig1, sig2  : 平滑后的两段信号
    """
    sig1 = sig1.copy().flatten()
    sig2 = sig2.copy().flatten()

    m_point = (sig1[-1] + sig2[0]) / 2.0

    L1 = len(sig1)
    L2 = len(sig2)
    L1_half = L1 // 2
    L2_half = L2 // 2

    dif_1 = np.linspace(0, 1, L1 - L1_half) ** smooth_para
    dif_2 = np.linspace(1, 0, L2_half) ** smooth_para

    dif_m1 = sig1[-1] - m_point
    dif_m2 = sig2[0] - m_point

    sig1[L1_half:] = sig1[L1_half:] - dif_1 * dif_m1
    sig2[:L2_half] = sig2[:L2_half] - dif_2 * dif_m2

    return sig1, sig2


def spa_eeg(
    data: np.ndarray,
    srate: float,
    threshold: float = 30e-6,
    win_size: float = 2.0,
    smooth_para: float = 2.0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """
    对连续 EEG 数据逐段做 PCA 去伪迹（SPA_EEG 的 Python 实现）。

    对应 MATLAB：SPA_EEG.m
    调用方式与 Manual 完全一致：SPA_EEG(EEG, 30, 2, 2)

    参数
    ----
    data              : ndarray, shape (n_channels, n_times)
    srate             : float，采样率（Hz）
    threshold         : float，PC 幅度阈值。若数据单位是 V（MNE 默认），
                        传 30e-6；若数据单位是 μV，传 30。默认 30e-6。
    win_size          : float，窗口大小（秒），默认 2 秒
    smooth_para       : float，平滑参数，默认 2
    progress_callback : 可选回调 f(current_seg, total_segs)，用于 UI 进度条

    返回
    ----
    data_new : ndarray, shape (n_channels, n_times)，去伪迹后的数据
    """
    s = int(win_size * srate)                  # 每段采样点数
    n_times = data.shape[1]
    segs = n_times // s                        # 总分段数

    data_new = data.copy()

    for j in range(segs - 1):
        # 当前段 j 和下一段 j+1
        start1 = j * s
        end1 = (j + 1) * s
        start2 = end1
        end2 = n_times if j == segs - 2 else (j + 2) * s

        temp1 = spa_segment(data[:, start1:end1], threshold)
        temp2 = spa_segment(data[:, start2:end2], threshold)

        for ch in range(data.shape[0]):
            sig1, sig2 = smooth_fusing_epochs(temp1[ch], temp2[ch], smooth_para)
            data_new[ch, start1:end1] = sig1
            data_new[ch, start2:end2] = sig2

        if progress_callback is not None:
            progress_callback(j + 1, segs - 1)

    return data_new


def compute_pc_amplitudes(data: np.ndarray, srate: float, win_size: float = 2.0) -> np.ndarray:
    """
    计算所有分段的 PC 幅度分布（用于论文 Fig 1 的双峰分布可视化）。

    参数
    ----
    data     : ndarray, shape (n_channels, n_times)
    srate    : float，采样率
    win_size : float，窗口大小（秒）

    返回
    ----
    amplitudes : ndarray, 所有片段所有 PC 的幅度值（sqrt(variance)）
    """
    s = int(win_size * srate)
    segs = data.shape[1] // s
    amps = []

    for j in range(segs):
        seg = data[:, j * s : (j + 1) * s]
        X = seg.T
        Xc = X - X.mean(axis=0)
        _, S, _ = np.linalg.svd(Xc, full_matrices=False)
        n = X.shape[0]
        latent = S ** 2 / (n - 1)
        amps.extend(np.sqrt(latent).tolist())

    return np.array(amps)
