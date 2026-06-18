"""
RR_Estimasi_PPG_ENS3.py
=======================
Perbandingan tiga algoritma estimasi Respiratory Rate dari sinyal PPG:

  PPG_EMD  : BPF(0.05–6 Hz) → EMD → pilih IMF terbaik via PSD (0.1–0.5 Hz) → peak count → RR
  PPG_EEMD : BPF(0.05–6 Hz) → EEMD → pilih IMF terbaik via PSD (0.1–0.5 Hz) → peak count → RR
  PPG_ENS3 : ENS3_SMART — tiga estimator independen (VMD + ENV + PSD) dengan
             harmonic-aware pairwise agreement voting

Tab ① Pra-Analisis — Muat sinyal, pilih/ekslusi window per subjek
Tab ② Analisis     — Jalankan analisis, log, progress
Tab ③ Hasil        — Tabel per-window untuk ketiga algoritma
Tab ④ Statistik    — Ringkasan MAE/RMSE, Bland-Altman, distribusi error, branch pie
"""

import os
import sys
import glob
import time as _time
import warnings
import json as _json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import butter, filtfilt, find_peaks, hilbert, welch
from scipy.stats import kurtosis as sp_kurtosis
from scipy.interpolate import CubicSpline

try:
    from vmdpy import VMD
    HAS_VMD = True
except ImportError:
    HAS_VMD = False

try:
    from PyEMD import EMD
    HAS_EMD = True
except ImportError:
    HAS_EMD = False

try:
    from PyEMD import EEMD
    HAS_EEMD = True
except ImportError:
    HAS_EEMD = False

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QTabWidget, QProgressBar,
    QAbstractItemView, QStatusBar, QTextEdit, QComboBox, QSizePolicy,
    QLineEdit, QCheckBox, QFrame,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QBrush

warnings.filterwarnings('ignore')


# =============================================================================
# MATPLOTLIB LIGHT THEME
# =============================================================================
matplotlib.rcParams.update({
    'figure.facecolor': '#f5f7fa', 'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#c8d2e0',   'axes.labelcolor': '#1a2233',
    'text.color': '#1a2233',       'xtick.color': '#1a2233',
    'ytick.color': '#1a2233',      'grid.color': '#dde3ee',
    'grid.linestyle': '--',        'grid.alpha': 0.6,
    'legend.facecolor': '#f5f7fa', 'legend.edgecolor': '#c8d2e0',
    'font.family': 'monospace',    'font.size': 10,
    'axes.titlesize': 11,          'axes.labelsize': 10,
})

C_BG    = '#f5f7fa'; C_BG2   = '#edf0f5'; C_BG3   = '#dde3ee'; C_BG4 = '#c8d2e0'
C_TEXT  = '#1a2233'; C_TEXT2 = '#334155'; C_ACCENT = '#1565c0'
C_GREEN = '#2e7d32'; C_AMBER = '#bf360c'; C_RED    = '#b71c1c'
PLOT_BG = '#ffffff'

_SS = f"""
QMainWindow,QWidget{{background:{C_BG};color:{C_TEXT};
    font-family:'Consolas','Courier New',monospace;font-size:12px;}}
QGroupBox{{border:1px solid {C_BG3};border-radius:5px;margin-top:14px;
    padding:8px 6px 6px 6px;font-weight:bold;color:{C_ACCENT};
    font-size:10px;letter-spacing:1px;}}
QGroupBox::title{{subcontrol-origin:margin;left:10px;padding:0 4px;}}
QTableWidget{{background:{C_BG2};alternate-background-color:{C_BG};
    gridline-color:{C_BG3};selection-background-color:#bbdefb;}}
QHeaderView::section{{background:{C_BG3};color:{C_TEXT};border:1px solid {C_BG4};
    padding:4px;font-size:10px;font-weight:bold;}}
QTabWidget::pane{{border:1px solid {C_BG3};background:{C_BG};}}
QTabBar::tab{{background:{C_BG2};color:{C_TEXT2};border:1px solid {C_BG3};
    padding:5px 14px;border-bottom:none;font-size:11px;}}
QTabBar::tab:selected{{background:{C_BG};color:{C_ACCENT};font-weight:bold;}}
QTextEdit{{background:{C_BG2};border:1px solid {C_BG3};border-radius:3px;}}
QComboBox{{background:{C_BG2};border:1px solid {C_BG3};border-radius:3px;
    padding:3px 8px;min-height:24px;}}
QProgressBar{{background:{C_BG3};border:1px solid {C_BG4};border-radius:3px;
    text-align:center;}}
QProgressBar::chunk{{background:{C_ACCENT};border-radius:2px;}}
QCheckBox{{spacing:6px;color:{C_TEXT};}}
QCheckBox::indicator{{width:14px;height:14px;border:1px solid {C_BG4};
    border-radius:2px;background:{C_BG};}}
QCheckBox::indicator:checked{{background:{C_ACCENT};border-color:{C_ACCENT};}}
QStatusBar{{background:{C_BG2};color:{C_TEXT2};font-size:10px;
    border-top:1px solid {C_BG3};}}
"""
_BTN = (
    f'QPushButton{{background:{C_ACCENT};border:none;color:#ffffff;'
    f'font-size:11px;border-radius:3px;padding:0 12px;min-height:28px;}}'
    f'QPushButton:hover{{background:#1976d2;}}'
    f'QPushButton:pressed{{background:#0d47a1;}}'
    f'QPushButton:disabled{{color:{C_BG4};background:{C_BG3};border-color:{C_BG4};}}'
)
_CELL_GOOD = (QColor('#e8f5e9'), QColor(C_GREEN))
_CELL_WARN = (QColor('#fff8e1'), QColor('#f57f17'))
_CELL_BAD  = (QColor('#ffebee'), QColor(C_RED))


# =============================================================================
# PATHS & CONSTANTS
# =============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))

BASE_PPG = (r"D:\File Umum\Tugas Akhir (Post Seminar Proposal)"
            r"\Akuisisi Data Tugas Akhir Haykal Nadhif"
            r"\Database Keseluruhan (Sudah Digabung Semua)")
BASE_GT  = (r"D:\File Umum\Tugas Akhir (Post Seminar Proposal)"
            r"\Akuisisi Data Tugas Akhir Haykal Nadhif"
            r"\Ekstraksi Ground Truth_Respiratory Rate"
            r"\Data Ground Truth Respiratory Rate")
BASE_BELT = (r"D:\File Umum\Tugas Akhir (Post Seminar Proposal)"
             r"\Akuisisi Data Tugas Akhir Haykal Nadhif"
             r"\Database Sinyal Respiration Belt"
             r"\Data CSV dari Software Vernier")
OUT_DIR = os.path.join(_HERE, 'Hasil Respiratory Rate PPG ENS3')

# Default ENS3/VMD parameters
VMD_K_DEFAULT     = 2
VMD_ALPHA_DEFAULT = 250
WIN_S_BASE        = 30.0   # durasi satu window GT
RESP_LO           = 0.1    # Hz (6 bpm)
RESP_HI           = 0.5    # Hz (30 bpm)
BPF_LO            = 0.05   # Hz — pre-filter untuk EMD/EEMD
BPF_HI            = 6.0    # Hz — pre-filter untuk EMD/EEMD
MIN_RR_BPM        = 3.0
MAX_RR_BPM        = 35.0
HARMONIC_CUTOFF   = 2.1
AGREE_TOL         = 3.0    # bpm
EEMD_TRIALS       = 10
EEMD_NOISE        = 0.05
_PRE_MAX_PTS      = 8000
PPG_EXCLUDED      = {'SUBJECT_001', 'SUBJECT_006', 'SUBJECT_007', 'SUBJECT_018'}


# =============================================================================
# HELPERS
# =============================================================================
def _col(df, candidates):
    cl = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cl: return cl[c.lower()]
    return None


def _unwrap_mcu(mcu_ms):
    mcu = np.array(mcu_ms, dtype=np.int64)
    if len(mcu) < 2: return mcu.astype(np.float64)
    diffs = np.diff(mcu); offsets = np.zeros(len(mcu), dtype=np.int64); cum = np.int64(0)
    for i, d in enumerate(diffs):
        if d < -1000: cum += np.int64(-d + 3)
        offsets[i + 1] = cum
    return (mcu + offsets).astype(np.float64)


def _estimate_fs(time_s):
    if len(time_s) < 2: return 1.0
    span = float(time_s[-1] - time_s[0])
    if span > 0: return (len(time_s) - 1) / span
    d = np.diff(time_s[:min(500, len(time_s))]); d = d[d > 0]
    return float(1.0 / np.median(d)) if len(d) else 1.0


def _discover_subjects(base_dir):
    if not os.path.isdir(base_dir):
        return [f'SUBJECT_{i:03d}' for i in range(1, 26)]
    entries = sorted(e for e in os.listdir(base_dir)
                     if e.upper().startswith('SUBJECT_') and
                     os.path.isdir(os.path.join(base_dir, e)))
    return entries if entries else [f'SUBJECT_{i:03d}' for i in range(1, 26)]


def load_ppg_sensor(folder):
    """Muat sinyal PPG mentah (IR/Red).

    Sinyal diinversi (negasi) terhadap mean-nya sebelum dikembalikan — sensor
    PPG yang dipakai merekam dicrotic notch lebih dulu dari systolic peak
    (polaritas terbalik dari konvensi PPG standar), sehingga semua estimator
    hilir (VMD/ENV/PSD/EMD/EEMD) butuh sinyal dengan polaritas yang benar.
    """
    files = glob.glob(os.path.join(folder, '*_full_PPG*.csv'))
    if not files: return None
    df = pd.read_csv(files[0])
    t_col   = next((c for c in ['ts_mcu_ms', 'time_ms', 'timestamp_ms'] if c in df.columns), None)
    ir_col  = next((c for c in df.columns if 'ir'  in c.lower()), None)
    red_col = next((c for c in df.columns if 'red' in c.lower()), None)
    if t_col is None or (ir_col is None and red_col is None): return None
    mcu = _unwrap_mcu(df[t_col].values); t = (mcu - mcu[0]) / 1000.0; fs = _estimate_fs(t)
    out = {'t': t, 'fs': fs}

    def _invert(x):
        x = x.astype(np.float64)
        return float(2.0 * np.mean(x)) - x   # negasi terhadap mean, jaga skala/baseline asli

    if ir_col:  out['IR']  = _invert(df[ir_col].values)
    if red_col: out['Red'] = _invert(df[red_col].values)
    return out


def load_belt(subject):
    folder = os.path.join(BASE_BELT, subject)
    files  = glob.glob(os.path.join(folder, '*_RespirationData.csv'))
    files  = [f for f in files if 'SebelumdiEdit' not in f]
    if not files: return None, None
    df = pd.read_csv(files[0])
    tcol = next((c for c in df.columns if 'Time'  in c), None)
    fcol = next((c for c in df.columns if 'Force' in c), None)
    if tcol is None or fcol is None: return None, None
    return (pd.to_numeric(df[tcol], errors='coerce').values.astype(np.float64),
            pd.to_numeric(df[fcol], errors='coerce').values.astype(np.float64))


def build_analysis_windows(gt_folder, win_dur_s=30.0):
    wf = glob.glob(os.path.join(gt_folder, '*_gt_windows.csv'))
    if not wf: return None
    gt_df = pd.read_csv(wf[0]).sort_values('window_idx').reset_index(drop=True)
    c_wi  = _col(gt_df, ['window_idx', 'window', 'idx'])
    c_s   = _col(gt_df, ['start_s', 'start'])
    c_e   = _col(gt_df, ['end_s', 'end'])
    c_gt  = _col(gt_df, ['rr_peak_bpm', 'rr_bpm', 'rr'])
    c_pts = _col(gt_df, ['peak_times_json', 'peak_times'])
    if None in (c_wi, c_s, c_e, c_gt): return None
    step = max(1, int(round(win_dur_s / WIN_S_BASE)))
    windows = []; rows = list(gt_df.iterrows()); i = 0; win_idx_out = 1
    while i < len(rows):
        chunk = [rows[j][1] for j in range(i, min(i + step, len(rows)))]
        if len(chunk) < step: break
        t_s = float(chunk[0][c_s]); t_e = float(chunk[-1][c_e])
        if step == 1:
            gt_rr = float(chunk[0][c_gt])
        else:
            all_peaks = []
            if c_pts is not None:
                for row in chunk:
                    raw = row[c_pts]
                    try:
                        pts = _json.loads(raw) if isinstance(raw, str) else []
                        all_peaks.extend([float(p) for p in pts])
                    except Exception: pass
            all_peaks.sort()
            if len(all_peaks) >= 2:
                span = float(all_peaks[-1] - all_peaks[0])
                gt_rr = ((len(all_peaks) - 1) / span * 60.0) if span > 0 else 0.0
            else:
                gt_rr = float(np.mean([float(r[c_gt]) for r in chunk]))
        windows.append(dict(window_idx=win_idx_out,
                            start_s=round(t_s, 1),
                            end_s=round(t_e, 1),
                            gt_rr_bpm=round(gt_rr, 4)))
        i += step; win_idx_out += 1
    return windows if windows else None


# =============================================================================
# SIGNAL PROCESSING PRIMITIVES
# =============================================================================
def _bandpass(x, lo, hi, fs, order=3):
    nyq = 0.5 * fs
    lo_n = max(1e-4, min(lo / nyq, 0.99)); hi_n = max(1e-4, min(hi / nyq, 0.99))
    if lo_n >= hi_n: return x
    try:
        b, a = butter(order, [lo_n, hi_n], btype='bandpass'); return filtfilt(b, a, x)
    except Exception: return x


def _lowpass(x, cutoff, fs, order=3):
    nyq = 0.5 * fs; wn = max(1e-4, min(cutoff / nyq, 0.99))
    try:
        b, a = butter(order, wn, btype='low'); return filtfilt(b, a, x)
    except Exception: return x


def _count_peaks_rr(sig, fs):
    if len(sig) < 4: return np.nan
    try: bp = _bandpass(sig, RESP_LO, RESP_HI, fs)
    except Exception: bp = sig
    min_d = max(1, int(fs / (MAX_RR_BPM / 60.0)))
    peaks, _ = find_peaks(bp, distance=min_d)
    dur = len(sig) / fs
    if dur <= 0 or len(peaks) < 1: return np.nan
    rr = (len(peaks) / dur) * 60.0
    return rr if MIN_RR_BPM <= rr <= MAX_RR_BPM else np.nan


def _fft_dominant_rr(sig, fs):
    if len(sig) < 4: return np.nan
    fft_v = np.abs(np.fft.rfft(sig - np.mean(sig))) ** 2
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    mask  = (freqs >= RESP_LO) & (freqs <= RESP_HI)
    if not np.any(mask): return np.nan
    f_dom = float(freqs[mask][np.argmax(fft_v[mask])])
    rr    = f_dom * 60.0
    return rr if MIN_RR_BPM <= rr <= MAX_RR_BPM else np.nan


def _welch_dominant_rr(sig, fs):
    if len(sig) < 8: return np.nan
    nperseg = max(8, min(len(sig), int(len(sig) * 0.5)))
    try:
        freqs, pxx = welch(sig - np.mean(sig), fs=fs, nperseg=nperseg,
                           noverlap=nperseg // 2, window='hann', detrend='constant')
    except Exception: return np.nan
    mask = (freqs >= RESP_LO) & (freqs <= RESP_HI)
    if not np.any(mask): return np.nan
    f_dom = float(freqs[mask][np.argmax(pxx[mask])])
    rr    = f_dom * 60.0
    return rr if MIN_RR_BPM <= rr <= MAX_RR_BPM else np.nan


def _dual_merge(rr_fft, rr_peaks):
    both = not np.isnan(rr_fft) and not np.isnan(rr_peaks)
    if both:
        if abs(rr_fft - rr_peaks) < 3.0: return float((rr_fft + rr_peaks) / 2.0)
        return rr_fft  # FFT lebih stabil dari peak counting untuk sinyal PPG noisy
    if not np.isnan(rr_fft):   return rr_fft
    if not np.isnan(rr_peaks): return rr_peaks
    return np.nan


def _select_imf_psd(imfs, fs):
    """Pilih IMF dengan energi PSD terbesar di band respirasi (0.1–0.5 Hz)."""
    best_idx = None; best_energy = -1.0
    for i, imf in enumerate(imfs):
        n = len(imf); fft_v = np.abs(np.fft.rfft(imf - np.mean(imf))) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        m = (freqs >= RESP_LO) & (freqs <= RESP_HI)
        energy = float(np.sum(fft_v[m])) if np.any(m) else 0.0
        if energy > best_energy: best_energy = energy; best_idx = i
    if best_idx is None: return None
    # Tolak jika energi terlalu rendah (< 5% dari IMF terbaik)
    if best_energy < 1e-12: return None
    return imfs[best_idx]


def _select_imf_kurtosis_freq(imfs, fs):
    """Pilih IMF untuk ENS3 VMD: strict/relaxed band, lalu kurtosis minimum."""
    all_scored = []
    for i, imf in enumerate(imfs):
        n = len(imf)
        fft_v = np.abs(np.fft.rfft(imf - np.mean(imf))) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        f_dom = float(freqs[np.argmax(fft_v)])
        kurt  = float(sp_kurtosis(imf, fisher=True))
        all_scored.append((i, f_dom, kurt))
    strict = [(i, k) for i, f, k in all_scored if RESP_LO <= f <= RESP_HI]
    if strict:
        return imfs[min(strict, key=lambda x: x[1])[0]]
    relaxed = [(i, k) for i, f, k in all_scored if 0.08 <= f <= 0.6]
    if relaxed:
        return imfs[min(relaxed, key=lambda x: x[1])[0]]
    nearest = min(all_scored, key=lambda x: abs(x[1] - 0.3))
    return imfs[nearest[0]]


# =============================================================================
# ALGORITHM 1: PPG_EMD
# =============================================================================
def est_emd(ppg_seg, fs):
    """BPF(0.05–6 Hz) → EMD → IMF selection via PSD → peak count → RR."""
    t0 = _time.perf_counter()
    if not HAS_EMD:
        return np.nan, (_time.perf_counter() - t0) * 1e3, None
    sig = _bandpass(ppg_seg - np.mean(ppg_seg), BPF_LO, BPF_HI, fs)
    try:
        emd = EMD()
        imfs = emd.emd(sig)
        if imfs is None or len(imfs) == 0:
            return np.nan, (_time.perf_counter() - t0) * 1e3, None
    except Exception:
        return np.nan, (_time.perf_counter() - t0) * 1e3, None
    best = _select_imf_psd(imfs, fs)
    if best is None:
        return np.nan, (_time.perf_counter() - t0) * 1e3, None
    rr = _count_peaks_rr(best, fs)
    return rr, (_time.perf_counter() - t0) * 1e3, best


# =============================================================================
# ALGORITHM 2: PPG_EEMD
# =============================================================================
def est_eemd(ppg_seg, fs):
    """BPF(0.05–6 Hz) → EEMD → IMF selection via PSD → peak count → RR."""
    t0 = _time.perf_counter()
    if not HAS_EEMD:
        return np.nan, (_time.perf_counter() - t0) * 1e3, None
    sig = _bandpass(ppg_seg - np.mean(ppg_seg), BPF_LO, BPF_HI, fs)
    try:
        eemd = EEMD(trials=EEMD_TRIALS, noise_width=EEMD_NOISE, noise_seed=42)
        imfs = eemd.eemd(sig, max_imf=10)
        if imfs is None or len(imfs) == 0:
            return np.nan, (_time.perf_counter() - t0) * 1e3, None
    except Exception:
        return np.nan, (_time.perf_counter() - t0) * 1e3, None
    best = _select_imf_psd(imfs, fs)
    if best is None:
        return np.nan, (_time.perf_counter() - t0) * 1e3, None
    rr = _count_peaks_rr(best, fs)
    return rr, (_time.perf_counter() - t0) * 1e3, best


# =============================================================================
# ALGORITHM 3: PPG_ENS3 — tiga sub-estimator
# =============================================================================
def _vmd_decompose(x, K, alpha):
    if not HAS_VMD: return None
    try:
        u, _, _ = VMD(x, alpha, 0.0, K, 0, 1, 1e-7)
        return u if u is not None and len(u) > 0 else None
    except Exception: return None


def _est_vmd(ppg_seg, fs, K, alpha):
    t0 = _time.perf_counter()
    imfs = _vmd_decompose(ppg_seg - np.mean(ppg_seg), K=K, alpha=alpha)
    if imfs is None: return np.nan, (_time.perf_counter() - t0) * 1e3
    best = _select_imf_kurtosis_freq(imfs, fs)
    if best is None: return np.nan, (_time.perf_counter() - t0) * 1e3
    return _count_peaks_rr(best, fs), (_time.perf_counter() - t0) * 1e3


def _est_dc(ppg_seg, fs):
    """DC baseline wander → Welch PSD → RR dominan di band respirasi.

    Modulasi DC akibat pergerakan dada saat bernapas — jalur paling langsung
    untuk PPG dari vest. Diekstrak dengan lowpass filter ketat di band respirasi.
    """
    t0 = _time.perf_counter()
    if len(ppg_seg) < 20: return np.nan, (_time.perf_counter() - t0) * 1e3
    dc = _lowpass(ppg_seg.astype(float), RESP_HI, fs)
    dc = dc - np.mean(dc)
    rr = _dual_merge(_welch_dominant_rr(dc, fs), _count_peaks_rr(dc, fs))
    return rr, (_time.perf_counter() - t0) * 1e3


def _est_env(ppg_seg, fs):
    t0 = _time.perf_counter()
    if len(ppg_seg) < 20: return np.nan, (_time.perf_counter() - t0) * 1e3
    nyq = 0.5 * fs
    lo_n = max(1e-4, min(0.7 / nyq, 0.99)); hi_n = max(1e-4, min(5.0 / nyq, 0.99))
    if lo_n >= hi_n: return np.nan, (_time.perf_counter() - t0) * 1e3
    try:
        b, a = butter(3, [lo_n, hi_n], btype='bandpass')
        ppg_bp = filtfilt(b, a, ppg_seg - np.mean(ppg_seg))
    except Exception:
        return np.nan, (_time.perf_counter() - t0) * 1e3
    env_filt = _lowpass(np.abs(hilbert(ppg_bp)), RESP_HI, fs)
    return _count_peaks_rr(env_filt, fs), (_time.perf_counter() - t0) * 1e3


def _build_ibi_tachogram(ppg_seg, fs, tachogram_fs=4.0):
    """Deteksi peak PPG → IBI tachogram (analog RRI tachogram di ECG)."""
    ppg_bp = _bandpass(ppg_seg - np.mean(ppg_seg), 0.5, 4.0, fs)
    min_d = max(1, int(fs * 0.35))  # min 350 ms antar detak (~171 bpm max)
    peaks, _ = find_peaks(ppg_bp, distance=min_d, prominence=0.1 * np.std(ppg_bp))
    if len(peaks) < 4: return None, None
    peak_times = peaks / fs
    ibi = np.diff(peak_times)
    tmd = (peak_times[:-1] + peak_times[1:]) / 2.0
    ok = (ibi >= 0.33) & (ibi <= 2.0)  # 30–180 bpm
    if np.sum(ok) < 3: return None, None
    iv = ibi[ok]; tv = tmd[ok]
    if tv[-1] <= tv[0]: return None, None
    try: cs = CubicSpline(tv, iv)
    except Exception: return None, None
    tu = np.arange(tv[0], tv[-1], 1.0 / tachogram_fs)
    if len(tu) < 8: return None, None
    return tu, cs(tu)


def _est_ibi(ppg_seg, fs):
    """IBI tachogram → Welch PSD → RR dominan di band respirasi (analog ECG_VMD pathway)."""
    t0 = _time.perf_counter()
    tu, tacho = _build_ibi_tachogram(ppg_seg, fs)
    if tu is None: return np.nan, (_time.perf_counter() - t0) * 1e3
    rr = _dual_merge(_welch_dominant_rr(tacho, 4.0), _count_peaks_rr(tacho, 4.0))
    return rr, (_time.perf_counter() - t0) * 1e3


def _est_psd(ppg_seg, fs):
    """Welch PSD dengan zero-padding (nfft > nperseg) untuk resolusi frekuensi
    lebih halus — tanpa zero-padding, nperseg besar (≈fs*8) menghasilkan
    resolusi sekasar ~7.5 brpm untuk fs≈100 Hz, cukup untuk membuat RR yang
    berbeda hanya beberapa brpm ter-snap ke bin yang sama (lihat temuan
    SUBJECT_021: PSD selalu jatuh ke 7.51 brpm terlepas dari GT 8-14 brpm).
    """
    t0 = _time.perf_counter()
    if len(ppg_seg) < 20: return np.nan, (_time.perf_counter() - t0) * 1e3
    resp_sig = _bandpass(ppg_seg - np.mean(ppg_seg), RESP_LO, RESP_HI, fs)
    try:
        nperseg = min(len(resp_sig), max(64, int(fs * 8)))
        nfft = nperseg * 4   # zero-padding 4x -> resolusi 4x lebih halus
        f, pxx = welch(resp_sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, nfft=nfft)
    except Exception:
        return np.nan, (_time.perf_counter() - t0) * 1e3
    mask = (f >= RESP_LO) & (f <= RESP_HI)
    if not np.any(mask) or np.all(pxx[mask] == 0):
        return np.nan, (_time.perf_counter() - t0) * 1e3
    f_peak = float(f[mask][np.argmax(pxx[mask])])
    rr = f_peak * 60.0
    return (rr if MIN_RR_BPM <= rr <= MAX_RR_BPM else np.nan), (_time.perf_counter() - t0) * 1e3


def ens3_smart(rr_v, rr_e, rr_p, ratio_cutoff=HARMONIC_CUTOFF, tol=AGREE_TOL):
    """Harmonic-aware pairwise agreement voting untuk tiga estimator PPG.

    rr_v : VMD pada segmen PPG
    rr_e : amplitudo envelope (Hilbert)
    rr_p : Welch PSD dominan
    """
    valid = [(v, n) for v, n in [(rr_v, 'v'), (rr_e, 'e'), (rr_p, 'p')]
             if not np.isnan(v)]
    if not valid: return np.nan, 'no_data'
    if len(valid) == 1: return float(valid[0][0]), 'fallback'
    if len(valid) == 2:
        v0, n0 = valid[0]; v1, n1 = valid[1]
        if abs(v0 - v1) < tol:
            return float((v0 + v1) / 2), f'pair_{"".join(sorted([n0, n1]))}'
        vmd_val = rr_v if not np.isnan(rr_v) else v0
        return float(vmd_val), 'fallback'
    ag_ve = abs(rr_v - rr_e) < tol
    ag_vp = abs(rr_v - rr_p) < tol
    ag_ep = abs(rr_e - rr_p) < tol
    if ag_ve and ag_vp:   return float(np.mean([rr_v, rr_e, rr_p])), 'all_agree'
    elif ag_ve:           return float((rr_v + rr_e) / 2), 'vmd+env'
    elif ag_vp:           return float((rr_v + rr_p) / 2), 'vmd+psd'
    elif ag_ep:
        ep_avg = (rr_e + rr_p) / 2
        return (float(ep_avg), 'harmonic->EP') if ep_avg / (rr_v + 1e-6) >= ratio_cutoff \
               else (float(rr_v), 'harmonic->VMD')
    else:                 return float(rr_v), 'all_disagree'


def est_ens3(ppg_seg, fs, K, alpha):
    """PPG ENS3_SMART: VMD + ENV + PSD → voting."""
    t0 = _time.perf_counter()
    rr_v, _ = _est_vmd(ppg_seg, fs, K, alpha)
    rr_e, _ = _est_env(ppg_seg, fs)
    rr_p, _ = _est_psd(ppg_seg, fs)
    rr_ens, branch = ens3_smart(rr_v, rr_e, rr_p)
    elapsed = (_time.perf_counter() - t0) * 1e3
    return rr_ens, elapsed, rr_v, rr_e, rr_p, branch


# =============================================================================
# STATISTICS
# =============================================================================
def compute_stats(df, est_col='rr_est_bpm'):
    gt = df['gt_rr_bpm'].values.astype(np.float64)
    est = df[est_col].values.astype(np.float64)
    valid = ~np.isnan(est) & ~np.isnan(gt) & (gt > 0)
    n_total = len(gt); n_valid = int(np.sum(valid))
    if n_valid == 0: return n_total, 0, np.nan, np.nan, np.nan, np.nan, np.nan
    gt_v = gt[valid]; est_v = est[valid]; err = est_v - gt_v; ae = np.abs(err)
    mae  = float(np.mean(ae)); rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(ae / gt_v) * 100.0)
    denom = np.abs(est_v) + np.abs(gt_v)
    smape = float(np.mean(np.where(denom > 0, 2.0 * ae / denom, 0.0)) * 100.0)
    mean_gt = float(np.mean(gt_v))
    return n_total, n_valid, mae, rmse, mape, smape, (mae / mean_gt if mean_gt > 0 else np.nan)


# =============================================================================
# TABLE SCHEMA — satu baris = satu window × satu kanal, berisi ketiga hasil
# =============================================================================
_COLUMNS = [
    'subject_id', 'window_idx', 'start_s', 'end_s', 'gt_rr_bpm', 'channel',
    # EMD
    'rr_emd', 'err_emd', 'ae_emd', 'ms_emd',
    # EEMD
    'rr_eemd', 'err_eemd', 'ae_eemd', 'ms_eemd',
    # ENS3 sub-estimators
    'rr_vmd', 'err_vmd', 'ae_vmd', 'rr_env', 'rr_psd_sub',
    # ENS3 output
    'rr_ens3', 'err_ens3', 'ae_ens3', 'ms_ens3', 'branch',
    'win_dur_s',
]
_HEADERS = [
    'Subject', 'Win', 'Start(s)', 'End(s)', 'GT(brpm)', 'Channel',
    'EMD(brpm)', 'Err EMD', '|Err| EMD', 'ms EMD',
    'EEMD(brpm)', 'Err EEMD', '|Err| EEMD', 'ms EEMD',
    'VMD sub', 'Err VMD', '|Err| VMD', 'ENV sub', 'PSD sub',
    'ENS3(brpm)', 'Err ENS3', '|Err| ENS3', 'ms ENS3', 'Branch',
    'WinDur(s)',
]


# =============================================================================
# WORKER
# =============================================================================
class AnalysisWorker(QThread):
    log_signal      = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    done_signal     = pyqtSignal(object)

    def __init__(self, subjects, channels, win_durs, vmd_k, vmd_alpha,
                 run_emd, run_eemd, window_filter):
        super().__init__()
        self.subjects       = subjects
        self.channels       = channels
        self.win_durs       = win_durs   # list of float, e.g. [30.0] or [30.0, 60.0]
        self.vmd_k          = vmd_k
        self.vmd_alpha      = vmd_alpha
        self.run_emd        = run_emd
        self.run_eemd       = run_eemd
        self.window_filter  = window_filter   # dict {subj: set of window_idx} or {}

    def run(self):
        def log(m): self.log_signal.emit(m)

        # Count total for progress bar
        total = 0
        for subj in self.subjects:
            for wd in self.win_durs:
                wins = build_analysis_windows(os.path.join(BASE_GT, subj), wd)
                if wins is None: continue
                wf = self.window_filter.get(subj)
                n  = len([w for w in wins if wf is None or w['window_idx'] in wf])
                total += n * len(self.channels)

        algs = ['ENS3']
        if self.run_emd:  algs.insert(0, 'EMD')
        if self.run_eemd: algs.insert(1 if self.run_emd else 0, 'EEMD')

        log(f'Algoritma: {" + ".join(algs)}')
        log(f'ENS3: K={self.vmd_k}  alpha={self.vmd_alpha}  tol={AGREE_TOL} bpm')
        log(f'Window={[f"{w:.0f}s" for w in self.win_durs]}  |  Kanal={self.channels}  |  Subjek={len(self.subjects)}')
        log('─' * 60)

        rows_by_dur = {wd: [] for wd in self.win_durs}
        done = 0

        for subj in self.subjects:
            ppg_folder = os.path.join(BASE_PPG, subj)
            gt_folder  = os.path.join(BASE_GT,  subj)
            log(f'\n▶  {subj}')
            ppg_data = load_ppg_sensor(ppg_folder)
            if ppg_data is None:
                log('   ⚠ File PPG tidak ditemukan — dilewati.')
                for wd in self.win_durs:
                    wins = build_analysis_windows(gt_folder, wd)
                    if wins is None: continue
                    wf = self.window_filter.get(subj)
                    done += len([w for w in wins if wf is None or w['window_idx'] in wf]) * len(self.channels)
                self.progress_signal.emit(done, total); continue

            t_ppg = ppg_data['t']; fs = ppg_data['fs']
            log(f'   fs≈{fs:.1f} Hz')

            for win_dur_s in self.win_durs:
                wins = build_analysis_windows(gt_folder, win_dur_s)
                if wins is None:
                    log(f'   ⚠ Ground truth tidak ditemukan ({win_dur_s:.0f}s) — dilewati.'); continue
                wf = self.window_filter.get(subj)
                n_active = len([w for w in wins if wf is None or w['window_idx'] in wf])
                log(f'   {n_active}/{len(wins)} window aktif × {win_dur_s:.0f}s')
                rows = rows_by_dur[win_dur_s]

                for w in wins:
                    wi = w['window_idx']
                    if wf is not None and wi not in wf:
                        continue
                    t_s = w['start_s']; t_e = w['end_s']; gt = w['gt_rr_bpm']
                    mask = (t_ppg >= t_s) & (t_ppg < t_e)

                    for ch in self.channels:
                        if ch not in ppg_data:
                            done += 1; self.progress_signal.emit(done, total); continue
                        ppg_seg = ppg_data[ch][mask]
                        if len(ppg_seg) < 20:
                            done += 1; self.progress_signal.emit(done, total); continue

                        def _err(rr):
                            se = (rr - gt) if not np.isnan(rr) else np.nan
                            return se, (abs(se) if not np.isnan(se) else np.nan)

                        # EMD
                        rr_emd = ms_emd = np.nan
                        if self.run_emd:
                            rr_emd, ms_emd, _ = est_emd(ppg_seg, fs)
                        se_emd, ae_emd = _err(rr_emd)

                        # EEMD
                        rr_eemd = ms_eemd = np.nan
                        if self.run_eemd:
                            rr_eemd, ms_eemd, _ = est_eemd(ppg_seg, fs)
                        se_eemd, ae_eemd = _err(rr_eemd)

                        # ENS3
                        rr_ens3, ms_ens3, rr_v, rr_e, rr_p, branch = \
                            est_ens3(ppg_seg, fs, self.vmd_k, self.vmd_alpha)
                        se_ens3, ae_ens3 = _err(rr_ens3)
                        se_vmd,  ae_vmd  = _err(rr_v)

                        def _r(v): return round(v, 4) if not np.isnan(v) else np.nan

                        rows.append({
                            'subject_id':  subj,  'window_idx': wi,
                            'start_s':     round(t_s, 1), 'end_s': round(t_e, 1),
                            'gt_rr_bpm':   round(gt, 4),  'channel': ch,
                            'rr_emd':      _r(rr_emd),    'err_emd':  _r(se_emd),   'ae_emd':  _r(ae_emd),   'ms_emd':  round(ms_emd, 3) if not np.isnan(ms_emd) else np.nan,
                            'rr_eemd':     _r(rr_eemd),   'err_eemd': _r(se_eemd),  'ae_eemd': _r(ae_eemd),  'ms_eemd': round(ms_eemd, 3) if not np.isnan(ms_eemd) else np.nan,
                            'rr_vmd':      _r(rr_v),      'err_vmd':  _r(se_vmd),   'ae_vmd':  _r(ae_vmd),
                            'rr_env':      _r(rr_e),      'rr_psd_sub': _r(rr_p),
                            'rr_ens3':     _r(rr_ens3),   'err_ens3': _r(se_ens3),  'ae_ens3': _r(ae_ens3),  'ms_ens3': round(ms_ens3, 3),
                            'branch':      branch,         'win_dur_s': win_dur_s,
                        })
                        done += 1
                        self.progress_signal.emit(done, total)

            log(f'   ✔ {subj} selesai')

        result = {wd: (pd.DataFrame(rows) if rows else pd.DataFrame(columns=_COLUMNS))
                  for wd, rows in rows_by_dur.items()}
        self.done_signal.emit(result)


# =============================================================================
# MATPLOTLIB CANVAS
# =============================================================================
class MplCanvas(FigureCanvas):
    def __init__(self, nrows=1, ncols=1):
        self.fig = Figure(facecolor=C_BG, tight_layout=True)
        if nrows == 1 and ncols == 1:
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.ax.set_facecolor(PLOT_BG)
            self.axes = [self.ax]
        else:
            self.axes = [self.fig.add_subplot(nrows, ncols, i + 1) for i in range(nrows * ncols)]
            for ax in self.axes: ax.set_facecolor(PLOT_BG)
            self.ax = self.axes[0]
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def clear_all(self):
        for ax in self.axes: ax.clear(); ax.set_facecolor(PLOT_BG)


# =============================================================================
# MAIN WINDOW
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Estimasi RR PPG — EMD vs EEMD vs ENS3_SMART')
        self.resize(1500, 940); self.setStyleSheet(_SS)
        self._df             = None
        self._worker         = None
        self._belt_cache:dict = {}
        self._window_filter: dict = {}   # {subj: set of window_idx}
        self._pre_sig_cache: dict = {}   # {subj: (t_disp, sig_disp, ch)}
        self._pre_gt_cache:  dict = {}   # {subj: (gt_df, c_wi, c_s, c_e, c_gt)}
        self._pre_sel_row    = -1
        self._pre_seg_idx    = 0
        self._pre_seg_total  = 1
        self._build_ui()

    # =========================================================================
    # TOP-LEVEL UI
    # =========================================================================
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        layout  = QVBoxLayout(central); layout.setContentsMargins(4, 4, 4, 4)
        self._main_tabs = QTabWidget()
        self._main_tabs.addTab(self._build_tab_pra(),       '  ① Pra-Analisis  ')
        self._main_tabs.addTab(self._build_tab_analisis(),  '  ② Analisis  ')
        self._main_tabs.addTab(self._build_tab_hasil(),     '  ③ Hasil  ')
        self._main_tabs.addTab(self._build_tab_statistik(), '  ④ Statistik & Plot  ')
        layout.addWidget(self._main_tabs)
        self._status = QStatusBar(); self.setStatusBar(self._status)
        self._status.showMessage('Siap. Mulai di Tab ① untuk memilih window yang akan dianalisis.')

    # =========================================================================
    # TAB ① — PRA-ANALISIS (pilih / ekslusi window)
    # =========================================================================
    def _build_tab_pra(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(6)

        # Controls row
        ctrl = QWidget(); cl = QHBoxLayout(ctrl); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(8)
        cl.addWidget(QLabel('Subjek:'))
        all_subj = [s for s in _discover_subjects(BASE_PPG) if s.upper() not in PPG_EXCLUDED]
        self._pre_subj_cb = QComboBox(); self._pre_subj_cb.addItems(all_subj)
        cl.addWidget(self._pre_subj_cb)
        cl.addWidget(QLabel('Kanal:'))
        self._pre_ch_cb = QComboBox(); self._pre_ch_cb.addItems(['IR', 'Red'])
        cl.addWidget(self._pre_ch_cb)
        self._btn_pre_load = QPushButton('Muat Sinyal'); self._btn_pre_load.setStyleSheet(_BTN)
        cl.addWidget(self._btn_pre_load)
        self._pre_info_lbl = QLabel('Pilih subjek & kanal, lalu klik Muat Sinyal.')
        self._pre_info_lbl.setStyleSheet(
            f'color:{C_TEXT};font-size:11px;padding:2px 6px;'
            f'background:{C_BG2};border:1px solid {C_BG3};border-radius:3px;')
        cl.addWidget(self._pre_info_lbl, stretch=1)
        v.addWidget(ctrl)

        # Signal plot
        self._pre_canvas = MplCanvas(); v.addWidget(self._pre_canvas, stretch=2)

        # Segment navigation
        nav = QWidget(); nl = QHBoxLayout(nav); nl.setContentsMargins(0, 0, 0, 0); nl.setSpacing(6)
        nav_lbl = QLabel('Navigasi (30s):')
        nav_lbl.setStyleSheet(f'color:{C_TEXT};font-size:11px;font-weight:bold;')
        self._btn_pre_prev = QPushButton('◀ Prev'); self._btn_pre_next = QPushButton('Next ▶')
        self._btn_pre_prev.setEnabled(False); self._btn_pre_next.setEnabled(False)
        self._btn_pre_prev.clicked.connect(self._on_pre_prev)
        self._btn_pre_next.clicked.connect(self._on_pre_next)
        self._pre_seg_lbl = QLabel('—')
        self._pre_seg_lbl.setStyleSheet(
            f'color:{C_ACCENT};font-size:11px;font-weight:bold;padding:2px 10px;'
            f'background:{C_BG2};border:1px solid {C_BG3};border-radius:3px;min-width:100px;')
        self._pre_seg_lbl.setAlignment(Qt.AlignCenter)
        self._btn_pre_all_view = QPushButton('Lihat Semua')
        self._btn_pre_all_view.setStyleSheet(_BTN); self._btn_pre_all_view.setEnabled(False)
        self._btn_pre_all_view.clicked.connect(self._pre_view_all)
        for wb in [nav_lbl, self._btn_pre_prev, self._pre_seg_lbl, self._btn_pre_next]:
            nl.addWidget(wb)
        nl.addStretch(); nl.addWidget(self._btn_pre_all_view)
        v.addWidget(nav)

        # Selection controls
        sel = QWidget(); sl = QHBoxLayout(sel); sl.setContentsMargins(0, 0, 0, 0); sl.setSpacing(6)
        self._pre_count_lbl = QLabel('')
        self._pre_count_lbl.setStyleSheet(
            f'color:{C_TEXT};font-weight:bold;font-size:11px;padding:2px 6px;'
            f'background:{C_BG2};border:1px solid {C_BG3};border-radius:3px;')
        sl.addWidget(self._pre_count_lbl); sl.addStretch()
        self._btn_pre_all    = QPushButton('Pilih Semua')
        self._btn_pre_none   = QPushButton('Hapus Semua')
        self._btn_pre_invert = QPushButton('Balik Pilihan')
        self._btn_pre_reset  = QPushButton('Reset Semua Subjek')
        for b in [self._btn_pre_all, self._btn_pre_none, self._btn_pre_invert, self._btn_pre_reset]:
            sl.addWidget(b)
        v.addWidget(sel)

        hint = QLabel('Centang window yang ingin diikutkan dalam analisis. Klik baris → zoom.')
        hint.setStyleSheet(f'color:{C_TEXT2};font-size:10px;'); v.addWidget(hint)

        self._pre_win_table = QTableWidget(0, 5)
        self._pre_win_table.setHorizontalHeaderLabels(['Sertakan', 'Win', 'Start(s)', 'End(s)', 'GT RR (brpm)'])
        self._pre_win_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._pre_win_table.setAlternatingRowColors(True)
        self._pre_win_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._pre_win_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._pre_win_table.setMaximumHeight(220)
        v.addWidget(self._pre_win_table)

        # Wire up
        self._btn_pre_load.clicked.connect(self._on_pre_load)
        self._btn_pre_all.clicked.connect(lambda: self._pre_set_all(True))
        self._btn_pre_none.clicked.connect(lambda: self._pre_set_all(False))
        self._btn_pre_invert.clicked.connect(self._pre_invert)
        self._btn_pre_reset.clicked.connect(self._pre_reset_all)
        self._pre_win_table.itemChanged.connect(self._on_pre_item_changed)
        self._pre_win_table.cellClicked.connect(self._on_pre_win_clicked)
        return w

    def _on_pre_load(self):
        subj = self._pre_subj_cb.currentText(); ch = self._pre_ch_cb.currentText()
        gt_df, _ = self._load_gt_raw(subj)
        if gt_df is None:
            self._pre_info_lbl.setText('⚠ Ground truth tidak ditemukan.'); return
        c_wi = _col(gt_df, ['window_idx', 'window', 'idx'])
        c_s  = _col(gt_df, ['start_s', 'start'])
        c_e  = _col(gt_df, ['end_s', 'end'])
        c_gt = _col(gt_df, ['rr_peak_bpm', 'rr_bpm', 'rr'])
        if None in (c_wi, c_s, c_e, c_gt):
            self._pre_info_lbl.setText('⚠ Kolom GT tidak dikenali.'); return
        self._pre_gt_cache[subj] = (gt_df, c_wi, c_s, c_e, c_gt)
        ppg_data = load_ppg_sensor(os.path.join(BASE_PPG, subj))
        if ppg_data is None or ch not in ppg_data:
            self._pre_info_lbl.setText(f'⚠ File PPG/{ch} tidak ditemukan.'); return
        t_raw = ppg_data['t']; sig_raw = ppg_data[ch]; fs = ppg_data['fs']
        sig_disp = _bandpass(sig_raw, 1.0, 5.0, fs)
        n = len(t_raw); step = max(1, n // _PRE_MAX_PTS)
        self._pre_sig_cache[subj] = (t_raw[::step], sig_disp[::step], ch)
        if subj not in self._window_filter:
            self._window_filter[subj] = {int(r[c_wi]) for _, r in gt_df.iterrows()}
        self._pre_win_table.blockSignals(True)
        self._pre_win_table.setRowCount(len(gt_df))
        included = self._window_filter[subj]
        for ri, (_, grow) in enumerate(gt_df.iterrows()):
            wi = int(grow[c_wi]); t_s = float(grow[c_s]); t_e = float(grow[c_e]); gt = float(grow[c_gt])
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            chk.setCheckState(Qt.Checked if wi in included else Qt.Unchecked)
            chk.setData(Qt.UserRole, wi)
            self._pre_win_table.setItem(ri, 0, chk)
            for ci, val in enumerate([str(wi), f'{t_s:.1f}', f'{t_e:.1f}', f'{gt:.2f}']):
                item = QTableWidgetItem(val); item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self._pre_win_table.setItem(ri, ci + 1, item)
        self._pre_win_table.blockSignals(False)
        self._pre_seg_total = max(1, int(np.ceil(t_raw[-1] / WIN_S_BASE)))
        self._pre_seg_idx = 0
        self._btn_pre_prev.setEnabled(False)
        self._btn_pre_next.setEnabled(self._pre_seg_total > 1)
        self._btn_pre_all_view.setEnabled(True)
        self._pre_update_seg_label(); self._pre_update_count()
        self._pre_update_plot(); self._pre_apply_xlim()
        self._pre_info_lbl.setText(
            f'{subj} [{ch}]  —  {len(gt_df)} window  |  fs≈{fs:.1f} Hz  |  ≈{t_raw[-1]:.0f}s')
        self._status.showMessage(f'{subj}/{ch} dimuat — {len(gt_df)} window.')

    def _load_gt_raw(self, subj):
        wf = glob.glob(os.path.join(BASE_GT, subj, '*_gt_windows.csv'))
        if not wf: return None, None
        return pd.read_csv(wf[0]), None

    def _on_pre_item_changed(self, item):
        if item.column() != 0: return
        subj = self._pre_subj_cb.currentText(); wi = item.data(Qt.UserRole)
        if wi is None or subj not in self._window_filter: return
        if item.checkState() == Qt.Checked: self._window_filter[subj].add(wi)
        else: self._window_filter[subj].discard(wi)
        self._pre_update_count(); self._pre_update_plot(); self._pre_apply_xlim()

    def _on_pre_win_clicked(self, row, _c):
        subj = self._pre_subj_cb.currentText()
        if subj not in self._pre_gt_cache: return
        gt_df, c_wi, c_s, c_e, c_gt = self._pre_gt_cache[subj]
        if row >= len(gt_df): return
        self._pre_sel_row = row
        t_s = float(gt_df.iloc[row][c_s])
        self._pre_seg_idx = max(0, min(int(t_s / WIN_S_BASE), self._pre_seg_total - 1))
        self._pre_update_seg_label(); self._pre_update_nav_btns()
        self._pre_update_plot(); self._pre_apply_xlim()

    def _pre_set_all(self, state):
        subj = self._pre_subj_cb.currentText()
        if subj not in self._pre_gt_cache: return
        gt_df, c_wi, _, _, _ = self._pre_gt_cache[subj]
        self._window_filter[subj] = (
            {int(r[c_wi]) for _, r in gt_df.iterrows()} if state else set())
        self._pre_reload_checks(); self._pre_update_count()
        self._pre_update_plot(); self._pre_apply_xlim()

    def _pre_invert(self):
        subj = self._pre_subj_cb.currentText()
        if subj not in self._pre_gt_cache: return
        gt_df, c_wi, _, _, _ = self._pre_gt_cache[subj]
        all_wi = {int(r[c_wi]) for _, r in gt_df.iterrows()}
        self._window_filter[subj] = all_wi - self._window_filter.get(subj, set())
        self._pre_reload_checks(); self._pre_update_count()
        self._pre_update_plot(); self._pre_apply_xlim()

    def _pre_reset_all(self):
        self._window_filter.clear()
        subj = self._pre_subj_cb.currentText()
        if subj in self._pre_gt_cache:
            gt_df, c_wi, _, _, _ = self._pre_gt_cache[subj]
            self._window_filter[subj] = {int(r[c_wi]) for _, r in gt_df.iterrows()}
            self._pre_reload_checks(); self._pre_update_count()
            self._pre_update_plot(); self._pre_apply_xlim()
        self._status.showMessage('Filter semua subjek direset — semua window aktif.')

    def _pre_reload_checks(self):
        included = self._window_filter.get(self._pre_subj_cb.currentText(), set())
        self._pre_win_table.blockSignals(True)
        for ri in range(self._pre_win_table.rowCount()):
            chk = self._pre_win_table.item(ri, 0)
            if chk: chk.setCheckState(Qt.Checked if chk.data(Qt.UserRole) in included else Qt.Unchecked)
        self._pre_win_table.blockSignals(False)

    def _pre_update_count(self):
        subj = self._pre_subj_cb.currentText()
        n = len(self._window_filter.get(subj, set()))
        tot = self._pre_win_table.rowCount()
        self._pre_count_lbl.setText(f'{n} / {tot} window dipilih')

    def _pre_update_plot(self):
        subj = self._pre_subj_cb.currentText()
        if subj not in self._pre_sig_cache or subj not in self._pre_gt_cache: return
        t_disp, sig_disp, ch_lbl = self._pre_sig_cache[subj]
        gt_df, c_wi, c_s, c_e, c_gt = self._pre_gt_cache[subj]
        included = self._window_filter.get(subj, set())
        self._pre_canvas.clear_all(); ax = self._pre_canvas.ax
        ax.plot(t_disp, sig_disp, color=C_ACCENT, lw=0.6, alpha=0.85, zorder=1)
        y_lo = float(np.min(sig_disp)); y_hi = float(np.max(sig_disp))
        y_range = (y_hi - y_lo) or 1.0
        for ri, (_, grow) in enumerate(gt_df.iterrows()):
            wi = int(grow[c_wi]); t_s = float(grow[c_s]); t_e = float(grow[c_e]); gt = float(grow[c_gt])
            is_sel = (ri == self._pre_sel_row)
            color, alpha_v, lc, lw = ('#43a047', 0.18, '#2e7d32', 1.0) if wi in included \
                else ('#9e9e9e', 0.12, '#757575', 0.8)
            ax.axvspan(t_s, t_e, alpha=alpha_v, color=color, zorder=2)
            ax.axvline(t_s, color=lc, lw=lw, alpha=0.6, zorder=3)
            ax.axvline(t_e, color=lc, lw=lw, alpha=0.6, zorder=3)
            if is_sel:
                ax.axvspan(t_s, t_e, alpha=0.25, color='#ff8f00', zorder=4)
                for xv in [t_s, t_e]: ax.axvline(xv, color='#e65100', lw=1.8, zorder=5)
            ax.text((t_s + t_e) / 2, y_hi + y_range * 0.02, f'W{wi}\n{gt:.1f}',
                    ha='center', va='bottom', fontsize=7,
                    color='#1b5e20' if wi in included else '#616161', zorder=6)
        ax.set_ylim(y_lo - y_range * 0.05, y_hi + y_range * 0.15)
        ax.set_xlabel('Waktu (s)', color=C_TEXT); ax.set_ylabel(f'PPG {ch_lbl}', color=C_TEXT)
        ax.set_title(f'Sinyal PPG [{ch_lbl}] — {subj}', color=C_TEXT, fontsize=10); ax.grid(True, alpha=0.4)
        self._pre_canvas.fig.tight_layout(pad=1.2); self._pre_canvas.draw()

    def _pre_apply_xlim(self):
        x0 = self._pre_seg_idx * WIN_S_BASE
        self._pre_canvas.ax.set_xlim(x0, x0 + WIN_S_BASE); self._pre_canvas.draw()

    def _pre_update_seg_label(self):
        self._pre_seg_lbl.setText(f'Seg {self._pre_seg_idx + 1} / {self._pre_seg_total}')

    def _pre_update_nav_btns(self):
        self._btn_pre_prev.setEnabled(self._pre_seg_idx > 0)
        self._btn_pre_next.setEnabled(self._pre_seg_idx < self._pre_seg_total - 1)

    def _on_pre_prev(self):
        if self._pre_seg_idx > 0:
            self._pre_seg_idx -= 1; self._pre_update_seg_label()
            self._pre_update_nav_btns(); self._pre_apply_xlim()

    def _on_pre_next(self):
        if self._pre_seg_idx < self._pre_seg_total - 1:
            self._pre_seg_idx += 1; self._pre_update_seg_label()
            self._pre_update_nav_btns(); self._pre_apply_xlim()

    def _pre_view_all(self):
        subj = self._pre_subj_cb.currentText()
        if subj not in self._pre_sig_cache: return
        t_disp, _, _ = self._pre_sig_cache[subj]
        self._pre_canvas.ax.set_xlim(0, t_disp[-1]); self._pre_canvas.draw()

    # =========================================================================
    # TAB ② — ANALISIS
    # =========================================================================
    def _build_tab_analisis(self):
        w = QWidget(); root = QHBoxLayout(w); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(8)

        # Left control panel
        left = QWidget(); left.setFixedWidth(270)
        lv   = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0); lv.setSpacing(8)

        # Algoritma checkboxes
        grp_alg = QGroupBox('ALGORITMA')
        av = QVBoxLayout(grp_alg); av.setSpacing(4)
        self._run_emd_cb  = QCheckBox('PPG_EMD  (BPF → EMD → PSD select)')
        self._run_eemd_cb = QCheckBox('PPG_EEMD (BPF → EEMD → PSD select)')
        self._run_emd_cb.setChecked(True); self._run_eemd_cb.setChecked(True)
        emd_note  = QLabel('✔ PyEMD tersedia' if HAS_EMD  else '⚠ pip install EMD-signal')
        eemd_note = QLabel('✔ PyEMD tersedia' if HAS_EEMD else '⚠ pip install EMD-signal')
        for lbl in [emd_note, eemd_note]:
            lbl.setStyleSheet(f'color:{C_GREEN if HAS_EMD else C_AMBER};font-size:9px;')
        ens3_info = QLabel('PPG_ENS3 (VMD+ENV+PSD voting) — selalu dijalankan')
        ens3_info.setStyleSheet(f'color:{C_TEXT};font-size:9px;'); ens3_info.setWordWrap(True)
        av.addWidget(self._run_emd_cb); av.addWidget(emd_note)
        av.addWidget(self._run_eemd_cb); av.addWidget(eemd_note)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f'color:{C_BG3};'); av.addWidget(sep)
        av.addWidget(ens3_info)

        # Channel
        grp_ch = QGroupBox('KANAL PPG')
        cv = QVBoxLayout(grp_ch); cv.setSpacing(4)
        self._ch_ir_cb  = QCheckBox('IR');  self._ch_ir_cb.setChecked(True)
        self._ch_red_cb = QCheckBox('Red'); self._ch_red_cb.setChecked(True)
        cv.addWidget(self._ch_ir_cb); cv.addWidget(self._ch_red_cb)

        # Window duration
        grp_win = QGroupBox('DURASI WINDOW')
        wv = QVBoxLayout(grp_win); wv.setSpacing(3)
        self._win_checks: dict = {}
        for wval, lbl_txt in [(30.0, '30 detik'), (60.0, '60 detik')]:
            cb = QCheckBox(lbl_txt); cb.setChecked(wval == 30.0)
            wv.addWidget(cb); self._win_checks[wval] = cb

        # Mini stats
        self._mini_stat = QTextEdit(); self._mini_stat.setReadOnly(True)
        self._mini_stat.setFixedHeight(140); self._mini_stat.setPlaceholderText('Belum ada analisis.')
        self._mini_stat.setStyleSheet(
            f'background:{C_BG2};color:{C_TEXT};border:1px solid {C_BG3};'
            f'border-radius:3px;font-size:10px;padding:4px;')

        self._btn_run  = QPushButton('▶  Jalankan Analisis')
        self._btn_save = QPushButton('💾  Simpan Excel')
        self._btn_run.setStyleSheet(_BTN); self._btn_save.setStyleSheet(_BTN)
        self._btn_save.setEnabled(False)
        self._btn_run.clicked.connect(self._on_run)
        self._btn_save.clicked.connect(self._on_save)
        self._progress = QProgressBar(); self._progress.setValue(0)
        self._progress.setFormat('%v / %m windows')



        grp_k = QGroupBox('K  (jumlah mode VMD untuk ENS3)')
        kv = QVBoxLayout(grp_k); kv.setSpacing(4)
        kv.addWidget(QLabel('Integer 2–10:'))
        self._k_input = QLineEdit(str(VMD_K_DEFAULT))
        self._k_input.setPlaceholderText('contoh: 2'); kv.addWidget(self._k_input)

        grp_alpha = QGroupBox('alpha  (bandwidth constraint ENS3)')
        alphav = QVBoxLayout(grp_alpha); alphav.setSpacing(4)
        alphav.addWidget(QLabel('Integer 100–50000:'))
        self._alpha_input = QLineEdit(str(VMD_ALPHA_DEFAULT))
        self._alpha_input.setPlaceholderText('contoh: 250'); alphav.addWidget(self._alpha_input)

        lv.addWidget(grp_alg); lv.addWidget(grp_k); lv.addWidget(grp_alpha)
        lv.addWidget(grp_ch); lv.addWidget(grp_win)
        lv.addWidget(self._mini_stat); lv.addStretch()
        lv.addWidget(self._btn_run)
        lv.addWidget(self._btn_save); lv.addWidget(self._progress)

        # Right: log
        right = QWidget(); rv = QVBoxLayout(right); rv.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel('Log Analisis:')
        lbl.setStyleSheet(f'color:{C_TEXT};font-size:11px;font-weight:bold;margin-bottom:2px;')
        rv.addWidget(lbl)
        self._log = QTextEdit(); self._log.setReadOnly(True); rv.addWidget(self._log)

        root.addWidget(left); root.addWidget(right, stretch=1)
        return w

    # =========================================================================
    # TAB ③ — HASIL
    # =========================================================================
    def _build_tab_hasil(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(4, 4, 4, 4); v.setSpacing(4)
        lbl = QLabel('Hasil per window (semua algoritma dalam satu baris).')
        lbl.setStyleSheet(f'color:{C_TEXT};font-size:10px;margin-bottom:2px;')
        v.addWidget(lbl)
        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        v.addWidget(self._table, stretch=1)
        return w

    # =========================================================================
    # TAB ④ — STATISTIK & PLOT
    # =========================================================================
    def _build_tab_statistik(self):
        w = QWidget(); root = QHBoxLayout(w); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(8)

        # Left: stats tables
        left = QWidget(); lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0); lv.setSpacing(6)
        left.setMinimumWidth(400)

        grp_stat = QGroupBox('STATISTIK KESELURUHAN (per Algoritma × Kanal)')
        sv = QVBoxLayout(grp_stat)
        _SC = ['Algoritma', 'Kanal', 'N Total', 'N Valid', 'MAE (brpm)', 'RMSE (brpm)', 'MAPE (%)', 'SMAPE (%)']
        self._stat_table = QTableWidget(0, len(_SC))
        self._stat_table.setHorizontalHeaderLabels(_SC)
        self._stat_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._stat_table.setAlternatingRowColors(True)
        self._stat_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._stat_table.setMaximumHeight(220); sv.addWidget(self._stat_table)
        lv.addWidget(grp_stat)

        grp_branch = QGroupBox('DISTRIBUSI BRANCH ENS3')
        bv = QVBoxLayout(grp_branch)
        self._branch_table = QTableWidget(0, 4)
        self._branch_table.setHorizontalHeaderLabels(['Branch', 'Kanal', 'Count', '%'])
        self._branch_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._branch_table.setAlternatingRowColors(True)
        self._branch_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._branch_table.setMaximumHeight(240); bv.addWidget(self._branch_table)
        lv.addWidget(grp_branch); lv.addStretch()

        # Right: plots
        right = QWidget(); rv = QVBoxLayout(right); rv.setContentsMargins(0, 0, 0, 0); rv.setSpacing(6)

        ctrl = QWidget(); cl = QHBoxLayout(ctrl); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(8)
        cl.addWidget(QLabel('Algoritma:'))
        self._stat_alg_cb = QComboBox(); self._stat_alg_cb.setMinimumWidth(120); cl.addWidget(self._stat_alg_cb)
        cl.addWidget(QLabel('Kanal:'))
        self._stat_ch_cb = QComboBox(); self._stat_ch_cb.setMinimumWidth(80); cl.addWidget(self._stat_ch_cb)
        self._btn_plot_ba   = QPushButton('Bland-Altman')
        self._btn_plot_err  = QPushButton('Error Distribution')
        self._btn_plot_br   = QPushButton('Branch Pie')
        self._btn_plot_comp = QPushButton('Perbandingan MAE')
        for btn in [self._btn_plot_ba, self._btn_plot_err, self._btn_plot_br, self._btn_plot_comp]:
            btn.setStyleSheet(_BTN); cl.addWidget(btn)
        cl.addStretch(); rv.addWidget(ctrl)
        self._stat_canvas = MplCanvas(); rv.addWidget(self._stat_canvas, stretch=1)

        self._btn_plot_ba.clicked.connect(self._plot_ba)
        self._btn_plot_err.clicked.connect(self._plot_err_dist)
        self._btn_plot_br.clicked.connect(self._plot_branch_pie)
        self._btn_plot_comp.clicked.connect(self._plot_mae_compare)

        root.addWidget(left, stretch=0); root.addWidget(right, stretch=1)
        return w

    # =========================================================================
    def _parse_k(self):
        try:
            v = int(self._k_input.text().strip())
            return v if 2 <= v <= 10 else None
        except ValueError: return None

    def _parse_alpha(self):
        try:
            v = int(self._alpha_input.text().strip())
            return v if 100 <= v <= 50000 else None
        except ValueError: return None

    # =========================================================================
    # ANALYSIS RUN
    # =========================================================================
    def _on_run(self):
        channels = []
        if self._ch_ir_cb.isChecked():  channels.append('IR')
        if self._ch_red_cb.isChecked(): channels.append('Red')
        if not channels:
            self._status.showMessage('Pilih minimal satu kanal.'); return
        vmd_k = self._parse_k()
        if vmd_k is None:
            self._status.showMessage('K tidak valid (2–10).'); return
        vmd_alpha = self._parse_alpha()
        if vmd_alpha is None:
            self._status.showMessage('alpha tidak valid (100–50000).'); return
        win_durs = sorted([wv for wv, cb in self._win_checks.items() if cb.isChecked()])
        if not win_durs:
            self._status.showMessage('Pilih durasi window.'); return

        all_subjects = [s for s in _discover_subjects(BASE_PPG) if s.upper() not in PPG_EXCLUDED]
        run_emd  = self._run_emd_cb.isChecked()
        run_eemd = self._run_eemd_cb.isChecked()

        self._btn_run.setEnabled(False); self._btn_save.setEnabled(False)
        self._log.clear(); self._mini_stat.clear(); self._table.setRowCount(0)
        self._stat_table.setRowCount(0); self._branch_table.setRowCount(0)
        self._progress.setValue(0); self._progress.setMaximum(0); self._df = None

        self._worker = AnalysisWorker(
            all_subjects, channels, win_durs, vmd_k, vmd_alpha,
            run_emd, run_eemd, dict(self._window_filter))
        self._worker.log_signal.connect(self._log.append)
        self._worker.progress_signal.connect(self._on_progress)
        self._worker.done_signal.connect(self._on_done)
        self._worker.start()
        dur_str = ' + '.join(f'{w:.0f}s' for w in win_durs)
        self._status.showMessage(
            f'Memproses {len(all_subjects)} subjek × {len(channels)} kanal × {dur_str} [K={vmd_k}, alpha={vmd_alpha}]...')

    def _on_progress(self, done, total):
        if self._progress.maximum() != total: self._progress.setMaximum(total)
        self._progress.setValue(done)
        self._status.showMessage(f'Memproses {done}/{total} window...')

    def _on_done(self, result):
        self._btn_run.setEnabled(True)
        if not result or all(v is None or len(v) == 0 for v in result.values()):
            self._status.showMessage('Selesai — tidak ada data.'); return
        self._df = result; self._btn_save.setEnabled(True)
        # Tampilkan durasi terkecil sebagai tampilan utama di GUI
        df = next((result[k] for k in sorted(result) if result[k] is not None and len(result[k]) > 0), None)
        if df is None: return
        self._populate_table(df); self._populate_stats(df)
        self._populate_branch_table(df); self._update_mini_stat(df)
        self._populate_stat_controls(df)
        self._main_tabs.setCurrentIndex(2)
        total_rows = sum(len(v) for v in result.values() if v is not None)
        self._status.showMessage(f'Selesai — {total_rows} baris total ({", ".join(f"{k:.0f}s" for k in sorted(result))})')

    # =========================================================================
    # TABLE POPULATION
    # =========================================================================
    def _populate_table(self, df):
        ae_cols = {'ae_emd', 'ae_eemd', 'ae_ens3'}
        self._table.setRowCount(len(df))
        for ri, (_, row) in enumerate(df.iterrows()):
            for ci, col in enumerate(_COLUMNS):
                val = row[col]
                txt = '—' if (val is None or (isinstance(val, float) and np.isnan(val))) else str(val)
                item = QTableWidgetItem(txt); item.setTextAlignment(Qt.AlignCenter)
                if col in ae_cols and txt != '—':
                    try:
                        e = float(txt)
                        bg, fg = _CELL_GOOD if e <= 2.0 else _CELL_WARN if e <= 5.0 else _CELL_BAD
                        item.setBackground(QBrush(bg)); item.setForeground(QBrush(fg))
                    except ValueError: pass
                if col == 'branch':
                    bcolors = {
                        'all_agree':     ('#e8f5e9', C_GREEN),
                        'vmd+env':       ('#e3f2fd', '#1565c0'),
                        'vmd+psd':       ('#e8eaf6', '#283593'),
                        'harmonic->EP':  ('#fff8e1', '#f57f17'),
                        'harmonic->VMD': ('#fce4ec', '#880e4f'),
                        'all_disagree':  ('#ffebee', C_RED),
                        'fallback':      ('#f3e5f5', '#6a1b9a'),
                    }
                    if txt in bcolors:
                        bg_h, fg_h = bcolors[txt]
                        item.setBackground(QBrush(QColor(bg_h)))
                        item.setForeground(QBrush(QColor(fg_h)))
                self._table.setItem(ri, ci, item)

    def _populate_stats(self, df):
        alg_map = {'PPG_VMD': 'ae_vmd', 'PPG_EMD': 'ae_emd', 'PPG_EEMD': 'ae_eemd', 'PPG_ENS3': 'ae_ens3'}
        # Only include alg if column has at least one non-nan
        rows_out = []
        for alg, ae_col in alg_map.items():
            if ae_col not in df.columns: continue
            for ch in sorted(df['channel'].unique()):
                grp = df[df['channel'] == ch].copy()
                grp = grp.rename(columns={ae_col: 'rr_est_bpm',
                                           ae_col.replace('ae_', 'rr_'): 'rr_est_bpm'})
                # compute stats using the error column directly
                sub = df[df['channel'] == ch]
                ae  = sub[ae_col].dropna().values if ae_col in sub.columns else np.array([])
                err_col = ae_col.replace('ae_', 'err_')
                err = sub[err_col].dropna().values if err_col in sub.columns else np.array([])
                n_total = len(sub); n_valid = len(ae)
                if n_valid == 0:
                    rows_out.append([alg, ch, str(n_total), '0', '—', '—', '—', '—'])
                    continue
                mae  = float(np.mean(ae)); rmse = float(np.sqrt(np.mean(err**2))) if len(err) else np.nan
                gt_v = sub.dropna(subset=[ae_col])['gt_rr_bpm'].values
                mape = float(np.mean(ae / gt_v) * 100) if len(gt_v) else np.nan
                denom = np.abs(sub.dropna(subset=[ae_col])['gt_rr_bpm'].values) + \
                        np.abs(sub.dropna(subset=[err_col])[err_col].values + sub.dropna(subset=[ae_col])['gt_rr_bpm'].values) \
                        if err_col in sub.columns else None
                smape_val = '—'
                rows_out.append([alg, ch, str(n_total), str(n_valid),
                                  f'{mae:.4f}', f'{rmse:.4f}' if not np.isnan(rmse) else '—',
                                  f'{mape:.2f}' if not np.isnan(mape) else '—', smape_val])

        self._stat_table.setRowCount(len(rows_out))
        for ri, cells in enumerate(rows_out):
            for ci, val in enumerate(cells):
                item = QTableWidgetItem(val); item.setTextAlignment(Qt.AlignCenter)
                if ci == 4 and val != '—':
                    try:
                        e = float(val)
                        bg, fg = _CELL_GOOD if e <= 2.0 else _CELL_WARN if e <= 5.0 else _CELL_BAD
                        item.setBackground(QBrush(bg)); item.setForeground(QBrush(fg))
                    except ValueError: pass
                self._stat_table.setItem(ri, ci, item)

    def _populate_branch_table(self, df):
        rows = []
        for ch in sorted(df['channel'].unique()):
            sub = df[df['channel'] == ch]
            bc  = sub['branch'].value_counts()
            total = len(sub)
            branch_order = ['all_agree', 'vmd+env', 'vmd+psd', 'harmonic->EP',
                            'harmonic->VMD', 'all_disagree', 'fallback', 'no_data']
            for b in sorted(bc.index, key=lambda x: (branch_order.index(x) if x in branch_order else 99)):
                cnt = int(bc[b]); pct = cnt / total * 100 if total else 0
                rows.append([b, ch, str(cnt), f'{pct:.1f}%'])
        self._branch_table.setRowCount(len(rows))
        for ri, cells in enumerate(rows):
            for ci, val in enumerate(cells):
                item = QTableWidgetItem(val); item.setTextAlignment(Qt.AlignCenter)
                self._branch_table.setItem(ri, ci, item)

    def _populate_stat_controls(self, df):
        algs = ['PPG_VMD', 'PPG_EMD', 'PPG_EEMD', 'PPG_ENS3']
        _ae_map = {'PPG_VMD': 'ae_vmd', 'PPG_EMD': 'ae_emd', 'PPG_EEMD': 'ae_eemd', 'PPG_ENS3': 'ae_ens3'}
        avail_algs = [a for a in algs if _ae_map[a] in df.columns and df[_ae_map[a]].notna().any()]
        self._stat_alg_cb.blockSignals(True); self._stat_alg_cb.clear()
        self._stat_alg_cb.addItems(avail_algs); self._stat_alg_cb.blockSignals(False)
        self._stat_ch_cb.blockSignals(True); self._stat_ch_cb.clear()
        self._stat_ch_cb.addItems(sorted(df['channel'].unique())); self._stat_ch_cb.blockSignals(False)

    def _update_mini_stat(self, df):
        lines = ['── Ringkasan ──']
        alg_map = {'PPG_VMD': 'ae_vmd', 'PPG_EMD': 'ae_emd', 'PPG_EEMD': 'ae_eemd', 'PPG_ENS3': 'ae_ens3'}
        for alg, ae_col in alg_map.items():
            if ae_col not in df.columns: continue
            for ch in sorted(df['channel'].unique()):
                sub = df[df['channel'] == ch]
                ae  = sub[ae_col].dropna().values
                if len(ae) == 0: continue
                err_col = ae_col.replace('ae_', 'err_')
                err = sub[err_col].dropna().values
                lines.append(f'\n{alg} [{ch}]:  MAE={np.mean(ae):.3f}'
                              f'  RMSE={float(np.sqrt(np.mean(err**2))):.3f}'
                              f'  N={len(ae)}/{len(sub)}')
        self._mini_stat.setText('\n'.join(lines))

    # =========================================================================
    # PLOTS
    # =========================================================================
    def _get_ae_col(self):
        alg = self._stat_alg_cb.currentText()
        return {'PPG_VMD': 'ae_vmd', 'PPG_EMD': 'ae_emd', 'PPG_EEMD': 'ae_eemd', 'PPG_ENS3': 'ae_ens3'}.get(alg)

    def _get_err_col(self):
        alg = self._stat_alg_cb.currentText()
        return {'PPG_VMD': 'err_vmd', 'PPG_EMD': 'err_emd', 'PPG_EEMD': 'err_eemd', 'PPG_ENS3': 'err_ens3'}.get(alg)

    def _get_rr_col(self):
        alg = self._stat_alg_cb.currentText()
        return {'PPG_VMD': 'rr_vmd', 'PPG_EMD': 'rr_emd', 'PPG_EEMD': 'rr_eemd', 'PPG_ENS3': 'rr_ens3'}.get(alg)

    def _get_df_view(self):
        if self._df is None: return None
        return next((self._df[k] for k in sorted(self._df) if self._df[k] is not None and len(self._df[k]) > 0), None)

    def _get_sub(self):
        df = self._get_df_view()
        if df is None: return None, None, None
        ch = self._stat_ch_cb.currentText(); alg = self._stat_alg_cb.currentText()
        if not ch or not alg: return None, None, None
        sub = df[df['channel'] == ch].copy()
        return sub, ch, alg

    def _plot_ba(self):
        sub, ch, alg = self._get_sub()
        if sub is None: return
        rr_col = self._get_rr_col(); err_col = self._get_err_col()
        if rr_col is None or rr_col not in sub.columns: return
        sub = sub.dropna(subset=[rr_col, 'gt_rr_bpm'])
        if len(sub) < 2: self._status.showMessage('Data tidak cukup.'); return
        est = sub[rr_col].values.astype(float); gt = sub['gt_rr_bpm'].values.astype(float)
        mean_vals = (est + gt) / 2.0; diff_vals = est - gt
        bias = float(np.mean(diff_vals)); sd = float(np.std(diff_vals, ddof=1))
        loa_hi = bias + 1.96 * sd; loa_lo = bias - 1.96 * sd
        self._stat_canvas.clear_all(); ax = self._stat_canvas.ax
        ax.scatter(mean_vals, diff_vals, color=C_ACCENT, alpha=0.72, s=38,
                   edgecolors='#0d47a1', linewidths=0.5, zorder=3)
        x_lo = float(mean_vals.min()) - 1.5; x_hi = float(mean_vals.max()) + 1.5
        ax.axhline(bias,   color=C_RED,   lw=1.8, linestyle='-',  label=f'Bias = {bias:+.3f}')
        ax.axhline(loa_hi, color=C_AMBER, lw=1.4, linestyle='--', label=f'+1.96 SD = {loa_hi:+.3f}')
        ax.axhline(loa_lo, color=C_AMBER, lw=1.4, linestyle='--', label=f'-1.96 SD = {loa_lo:+.3f}')
        ax.axhline(0.0, color='#90a4ae', lw=0.8, linestyle=':')
        ax.fill_between([x_lo, x_hi], loa_lo, loa_hi, color=C_AMBER, alpha=0.08)
        for y_val, lbl, col in [(bias, f'Bias\n{bias:+.2f}', C_RED),
                                  (loa_hi, f'+LoA\n{loa_hi:+.2f}', C_AMBER),
                                  (loa_lo, f'-LoA\n{loa_lo:+.2f}', C_AMBER)]:
            ax.annotate(lbl, xy=(x_hi, y_val), xytext=(5, 0), textcoords='offset points',
                        va='center', color=col, fontsize=8, annotation_clip=False)
        ax.set_xlim(x_lo, x_hi)
        ax.set_title(f'Bland-Altman — {alg} [{ch}]   (N={len(sub)})', color=C_TEXT, fontsize=11, pad=8)
        ax.set_xlabel('Mean of Estimated & GT RR (brpm)', color=C_TEXT)
        ax.set_ylabel('Estimated − GT RR (brpm)', color=C_TEXT)
        ax.legend(fontsize=9); ax.grid(True)
        self._stat_canvas.fig.tight_layout(pad=1.8); self._stat_canvas.draw()
        self._status.showMessage(f'BA {alg}[{ch}]  Bias={bias:+.3f}  LoA=[{loa_lo:+.3f},{loa_hi:+.3f}]')

    def _plot_err_dist(self):
        sub, ch, alg = self._get_sub()
        if sub is None: return
        ae_col = self._get_ae_col()
        if ae_col is None or ae_col not in sub.columns: return
        ae = sub[ae_col].dropna().values
        if len(ae) == 0: return
        self._stat_canvas.clear_all(); ax = self._stat_canvas.ax
        ax.hist(ae, bins=30, color=C_ACCENT, edgecolor='#0d47a1', alpha=0.85, zorder=2)
        ax.axvline(float(np.mean(ae)),   color=C_RED,   lw=2, linestyle='-',
                   label=f'MAE = {float(np.mean(ae)):.3f}')
        ax.axvline(float(np.median(ae)), color=C_GREEN, lw=2, linestyle='--',
                   label=f'Median = {float(np.median(ae)):.3f}')
        ax.set_title(f'Distribusi |Error| — {alg} [{ch}]  (N={len(ae)})', color=C_TEXT, fontsize=11, pad=8)
        ax.set_xlabel('|Error| (brpm)', color=C_TEXT); ax.set_ylabel('Count', color=C_TEXT)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
        self._stat_canvas.fig.tight_layout(pad=1.8); self._stat_canvas.draw()

    def _plot_branch_pie(self):
        df = self._get_df_view()
        if df is None: return
        ch = self._stat_ch_cb.currentText()
        sub = df[df['channel'] == ch] if ch else df
        counts = sub['branch'].value_counts()
        if counts.empty: return
        palette = {
            'all_agree': '#4caf50', 'vmd+env': '#1976d2', 'vmd+psd': '#3949ab',
            'harmonic->EP': '#ffa000', 'harmonic->VMD': '#e91e63',
            'all_disagree': '#e53935', 'fallback': '#9c27b0', 'no_data': '#9e9e9e',
        }
        self._stat_canvas.clear_all(); ax = self._stat_canvas.ax
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index,
            colors=[palette.get(b, '#607d8b') for b in counts.index],
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 9, 'color': C_TEXT})
        for at in autotexts: at.set_fontsize(8)
        ax.set_title(f'Branch ENS3 [{ch}]  (N={len(sub)})', color=C_TEXT, fontsize=11, pad=12)
        self._stat_canvas.fig.tight_layout(pad=1.8); self._stat_canvas.draw()

    def _plot_mae_compare(self):
        df = self._get_df_view()
        if df is None: return
        alg_map = {'PPG_VMD': 'ae_vmd', 'PPG_EMD': 'ae_emd', 'PPG_EEMD': 'ae_eemd', 'PPG_ENS3': 'ae_ens3'}
        channels = sorted(df['channel'].unique())
        algs     = [a for a, c in alg_map.items() if c in df.columns and df[c].notna().any()]
        if not algs: return
        self._stat_canvas.clear_all(); ax = self._stat_canvas.ax
        x = np.arange(len(channels)); width = 0.20
        colors_bar = ['#9c27b0', C_ACCENT, C_AMBER, C_GREEN]
        for idx, (alg, color) in enumerate(zip(algs, colors_bar)):
            ae_col = alg_map[alg]
            maes = [float(np.mean(df[df['channel'] == ch][ae_col].dropna().values))
                    if len(df[df['channel'] == ch][ae_col].dropna()) > 0 else 0.0
                    for ch in channels]
            bars = ax.bar(x + idx * width, maes, width, label=alg, color=color, alpha=0.85, edgecolor='white')
            for bar, mae in zip(bars, maes):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f'{mae:.2f}', ha='center', va='bottom', fontsize=8, color=C_TEXT)
        ax.set_xticks(x + width * (len(algs) - 1) / 2); ax.set_xticklabels(channels)
        ax.set_xlabel('Kanal PPG', color=C_TEXT); ax.set_ylabel('MAE (brpm)', color=C_TEXT)
        ax.set_title('Perbandingan MAE — VMD vs EMD vs EEMD vs ENS3', color=C_TEXT, fontsize=11, pad=8)
        ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.4)
        self._stat_canvas.fig.tight_layout(pad=1.8); self._stat_canvas.draw()

    # =========================================================================
    # SAVE EXCEL
    # =========================================================================
    def _on_save(self):
        if self._df is None: return
        os.makedirs(OUT_DIR, exist_ok=True)
        ts   = _time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(OUT_DIR, f'RR_PPG_ENS3_{ts}.xlsx')

        alg_map = {'PPG_VMD':  ('rr_vmd',  'ae_vmd',  'err_vmd'),
                   'PPG_EMD':  ('rr_emd',  'ae_emd',  'err_emd'),
                   'PPG_EEMD': ('rr_eemd', 'ae_eemd', 'err_eemd'),
                   'PPG_ENS3': ('rr_ens3', 'ae_ens3', 'err_ens3')}

        def _make_sheets(df):
            rows_sub = []
            for (subj, ch), grp in df.groupby(['subject_id', 'channel'], sort=True):
                for alg, (_, ae_col, err_col) in alg_map.items():
                    if ae_col not in grp.columns: continue
                    ae  = grp[ae_col].dropna().values
                    if len(ae) == 0: continue
                    err = grp[err_col].dropna().values if err_col in grp.columns else np.array([])
                    gt_v = grp.dropna(subset=[ae_col])['gt_rr_bpm'].values
                    mae  = float(np.mean(ae))
                    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else np.nan
                    mape = float(np.mean(ae / gt_v) * 100) if len(gt_v) else np.nan
                    rows_sub.append({'subject_id': subj, 'channel': ch, 'algorithm': alg,
                                     'n_windows': len(grp), 'n_valid': len(ae),
                                     'MAE': round(mae, 4), 'RMSE': round(rmse, 4) if not np.isnan(rmse) else np.nan,
                                     'MAPE': round(mape, 4) if not np.isnan(mape) else np.nan})
            rows_all = []
            for ch in sorted(df['channel'].unique()):
                grp = df[df['channel'] == ch]
                for alg, (_, ae_col, err_col) in alg_map.items():
                    if ae_col not in grp.columns: continue
                    ae  = grp[ae_col].dropna().values
                    if len(ae) == 0: continue
                    err = grp[err_col].dropna().values if err_col in grp.columns else np.array([])
                    gt_v = grp.dropna(subset=[ae_col])['gt_rr_bpm'].values
                    mae  = float(np.mean(ae))
                    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else np.nan
                    mape = float(np.mean(ae / gt_v) * 100) if len(gt_v) else np.nan
                    rows_all.append({'channel': ch, 'algorithm': alg,
                                     'n_windows': len(grp), 'n_valid': len(ae),
                                     'MAE': round(mae, 4), 'RMSE': round(rmse, 4) if not np.isnan(rmse) else np.nan,
                                     'MAPE': round(mape, 4) if not np.isnan(mape) else np.nan})
            rows_branch = []
            for ch in sorted(df['channel'].unique()):
                bc = df[df['channel'] == ch]['branch'].value_counts()
                tot = len(df[df['channel'] == ch])
                for branch, cnt in bc.items():
                    rows_branch.append({'channel': ch, 'branch': branch,
                                        'count': int(cnt), 'pct': round(cnt / tot * 100, 2)})
            return pd.DataFrame(rows_sub), pd.DataFrame(rows_all), pd.DataFrame(rows_branch)

        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            for wd in sorted(self._df.keys()):
                df = self._df[wd]
                if df is None or len(df) == 0: continue
                tag = f'{int(wd)}s'
                df_sub, df_all, df_branch = _make_sheets(df)
                df.to_excel(       writer, sheet_name=f'Raw_{tag}',         index=False)
                df_all.to_excel(   writer, sheet_name=f'Summary_{tag}',     index=False)
                df_sub.to_excel(   writer, sheet_name=f'PerSubjek_{tag}',   index=False)
                df_branch.to_excel(writer, sheet_name=f'Branch_{tag}',      index=False)

        self._log.append(f'\nFile tersimpan: {path}')
        self._status.showMessage(f'Tersimpan: {path}')


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
