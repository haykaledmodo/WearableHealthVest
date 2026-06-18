"""
RR_Estimasi_PPG_BIDMC_ENS3.py
==============================
Evaluasi algoritma PPG_VMD/PPG_EMD/PPG_EEMD/PPG_ENS3 (identik dengan
RR_Estimasi_PPG_ENS3.py) pada dataset sekunder BIDMC (53 subjek ICU,
sinyal PLETH @125 Hz, anotasi napas manual ann1/ann2).

Seluruh pipeline estimasi (VMD, EMD, EEMD, ENV, PSD, voting ENS3_SMART) SAMA
PERSIS dengan program utama — hanya loader data dan ground truth yang
berbeda (BIDMC WFDB format, kanal tunggal PLETH, bukan IR/Red dataset primer).

Tab ① — Analisis   : jalankan PPG_VMD + PPG_EMD + PPG_EEMD + PPG_ENS3 pada 53 subjek
Tab ② — Hasil      : tabel per window dengan branch decision ENS3
Tab ③ — Statistik  : MAE/RMSE/MAPE + Bland-Altman plot
"""

import os, sys, glob, time as _time, warnings
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
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QTabWidget, QProgressBar,
    QAbstractItemView, QStatusBar, QTextEdit, QComboBox, QSizePolicy, QLineEdit,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QBrush

warnings.filterwarnings('ignore')


# =============================================================================
# THEME (identik dgn ENS3 primer/ECG_BIDMC)
# =============================================================================
matplotlib.rcParams.update({
    'figure.facecolor': '#f5f7fa', 'axes.facecolor': '#ffffff',
    'axes.edgecolor':   '#c8d2e0', 'axes.labelcolor': '#1a2233',
    'text.color':       '#1a2233', 'xtick.color':     '#1a2233',
    'ytick.color':      '#1a2233', 'grid.color':       '#dde3ee',
    'grid.linestyle':   '--',      'grid.alpha':        0.6,
    'legend.facecolor': '#f5f7fa', 'legend.edgecolor': '#c8d2e0',
    'font.family':      'monospace', 'font.size':        9,
    'axes.titlesize':   10,          'axes.labelsize':   9,
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
QProgressBar{{background:{C_BG3};border:1px solid {C_BG4};border-radius:3px;text-align:center;}}
QProgressBar::chunk{{background:{C_ACCENT};border-radius:2px;}}
QStatusBar{{background:{C_BG2};color:{C_TEXT2};font-size:10px;border-top:1px solid {C_BG3};}}
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

_BRANCH_COLORS = {
    'all_agree':    QColor('#e3f2fd'),
    'vmd+env':      QColor('#f3e5f5'),
    'vmd+psd':      QColor('#e8f5e9'),
    'harmonic->EP': QColor('#fff3e0'),
    'harmonic->VMD':QColor('#fce4ec'),
    'all_disagree': QColor('#ffebee'),
    'fallback':     QColor('#f5f5f5'),
}


# =============================================================================
# PATHS & CONSTANTS
# =============================================================================
BASE_BIDMC = (r"D:\File Umum\Tugas Akhir (Post Seminar Proposal)"
              r"\Akuisisi Data Tugas Akhir Haykal Nadhif"
              r"\Analisis Data Tugas Akhir Haykal Nadhif"
              r"\Analisis Respiratory Rate\Dataset")
OUT_DIR    = (r"D:\File Umum\Tugas Akhir (Post Seminar Proposal)"
              r"\Akuisisi Data Tugas Akhir Haykal Nadhif"
              r"\Analisis Data Tugas Akhir Haykal Nadhif"
              r"\Analisis Respiratory Rate\Hasil Respiratory Rate PPG BIDMC ENS3")

VMD_K_DEFAULT     = 2
VMD_ALPHA_DEFAULT = 250
RESP_LO           = 0.1    # Hz (6 bpm)
RESP_HI           = 0.5    # Hz (30 bpm)
BPF_LO            = 0.05   # Hz — pre-filter EMD/EEMD
BPF_HI            = 6.0    # Hz — pre-filter EMD/EEMD
MIN_RR_BPM        = 3.0
MAX_RR_BPM        = 35.0
HARMONIC_CUTOFF   = 2.1
AGREE_TOL         = 3.0    # bpm
EEMD_TRIALS       = 10
EEMD_NOISE        = 0.05
PLETH_CHANNEL_CANDIDATES = ['PLETH,', 'PLETH']


def _discover_bidmc_subjects(base_dir):
    if not os.path.isdir(base_dir): return []
    heas = sorted(glob.glob(os.path.join(base_dir, 'bidmc*.hea')))
    subjs = []
    for h in heas:
        name = os.path.splitext(os.path.basename(h))[0]
        if name.endswith('n'): continue
        subjs.append(name)
    return subjs


BIDMC_SUBJECTS = _discover_bidmc_subjects(BASE_BIDMC)


# =============================================================================
# DATA LOADING — BIDMC (WFDB)
# =============================================================================
def load_bidmc_pleth(record_name):
    """Load sinyal PLETH (PPG) dari record BIDMC. Return (ppg_raw, fs) atau None."""
    if not HAS_WFDB: return None
    base = os.path.join(BASE_BIDMC, record_name)
    try:
        rec = wfdb.rdrecord(base)
    except Exception:
        return None
    sig_names = [s.strip() for s in rec.sig_name]
    ch_idx = None
    for cand in PLETH_CHANNEL_CANDIDATES:
        cand_s = cand.strip().rstrip(',')
        for i, sn in enumerate(sig_names):
            if sn.rstrip(',') == cand_s:
                ch_idx = i; break
        if ch_idx is not None: break
    if ch_idx is None: return None
    ppg_raw = rec.p_signal[:, ch_idx].astype(np.float64)
    return ppg_raw, float(rec.fs)


def load_bidmc_breath_gt(record_name):
    """Load anotasi napas (ann1+ann2) BIDMC — identik dengan ECG_BIDMC."""
    if not HAS_WFDB: return None
    base = os.path.join(BASE_BIDMC, record_name)
    try:
        ann = wfdb.rdann(base, 'breath')
    except Exception:
        return None
    fs = float(ann.fs) if ann.fs else 125.0
    times_ann1 = np.array([s for s, a in zip(ann.sample, ann.aux_note) if a == 'ann1']) / fs
    times_ann2 = np.array([s for s, a in zip(ann.sample, ann.aux_note) if a == 'ann2']) / fs
    if len(times_ann1) < 2 and len(times_ann2) < 2:
        return None
    return {'ann1': times_ann1, 'ann2': times_ann2}


def build_bidmc_windows(breath_times_dict, total_dur_s, win_dur_s=30.0):
    """Bangun window non-overlap, GT RR dari breath onset ann1 (fallback ann2)."""
    ann1 = breath_times_dict['ann1']; ann2 = breath_times_dict['ann2']
    windows = []
    n_win = int(total_dur_s // win_dur_s)
    for wi in range(n_win):
        t_s = wi * win_dur_s; t_e = t_s + win_dur_s
        pk1 = ann1[(ann1 >= t_s) & (ann1 < t_e)]
        pk2 = ann2[(ann2 >= t_s) & (ann2 < t_e)]
        pk  = pk1 if len(pk1) >= 2 else pk2
        if len(pk) >= 2:
            span = float(pk[-1] - pk[0])
            gt_rr = ((len(pk) - 1) / span * 60.0) if span > 0 else np.nan
        else:
            gt_rr = np.nan
        if not np.isnan(gt_rr) and MIN_RR_BPM <= gt_rr <= MAX_RR_BPM:
            windows.append(dict(window_idx=wi + 1,
                                 start_s=round(t_s, 1), end_s=round(t_e, 1),
                                 gt_rr_bpm=round(gt_rr, 4)))
    return windows if windows else None


# =============================================================================
# SIGNAL PROCESSING PRIMITIVES  (identik persis dgn ENS3 primer)
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
        return rr_fft
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
# ALGORITHM 1: PPG_EMD  (identik persis)
# =============================================================================
def est_emd(ppg_seg, fs):
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
# ALGORITHM 2: PPG_EEMD  (identik persis)
# =============================================================================
def est_eemd(ppg_seg, fs):
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
# ALGORITHM 3: PPG_ENS3 — tiga sub-estimator  (identik persis)
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


def _est_psd(ppg_seg, fs):
    t0 = _time.perf_counter()
    if len(ppg_seg) < 20: return np.nan, (_time.perf_counter() - t0) * 1e3
    resp_sig = _bandpass(ppg_seg - np.mean(ppg_seg), RESP_LO, RESP_HI, fs)
    try:
        nperseg = min(len(resp_sig), max(64, int(fs * 8)))
        f, pxx = welch(resp_sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    except Exception:
        return np.nan, (_time.perf_counter() - t0) * 1e3
    mask = (f >= RESP_LO) & (f <= RESP_HI)
    if not np.any(mask) or np.all(pxx[mask] == 0):
        return np.nan, (_time.perf_counter() - t0) * 1e3
    f_peak = float(f[mask][np.argmax(pxx[mask])])
    rr = f_peak * 60.0
    return (rr if MIN_RR_BPM <= rr <= MAX_RR_BPM else np.nan), (_time.perf_counter() - t0) * 1e3


def ens3_smart(rr_v, rr_e, rr_p, ratio_cutoff=HARMONIC_CUTOFF, tol=AGREE_TOL):
    """Harmonic-aware pairwise agreement voting — identik persis dgn primer.

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
# TABLE SCHEMA
# =============================================================================
_COLUMNS = [
    'subject_id', 'window_idx', 'start_s', 'end_s', 'gt_rr_bpm',
    'rr_emd', 'err_emd', 'ae_emd', 'ms_emd',
    'rr_eemd', 'err_eemd', 'ae_eemd', 'ms_eemd',
    'rr_vmd', 'err_vmd', 'ae_vmd', 'rr_env', 'rr_psd_sub',
    'rr_ens3', 'err_ens3', 'ae_ens3', 'ms_ens3', 'branch',
    'win_dur_s',
]
_HEADERS = [
    'Subject', 'Win', 'Start(s)', 'End(s)', 'GT(brpm)',
    'EMD(brpm)', 'Err EMD', '|Err| EMD', 'ms EMD',
    'EEMD(brpm)', 'Err EEMD', '|Err| EEMD', 'ms EEMD',
    'VMD sub', 'Err VMD', '|Err| VMD', 'ENV sub', 'PSD sub',
    'ENS3(brpm)', 'Err ENS3', '|Err| ENS3', 'ms ENS3', 'Branch',
    'WinDur(s)',
]
_STAT_COLS = ['Win Dur', 'Algorithm', 'N Total', 'N Valid', 'Coverage(%)',
              'MAE (brpm)', 'RMSE (brpm)', 'MAPE (%)', 'SMAPE (%)', 'Rel. MAE']


# =============================================================================
# ANALYSIS WORKER
# =============================================================================
class AnalysisWorker(QThread):
    log_signal      = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    done_signal     = pyqtSignal(object)

    def __init__(self, vmd_k, vmd_alpha, run_emd, run_eemd):
        super().__init__()
        self.vmd_k     = vmd_k
        self.vmd_alpha = vmd_alpha
        self.run_emd   = run_emd
        self.run_eemd  = run_eemd

    def run(self):
        def log(m): self.log_signal.emit(m)
        subjects = BIDMC_SUBJECTS
        win_durs = [30.0, 60.0]

        algs = ['ENS3']
        if self.run_emd:  algs.insert(0, 'EMD')
        if self.run_eemd: algs.insert(1 if self.run_emd else 0, 'EEMD')

        log(f'Algoritma: {" + ".join(algs)}')
        log(f'ENS3: K={self.vmd_k}  alpha={self.vmd_alpha}  tol={AGREE_TOL} bpm  window=30s + 60s')
        log(f'vmdpy: {"OK" if HAS_VMD else "TIDAK TERSEDIA"}   wfdb: {"OK" if HAS_WFDB else "TIDAK TERSEDIA"}')
        log(f'Subjek BIDMC ditemukan: {len(subjects)}')
        log('─' * 60)

        if not HAS_WFDB:
            log('[FATAL] Package wfdb tidak terpasang. pip install wfdb')
            self.done_signal.emit(None); return

        # Pra-hitung total ops untuk progress bar
        cache = {}
        total_ops = 0
        for subj in subjects:
            ppg_data = load_bidmc_pleth(subj)
            breath   = load_bidmc_breath_gt(subj)
            if ppg_data is None or breath is None:
                cache[subj] = None; continue
            ppg_raw, fs = ppg_data
            total_dur = len(ppg_raw) / fs
            wins_by_dur = {wd: build_bidmc_windows(breath, total_dur, wd) for wd in win_durs}
            cache[subj] = (ppg_raw, fs, wins_by_dur)
            total_ops += sum(len(w or []) for w in wins_by_dur.values())

        done_ops = 0
        rows_by_dur = {30.0: [], 60.0: []}

        for subj in subjects:
            entry = cache.get(subj)
            if entry is None:
                log(f'▶  {subj}  [SKIP] PLETH/anotasi tidak ditemukan.'); continue
            ppg_raw, fs, wins_by_dur = entry
            log(f'▶  {subj}  ({len(ppg_raw):,} spl @ {fs:.0f} Hz)')

            for win_dur_s in win_durs:
                wins = wins_by_dur.get(win_dur_s)
                if wins is None:
                    log(f'   [SKIP] Ground truth tidak cukup ({win_dur_s:.0f}s).'); continue
                log(f'   {len(wins)} window × {win_dur_s:.0f}s')
                rows = rows_by_dur[win_dur_s]

                for win in wins:
                    wi  = win['window_idx']
                    t_s = win['start_s']; t_e = win['end_s']; gt = win['gt_rr_bpm']
                    i_s = int(t_s * fs); i_e = int(t_e * fs)
                    ppg_seg = ppg_raw[i_s:min(i_e, len(ppg_raw))]
                    if len(ppg_seg) < 20:
                        done_ops += 1; self.progress_signal.emit(done_ops, total_ops); continue

                    def _err(rr):
                        se = (rr - gt) if not np.isnan(rr) else np.nan
                        return se, (abs(se) if not np.isnan(se) else np.nan)

                    rr_emd = ms_emd = np.nan
                    if self.run_emd:
                        rr_emd, ms_emd, _ = est_emd(ppg_seg, fs)
                    se_emd, ae_emd = _err(rr_emd)

                    rr_eemd = ms_eemd = np.nan
                    if self.run_eemd:
                        rr_eemd, ms_eemd, _ = est_eemd(ppg_seg, fs)
                    se_eemd, ae_eemd = _err(rr_eemd)

                    rr_ens3, ms_ens3, rr_v, rr_e, rr_p, branch = \
                        est_ens3(ppg_seg, fs, self.vmd_k, self.vmd_alpha)
                    se_ens3, ae_ens3 = _err(rr_ens3)
                    se_vmd, ae_vmd = _err(rr_v)

                    def _r(v): return round(v, 4) if not np.isnan(v) else np.nan

                    rows.append({
                        'subject_id': subj, 'window_idx': wi,
                        'start_s': round(t_s, 1), 'end_s': round(t_e, 1),
                        'gt_rr_bpm': round(gt, 4),
                        'rr_emd':  _r(rr_emd),  'err_emd':  _r(se_emd),  'ae_emd':  _r(ae_emd),
                        'ms_emd':  round(ms_emd, 3) if not np.isnan(ms_emd) else np.nan,
                        'rr_eemd': _r(rr_eemd), 'err_eemd': _r(se_eemd), 'ae_eemd': _r(ae_eemd),
                        'ms_eemd': round(ms_eemd, 3) if not np.isnan(ms_eemd) else np.nan,
                        'rr_vmd': _r(rr_v), 'err_vmd': _r(se_vmd), 'ae_vmd': _r(ae_vmd),
                        'rr_env': _r(rr_e), 'rr_psd_sub': _r(rr_p),
                        'rr_ens3': _r(rr_ens3), 'err_ens3': _r(se_ens3), 'ae_ens3': _r(ae_ens3),
                        'ms_ens3': round(ms_ens3, 3),
                        'branch': branch, 'win_dur_s': win_dur_s,
                    })
                    done_ops += 1
                    self.progress_signal.emit(done_ops, total_ops)
            log('   ✔ selesai')

        log('─' * 60)
        all_rows = rows_by_dur[30.0] + rows_by_dur[60.0]
        log(f'Total baris: {len(all_rows):,}  (30s: {len(rows_by_dur[30.0])}, 60s: {len(rows_by_dur[60.0])})')
        result = {
            'df_30': pd.DataFrame(rows_by_dur[30.0]) if rows_by_dur[30.0] else None,
            'df_60': pd.DataFrame(rows_by_dur[60.0]) if rows_by_dur[60.0] else None,
        }
        self.done_signal.emit(result)


# =============================================================================
# CANVAS HELPERS
# =============================================================================
class MplCanvas(FigureCanvas):
    def __init__(self, nrows=1, ncols=1, figsize=None):
        kw = dict(facecolor=C_BG, tight_layout=True)
        if figsize: kw['figsize'] = figsize
        self.fig = Figure(**kw)
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
        self.setWindowTitle('Estimasi RR — PPG ENS3 (Dataset BIDMC, kanal PLETH)')
        self.resize(1550, 950); self.setStyleSheet(_SS)
        self._df     = None
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(4, 4, 4, 4)
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_tab_analysis(), '  ① Analisis  ')
        self._tabs.addTab(self._build_tab_result(),   '  ② Hasil  ')
        self._tabs.addTab(self._build_tab_stat(),     '  ③ Statistik & Plot  ')
        root.addWidget(self._tabs)
        self._status = QStatusBar(); self.setStatusBar(self._status)
        self._status.showMessage('Siap. Klik Jalankan Analisis untuk menguji pada 53 subjek BIDMC (kanal PLETH).')

    # =========================================================================
    # TAB ① ANALISIS
    # =========================================================================
    def _build_tab_analysis(self):
        w = QWidget(); root = QHBoxLayout(w)
        root.setContentsMargins(8, 8, 8, 8); root.setSpacing(8)

        left = QWidget(); left.setFixedWidth(270)
        lv   = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0); lv.setSpacing(8)

        grp_alg = QGroupBox('ALGORITMA (identik dgn ENS3 primer)')
        av = QVBoxLayout(grp_alg); av.setSpacing(3)
        self._run_emd_cb  = self._mk_checkbox('PPG_EMD  (BPF → EMD → PSD select)', True)
        self._run_eemd_cb = self._mk_checkbox('PPG_EEMD (BPF → EEMD → PSD select)', True)
        av.addWidget(self._run_emd_cb); av.addWidget(self._run_eemd_cb)
        for line in [
            '', 'PPG_ENS3 (selalu dijalankan):',
            '  VMD : VMD pada PPG → IMF → RR',
            '  ENV : amplitudo envelope (Hilbert)',
            '  PSD : Welch PSD dominan',
            '  Voting: pairwise agree (3 bpm tol)',
            '  Guard : harmonic ratio ≥ 2.1',
            '', 'Window: 30s dan 60s (simultan)',
            'Dataset: BIDMC (kanal PLETH @125Hz)',
            'GT: anotasi napas manual ann1/ann2',
        ]:
            lbl = QLabel(line); lbl.setStyleSheet(f'color:{C_TEXT};font-size:10px;')
            av.addWidget(lbl)
        vmd_lbl = QLabel('vmdpy: tersedia' if HAS_VMD else 'vmdpy: TIDAK TERSEDIA')
        vmd_lbl.setStyleSheet(f'color:{C_GREEN if HAS_VMD else C_AMBER};font-size:10px;font-weight:bold;')
        av.addWidget(vmd_lbl)
        emd_lbl = QLabel('PyEMD: tersedia' if (HAS_EMD and HAS_EEMD) else 'PyEMD: TIDAK TERSEDIA')
        emd_lbl.setStyleSheet(f'color:{C_GREEN if (HAS_EMD and HAS_EEMD) else C_AMBER};font-size:10px;font-weight:bold;')
        av.addWidget(emd_lbl)
        wfdb_lbl = QLabel('wfdb: tersedia' if HAS_WFDB else 'wfdb: TIDAK TERSEDIA (pip install wfdb)')
        wfdb_lbl.setStyleSheet(f'color:{C_GREEN if HAS_WFDB else C_AMBER};font-size:10px;font-weight:bold;')
        av.addWidget(wfdb_lbl)

        grp_vmd = QGroupBox('PARAMETER VMD')
        vv = QVBoxLayout(grp_vmd); vv.setSpacing(4)
        k_row = QWidget(); kl = QHBoxLayout(k_row); kl.setContentsMargins(0,0,0,0)
        kl.addWidget(QLabel('K :')); self._k_input = QLineEdit(str(VMD_K_DEFAULT))
        self._k_input.setFixedWidth(60); kl.addWidget(self._k_input); kl.addStretch()
        a_row = QWidget(); al = QHBoxLayout(a_row); al.setContentsMargins(0,0,0,0)
        al.addWidget(QLabel('α :')); self._alpha_input = QLineEdit(str(VMD_ALPHA_DEFAULT))
        self._alpha_input.setFixedWidth(80); al.addWidget(self._alpha_input); al.addStretch()
        vv.addWidget(k_row); vv.addWidget(a_row)

        grp_subj = QGroupBox('SUBJEK')
        sv = QVBoxLayout(grp_subj); sv.setSpacing(2)
        lbl_s = QLabel(f'Ditemukan: {len(BIDMC_SUBJECTS)} subjek BIDMC')
        lbl_s.setStyleSheet(f'color:{C_TEXT};font-size:10px;')
        sv.addWidget(lbl_s)

        self._mini_stat = QTextEdit()
        self._mini_stat.setReadOnly(True); self._mini_stat.setFixedHeight(100)
        self._mini_stat.setPlaceholderText('Statistik ringkas muncul setelah analisis.')
        self._mini_stat.setStyleSheet(
            f'background:{C_BG2};color:{C_TEXT};border:1px solid {C_BG3};'
            f'border-radius:3px;font-size:10px;padding:4px;')

        self._btn_run  = QPushButton('  Jalankan Analisis')
        self._btn_save = QPushButton('  Simpan Hasil (.xlsx)')
        self._btn_run.setStyleSheet(_BTN); self._btn_save.setStyleSheet(_BTN)
        self._btn_save.setEnabled(False)
        self._btn_run.clicked.connect(self._on_run)
        self._btn_save.clicked.connect(self._on_save)
        self._progress = QProgressBar(); self._progress.setValue(0)

        lv.addWidget(grp_alg); lv.addWidget(grp_vmd); lv.addWidget(grp_subj)
        lv.addWidget(self._mini_stat); lv.addStretch()
        lv.addWidget(self._btn_run); lv.addWidget(self._btn_save)
        lv.addWidget(self._progress)

        right = QWidget(); rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0); rv.setSpacing(4)
        lbl2 = QLabel('Log:'); lbl2.setStyleSheet(f'color:{C_TEXT};font-weight:bold;')
        self._log = QTextEdit(); self._log.setReadOnly(True)
        self._log.setStyleSheet(
            f'background:{C_BG2};color:{C_TEXT};border:1px solid {C_BG3};'
            f'border-radius:3px;font-size:10px;font-family:Consolas,monospace;')
        rv.addWidget(lbl2); rv.addWidget(self._log)

        root.addWidget(left); root.addWidget(right, stretch=1)
        return w

    def _mk_checkbox(self, text, checked):
        from PyQt5.QtWidgets import QCheckBox
        cb = QCheckBox(text); cb.setChecked(checked)
        return cb

    # =========================================================================
    # TAB ② HASIL
    # =========================================================================
    def _build_tab_result(self):
        w = QWidget(); v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8); v.setSpacing(4)

        top = QWidget(); tl = QHBoxLayout(top); tl.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel('Hasil per window — warna Branch = jalur keputusan ENS3_SMART:')
        lbl.setStyleSheet(f'color:{C_TEXT};font-weight:bold;font-size:11px;')
        tl.addWidget(lbl); tl.addStretch()
        for branch, desc in [
            ('all_agree','Semua agree'), ('vmd+env','VMD+ENV'),
            ('vmd+psd','VMD+PSD'), ('harmonic->EP','Harmonic→EP'),
            ('harmonic->VMD','Harmonic→VMD'), ('all_disagree','All disagree'),
        ]:
            chip = QLabel(f'  {desc}  ')
            chip.setStyleSheet(
                f'background:{_BRANCH_COLORS[branch].name()};color:{C_TEXT};'
                f'font-size:9px;border-radius:3px;padding:2px 4px;')
            tl.addWidget(chip)
        v.addWidget(top)

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        v.addWidget(self._table)
        return w

    # =========================================================================
    # TAB ③ STATISTIK & PLOT
    # =========================================================================
    def _build_tab_stat(self):
        w = QWidget(); v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8); v.setSpacing(6)

        lbl = QLabel('Ringkasan statistik per algoritma:')
        lbl.setStyleSheet(f'color:{C_TEXT};font-weight:bold;font-size:11px;')
        v.addWidget(lbl)
        self._stat_table = QTableWidget(0, len(_STAT_COLS))
        self._stat_table.setHorizontalHeaderLabels(_STAT_COLS)
        self._stat_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._stat_table.setAlternatingRowColors(True)
        self._stat_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._stat_table.setMaximumHeight(160)
        v.addWidget(self._stat_table)

        branch_lbl = QLabel('Distribusi branch ENS3_SMART:')
        branch_lbl.setStyleSheet(f'color:{C_TEXT};font-weight:bold;font-size:11px;')
        v.addWidget(branch_lbl)
        self._branch_table = QTableWidget(0, 4)
        self._branch_table.setHorizontalHeaderLabels(
            ['Branch', 'N Windows', 'MAE ENS3 (brpm)', 'MAE VMD (brpm)'])
        self._branch_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._branch_table.setAlternatingRowColors(True)
        self._branch_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._branch_table.setMaximumHeight(180)
        v.addWidget(self._branch_table)

        lbl2 = QLabel('Bland-Altman Plot:')
        lbl2.setStyleSheet(f'color:{C_TEXT};font-weight:bold;font-size:11px;')
        v.addWidget(lbl2)
        plot_ctrl = QWidget(); pc = QHBoxLayout(plot_ctrl); pc.setContentsMargins(0,0,0,0); pc.setSpacing(6)
        pc.addWidget(QLabel('Durasi:'))
        self._dur_plot_cb = QComboBox(); self._dur_plot_cb.setFixedWidth(90)
        self._dur_plot_cb.currentIndexChanged.connect(self._refresh_plot)
        pc.addWidget(self._dur_plot_cb)
        pc.addWidget(QLabel('Algoritma:'))
        self._alg_plot_cb = QComboBox()
        self._alg_plot_cb.currentIndexChanged.connect(self._refresh_plot)
        pc.addWidget(self._alg_plot_cb); pc.addStretch()
        v.addWidget(plot_ctrl)
        self._canvas = MplCanvas()
        self._canvas.setMinimumHeight(260)
        v.addWidget(self._canvas, stretch=1)
        return w

    # =========================================================================
    # RUN / LOG / PROGRESS
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

    def _on_run(self):
        vmd_k = self._parse_k()
        if vmd_k is None:
            self._status.showMessage('K tidak valid (2–10).'); return
        vmd_alpha = self._parse_alpha()
        if vmd_alpha is None:
            self._status.showMessage('alpha tidak valid (100–50000).'); return

        self._btn_run.setEnabled(False); self._btn_save.setEnabled(False)
        self._log.clear(); self._mini_stat.clear()
        self._table.setRowCount(0); self._stat_table.setRowCount(0)
        self._branch_table.setRowCount(0)
        self._progress.setValue(0); self._progress.setMaximum(0)
        self._alg_plot_cb.clear()

        run_emd  = self._run_emd_cb.isChecked()
        run_eemd = self._run_eemd_cb.isChecked()

        self._worker = AnalysisWorker(vmd_k, vmd_alpha, run_emd, run_eemd)
        self._worker.log_signal.connect(self._on_log)
        self._worker.progress_signal.connect(self._on_progress)
        self._worker.done_signal.connect(self._on_done)
        self._worker.start()

    def _on_log(self, msg): self._log.append(msg)

    def _on_progress(self, done, total):
        if self._progress.maximum() != total: self._progress.setMaximum(total)
        self._progress.setValue(done)
        self._status.showMessage(f'Memproses {done}/{total} windows...')

    def _on_done(self, result):
        self._btn_run.setEnabled(True)
        if result is None or (result.get('df_30') is None and result.get('df_60') is None):
            self._status.showMessage('Selesai — tidak ada data.'); return
        self._df = result
        self._btn_save.setEnabled(True)

        df = result.get('df_60') if result.get('df_60') is not None else result.get('df_30')

        self._table.setRowCount(len(df))
        for ri, (_, row) in enumerate(df.iterrows()):
            branch_val   = str(row.get('branch', ''))
            branch_color = _BRANCH_COLORS.get(branch_val, None)
            for ci, col in enumerate(_COLUMNS):
                val = row.get(col, None)
                txt = '—' if (val is None or (isinstance(val, float) and np.isnan(val))) \
                      else str(val)
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 'ae_ens3' and txt != '—':
                    try:
                        e = float(txt)
                        bg, fg = (_CELL_GOOD if e <= 2.0 else
                                  _CELL_WARN if e <= 5.0 else _CELL_BAD)
                        item.setBackground(QBrush(bg)); item.setForeground(QBrush(fg))
                    except Exception: pass
                elif col == 'branch' and branch_color is not None:
                    item.setBackground(QBrush(branch_color))
                self._table.setItem(ri, ci, item)

        alg_map = {'PPG_VMD': 'rr_vmd', 'PPG_EMD': 'rr_emd', 'PPG_EEMD': 'rr_eemd', 'PPG_ENS3': 'rr_ens3'}
        stat_rows = []; mini_lines = []
        for dur_label, dur_key in [('60s', 'df_60'), ('30s', 'df_30')]:
            df_dur = result.get(dur_key)
            if df_dur is None: continue
            for alg, est_col in alg_map.items():
                if est_col not in df_dur.columns: continue
                n_tot, n_val, mae, rmse, mape, smape, rel_mae = compute_stats(df_dur, est_col)
                if n_val == 0 and df_dur[est_col].isna().all(): continue
                cov = round(n_val / n_tot * 100, 1) if n_tot > 0 else 0.0
                stat_rows.append({
                    'Win Dur': dur_label, 'Algorithm': alg, 'N Total': n_tot, 'N Valid': n_val,
                    'Coverage(%)': cov,
                    'MAE (brpm)':  round(mae,     4) if not np.isnan(mae)     else np.nan,
                    'RMSE (brpm)': round(rmse,    4) if not np.isnan(rmse)    else np.nan,
                    'MAPE (%)':    round(mape,    2) if not np.isnan(mape)    else np.nan,
                    'SMAPE (%)':   round(smape,   2) if not np.isnan(smape)   else np.nan,
                    'Rel. MAE':    round(rel_mae, 4) if not np.isnan(rel_mae) else np.nan,
                })
                mae_s = f'{mae:.4f}' if not np.isnan(mae) else '—'
                mini_lines.append(f'[{dur_label}] {alg}: MAE={mae_s}  N={n_val}/{n_tot}')

        self._stat_table.setRowCount(len(stat_rows))
        for ri, sr in enumerate(stat_rows):
            vals = [sr['Win Dur'], str(sr['Algorithm']), str(sr['N Total']), str(sr['N Valid']),
                    f'{sr["Coverage(%)"]:.1f}',
                    f'{sr["MAE (brpm)"]:.4f}'  if not np.isnan(sr['MAE (brpm)'])  else '—',
                    f'{sr["RMSE (brpm)"]:.4f}' if not np.isnan(sr['RMSE (brpm)']) else '—',
                    f'{sr["MAPE (%)"]:.2f}'    if not np.isnan(sr['MAPE (%)'])    else '—',
                    f'{sr["SMAPE (%)"]:.2f}'   if not np.isnan(sr['SMAPE (%)'])   else '—',
                    f'{sr["Rel. MAE"]:.4f}'    if not np.isnan(sr['Rel. MAE'])    else '—']
            for ci, val in enumerate(vals):
                item = QTableWidgetItem(val); item.setTextAlignment(Qt.AlignCenter)
                if sr['Algorithm'] == 'PPG_ENS3':
                    item.setBackground(QBrush(QColor('#e8f5e9')))
                    item.setForeground(QBrush(QColor(C_GREEN)))
                elif sr['Algorithm'] == 'PPG_VMD':
                    item.setBackground(QBrush(QColor('#e3f2fd')))
                    item.setForeground(QBrush(QColor('#1565c0')))
                self._stat_table.setItem(ri, ci, item)

        if 'branch' in df.columns:
            bc = df['branch'].value_counts()
            branch_order = ['all_agree', 'vmd+env', 'vmd+psd', 'harmonic->EP',
                             'harmonic->VMD', 'all_disagree', 'fallback', 'no_data']
            ordered = sorted(bc.index, key=lambda x: (branch_order.index(x) if x in branch_order else 99))
            self._branch_table.setRowCount(len(ordered))
            for ri, bname in enumerate(ordered):
                sub = df[df['branch'] == bname]
                mae_ens3 = sub['ae_ens3'].dropna().mean() if 'ae_ens3' in sub.columns else np.nan
                mae_vmd  = sub['ae_vmd'].dropna().mean()  if 'ae_vmd'  in sub.columns else np.nan
                vals = [bname, str(int(bc[bname])),
                        f'{mae_ens3:.4f}' if not np.isnan(mae_ens3) else '—',
                        f'{mae_vmd:.4f}'  if not np.isnan(mae_vmd)  else '—']
                color = _BRANCH_COLORS.get(bname, None)
                for ci, val in enumerate(vals):
                    item = QTableWidgetItem(val); item.setTextAlignment(Qt.AlignCenter)
                    if color: item.setBackground(QBrush(color))
                    self._branch_table.setItem(ri, ci, item)

        self._mini_stat.setPlainText('\n'.join(mini_lines))

        self._dur_plot_cb.blockSignals(True); self._dur_plot_cb.clear()
        if result.get('df_60') is not None: self._dur_plot_cb.addItem('60s', userData='df_60')
        if result.get('df_30') is not None: self._dur_plot_cb.addItem('30s', userData='df_30')
        self._dur_plot_cb.blockSignals(False)

        self._alg_plot_cb.blockSignals(True); self._alg_plot_cb.clear()
        for alg, est_col in alg_map.items():
            if est_col in df.columns and df[est_col].notna().any():
                self._alg_plot_cb.addItem(alg)
        self._alg_plot_cb.blockSignals(False)
        self._refresh_plot()

        n30 = len(result['df_30']) if result.get('df_30') is not None else 0
        n60 = len(result['df_60']) if result.get('df_60') is not None else 0
        self._tabs.setCurrentIndex(2)
        self._status.showMessage(f'Selesai — 30s: {n30} baris  |  60s: {n60} baris')

    # =========================================================================
    # BLAND-ALTMAN
    # =========================================================================
    def _refresh_plot(self, _=None):
        if self._df is None: return
        alg     = self._alg_plot_cb.currentText()
        dur_key = self._dur_plot_cb.currentData()
        if not alg or not dur_key: return
        df = self._df.get(dur_key)
        if df is None: return
        alg_map = {'PPG_VMD': 'rr_vmd', 'PPG_EMD': 'rr_emd', 'PPG_EEMD': 'rr_eemd', 'PPG_ENS3': 'rr_ens3'}
        est_col = alg_map.get(alg)
        if est_col is None or est_col not in df.columns: return
        gt    = df['gt_rr_bpm'].values.astype(np.float64)
        est   = df[est_col].values.astype(np.float64)
        valid = ~np.isnan(gt) & ~np.isnan(est)
        gt_v  = gt[valid]; est_v = est[valid]

        self._canvas.clear_all(); ax = self._canvas.ax
        if len(gt_v) == 0:
            ax.text(0.5, 0.5, 'Tidak ada data valid', ha='center', va='center',
                    transform=ax.transAxes, color=C_TEXT)
            self._canvas.draw(); return

        mean_vals = (gt_v + est_v) / 2.0
        diff_vals = est_v - gt_v
        mean_d    = float(np.mean(diff_vals))
        std_d     = float(np.std(diff_vals))
        loa_hi    = mean_d + 1.96 * std_d
        loa_lo    = mean_d - 1.96 * std_d

        if alg == 'PPG_ENS3' and 'branch' in df.columns:
            branches = df['branch'].values[valid]
            for branch, color in [
                ('all_agree','#1565c0'), ('harmonic->EP','#e65100'),
                ('harmonic->VMD','#6a1b9a'), ('vmd+env','#2e7d32'),
                ('vmd+psd','#00695c'), ('all_disagree','#b71c1c'), ('fallback','#546e7a'),
            ]:
                mask = branches == branch
                if np.any(mask):
                    ax.scatter(mean_vals[mask], diff_vals[mask], alpha=0.7, s=22,
                               color=color, label=branch, zorder=3)
        else:
            ax.scatter(mean_vals, diff_vals, alpha=0.5, s=20, color=C_ACCENT, zorder=3)

        ax.axhline(mean_d, color=C_GREEN, linewidth=1.8, linestyle='-',
                   label=f'Bias = {mean_d:+.2f}')
        ax.axhline(loa_hi, color=C_RED, linewidth=1.3, linestyle='--',
                   label=f'+1.96σ = {loa_hi:+.2f}')
        ax.axhline(loa_lo, color=C_RED, linewidth=1.3, linestyle='--',
                   label=f'-1.96σ = {loa_lo:+.2f}')
        ax.axhline(0, color=C_TEXT2, linewidth=0.8, linestyle=':')
        ax.set_xlabel('Mean (GT + Est) / 2  (brpm)')
        ax.set_ylabel('Est − GT  (brpm)')
        ax.set_title(f'Bland-Altman — {alg}  [{self._dur_plot_cb.currentText()}]  (n={len(gt_v)})', color=C_TEXT)
        ax.legend(fontsize=8, loc='upper right'); ax.grid(True)
        self._canvas.fig.tight_layout(pad=1.2)
        self._canvas.draw()

    # =========================================================================
    # SAVE
    # =========================================================================
    def _on_save(self):
        if self._df is None: return
        os.makedirs(OUT_DIR, exist_ok=True)
        vmd_k     = self._parse_k()     or VMD_K_DEFAULT
        vmd_alpha = self._parse_alpha() or VMD_ALPHA_DEFAULT
        ts  = _time.strftime('%Y%m%d_%H%M%S')
        tag = f'BIDMC_PPG_ENS3_K{vmd_k}a{vmd_alpha}_{ts}'
        out = os.path.join(OUT_DIR, f'{tag}.xlsx')

        alg_map = {'PPG_VMD': 'rr_vmd', 'PPG_EMD': 'rr_emd', 'PPG_EEMD': 'rr_eemd', 'PPG_ENS3': 'rr_ens3'}

        def _make_summary(df):
            rows = []
            for alg, est_col in alg_map.items():
                if est_col not in df.columns: continue
                n_tot, n_val, mae, rmse, mape, smape, rel_mae = compute_stats(df, est_col)
                cov = round(n_val / n_tot * 100, 1) if n_tot > 0 else 0.0
                rows.append({
                    'Algorithm': alg, 'N_Total': n_tot, 'N_Valid': n_val,
                    'Coverage(%)': cov, 'MAE(brpm)': mae, 'RMSE(brpm)': rmse,
                    'MAPE(%)': mape, 'SMAPE(%)': smape, 'RelMAE': rel_mae,
                })
            return pd.DataFrame(rows)

        df30 = self._df.get('df_30')
        df60 = self._df.get('df_60')

        with pd.ExcelWriter(out, engine='openpyxl') as writer:
            if df60 is not None:
                df60.to_excel(writer, sheet_name='Raw_60s', index=False)
                _make_summary(df60).to_excel(writer, sheet_name='Summary_60s', index=False)
            if df30 is not None:
                df30.to_excel(writer, sheet_name='Raw_30s', index=False)
                _make_summary(df30).to_excel(writer, sheet_name='Summary_30s', index=False)

        self._status.showMessage(f'Tersimpan: {out}')
        self._log.append(f'\n[SAVED] {out}')


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
