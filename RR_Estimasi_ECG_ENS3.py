"""
RR_Estimasi_ECG_ENS3.py
========================
Estimasi Respiratory Rate dari sinyal ECG menggunakan ENS3_SMART.

Tab ① — Analisis   : jalankan ENS3_SMART + ECG_VMD pada seluruh subjek
Tab ② — Hasil      : tabel per window dengan branch decision
Tab ③ — Statistik  : MAE/RMSE/MAPE + Bland-Altman plot
Tab ④ — ECG Viewer : lihat sinyal ECG + R-peak + tachogram per window
"""

import os, sys, glob, json, time as _time, warnings
import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import butter, filtfilt, find_peaks, resample, welch
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d
from scipy.stats import kurtosis as sp_kurtosis

try:
    from vmdpy import VMD
    HAS_VMD = True
except ImportError:
    HAS_VMD = False

try:
    import openpyxl  # noqa
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QTabWidget, QProgressBar,
    QAbstractItemView, QStatusBar, QTextEdit, QComboBox, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QBrush

warnings.filterwarnings('ignore')


# =============================================================================
# THEME
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
    'all_agree':     QColor('#e3f2fd'),
    'vmd+qrs':       QColor('#f3e5f5'),
    'vmd+ampl':      QColor('#e8f5e9'),
    'harmonic→QA':   QColor('#fff3e0'),
    'harmonic→VMD':  QColor('#fce4ec'),
    'all_disagree':  QColor('#ffebee'),
    'fallback':      QColor('#f5f5f5'),
    # branch khusus ENS3_AMPL (anchor R-AMPL, lihat ens3_smart_ampl_anchor)
    'qrs+ampl':      QColor('#e0f2f1'),
    'harmonic→VQ':   QColor('#ede7f6'),
    'harmonic→AMPL': QColor('#fbe9e7'),
}


# =============================================================================
# PATHS & CONSTANTS
# =============================================================================
BASE_PPG = (r"D:\File Umum\Tugas Akhir (Post Seminar Proposal)"
            r"\Akuisisi Data Tugas Akhir Haykal Nadhif"
            r"\Database Keseluruhan (Sudah Digabung Semua)")
BASE_GT  = (r"D:\File Umum\Tugas Akhir (Post Seminar Proposal)"
            r"\Akuisisi Data Tugas Akhir Haykal Nadhif"
            r"\Ekstraksi Ground Truth_Respiratory Rate"
            r"\Data Ground Truth Respiratory Rate")
OUT_DIR  = (r"D:\File Umum\Tugas Akhir (Post Seminar Proposal)"
            r"\Akuisisi Data Tugas Akhir Haykal Nadhif"
            r"\Analisis Data Tugas Akhir Haykal Nadhif"
            r"\Analisis Respiratory Rate\Hasil Respiratory Rate ECG ENS3")

ECG_EXCLUDED    = {'SUBJECT_001', 'SUBJECT_006', 'SUBJECT_007', 'SUBJECT_018'}
TARGET_FS       = 300.0
WIN_SAMPLES_ECG = int(10.0 * TARGET_FS)
TACHOGRAM_FS    = 4.0
RESP_LO         = 0.1
RESP_HI         = 0.5
WIN_S_BASE      = 30.0
MIN_RR_BPM      = 3.0
MAX_RR_BPM      = 35.0
VMD_K           = 3
VMD_ALPHA       = 1250
HARMONIC_CUTOFF = 2.1
AGREE_TOL       = 3.0


def _discover_subjects(base_dir):
    if not os.path.isdir(base_dir):
        return [f'SUBJECT_{i:03d}' for i in range(1, 26)]
    entries = sorted(e for e in os.listdir(base_dir)
                     if e.upper().startswith('SUBJECT_') and
                     os.path.isdir(os.path.join(base_dir, e)))
    return entries if entries else [f'SUBJECT_{i:03d}' for i in range(1, 26)]


SUBJECTS = _discover_subjects(BASE_PPG)


# =============================================================================
# DATA LOADING
# =============================================================================
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

def _col(df, candidates):
    cl = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cl: return cl[c.lower()]
    return None

def load_ecg_sensor(folder):
    files = glob.glob(os.path.join(folder, '*_full_ECG*.csv'))
    if not files: return None
    df = pd.read_csv(files[0])
    if 'ts_mcu_ms' not in df.columns or 'ecg_raw' not in df.columns: return None
    mcu = _unwrap_mcu(df['ts_mcu_ms'].values)
    t   = (mcu - mcu[0]) / 1000.0
    return t, df['ecg_raw'].values.astype(np.float64), _estimate_fs(t)

def build_analysis_windows(gt_folder, win_dur_s=60.0):
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
    windows = []; rows = list(gt_df.iterrows()); i = 0; wi_out = 1
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
                        pts = json.loads(raw) if isinstance(raw, str) else []
                        all_peaks.extend([float(p) for p in pts])
                    except Exception: pass
            all_peaks.sort()
            if len(all_peaks) >= 2:
                span = float(all_peaks[-1] - all_peaks[0])
                gt_rr = ((len(all_peaks) - 1) / span * 60.0) if span > 0 else 0.0
            else:
                gt_rr = float(np.mean([float(r[c_gt]) for r in chunk]))
        windows.append(dict(window_idx=wi_out,
                            start_s=round(t_s, 1), end_s=round(t_e, 1),
                            gt_rr_bpm=round(gt_rr, 4)))
        i += step; wi_out += 1
    return windows if windows else None


# =============================================================================
# R-PEAK DETECTION
# =============================================================================
def _ecg_process_window(segment, start_idx_global, last_r_peak, last_r_slope,
                         thresh_mult=0.75):
    nyq  = 0.5 * TARGET_FS
    b, a = butter(2, [5 / nyq, 20 / nyq], btype='bandpass')
    filt = filtfilt(b, a, segment)
    coeffs = pywt.wavedec(filt, 'sym4', level=5)
    n  = len(segment)
    d3 = pywt.upcoef('d', coeffs[3], 'sym4', level=3, take=n)
    d4 = pywt.upcoef('d', coeffs[4], 'sym4', level=4, take=n)
    d5 = pywt.upcoef('d', coeffs[5], 'sym4', level=5, take=n)
    chosen   = d3 + d4 + d5
    diff_sig = np.append(np.diff(chosen), 0)
    squared  = diff_sig ** 2
    mwi      = uniform_filter1d(squared, size=int(0.15 * TARGET_FS))
    lmean    = uniform_filter1d(mwi,     size=int(1.5  * TARGET_FS))
    thresh   = lmean * thresh_mult
    rough, _ = find_peaks(mwi, height=thresh, distance=int(0.20 * TARGET_FS))
    r_peaks, slopes = [], []
    sw = int(0.10 * TARGET_FS)
    for pk in rough:
        s = max(0, pk - sw); e = min(n, pk + sw)
        if s >= e: continue
        seg_loc = filt[s:e]
        pos_max = float(np.max(np.maximum(seg_loc, 0.0)))
        abs_max = float(np.max(np.abs(seg_loc)))
        exact   = s + (int(np.argmax(np.maximum(seg_loc, 0.0)))
                       if pos_max >= 0.25 * abs_max
                       else int(np.argmax(np.abs(seg_loc))))
        eg = start_idx_global + exact
        sl = float(np.max(np.abs(diff_sig[s:e])))
        cur_last  = r_peaks[-1] if r_peaks else last_r_peak
        cur_slope = slopes[-1]  if slopes  else last_r_slope
        if cur_last == -9999:
            r_peaks.append(eg); slopes.append(sl); continue
        jms = ((eg - cur_last) / TARGET_FS) * 1000.0
        if jms < 200: continue
        if 200 <= jms <= 360 and sl < 0.5 * cur_slope: continue
        r_peaks.append(eg); slopes.append(sl)
    return (r_peaks,
            r_peaks[-1] if r_peaks else last_r_peak,
            slopes[-1]  if slopes  else last_r_slope,
            filt)

def _dedup(peaks, tol):
    if len(peaks) < 2: return peaks
    peaks = np.sort(peaks); keep = [int(peaks[0])]
    for p in peaks[1:]:
        if int(p) - keep[-1] > tol: keep.append(int(p))
    return np.array(keep, dtype=int)

def _remove_twave(peaks, filt):
    if len(peaks) < 2: return peaks
    thr  = int(450 * TARGET_FS / 1000)
    vals = np.array([float(filt[int(p)]) for p in peaks if 0 <= int(p) < len(filt)])
    pv   = vals[vals > 0]
    ref  = float(np.median(pv)) if len(pv) >= 3 else float(np.median(np.abs(vals)))
    amp_thr = 0.30 * ref
    keep = np.ones(len(peaks), dtype=bool)
    for i in range(1, len(peaks)):
        if int(peaks[i]) - int(peaks[i - 1]) >= thr: continue
        idx = int(peaks[i])
        if not (0 <= idx < len(filt)): continue
        v = float(filt[idx])
        if v < 0 or abs(v) < amp_thr: keep[i] = False
    return peaks[keep]

def _fill_gaps(sig_300, peaks):
    if len(peaks) < 3: return peaks
    rr  = np.diff(peaks.astype(np.float64)); med = float(np.median(rr))
    if med <= 0: return peaks
    mn  = int(0.25 * TARGET_FS); new = list(peaks)
    for i in range(len(peaks) - 1):
        if rr[i] <= 1.5 * med: continue
        zs = int(peaks[i]) + mn; ze = int(peaks[i + 1]) - mn
        if ze - zs < TARGET_FS // 2 or zs < 0 or ze > len(sig_300): continue
        found, _, _, _ = _ecg_process_window(sig_300[zs:ze], zs, int(peaks[i]), 1e9,
                                              thresh_mult=0.40)
        for p in found:
            if (p - int(peaks[i])) >= mn and (int(peaks[i + 1]) - p) >= mn:
                new.append(p)
    if len(new) == len(peaks): return peaks
    return _dedup(np.array(sorted(new), dtype=int), tol=int(0.15 * TARGET_FS))

def detect_r_peaks(ecg_raw, fs_orig):
    sig = ecg_raw.astype(np.float64)
    if abs(fs_orig - TARGET_FS) > 1e-6 and len(sig) > 1:
        sig = resample(sig, int(len(sig) * TARGET_FS / fs_orig))
    n = len(sig); fall = np.zeros(n); rall = []; lrp = -9999; lrs = 0.0
    PAD = int(0.75 * TARGET_FS)
    for start in range(0, n, WIN_SAMPLES_ECG):
        end = min(start + WIN_SAMPLES_ECG, n)
        if end - start < TARGET_FS: break
        pl = min(PAD, start); pr = min(PAD, n - end)
        ss = start - pl;      se = end + pr
        pg, nlrp, nlrs, filt = _ecg_process_window(sig[ss:se], ss, lrp, lrs)
        core = [p for p in pg if start <= p < end]
        fall[start:end] = filt[pl: pl + (end - start)]
        if core: lrp = core[-1]; lrs = nlrs
        elif nlrp != lrp: lrp = nlrp; lrs = nlrs
        rall.extend(core)
    pf = np.array(rall, dtype=int)
    pf = _fill_gaps(sig, pf)
    pf = _remove_twave(pf, fall)
    return pf, sig


# =============================================================================
# TACHOGRAM BUILDERS
# =============================================================================
def _build_rri_tachogram(r_times):
    if len(r_times) < 4: return None, None
    rri = np.diff(r_times)
    tmd = (r_times[:-1] + r_times[1:]) / 2.0
    ok  = (rri >= 0.30) & (rri <= 3.0)
    if np.sum(ok) < 3: return None, None
    rv = rri[ok]; tv = tmd[ok]
    if tv[-1] <= tv[0]: return None, None
    try: cs = CubicSpline(tv, rv)
    except Exception: return None, None
    tu = np.arange(tv[0], tv[-1], 1.0 / TACHOGRAM_FS)
    if len(tu) < 8: return None, None
    return tu, cs(tu)

def _build_qrs_rms_tachogram(ecg_300, r_peaks, hw_ms=40):
    hw = int(hw_ms / 1000.0 * TARGET_FS)
    amps, times = [], []
    for pk in r_peaks:
        s = max(0, pk - hw); e = min(len(ecg_300), pk + hw)
        if e - s < 4: continue
        amps.append(float(np.sqrt(np.mean(ecg_300[s:e] ** 2))))
        times.append(pk / float(TARGET_FS))
    if len(times) < 8: return None, None
    t = np.array(times); a = np.array(amps)
    if t[-1] <= t[0]: return None, None
    try: cs = CubicSpline(t, a)
    except Exception: return None, None
    tu = np.arange(t[0], t[-1], 1.0 / TACHOGRAM_FS)
    return (None, None) if len(tu) < 8 else (tu, cs(tu))

def _build_r_amplitude_tachogram(ecg_300, r_peaks):
    nyq = 0.5 * TARGET_FS
    b, a = butter(2, [5 / nyq, 20 / nyq], btype='bandpass')
    try: filt = filtfilt(b, a, ecg_300)
    except Exception: filt = ecg_300
    amps, times = [], []
    for pk in r_peaks:
        if 0 <= pk < len(filt):
            amps.append(float(filt[pk]))
            times.append(pk / float(TARGET_FS))
    if len(times) < 8: return None, None
    t = np.array(times); a = np.array(amps)
    if t[-1] <= t[0]: return None, None
    try: cs = CubicSpline(t, a)
    except Exception: return None, None
    tu = np.arange(t[0], t[-1], 1.0 / TACHOGRAM_FS)
    return (None, None) if len(tu) < 8 else (tu, cs(tu))


# =============================================================================
# VMD CORE
# =============================================================================
def _vmd_decompose(x, K=VMD_K, alpha=VMD_ALPHA):
    if not HAS_VMD: return None
    try:
        u, _, _ = VMD(x, alpha, 0.0, K, 0, 1, 1e-7)
        return u if u is not None and len(u) > 0 else None
    except Exception: return None

def _select_imf_kurtosis_freq(imfs, fs):
    all_scored = []
    for i, imf in enumerate(imfs):
        fft_v = np.abs(np.fft.rfft(imf - np.mean(imf))) ** 2
        freqs = np.fft.rfftfreq(len(imf), d=1.0 / fs)
        f_dom = float(freqs[np.argmax(fft_v)])
        kurt  = float(sp_kurtosis(imf, fisher=True))
        all_scored.append((i, f_dom, kurt))
    strict = [(i, k) for i, f, k in all_scored if RESP_LO <= f <= RESP_HI]
    if strict: return imfs[min(strict, key=lambda x: x[1])[0]]
    relaxed = [(i, k) for i, f, k in all_scored if 0.08 <= f <= 0.6]
    if relaxed: return imfs[min(relaxed, key=lambda x: x[1])[0]]
    nearest = min(all_scored, key=lambda x: abs(x[1] - 0.3))
    return imfs[nearest[0]]


# =============================================================================
# RR ESTIMATION HELPERS
# =============================================================================
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

def _peak_count_rr(sig, fs):
    if len(sig) < 4: return np.nan
    nyq  = 0.5 * fs
    lo_n = max(1e-4, min(RESP_LO / nyq, 0.99))
    hi_n = max(1e-4, min(RESP_HI / nyq, 0.99))
    if lo_n < hi_n:
        try:
            b, a = butter(3, [lo_n, hi_n], btype='bandpass')
            bp   = filtfilt(b, a, sig)
        except Exception: bp = sig
    else: bp = sig
    min_d = max(1, int(fs / (MAX_RR_BPM / 60.0)))
    peaks, _ = find_peaks(bp, distance=min_d)
    dur = len(sig) / fs
    if dur <= 0 or len(peaks) < 1: return np.nan
    rr = (len(peaks) / dur) * 60.0
    return rr if MIN_RR_BPM <= rr <= MAX_RR_BPM else np.nan

def _dual_merge(rr_fft, rr_peaks):
    both = not np.isnan(rr_fft) and not np.isnan(rr_peaks)
    if both:
        if abs(rr_fft - rr_peaks) < 3.0: return float((rr_fft + rr_peaks) / 2.0)
        return rr_fft if rr_fft < 14.0 else rr_peaks
    if not np.isnan(rr_fft):   return rr_fft
    if not np.isnan(rr_peaks): return rr_peaks
    return np.nan


# =============================================================================
# THREE ESTIMATORS
# =============================================================================
def est_vmd(r_times, vmd_k=VMD_K, vmd_alpha=VMD_ALPHA):
    tu, rru = _build_rri_tachogram(r_times)
    if tu is None: return np.nan
    imfs = _vmd_decompose(rru - np.mean(rru), K=vmd_k, alpha=vmd_alpha)
    if imfs is None: return np.nan
    best = _select_imf_kurtosis_freq(imfs, TACHOGRAM_FS)
    if best is None: return np.nan
    return _dual_merge(_fft_dominant_rr(best, TACHOGRAM_FS),
                       _peak_count_rr(best, TACHOGRAM_FS))

def est_qrs_rms(ecg_300, r_peaks):
    tu, amp_u = _build_qrs_rms_tachogram(ecg_300, r_peaks)
    if tu is None: return np.nan
    return _dual_merge(_welch_dominant_rr(amp_u, TACHOGRAM_FS),
                       _peak_count_rr(amp_u, TACHOGRAM_FS))

def est_r_ampl(ecg_300, r_peaks):
    tu, amp_u = _build_r_amplitude_tachogram(ecg_300, r_peaks)
    if tu is None: return np.nan
    return _dual_merge(_welch_dominant_rr(amp_u, TACHOGRAM_FS),
                       _peak_count_rr(amp_u, TACHOGRAM_FS))


def est_mpf_hf(r_times):
    """Mean Power Frequency pada HF band (0.15–0.4 Hz) dari RRI tachogram.

    Metode referensi kakak tingkat. Rentang valid: 9–24 napas/menit.
    RR (brpm) = MPF × 60.
    """
    HF_LO = 0.15; HF_HI = 0.40
    tu, rru = _build_rri_tachogram(r_times)
    if tu is None or len(rru) < 8: return np.nan
    try:
        nperseg = min(len(rru), max(32, int(TACHOGRAM_FS * 8)))
        f, pxx = welch(rru - np.mean(rru), fs=TACHOGRAM_FS,
                       nperseg=nperseg, noverlap=nperseg // 2)
    except Exception: return np.nan
    mask = (f >= HF_LO) & (f <= HF_HI)
    if not np.any(mask) or np.sum(pxx[mask]) < 1e-18: return np.nan
    mpf = float(np.sum(f[mask] * pxx[mask]) / np.sum(pxx[mask]))
    rr  = mpf * 60.0
    # Batasi sesuai keterbatasan metode (9–24 brpm)
    return rr if 9.0 <= rr <= 24.0 else np.nan


# =============================================================================
# ENS3_SMART
# =============================================================================
def ens3_smart(rr_v, rr_q, rr_a, ratio_cutoff=HARMONIC_CUTOFF, tol=AGREE_TOL):
    valid = [(v, n) for v, n in [(rr_v,'v'),(rr_q,'q'),(rr_a,'a')] if not np.isnan(v)]
    if not valid:
        return np.nan, 'no_data'
    if len(valid) == 1:
        return float(valid[0][0]), 'fallback'
    if len(valid) == 2:
        v0, v1 = valid[0][0], valid[1][0]
        if abs(v0 - v1) < tol: return float((v0 + v1) / 2.0), 'fallback'
        return (float(rr_v), 'fallback') if not np.isnan(rr_v) else (float(v0), 'fallback')
    ag01 = abs(rr_v - rr_q) < tol
    ag02 = abs(rr_v - rr_a) < tol
    ag12 = abs(rr_q - rr_a) < tol
    if ag01 and ag02:
        return float(np.mean([rr_v, rr_q, rr_a])), 'all_agree'
    elif ag01 and not ag12:
        return float((rr_v + rr_q) / 2.0), 'vmd+qrs'
    elif ag02 and not ag12:
        return float((rr_v + rr_a) / 2.0), 'vmd+ampl'
    elif ag12 and not ag01:
        qa    = (rr_q + rr_a) / 2.0
        ratio = qa / (rr_v + 1e-6)
        if ratio >= ratio_cutoff:
            return float(qa), 'harmonic→QA'
        return float(rr_v), 'harmonic→VMD'
    else:
        return float(rr_v), 'all_disagree'


def ens3_smart_ampl_anchor(rr_v, rr_q, rr_a, ratio_cutoff=HARMONIC_CUTOFF, tol=AGREE_TOL):
    """Varian ENS3_SMART dengan R-AMPL sebagai anchor/fallback (bukan VMD).

    Voting logic identik (pairwise agreement, harmonic guard 2-arah), hanya
    estimator default saat 2-of-3 disagree atau all_disagree adalah rr_a,
    bukan rr_v. Motivasi: VMD bergantung pada modulasi RSA di RRI tachogram,
    yang ternyata kurang reliable pada populasi/sensor tertentu (lihat temuan
    eksplorasi BIDMC) — sementara modulasi amplitudo R-peak akibat respirasi
    teramati lebih konsisten secara visual. Tetap dipertahankan berdampingan
    dengan ens3_smart (anchor VMD) untuk perbandingan, tidak menggantikannya.
    """
    valid = [(v, n) for v, n in [(rr_v, 'v'), (rr_q, 'q'), (rr_a, 'a')] if not np.isnan(v)]
    if not valid:
        return np.nan, 'no_data'
    if len(valid) == 1:
        return float(valid[0][0]), 'fallback'
    if len(valid) == 2:
        v0, v1 = valid[0][0], valid[1][0]
        if abs(v0 - v1) < tol: return float((v0 + v1) / 2.0), 'fallback'
        return (float(rr_a), 'fallback') if not np.isnan(rr_a) else (float(v0), 'fallback')
    ag01 = abs(rr_v - rr_q) < tol   # VMD vs QRS
    ag02 = abs(rr_v - rr_a) < tol   # VMD vs AMPL
    ag12 = abs(rr_q - rr_a) < tol   # QRS  vs AMPL
    if ag01 and ag02:
        return float(np.mean([rr_v, rr_q, rr_a])), 'all_agree'
    elif ag12 and not ag02:
        return float((rr_q + rr_a) / 2.0), 'qrs+ampl'
    elif ag02 and not ag01:
        return float((rr_v + rr_a) / 2.0), 'vmd+ampl'
    elif ag01 and not ag02:
        vq    = (rr_v + rr_q) / 2.0
        ratio = vq / (rr_a + 1e-6)
        if ratio >= ratio_cutoff:
            return float(vq), 'harmonic→VQ'
        return float(rr_a), 'harmonic→AMPL'
    else:
        return float(rr_a), 'all_disagree'


# =============================================================================
# STATISTICS
# =============================================================================
def compute_stats(group_df):
    gt    = group_df['gt_rr_bpm'].values.astype(np.float64)
    est   = group_df['rr_est_bpm'].values.astype(np.float64)
    valid = ~np.isnan(est) & ~np.isnan(gt) & (gt > 0)
    n_total = len(gt); n_valid = int(np.sum(valid))
    if n_valid == 0: return n_total, 0, np.nan, np.nan, np.nan, np.nan, np.nan
    gt_v  = gt[valid]; est_v = est[valid]; err = est_v - gt_v; ae = np.abs(err)
    mae   = float(np.mean(ae)); rmse = float(np.sqrt(np.mean(err ** 2)))
    mape  = float(np.mean(ae / gt_v) * 100.0)
    denom = np.abs(est_v) + np.abs(gt_v)
    smape = float(np.mean(np.where(denom > 0, 2.0 * ae / denom, 0.0)) * 100.0)
    mean_gt = float(np.mean(gt_v))
    return n_total, n_valid, mae, rmse, mape, smape, (mae / mean_gt if mean_gt > 0 else np.nan)


# =============================================================================
# ANALYSIS WORKER
# =============================================================================
class AnalysisWorker(QThread):
    log_signal      = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    done_signal     = pyqtSignal(object)

    def __init__(self, vmd_k, vmd_alpha):
        super().__init__()
        self.vmd_k     = vmd_k
        self.vmd_alpha = vmd_alpha

    def run(self):
        def log(m): self.log_signal.emit(m)
        subjects    = [s for s in SUBJECTS if s not in ECG_EXCLUDED]
        win_durs    = [30.0, 60.0]
        total_ops   = sum(
            len(build_analysis_windows(os.path.join(BASE_GT, s), wd) or [])
            for s in subjects for wd in win_durs)
        done_ops    = 0
        rows_by_dur = {30.0: [], 60.0: []}

        log(f'ENS3_SMART  K={self.vmd_k}  α={self.vmd_alpha}  window=30s + 60s')
        log(f'Harmonic cutoff={HARMONIC_CUTOFF}  Agree tol={AGREE_TOL} bpm')
        log(f'vmdpy: {"OK" if HAS_VMD else "TIDAK TERSEDIA"}')
        log('─' * 60)

        for subj in subjects:
            log(f'▶  {subj}')
            ecg_data = load_ecg_sensor(os.path.join(BASE_PPG, subj))
            if ecg_data is None:
                log('   [SKIP] File ECG tidak ditemukan.'); continue
            _, ecg_raw, ecg_fs = ecg_data
            log(f'   ECG: {len(ecg_raw):,} spl  fs={ecg_fs:.1f} Hz  →  deteksi R-peak...')
            try:
                r_peaks, ecg_300 = detect_r_peaks(ecg_raw, ecg_fs)
                r_peak_times     = r_peaks / float(TARGET_FS)
                log(f'   {len(r_peaks)} R-peak terdeteksi')
            except Exception as ex:
                log(f'   [ERROR] R-peak gagal: {ex}'); continue

            for win_dur_s in win_durs:
                wins = build_analysis_windows(os.path.join(BASE_GT, subj), win_dur_s)
                if wins is None:
                    log(f'   [SKIP] Ground truth tidak ditemukan ({win_dur_s:.0f}s).'); continue
                log(f'   {len(wins)} window × {win_dur_s:.0f}s')
                rows = rows_by_dur[win_dur_s]

                for win in wins:
                    wi  = win['window_idx']
                    t_s = win['start_s']; t_e = win['end_s']; gt = win['gt_rr_bpm']
                    wp      = r_peak_times[(r_peak_times >= t_s) & (r_peak_times < t_e)]
                    i_s     = int(t_s * TARGET_FS); i_e = int(t_e * TARGET_FS)
                    ecg_seg = ecg_300[i_s:min(i_e, len(ecg_300))]
                    peaks_s = r_peaks[(r_peaks >= i_s) & (r_peaks < i_e)] - i_s

                    def safe(fn, *args):
                        try:    return fn(*args)
                        except: return np.nan

                    t0    = _time.perf_counter()
                    rr_v  = safe(est_vmd,     wp, self.vmd_k, self.vmd_alpha)
                    rr_q  = safe(est_qrs_rms, ecg_seg, peaks_s)
                    rr_a  = safe(est_r_ampl,  ecg_seg, peaks_s)
                    rr_e,      branch      = ens3_smart(rr_v, rr_q, rr_a)
                    rr_e_ampl, branch_ampl = ens3_smart_ampl_anchor(rr_v, rr_q, rr_a)
                    rr_m  = safe(est_mpf_hf,  wp)
                    ms    = (_time.perf_counter() - t0) * 1e3

                    def _row(alg, rr_est, br='—'):
                        se = (rr_est - gt) if not (np.isnan(rr_est) or np.isnan(gt)) else np.nan
                        ae = abs(se) if not np.isnan(se) else np.nan
                        return dict(
                            subject_id=subj, window_idx=wi,
                            start_s=round(t_s, 1), end_s=round(t_e, 1),
                            gt_rr_bpm=round(gt, 4), algorithm=alg,
                            rr_vmd=round(rr_v, 4)  if not np.isnan(rr_v)  else np.nan,
                            rr_qrs=round(rr_q, 4)  if not np.isnan(rr_q)  else np.nan,
                            rr_ampl=round(rr_a, 4) if not np.isnan(rr_a)  else np.nan,
                            rr_mpf=round(rr_m, 4)  if not np.isnan(rr_m)  else np.nan,
                            rr_est_bpm=round(rr_est, 4) if not np.isnan(rr_est) else np.nan,
                            signed_error_bpm=round(se, 4) if not np.isnan(se) else np.nan,
                            abs_error_bpm=round(ae, 4)   if not np.isnan(ae) else np.nan,
                            elapsed_ms=round(ms, 3),
                            branch=br,
                            win_dur_s=win_dur_s,
                        )

                    rows.append(_row('ENS3_SMART', rr_e,      branch))
                    rows.append(_row('ENS3_AMPL',  rr_e_ampl, branch_ampl))
                    rows.append(_row('ECG_VMD',    rr_v))
                    rows.append(_row('ECG_MPF_HF', rr_m))
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
# TABLE SCHEMAS
# =============================================================================
_COLUMNS = ['subject_id', 'window_idx', 'start_s', 'end_s', 'gt_rr_bpm',
            'algorithm', 'rr_vmd', 'rr_qrs', 'rr_ampl', 'rr_mpf',
            'rr_est_bpm', 'signed_error_bpm', 'abs_error_bpm',
            'elapsed_ms', 'branch', 'win_dur_s']
_HEADERS = ['Subject', 'Win', 'Start(s)', 'End(s)', 'GT(brpm)',
            'Algorithm', 'VMD', 'QRS', 'AMPL', 'MPF-HF',
            'RR Est(brpm)', 'Error(brpm)', '|Error|(brpm)',
            'Time(ms)', 'Branch', 'WinDur(s)']
_STAT_COLS = ['Win Dur', 'Algorithm', 'N Total', 'N Valid', 'Coverage(%)',
              'MAE (brpm)', 'RMSE (brpm)', 'MAPE (%)', 'SMAPE (%)', 'Rel. MAE']


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
            self.axes = [self.fig.add_subplot(nrows, ncols, i + 1)
                         for i in range(nrows * ncols)]
            for ax in self.axes: ax.set_facecolor(PLOT_BG)
            self.ax = self.axes[0]
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def clear_all(self):
        for ax in self.axes:
            ax.clear(); ax.set_facecolor(PLOT_BG)


# =============================================================================
# MAIN WINDOW
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Estimasi RR — ECG ENS3_SMART')
        self.resize(1550, 950); self.setStyleSheet(_SS)
        self._df      = None   # dict {'df_30': ..., 'df_60': ...}
        self._worker  = None
        self._ecg_cache = {}   # subj → (ecg_300, r_peaks)
        self._build_ui()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(4, 4, 4, 4)
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_tab_analysis(), '  ① Analisis  ')
        self._tabs.addTab(self._build_tab_result(),   '  ② Hasil  ')
        self._tabs.addTab(self._build_tab_stat(),     '  ③ Statistik & Plot  ')
        self._tabs.addTab(self._build_tab_viewer(),   '  ④ ECG Viewer  ')
        root.addWidget(self._tabs)
        self._status = QStatusBar(); self.setStatusBar(self._status)
        self._status.showMessage('Siap. Pilih durasi window lalu klik Jalankan Analisis.')

    # =========================================================================
    # TAB ① ANALISIS
    # =========================================================================
    def _build_tab_analysis(self):
        w = QWidget(); root = QHBoxLayout(w)
        root.setContentsMargins(8, 8, 8, 8); root.setSpacing(8)

        left = QWidget(); left.setFixedWidth(270)
        lv   = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0); lv.setSpacing(8)

        grp_alg = QGroupBox('ALGORITMA')
        av = QVBoxLayout(grp_alg); av.setSpacing(3)
        for line in [
            'ENS3_SMART (output utama):',
            '  VMD  : tachogram → VMD → IMF → RR',
            '  QRS  : QRS-RMS modulation → RR',
            '  AMPL : R-peak amplitude → RR',
            '  Voting: pairwise agree (3 bpm tol)',
            '  Guard : harmonic ratio ≥ 2.1',
            '', 'ECG_VMD (baseline):',
            '  RRI tachogram → VMD → kurtosis+freq',
            '', 'ECG_MPF_HF (referensi):',
            '  HF band 0.15–0.4 Hz → MPF × 60',
            '', 'Window: 30s dan 60s (simultan)',
        ]:
            lbl = QLabel(line); lbl.setStyleSheet(f'color:{C_TEXT};font-size:10px;')
            av.addWidget(lbl)
        vmd_lbl = QLabel('vmdpy: tersedia' if HAS_VMD else 'vmdpy: TIDAK TERSEDIA')
        vmd_lbl.setStyleSheet(
            f'color:{C_GREEN if HAS_VMD else C_AMBER};font-size:10px;font-weight:bold;')
        av.addWidget(vmd_lbl)

        grp_vmd = QGroupBox('PARAMETER VMD')
        vv = QVBoxLayout(grp_vmd); vv.setSpacing(4)
        from PyQt5.QtWidgets import QLineEdit as _QLE
        k_row = QWidget(); kl = QHBoxLayout(k_row); kl.setContentsMargins(0,0,0,0)
        kl.addWidget(QLabel('K :')); self._k_input = _QLE(str(VMD_K))
        self._k_input.setFixedWidth(60); kl.addWidget(self._k_input); kl.addStretch()
        a_row = QWidget(); al = QHBoxLayout(a_row); al.setContentsMargins(0,0,0,0)
        al.addWidget(QLabel('α :')); self._alpha_input = _QLE(str(VMD_ALPHA))
        self._alpha_input.setFixedWidth(80); al.addWidget(self._alpha_input); al.addStretch()
        vv.addWidget(k_row); vv.addWidget(a_row)

        grp_subj = QGroupBox('SUBJEK')
        sv = QVBoxLayout(grp_subj); sv.setSpacing(2)
        valid_subj = [s for s in SUBJECTS if s not in ECG_EXCLUDED]
        excl_subj  = sorted(ECG_EXCLUDED)
        for txt in [f'Dianalisis: {len(valid_subj)} subjek',
                    f'Dikecualikan: {", ".join(excl_subj)}']:
            lbl = QLabel(txt); lbl.setStyleSheet(f'color:{C_TEXT};font-size:10px;')
            sv.addWidget(lbl)

        self._mini_stat = QTextEdit()
        self._mini_stat.setReadOnly(True); self._mini_stat.setFixedHeight(80)
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
        lbl = QLabel('Log:'); lbl.setStyleSheet(f'color:{C_TEXT};font-weight:bold;')
        self._log = QTextEdit(); self._log.setReadOnly(True)
        self._log.setStyleSheet(
            f'background:{C_BG2};color:{C_TEXT};border:1px solid {C_BG3};'
            f'border-radius:3px;font-size:10px;font-family:Consolas,monospace;')
        rv.addWidget(lbl); rv.addWidget(self._log)

        root.addWidget(left); root.addWidget(right, stretch=1)
        return w

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
            ('all_agree','Semua agree'), ('vmd+qrs','VMD+QRS'),
            ('vmd+ampl','VMD+AMPL'), ('harmonic→QA','Harmonic→QA'),
            ('harmonic→VMD','Harmonic→VMD'), ('all_disagree','All disagree'),
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
        self._stat_table.setMaximumHeight(120)
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
    # TAB ④ ECG VIEWER
    # =========================================================================
    def _build_tab_viewer(self):
        w = QWidget(); root = QVBoxLayout(w)
        root.setContentsMargins(8, 8, 8, 8); root.setSpacing(6)

        # ---- Control bar ----
        ctrl = QWidget(); cl = QHBoxLayout(ctrl); cl.setContentsMargins(0, 0, 0, 0)

        cl.addWidget(QLabel('Subjek:'))
        self._vwr_subj_cb = QComboBox()
        self._vwr_subj_cb.setMinimumWidth(160)
        valid_subj = [s for s in SUBJECTS if s not in ECG_EXCLUDED]
        self._vwr_subj_cb.addItems(valid_subj)
        self._vwr_subj_cb.currentIndexChanged.connect(self._vwr_on_subj_changed)
        cl.addWidget(self._vwr_subj_cb)

        cl.addWidget(QLabel('  Window:'))
        self._vwr_win_cb = QComboBox(); self._vwr_win_cb.setMinimumWidth(200)
        cl.addWidget(self._vwr_win_cb)

        self._btn_vwr = QPushButton('  Tampilkan')
        self._btn_vwr.setStyleSheet(_BTN)
        self._btn_vwr.clicked.connect(self._vwr_plot)
        cl.addWidget(self._btn_vwr)
        cl.addStretch()

        # Info box
        self._vwr_info = QLabel('— Pilih subjek & window lalu klik Tampilkan —')
        self._vwr_info.setStyleSheet(
            f'background:{C_BG2};border:1px solid {C_BG3};border-radius:3px;'
            f'padding:4px 8px;font-size:10px;color:{C_TEXT};')
        self._vwr_info.setWordWrap(True)

        # 4-panel canvas: ECG + RRI + QRS-RMS + R-AMPL
        self._vwr_canvas = MplCanvas(nrows=4, ncols=1, figsize=(14, 10))
        self._vwr_canvas.setMinimumHeight(500)

        root.addWidget(ctrl)
        root.addWidget(self._vwr_info)
        root.addWidget(self._vwr_canvas, stretch=1)

        # populate window combo on startup with placeholder
        self._vwr_subj_cb.setCurrentIndex(0)
        self._vwr_on_subj_changed()
        return w

    def _vwr_on_subj_changed(self):
        """Populate window combo based on current analysis results or fallback."""
        subj = self._vwr_subj_cb.currentText()
        self._vwr_win_cb.blockSignals(True)
        self._vwr_win_cb.clear()
        df_view = None
        if self._df is not None:
            df_view = self._df.get('df_60') if self._df.get('df_60') is not None else self._df.get('df_30')
        if df_view is not None:
            subj_df = df_view[(df_view['subject_id'] == subj) &
                              (df_view['algorithm'] == 'ENS3_SMART')]
            for _, row in subj_df.iterrows():
                ae = row.get('abs_error_bpm', np.nan)
                ae_s = f'{ae:.2f}' if not (isinstance(ae, float) and np.isnan(ae)) else '—'
                label = (f"Win {int(row['window_idx'])}  "
                         f"{row['start_s']:.0f}–{row['end_s']:.0f}s  "
                         f"GT={row['gt_rr_bpm']:.1f}  "
                         f"ENS3={row['rr_est_bpm'] if not (isinstance(row['rr_est_bpm'], float) and np.isnan(row['rr_est_bpm'])) else '—'}  "
                         f"|err|={ae_s}")
                self._vwr_win_cb.addItem(label,
                    userData=(row['start_s'], row['end_s'],
                              row['gt_rr_bpm'], row.get('rr_est_bpm', np.nan),
                              row.get('rr_vmd', np.nan), row.get('rr_qrs', np.nan),
                              row.get('rr_ampl', np.nan), row.get('branch', '—')))
        else:
            # fallback: buat list window dari GT file
            wins = build_analysis_windows(os.path.join(BASE_GT, subj), 60.0)
            if wins:
                for win in wins:
                    label = f"Win {win['window_idx']}  {win['start_s']:.0f}–{win['end_s']:.0f}s  GT={win['gt_rr_bpm']:.1f}"
                    self._vwr_win_cb.addItem(label,
                        userData=(win['start_s'], win['end_s'],
                                  win['gt_rr_bpm'], np.nan,
                                  np.nan, np.nan, np.nan, '—'))
        self._vwr_win_cb.blockSignals(False)

    def _vwr_plot(self):
        subj = self._vwr_subj_cb.currentText()
        ud   = self._vwr_win_cb.currentData()
        if ud is None:
            self._status.showMessage('Pilih window terlebih dahulu.'); return
        t_s, t_e, gt, rr_e, rr_v, rr_q, rr_a, branch = ud

        # Load / cache ECG
        if subj not in self._ecg_cache:
            self._status.showMessage(f'Memuat ECG {subj}...')
            QApplication.processEvents()
            ecg_data = load_ecg_sensor(os.path.join(BASE_PPG, subj))
            if ecg_data is None:
                self._status.showMessage(f'{subj}: file ECG tidak ditemukan.'); return
            _, ecg_raw, ecg_fs = ecg_data
            try:
                r_peaks, ecg_300 = detect_r_peaks(ecg_raw, ecg_fs)
                self._ecg_cache[subj] = (ecg_300, r_peaks)
            except Exception as ex:
                self._status.showMessage(f'R-peak gagal: {ex}'); return

        ecg_300, r_peaks = self._ecg_cache[subj]

        i_s     = int(t_s * TARGET_FS)
        i_e     = int(t_e * TARGET_FS)
        ecg_seg = ecg_300[i_s:min(i_e, len(ecg_300))]
        t_ax    = np.linspace(t_s, t_e, len(ecg_seg))

        # R-peaks in window
        mask_pk = (r_peaks >= i_s) & (r_peaks < min(i_e, len(ecg_300)))
        pk_abs  = r_peaks[mask_pk]
        pk_t    = pk_abs / TARGET_FS
        pk_v    = ecg_300[pk_abs]

        # Tachogram signals
        wp        = pk_t
        peaks_s   = pk_abs - i_s
        _, rri_u  = _build_rri_tachogram(wp)
        tu_rri, _ = _build_rri_tachogram(wp)
        tu_qrs, qrs_u = _build_qrs_rms_tachogram(ecg_seg, peaks_s)
        tu_amp, amp_u = _build_r_amplitude_tachogram(ecg_seg, peaks_s)
        if tu_rri is not None: tu_rri = tu_rri + t_s
        if tu_qrs is not None: tu_qrs = tu_qrs + t_s
        if tu_amp is not None: tu_amp = tu_amp + t_s

        # Build info text
        def fmt(v): return f'{v:.2f}' if not (isinstance(v, float) and np.isnan(v)) else '—'
        info_parts = [
            f'Subjek: {subj}',
            f'Window: {t_s:.0f}–{t_e:.0f}s',
            f'GT={fmt(gt)} brpm',
            f'ENS3={fmt(rr_e)} brpm',
            f'VMD={fmt(rr_v)}',
            f'QRS={fmt(rr_q)}',
            f'AMPL={fmt(rr_a)}',
            f'Branch: {branch}',
            f'R-peaks terdeteksi: {len(pk_abs)}',
        ]
        if not (isinstance(rr_e, float) and np.isnan(rr_e)) and \
           not (isinstance(gt, float) and np.isnan(gt)):
            err = rr_e - gt
            c   = C_GREEN if abs(err) <= 2 else C_AMBER if abs(err) <= 5 else C_RED
            info_parts.append(f'  |error|={abs(err):.2f} brpm')
            self._vwr_info.setStyleSheet(
                f'background:{C_BG2};border:1px solid {c};border-radius:3px;'
                f'padding:4px 8px;font-size:10px;color:{C_TEXT};')
        else:
            self._vwr_info.setStyleSheet(
                f'background:{C_BG2};border:1px solid {C_BG3};border-radius:3px;'
                f'padding:4px 8px;font-size:10px;color:{C_TEXT};')
        self._vwr_info.setText('   |   '.join(info_parts))

        # Plot
        self._vwr_canvas.clear_all()
        axes = self._vwr_canvas.axes

        # Panel 0 — Raw ECG + R-peaks
        ax0 = axes[0]
        ax0.plot(t_ax, ecg_seg, color='#1565c0', linewidth=0.7, label='ECG (300 Hz)')
        if len(pk_t) > 0:
            ax0.scatter(pk_t, pk_v, color=C_RED, s=30, zorder=5, label=f'R-peak ({len(pk_t)})')
        ax0.axvspan(t_s, t_e, alpha=0.04, color=C_ACCENT)
        ax0.set_ylabel('Amplitude')
        ax0.set_title(f'ECG — {subj}  |  {t_s:.0f}–{t_e:.0f}s  |  '
                      f'GT={fmt(gt)} brpm  ENS3={fmt(rr_e)} brpm  '
                      f'branch={branch}', color=C_TEXT)
        ax0.legend(fontsize=8, loc='upper right'); ax0.grid(True)

        # Panel 1 — RRI tachogram
        ax1 = axes[1]
        if tu_rri is not None and rri_u is not None:
            ax1.plot(tu_rri, rri_u, color='#6a1b9a', linewidth=1.2, label='RRI tachogram')
            ax1.set_ylabel('RRI (s)')
        else:
            ax1.text(0.5, 0.5, 'RRI tachogram: tidak tersedia',
                     ha='center', va='center', transform=ax1.transAxes, color=C_TEXT2)
        ax1.set_title(f'RRI Tachogram  →  ECG_VMD = {fmt(rr_v)} brpm', color=C_TEXT)
        ax1.legend(fontsize=8); ax1.grid(True)

        # Panel 2 — QRS-RMS
        ax2 = axes[2]
        if tu_qrs is not None and qrs_u is not None:
            ax2.plot(tu_qrs, qrs_u, color='#2e7d32', linewidth=1.2, label='QRS-RMS modulation')
            ax2.set_ylabel('RMS Amplitude')
        else:
            ax2.text(0.5, 0.5, 'QRS-RMS: tidak tersedia',
                     ha='center', va='center', transform=ax2.transAxes, color=C_TEXT2)
        ax2.set_title(f'QRS-RMS Tachogram  →  ECG_QRS = {fmt(rr_q)} brpm', color=C_TEXT)
        ax2.legend(fontsize=8); ax2.grid(True)

        # Panel 3 — R-AMPL
        ax3 = axes[3]
        if tu_amp is not None and amp_u is not None:
            ax3.plot(tu_amp, amp_u, color='#e65100', linewidth=1.2, label='R-peak amplitude')
            ax3.set_ylabel('Amplitude')
        else:
            ax3.text(0.5, 0.5, 'R-AMPL: tidak tersedia',
                     ha='center', va='center', transform=ax3.transAxes, color=C_TEXT2)
        ax3.set_title(f'R-AMPL Tachogram  →  ECG_AMPL = {fmt(rr_a)} brpm', color=C_TEXT)
        ax3.set_xlabel('Waktu (s)'); ax3.legend(fontsize=8); ax3.grid(True)

        self._vwr_canvas.fig.tight_layout(pad=1.5)
        self._vwr_canvas.draw()
        self._status.showMessage(f'Viewer: {subj}  win {t_s:.0f}–{t_e:.0f}s')

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
        self._ecg_cache.clear()

        self._worker = AnalysisWorker(vmd_k, vmd_alpha)
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

        # Gunakan df_60 sebagai tampilan utama di tabel & statistik; fallback ke df_30
        df = result.get('df_60') if result.get('df_60') is not None else result.get('df_30')

        # Isi tabel hasil
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
                if col == 'abs_error_bpm' and txt != '—':
                    try:
                        e = float(txt)
                        bg, fg = (_CELL_GOOD if e <= 2.0 else
                                  _CELL_WARN if e <= 5.0 else _CELL_BAD)
                        item.setBackground(QBrush(bg)); item.setForeground(QBrush(fg))
                    except Exception: pass
                elif col == 'branch' and branch_color is not None \
                     and row.get('algorithm') == 'ENS3_SMART':
                    item.setBackground(QBrush(branch_color))
                self._table.setItem(ri, ci, item)

        # Statistik — tampilkan 60s dan 30s sekaligus dalam satu tabel
        stat_rows = []; mini_lines = []
        for dur_label, dur_key in [('60s', 'df_60'), ('30s', 'df_30')]:
            df_dur = result.get(dur_key)
            if df_dur is None: continue
            for alg in ['ENS3_SMART', 'ENS3_AMPL', 'ECG_VMD', 'ECG_MPF_HF']:
                sub = df_dur[df_dur['algorithm'] == alg]
                if len(sub) == 0: continue
                n_tot, n_val, mae, rmse, mape, smape, rel_mae = compute_stats(sub)
                cov = round(n_val / n_tot * 100, 1) if n_tot > 0 else 0.0
                stat_rows.append({
                    'Win Dur': dur_label,
                    'Algorithm': alg, 'N Total': n_tot, 'N Valid': n_val, 'Coverage(%)': cov,
                    'MAE (brpm)':  round(mae,     4) if not np.isnan(mae)     else np.nan,
                    'RMSE (brpm)': round(rmse,    4) if not np.isnan(rmse)    else np.nan,
                    'MAPE (%)':    round(mape,    2) if not np.isnan(mape)    else np.nan,
                    'SMAPE (%)':   round(smape,   2) if not np.isnan(smape)   else np.nan,
                    'Rel. MAE':    round(rel_mae, 4) if not np.isnan(rel_mae) else np.nan,
                })
                mae_s = f'{mae:.4f}' if not np.isnan(mae) else '—'
                mini_lines.append(f'[{dur_label}] {alg}: MAE={mae_s}  RMSE={rmse:.4f}  N={n_val}/{n_tot}')

        self._stat_table.setRowCount(len(stat_rows))
        self._stat_table.setMaximumHeight(220)
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
                if sr['Algorithm'] == 'ENS3_SMART':
                    item.setBackground(QBrush(QColor('#e8f5e9')))
                    item.setForeground(QBrush(QColor(C_GREEN)))
                elif sr['Algorithm'] == 'ECG_MPF_HF':
                    item.setBackground(QBrush(QColor('#e3f2fd')))
                    item.setForeground(QBrush(QColor('#1565c0')))
                self._stat_table.setItem(ri, ci, item)

        # Branch distribution
        ens3_df = df[df['algorithm'] == 'ENS3_SMART'].copy()
        vmd_df  = df[df['algorithm'] == 'ECG_VMD'].copy()
        if len(ens3_df) > 0:
            merged = ens3_df[['subject_id','window_idx','branch','abs_error_bpm']].merge(
                vmd_df[['subject_id','window_idx','abs_error_bpm']].rename(
                    columns={'abs_error_bpm': 'err_vmd'}),
                on=['subject_id','window_idx'], how='left')
            bsum = (merged.groupby('branch')
                    .agg(n=('abs_error_bpm','count'),
                         mae_ens3=('abs_error_bpm','mean'),
                         mae_vmd=('err_vmd','mean'))
                    .reset_index().sort_values('n', ascending=False))
            self._branch_table.setRowCount(len(bsum))
            for ri, (_, brow) in enumerate(bsum.iterrows()):
                bname = str(brow['branch'])
                vals  = [bname, str(int(brow['n'])),
                         f'{brow["mae_ens3"]:.4f}' if not np.isnan(brow['mae_ens3']) else '—',
                         f'{brow["mae_vmd"]:.4f}'  if not np.isnan(brow['mae_vmd'])  else '—']
                bc = _BRANCH_COLORS.get(bname, None)
                for ci, val in enumerate(vals):
                    item = QTableWidgetItem(val); item.setTextAlignment(Qt.AlignCenter)
                    if bc: item.setBackground(QBrush(bc))
                    self._branch_table.setItem(ri, ci, item)

        self._mini_stat.setPlainText('\n'.join(mini_lines))

        # Refresh plot combos
        self._dur_plot_cb.blockSignals(True); self._dur_plot_cb.clear()
        if result.get('df_60') is not None: self._dur_plot_cb.addItem('60s', userData='df_60')
        if result.get('df_30') is not None: self._dur_plot_cb.addItem('30s', userData='df_30')
        self._dur_plot_cb.blockSignals(False)

        self._alg_plot_cb.blockSignals(True); self._alg_plot_cb.clear()
        for alg in ['ENS3_SMART', 'ENS3_AMPL', 'ECG_VMD', 'ECG_MPF_HF']:
            if alg in df['algorithm'].values:
                self._alg_plot_cb.addItem(alg)
        self._alg_plot_cb.blockSignals(False)
        self._refresh_plot()

        # Refresh viewer window list for current subject
        self._vwr_on_subj_changed()

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
        sub   = df[df['algorithm'] == alg]
        gt    = sub['gt_rr_bpm'].values.astype(np.float64)
        est   = sub['rr_est_bpm'].values.astype(np.float64)
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

        if alg == 'ENS3_SMART' and 'branch' in sub.columns:
            branches = sub['branch'].values[valid]
            for branch, color in [
                ('all_agree','#1565c0'), ('harmonic→QA','#e65100'),
                ('harmonic→VMD','#6a1b9a'), ('vmd+qrs','#2e7d32'),
                ('vmd+ampl','#00695c'), ('all_disagree','#b71c1c'), ('fallback','#546e7a'),
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
        vmd_k     = self._parse_k()     or VMD_K
        vmd_alpha = self._parse_alpha() or VMD_ALPHA
        ts  = _time.strftime('%Y%m%d_%H%M%S')
        tag = f'ENS3_K{vmd_k}a{vmd_alpha}_{ts}'
        out = os.path.join(OUT_DIR, f'ENS3_{tag}.xlsx')

        def _make_summary(df):
            rows = []
            for alg in ['ENS3_SMART', 'ENS3_AMPL', 'ECG_VMD', 'ECG_MPF_HF']:
                sub = df[df['algorithm'] == alg]
                if len(sub) == 0: continue
                n_tot, n_val, mae, rmse, mape, smape, rel_mae = compute_stats(sub)
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
