"""
7-Fold Cross-Validation — PPGBGL_v3 Baseline Mas Aldo  [BGL-1]
=========================================================================
Arsitektur : PPGBGL_v3 (Gambar 3.9 Buku TA Mas Aldo) — Pure CNN
             Conv1D(64,k=5,s=2,ReLU) → MaxPool(2) → BN
             → Conv1D(96,k=3,ReLU)
             → Conv1D(208,k=3,ReLU) → MaxPool(3)
             → Conv1D(64,k=3,ReLU) → MaxPool(3)
             → GlobalAvgPool → Dropout(0.3) → Dense(1)

Input   : 6 channel [IR, Red, dIR/dt, dRed/dt, d²IR/dt², d²Red/dt²]
          Window 2000 sampel @ 100 Hz = 20 detik
Dataset : 27 subjek (001-025 minus 005 + A_1,A_2,A_3,A_5,A_6, eksklusi A_4)
Fold    : 7-Fold GroupKFold, pasangan duplikat satu fold
Evaluasi: MAE, RMSE, Clarke Error Grid Analysis (Zone A/B/C/D/E)
Output  : Hasil_Eksperimen_BGL/BGL_1_Aldo_v3/
"""

import os, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras

# ─────────────────────────────────────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────────────────────────────────────

ALL_SUBJECTS = [f'SUBJECT_{i:03d}' for i in range(1, 26)] + [
    'SUBJECT_A_1', 'SUBJECT_A_2', 'SUBJECT_A_3',
    'SUBJECT_A_4', 'SUBJECT_A_5', 'SUBJECT_A_6',
]
EXCLUDED_SUBJECTS = {'SUBJECT_005', 'SUBJECT_A_4'}

DUPLICATE_PAIRS = [
    ('SUBJECT_004', 'SUBJECT_A_1'),
    ('SUBJECT_002', 'SUBJECT_A_2'),
    ('SUBJECT_009', 'SUBJECT_A_3'),
    ('SUBJECT_013', 'SUBJECT_A_4'),
    ('SUBJECT_001', 'SUBJECT_A_5'),
    ('SUBJECT_011', 'SUBJECT_A_6'),
]

EXCLUDED_SEGMENTS = {
    'SUBJECT_001': [3], 'SUBJECT_007': [3, 5], 'SUBJECT_009': [4],
    'SUBJECT_010': [5], 'SUBJECT_017': [4], 'SUBJECT_018': [3],
    'SUBJECT_022': [1, 4], 'SUBJECT_023': [2], 'SUBJECT_025': [3, 5],
}

FS          = 100
WIN_SAMPLES = 2000
STEP_TRAIN  = 100
STEP_TEST   = 500
N_CHANNELS  = 6
N_FOLDS     = 7
EPOCHS      = 200
LR          = 1e-4
BATCH_SIZE  = 32
DROPOUT     = 0.3
SEED        = 42

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_ANALISIS_DIR= os.path.dirname(_SCRIPT_DIR)
_ROOT_DIR    = os.path.dirname(_ANALISIS_DIR)
DATA_DIR     = os.path.join(_ROOT_DIR, 'Database Keseluruhan (Sudah Digabung Semua)')

OUT_DIR   = os.path.join(_SCRIPT_DIR, 'Hasil_Eksperimen_BGL', 'BGL_1_Aldo_v3')
MODEL_DIR = os.path.join(OUT_DIR, 'models')
LOG_PATH  = os.path.join(OUT_DIR, 'log.txt')
RESULT_CSV= os.path.join(OUT_DIR, 'results.csv')
BHS_TXT   = os.path.join(OUT_DIR, 'metrics.txt')

os.makedirs(MODEL_DIR, exist_ok=True)

_log_fh = open(LOG_PATH, 'w', encoding='utf-8')
def log(msg=''):
    line = '[' + time.strftime('%Y-%m-%d %H:%M:%S') + '] ' + str(msg)
    print(line, flush=True)
    _log_fh.write(line + '\n')
    _log_fh.flush()

# ─────────────────────────────────────────────────────────────────────────────
# Model PPGBGL_v3 (identik Gambar 3.9 Mas Aldo)
# ─────────────────────────────────────────────────────────────────────────────

def build_model():
    inputs = keras.Input(shape=(WIN_SAMPLES, N_CHANNELS), name='ppg_input')
    x = keras.layers.Conv1D(64, kernel_size=5, strides=2, activation='relu')(inputs)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(96, kernel_size=3, activation='relu')(x)
    x = keras.layers.Conv1D(208, kernel_size=3, activation='relu')(x)
    x = keras.layers.MaxPooling1D(pool_size=3)(x)
    x = keras.layers.Conv1D(64, kernel_size=3, activation='relu')(x)
    x = keras.layers.MaxPooling1D(pool_size=3)(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    outputs = keras.layers.Dense(1, name='glucose')(x)
    return keras.Model(inputs, outputs, name='PPGBGL_v3')

# ─────────────────────────────────────────────────────────────────────────────
# Signal processing
# ─────────────────────────────────────────────────────────────────────────────

def bpf(sig, lo=1.0, hi=5.0, fs=FS, order=4):
    nyq = fs / 2
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, sig.astype(np.float64))

def minmax_norm(arr, vmin=0.0, vmax=5.0):
    lo, hi = arr.min(), arr.max()
    if hi == lo: return np.zeros_like(arr)
    return (arr - lo) / (hi - lo) * (vmax - vmin) + vmin

def make_6ch_window(ir_filt, red_filt):
    """[IR, Red, dIR, dRed, d²IR, d²Red] semua di-minmax ke [0,5]."""
    d_ir  = np.gradient(ir_filt)
    d_red = np.gradient(red_filt)
    d2_ir = np.gradient(d_ir)
    d2_red= np.gradient(d_red)
    ch = [minmax_norm(c) for c in [ir_filt, red_filt, d_ir, d_red, d2_ir, d2_red]]
    return np.stack(ch, axis=-1).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def active_segs(subj):
    excl = EXCLUDED_SEGMENTS.get(subj, [])
    if excl == 'all': return []
    return [s for s in range(1, 6) if s not in excl]

def load_subject(subj):
    folder = os.path.join(DATA_DIR, subj)
    if not os.path.isdir(folder): return None
    meta_files = [f for f in os.listdir(folder) if f.endswith('_metadata.csv')]
    if not meta_files: return None
    df_meta = pd.read_csv(os.path.join(folder, meta_files[0]))
    glucose = float(df_meta.iloc[0]['glucose_mgdl'])
    prefix  = meta_files[0][:-len('_metadata.csv')]
    segs    = active_segs(subj)
    if not segs: return None
    parts_ir, parts_red = [], []
    for seg in range(1, 6):
        if seg not in segs: continue
        ppg_path = os.path.join(folder, f'{prefix}_seg{seg:02d}_PPG.csv')
        if not os.path.exists(ppg_path): continue
        df = pd.read_csv(ppg_path)
        parts_ir.append(-df['ir_raw'].values.astype(np.float64))
        parts_red.append(-df['red_raw'].values.astype(np.float64))
    if not parts_ir: return None
    ir_full  = np.concatenate(parts_ir)
    red_full = np.concatenate(parts_red)
    if len(ir_full) < WIN_SAMPLES: return None
    return {
        'subj': subj,
        'glucose': glucose,
        'ir_filt':  bpf(ir_full),
        'red_filt': bpf(red_full),
        'n': len(ir_full),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Windows & Folds
# ─────────────────────────────────────────────────────────────────────────────

def build_windows(d, step):
    ir_f = d['ir_filt']; red_f = d['red_filt']
    n = len(ir_f); g = d['glucose']
    X, y = [], []
    for start in range(0, n - WIN_SAMPLES + 1, step):
        win = make_6ch_window(ir_f[start:start+WIN_SAMPLES], red_f[start:start+WIN_SAMPLES])
        X.append(win); y.append(g)
    if not X:
        return np.empty((0, WIN_SAMPLES, N_CHANNELS), np.float32), np.array([])
    return np.array(X, np.float32), np.array(y, np.float32)

def make_folds(active_loaded, n_folds=N_FOLDS, seed=SEED):
    pair_map = {}
    for a, b in DUPLICATE_PAIRS:
        if a in active_loaded and b in active_loaded:
            pair_map[a]=(a,b); pair_map[b]=(a,b)
    seen, units = set(), []
    for s in active_loaded:
        if s in seen: continue
        if s in pair_map:
            pair=pair_map[s]; units.append(list(pair)); seen.update(pair)
        else:
            units.append([s]); seen.add(s)
    rng = np.random.RandomState(seed)
    rng.shuffle(units)
    folds = [[] for _ in range(n_folds)]
    for i, unit in enumerate(units):
        folds[i % n_folds].extend(unit)
    return folds

# ─────────────────────────────────────────────────────────────────────────────
# Clarke Error Grid Analysis
# ─────────────────────────────────────────────────────────────────────────────

def clarke_zone(ref, pred):
    """Return zone A/B/C/D/E untuk satu pasang (ref, pred) dalam mg/dL."""
    if (ref <= 70 and pred <= 70) or abs(pred - ref) / (ref + 1e-9) <= 0.20:
        return 'A'
    if ref >= 180 and pred <= 70: return 'E'
    if ref <= 70 and pred >= 180: return 'E'
    if ref >= 240 and (70 <= pred <= 180): return 'C'
    if ref <= 70 and (180 >= pred > 70 * 1.2): return 'C'
    if (ref >= 180 and pred < 70) or (ref <= 70 and pred > 180): return 'E'
    if ref >= 180 and 70 <= pred <= 180: return 'D'
    if ref <= 70 and pred >= 70: return 'D'
    return 'B'

def plot_clarke(gt_list, pred_list, out_path, title='Clarke EGA'):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 400); ax.set_ylim(0, 400)
    ax.plot([0, 400], [0, 400], 'k--', lw=0.8)
    ax.plot([0, 400], [0, 400*1.2], 'k-', lw=0.5)
    ax.plot([0, 400], [0, 400*0.8], 'k-', lw=0.5)
    colors = {'A':'#2ecc71','B':'#3498db','C':'#f39c12','D':'#e74c3c','E':'#8e44ad'}
    zones  = [clarke_zone(r, p) for r, p in zip(gt_list, pred_list)]
    for z in 'ABCDE':
        idx = [i for i, zz in enumerate(zones) if zz == z]
        if idx:
            ax.scatter([gt_list[i] for i in idx], [pred_list[i] for i in idx],
                       c=colors[z], label=f'Zone {z} (n={len(idx)})', s=60, alpha=0.8)
    ax.set_xlabel('Reference BGL (mg/dL)'); ax.set_ylabel('Predicted BGL (mg/dL)')
    ax.set_title(title); ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log(f'  Clarke EGA disimpan: {out_path}')
    cnt = {z: zones.count(z) for z in 'ABCDE'}
    total = len(zones)
    for z in 'ABCDE':
        log(f'    Zone {z}: {cnt[z]}/{total} ({cnt[z]/total*100:.1f}%)')

# ─────────────────────────────────────────────────────────────────────────────
# DelayedEarlyStopping
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(all_histories, out_path, title='Training Curves'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors
    for i, h in enumerate(all_histories):
        c = colors[i % len(colors)]
        lbl = f'Fold {i+1}'
        axes[0].plot(h.get('loss', []),     color=c, alpha=0.8, label=lbl)
        axes[1].plot(h.get('val_loss', []), color=c, alpha=0.8, label=lbl)
    for ax, ttl in zip(axes, ['Train Loss (Huber)', 'Val Loss (Huber)']):
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title(f'{title}\n{ttl}')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log(f'  Training curves disimpan: {out_path}')


class DelayedEarlyStopping(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', patience=20, min_delta=1.0,
                 start_epoch=10, restore_best_weights=True):
        super().__init__()
        self.monitor=monitor; self.patience=patience; self.min_delta=min_delta
        self.start_epoch=start_epoch; self.restore_best=restore_best_weights
        self.best=np.inf; self.best_weights=None; self.wait=0
    def on_epoch_end(self, epoch, logs=None):
        current=(logs or {}).get(self.monitor, np.inf)
        if current < self.best - self.min_delta:
            self.best=current; self.wait=0
            if self.restore_best: self.best_weights=self.model.get_weights()
        else:
            self.wait+=1
            if epoch>=self.start_epoch and self.wait>=self.patience:
                self.model.stop_training=True
                if self.restore_best and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['PYTHONHASHSEED'] = str(SEED)
    keras.utils.set_random_seed(SEED)

    log('='*60)
    log('  BGL-1: PPGBGL_v3 Baseline Mas Aldo — 7-Fold 27 subj')
    log('='*60)

    active_subjects = [s for s in ALL_SUBJECTS if s not in EXCLUDED_SUBJECTS]
    subject_data, active_loaded = {}, []
    for subj in active_subjects:
        d = load_subject(subj)
        if d is None:
            log(f'  [SKIP] {subj}')
            continue
        subject_data[subj]=d; active_loaded.append(subj)
        log(f'  Loaded {subj}  glucose={d["glucose"]:.0f} mg/dL  n={d["n"]}')

    log(f'\n  Total aktif: {len(active_loaded)}')
    folds = make_folds(active_loaded)
    for i, f in enumerate(folds):
        log(f'  Fold {i+1}: {f}')

    all_records   = []
    all_histories = []

    for fold_idx, test_group in enumerate(folds):
        fold_t0 = time.time()
        train_subjects = [s for s in active_loaded if s not in test_group]
        val_candidates = [s for s in train_subjects
                          if not any(s in p for p in DUPLICATE_PAIRS
                                     if all(x in active_loaded for x in p))]
        if not val_candidates: val_candidates = train_subjects
        vi1=fold_idx%len(val_candidates)
        vi2=(fold_idx+len(val_candidates)//2)%len(val_candidates)
        if vi1==vi2: vi2=(vi1+1)%len(val_candidates)
        val_names=[val_candidates[vi1], val_candidates[vi2]]
        fit_names=[s for s in train_subjects if s not in val_names]

        log(f'\n  ══════ Fold {fold_idx+1}/{N_FOLDS} ══════')
        log(f'  Test: {test_group}  Val: {val_names}  Train: {len(fit_names)} subj')

        X_tr, y_tr = [], []
        for s in fit_names:
            Xw, yw = build_windows(subject_data[s], STEP_TRAIN)
            if len(Xw): X_tr.append(Xw); y_tr.append(yw)
        if not X_tr: continue

        X_train=np.concatenate(X_tr); y_train=np.concatenate(y_tr)
        idx=np.random.RandomState(SEED+fold_idx).permutation(len(X_train))
        X_train=X_train[idx]; y_train=y_train[idx]

        Xv_l, yv_l = [], []
        for vn in val_names:
            Xv, yv = build_windows(subject_data[vn], STEP_TEST)
            if len(Xv): Xv_l.append(Xv); yv_l.append(yv)
        X_val=np.concatenate(Xv_l) if Xv_l else np.array([])
        y_val=np.concatenate(yv_l) if yv_l else np.array([])

        log(f'  Train: {len(X_train):,} windows  |  Val: {len(X_val)} windows')

        model = build_model()
        model.compile(optimizer=keras.optimizers.Adam(LR),
                      loss=keras.losses.Huber(delta=10.0))
        val_data=(X_val, y_val) if len(X_val)>0 else None
        cbs=[
            DelayedEarlyStopping(patience=20, min_delta=1.0, start_epoch=10),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6, verbose=0),
        ]
        history=model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                          validation_data=val_data, callbacks=cbs, verbose=0)
        vloss=history.history.get('val_loss',[0])
        best_ep=int(np.argmin(vloss))+1; best_val=float(min(vloss))
        all_histories.append(history.history)
        log(f'  Selesai epoch={best_ep}  val_loss={best_val:.4f}  durasi={time.time()-fold_t0:.0f}s')

        for test_name in test_group:
            X_test, _ = build_windows(subject_data[test_name], STEP_TEST)
            d_test = subject_data[test_name]
            if len(X_test)==0: continue
            preds = model.predict(X_test, batch_size=64, verbose=0).flatten()
            pred_g = float(np.median(preds))
            err_g  = pred_g - d_test['glucose']
            zone   = clarke_zone(d_test['glucose'], pred_g)
            log(f'  HASIL {test_name}  GT:{d_test["glucose"]:.0f}'
                f'  Pred:{pred_g:.1f}  Err:{err_g:+.1f}  Zone:{zone}')
            rec={'fold':fold_idx+1,'subject_id':test_name,
                 'gt_glucose':d_test['glucose'],'pred_glucose':round(pred_g,2),
                 'error_mgdl':round(err_g,2),'clarke_zone':zone,
                 'best_epoch':best_ep,'val_loss':round(best_val,4)}
            all_records.append(rec)
            pd.DataFrame(all_records).to_csv(RESULT_CSV, index=False)

        model.save(os.path.join(MODEL_DIR, f'fold_{fold_idx+1:02d}.keras'))

    if all_records:
        df = pd.DataFrame(all_records)
        mae_v  = float(np.mean(np.abs(df.error_mgdl)))
        rmse_v = float(np.sqrt(np.mean(df.error_mgdl**2)))
        log(f'\n  ╔══════════════════════════════════════════╗')
        log(f'  ║  RINGKASAN FINAL BGL-1 (27 subjek)')
        log(f'  ║  MAE={mae_v:.2f} mg/dL  RMSE={rmse_v:.2f} mg/dL')
        for z in 'ABCDE':
            n = (df.clarke_zone==z).sum()
            log(f'  ║  Zone {z}: {n}/{len(df)} ({n/len(df)*100:.1f}%)')
        log(f'  ╚══════════════════════════════════════════╝')

        gt_list   = df.gt_glucose.tolist()
        pred_list = df.pred_glucose.tolist()
        plot_clarke(gt_list, pred_list,
                    os.path.join(OUT_DIR, 'clarke_ega.png'),
                    title='Clarke EGA — BGL-1 Baseline Aldo v3')

        with open(BHS_TXT, 'w', encoding='utf-8') as f:
            f.write(f'BGL-1: PPGBGL_v3 Baseline Mas Aldo\n')
            f.write(f'{N_FOLDS}-Fold CV, 27 subjek\n\n')
            f.write(f'MAE  : {mae_v:.2f} mg/dL\n')
            f.write(f'RMSE : {rmse_v:.2f} mg/dL\n\n')
            f.write('Clarke EGA:\n')
            for z in 'ABCDE':
                n=(df.clarke_zone==z).sum()
                f.write(f'  Zone {z}: {n}/{len(df)} ({n/len(df)*100:.1f}%)\n')

    if all_histories:
        plot_training_curves(all_histories,
                             os.path.join(OUT_DIR, 'training_curves.png'),
                             title='BGL-1: PPGBGL_v3 Baseline Mas Aldo')

    log('\nSELESAI.')
    _log_fh.close()

if __name__ == '__main__':
    main()
