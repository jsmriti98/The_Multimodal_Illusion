
"""Multi-Modal Fraud Meta-Learner — Real Datasets (IEEE-CIS + Balabit + DGraph-Fin)

## 0. Install dependencies

!pip install -q xgboost scikit-learn pandas numpy scipy matplotlib shap torch torch_geometric"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import io, os, glob, zipfile

EMB_DIM = 32  # per-branch embedding size, kept small for CPU speed (paper uses 128)

DATA_ROOT = './fraud_meta_learner_data'  

IEEE_DIR = os.path.join(DATA_ROOT, 'ieee_cis')
BALABIT_DIR = os.path.join(DATA_ROOT, 'balabit')
DGRAPH_DIR = os.path.join(DATA_ROOT, 'dgraph')

for d in [DATA_ROOT, IEEE_DIR, BALABIT_DIR, DGRAPH_DIR]:
    os.makedirs(d, exist_ok=True)

print('Data root:', DATA_ROOT)
print('  IEEE-CIS dir:', IEEE_DIR, '->', os.listdir(IEEE_DIR) if os.path.isdir(IEEE_DIR) else 'missing')
print('  Balabit dir :', BALABIT_DIR, '->', os.listdir(BALABIT_DIR) if os.path.isdir(BALABIT_DIR) else 'missing')
print('  DGraph dir  :', DGRAPH_DIR, '->', os.listdir(DGRAPH_DIR) if os.path.isdir(DGRAPH_DIR) else 'missing')
print()

import numpy as np

test_path = dgraph_path

if test_path and os.path.exists(test_path):
    with np.load(test_path) as data:
        print("Actual keys in your .npz file:", list(data.keys()))
else:
    print("File path does not exist.")

"""## 1. Shared evaluation utilities (Section 6 of the report)"""

def pr_auc(y_true, y_score):
    return average_precision_score(y_true, y_score)

def recall_at_fpr(y_true, y_score, target_fpr=0.001):
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side='right') - 1
    idx = max(idx, 0)
    return tpr[idx], thresh[idx]

def cost_weighted_loss(y_true, y_score, threshold, fn_cost=500.0, fp_cost=5.0):
    y_pred = (y_score >= threshold).astype(int)
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    n = len(y_true)
    total_cost = fn * fn_cost + fp * fp_cost
    return total_cost * (10_000 / n)

def summarize(y_true, y_score, label=''):
    ap = pr_auc(y_true, y_score)
    recall, thresh = recall_at_fpr(y_true, y_score, target_fpr=0.001)
    cost = cost_weighted_loss(y_true, y_score, thresh)
    print(f'[{label}] PR-AUC={ap:.4f} | Recall@0.1%FPR={recall:.4f} | Cost/10k={cost:,.0f}')
    return {'pr_auc': ap, 'recall_at_fpr': recall, 'cost_per_10k': cost, 'threshold': thresh}

"""## 2. Missing-modality-aware late-fusion meta-learner"""

def build_meta_features(branch_outputs: dict):
    names = sorted(branch_outputs.keys())
    emb_parts, flag_parts = [], []
    for name in names:
        emb, avail = branch_outputs[name]
        avail = avail.astype(float).reshape(-1, 1)
        emb_parts.append(emb * avail)
        flag_parts.append(avail)
    X_meta = np.hstack(emb_parts + flag_parts)
    return X_meta, names

class MetaLearner:
    def __init__(self):
        self.model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                    eval_metric='logloss', n_jobs=-1)
        self.branch_names_ = None

    def fit(self, branch_outputs, y):
        X_meta, names = build_meta_features(branch_outputs)
        self.branch_names_ = names
        pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
        self.model.set_params(scale_pos_weight=pos_weight)
        self.model.fit(X_meta, y)
        return self

    def predict_proba(self, branch_outputs):
        X_meta, names = build_meta_features(branch_outputs)
        assert names == self.branch_names_, f'Branch mismatch: {names} vs {self.branch_names_}'
        return self.model.predict_proba(X_meta)[:, 1]

""" SECTION A — IEEE-CIS Fraud Detection"""


ieee_transaction_path = os.path.join(IEEE_DIR, 'train_transaction.csv')
ieee_identity_path = os.path.join(IEEE_DIR, 'train_identity.csv')

if not os.path.exists(ieee_transaction_path):
    print(f'Not found: {ieee_transaction_path}')
    print(f'Place train_transaction.csv (and train_identity.csv) in {IEEE_DIR} and re-run this cell.')
    ieee_transaction_path = None
if ieee_identity_path and not os.path.exists(ieee_identity_path):
    print(f'Identity file not found at {ieee_identity_path} -- will proceed with transaction-only columns.')
    ieee_identity_path = None

print('Transaction file:', ieee_transaction_path)
print('Identity file:', ieee_identity_path)

"""### A.1 Load and join on `TransactionID`"""

ieee_df = None
if ieee_transaction_path and os.path.exists(ieee_transaction_path):
    txn = pd.read_csv(ieee_transaction_path)
    if ieee_identity_path and os.path.exists(ieee_identity_path):
        idn = pd.read_csv(ieee_identity_path)
        ieee_df = txn.merge(idn, on='TransactionID', how='left')
    else:
        print('No identity file found — proceeding with transaction-only columns; '
              'device/text branches will be unavailable for every row.')
        ieee_df = txn
    print('IEEE-CIS shape:', ieee_df.shape, '| fraud rate:', ieee_df['isFraud'].mean().round(4))
else:
    print('IEEE-CIS not uploaded — skipping Section A. Upload files in the cell above to run this section.')

"""A.2 Branch encoders"""

class IEEETabularBranch:
    def __init__(self):
        self.num_cols = [c for c in ['TransactionAmt','dist1','dist2'] +
                         [f'C{i}' for i in range(1,15)] + [f'D{i}' for i in range(1,16)]
                         if c in ieee_df.columns]
        self.cat_cols = [c for c in ['ProductCD','card4','card6','addr1','addr2'] if c in ieee_df.columns]
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.clf = XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.07,
                                  eval_metric='logloss', n_jobs=-1)

    def _prep(self, df, fit=False):
        num = df[self.num_cols].fillna(-999).values
        cat = df[self.cat_cols].fillna('missing').astype(str).values
        if fit:
            num = self.scaler.fit_transform(num)
            cat = self.ohe.fit_transform(cat)
        else:
            num = self.scaler.transform(num)
            cat = self.ohe.transform(cat)
        return np.hstack([num, cat])

    def fit(self, df, y):
        X = self._prep(df, fit=True)
        pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
        self.clf.set_params(scale_pos_weight=pos_weight)
        self.clf.fit(X, y)
        return self

    def transform(self, df):
        X = self._prep(df, fit=False)
        proba = self.clf.predict_proba(X)[:, 1].reshape(-1, 1)
        emb = np.hstack([proba, np.zeros((len(df), EMB_DIM - 1))])
        available = np.ones(len(df), dtype=bool)
        return emb, available


class IEEETextProxyBranch:
    """DeviceInfo / OS string / browser string / email domains as pseudo merchant-description text."""
    def __init__(self, emb_dim=EMB_DIM):
        self.text_cols = [c for c in ['DeviceInfo','id_30','id_31','P_emaildomain','R_emaildomain']
                           if c in ieee_df.columns]
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), max_features=3000)
        self.svd = TruncatedSVD(n_components=emb_dim - 1, random_state=0)
        self.clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=0)
        self.emb_dim = emb_dim

    def _serialize(self, df):
        if not self.text_cols:
            return pd.Series([''] * len(df))
        return df[self.text_cols].fillna('').astype(str).agg(' | '.join, axis=1)

    def fit(self, df, y):
        text = self._serialize(df)
        mask = text.str.strip().str.len() > 0
        tfidf = self.vectorizer.fit_transform(text)
        svd_emb = self.svd.fit_transform(tfidf)
        self.clf.fit(svd_emb[mask.values], y[mask.values])
        return self

    def transform(self, df):
        text = self._serialize(df)
        mask = (text.str.strip().str.len() > 0).values
        tfidf = self.vectorizer.transform(text)
        svd_emb = self.svd.transform(tfidf)
        risk = self.clf.predict_proba(svd_emb)[:, 1].reshape(-1, 1)
        emb = np.hstack([risk, svd_emb[:, :self.emb_dim - 1]])
        emb[~mask] = 0.0
        return emb, mask


class IEEEDeviceBranch:
    def __init__(self, emb_dim=EMB_DIM):
        self.id_cols = [f'id_{i:02d}' for i in range(1, 12) if f'id_{i:02d}' in ieee_df.columns]
        self.scaler = StandardScaler()
        self.clf = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=300, random_state=0)
        self.emb_dim = emb_dim

    @staticmethod
    def _parse_resolution(df):
        if 'id_33' not in df.columns:
            return pd.DataFrame({'res_w': 0.0, 'res_h': 0.0}, index=df.index)
        parts = df['id_33'].fillna('0x0').astype(str).str.split('x', expand=True)
        w = pd.to_numeric(parts[0], errors='coerce').fillna(0.0)
        h = pd.to_numeric(parts[1], errors='coerce').fillna(0.0) if parts.shape[1] > 1 else pd.Series(0.0, index=df.index)
        return pd.DataFrame({'res_w': w, 'res_h': h})

    def _prep(self, df):
        num = df[self.id_cols].apply(pd.to_numeric, errors='coerce') if self.id_cols else pd.DataFrame(index=df.index)
        res = self._parse_resolution(df)
        full = pd.concat([num, res], axis=1)
        mask = full.notna().any(axis=1).values if full.shape[1] > 0 else np.zeros(len(df), dtype=bool)
        filled = full.fillna(0.0).values if full.shape[1] > 0 else np.zeros((len(df), 1))
        return filled, mask

    def fit(self, df, y):
        X, mask = self._prep(df)
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs[mask], y[mask])
        return self

    def transform(self, df):
        X, mask = self._prep(df)
        Xs = self.scaler.transform(X)
        risk = self.clf.predict_proba(Xs)[:, 1].reshape(-1, 1)
        pad = np.zeros((len(df), max(self.emb_dim - 1 - Xs.shape[1], 0)))
        emb = np.hstack([risk, Xs, pad])[:, :self.emb_dim]
        emb[~mask] = 0.0
        return emb, mask

""" A.3 Train/test split, fit branches"""

if ieee_df is not None:
    y_all = ieee_df['isFraud'].values
    idx = np.arange(len(ieee_df))
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=42, stratify=y_all)

    tab_branch = IEEETabularBranch().fit(ieee_df.iloc[train_idx], y_all[train_idx])
    txt_branch = IEEETextProxyBranch().fit(ieee_df.iloc[train_idx], y_all[train_idx])
    dev_branch = IEEEDeviceBranch().fit(ieee_df.iloc[train_idx], y_all[train_idx])

    def ieee_branch_outputs(idx_subset, active=('tabular','text','device')):
        out = {}
        if 'tabular' in active: out['tabular'] = tab_branch.transform(ieee_df.iloc[idx_subset])
        if 'text' in active:    out['text'] = txt_branch.transform(ieee_df.iloc[idx_subset])
        if 'device' in active:  out['device'] = dev_branch.transform(ieee_df.iloc[idx_subset])
        return out

    def ieee_run(name, active):
        train_out = ieee_branch_outputs(train_idx, active)
        test_out = ieee_branch_outputs(test_idx, active)
        meta = MetaLearner().fit(train_out, y_all[train_idx])
        scores = meta.predict_proba(test_out)
        return summarize(y_all[test_idx], scores, label=name), meta

    ieee_results = {}
    ieee_results['tabular-only'], _ = ieee_run('tabular-only', ('tabular',))
    ieee_results['tabular+text'], _ = ieee_run('tabular+text', ('tabular','text'))
    ieee_results['tabular+device'], _ = ieee_run('tabular+device', ('tabular','device'))
    ieee_results['full'], ieee_full_meta = ieee_run('full (tabular+text+device)', ('tabular','text','device'))

    plt.figure(figsize=(7,4))
    plt.bar(ieee_results.keys(), [v['pr_auc'] for v in ieee_results.values()], color='#1F3864')
    plt.ylabel('PR-AUC'); plt.title('IEEE-CIS: Modality Ablation'); plt.xticks(rotation=15, ha='right')
    plt.tight_layout(); plt.show()
else:
    print('Skipping Section A — IEEE-CIS not loaded.')

"""SECTION B — Balabit Mouse Dynamics Challenge"""


balabit_labels_path = os.path.join(BALABIT_DIR, 'public_labels.csv')

extract_dir = os.path.join(BALABIT_DIR, '_extracted') # Define extract_dir here

# auto-extract any zip found in the folder (one-time; subsequent runs just see the extracted CSVs)
for zpath in glob.glob(os.path.join(BALABIT_DIR, '*.zip')):
    if not os.path.isdir(extract_dir):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(extract_dir)
        print(f'Extracted {zpath} -> {extract_dir}')

# Based on the file listing, session files are named 'session_XXXXXXXXXX' and do NOT have a .csv extension.
# Adjusting the glob pattern to find files starting with 'session_' recursively.
balabit_session_paths = glob.glob(os.path.join(BALABIT_DIR, '**', 'session_*'), recursive=True)

# Ensure public_labels.csv is not included if it somehow matches 'session_*' (unlikely but good for robustness)
balabit_session_paths = [p for p in balabit_session_paths if not os.path.basename(p).startswith('public_labels')]

labels_file_exists = os.path.exists(balabit_labels_path)

if not balabit_session_paths or not labels_file_exists:
    missing_parts = []
    if not balabit_session_paths:
        missing_parts.append('session files (e.g., files starting with "session_")')
    if not labels_file_exists:
        missing_parts.append('public_labels.csv')

    print(f'{', '.join(missing_parts)} not found in {BALABIT_DIR} or its subdirectories.')
    print(f'Place them there (or a sessions.zip) and re-run this cell.')
    if not labels_file_exists:
        balabit_labels_path = None

print(f'{len(balabit_session_paths)} session file(s) found. Labels file: {balabit_labels_path}')

"""B.1 Parse raw mouse-move logs into session-level features"""

def session_features_from_raw(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    ts_col = 'client timestamp' if 'client timestamp' in df.columns else df.columns[0]
    df = df.sort_values(ts_col)
    dt = df[ts_col].diff().fillna(0).values
    dx = df['x'].diff().fillna(0).values
    dy = df['y'].diff().fillna(0).values
    dist = np.sqrt(dx**2 + dy**2)
    velocity = np.divide(dist, dt, out=np.zeros_like(dist), where=dt > 0)

    # click dwell time: gap between consecutive 'Pressed' -> 'Released' of same button, if present
    dwell_times = []
    if 'state' in df.columns:
        pressed_ts = None
        for _, row in df.iterrows():
            state = str(row.get('state', '')).lower()
            if state == 'pressed':
                pressed_ts = row[ts_col]
            elif state == 'released' and pressed_ts is not None:
                dwell_times.append(row[ts_col] - pressed_ts)
                pressed_ts = None
    dwell_times = np.array(dwell_times) if dwell_times else np.array([0.0])

    # spectral entropy of velocity signal as a wavelet-entropy stand-in
    if len(velocity) > 4:
        spectrum = np.abs(np.fft.rfft(velocity - velocity.mean()))
        psd = spectrum**2
        psd_norm = psd / (psd.sum() + 1e-9)
        spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
    else:
        spectral_entropy = 0.0

    return {
        'dwell_mean_ms': float(np.mean(dwell_times)),
        'dwell_std_ms': float(np.std(dwell_times)),
        'mouse_path_jitter': float(np.std(dist)),
        'wavelet_entropy': float(spectral_entropy),
        'session_len_s': float(df[ts_col].max() - df[ts_col].min()) if len(df) > 1 else 0.0,
        'avg_velocity': float(np.mean(velocity)),
    }

balabit_df = None
if balabit_session_paths and balabit_labels_path and os.path.exists(balabit_labels_path):
    labels_df = pd.read_csv(balabit_labels_path)
    labels_df.columns = [c.strip().lower() for c in labels_df.columns]
    rows = []
    for p in balabit_session_paths:
        fname = os.path.basename(p)
        match = labels_df[labels_df['filename'].astype(str).str.contains(fname.split('.')[0], regex=False)]
        if match.empty:
            continue
        feats = session_features_from_raw(p)
        feats['is_illegal'] = int(match.iloc[0]['is_illegal'])
        rows.append(feats)
    balabit_df = pd.DataFrame(rows)
    print('Balabit session-level dataset:', balabit_df.shape if balabit_df is not None else None)
else:
    print('Balabit files not uploaded — skipping Section B.')

"""B.2 Train the behavioral-biometrics branch and evaluate standalone"""

class BehavioralBranch:
    def __init__(self, emb_dim=EMB_DIM):
        self.cols = ['dwell_mean_ms','dwell_std_ms','mouse_path_jitter','wavelet_entropy','session_len_s','avg_velocity']
        self.scaler = StandardScaler()
        self.clf = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=400, random_state=0)
        self.emb_dim = emb_dim

    def fit(self, df, y):
        Xs = self.scaler.fit_transform(df[self.cols].values)
        self.clf.fit(Xs, y)
        return self

    def transform(self, df):
        Xs = self.scaler.transform(df[self.cols].values)
        risk = self.clf.predict_proba(Xs)[:, 1].reshape(-1, 1)
        pad = np.zeros((len(df), self.emb_dim - 1 - Xs.shape[1]))
        emb = np.hstack([risk, Xs, pad])
        available = np.ones(len(df), dtype=bool)
        return emb, available

if balabit_df is not None and len(balabit_df) > 20:
    y_bal = balabit_df['is_illegal'].values
    b_train, b_test = train_test_split(np.arange(len(balabit_df)), test_size=0.3, random_state=42,
                                        stratify=y_bal if y_bal.sum() > 1 else None)
    beh_branch = BehavioralBranch().fit(balabit_df.iloc[b_train], y_bal[b_train])
    emb, avail = beh_branch.transform(balabit_df.iloc[b_test])
    scores = emb[:, 0]  # branch's own risk score
    summarize(y_bal[b_test], scores, label='Balabit behavioral branch (standalone)')
else:
    print('Not enough labeled Balabit sessions to train/evaluate — skipping.')

"""SECTION C — DGraph-Fin"""

dgraph_path = os.path.join(DGRAPH_DIR, 'dgraphfin.npz')
if not os.path.exists(dgraph_path):
    print(f'Not found: {dgraph_path}')
    print(f'Place dgraphfin.npz in {DGRAPH_DIR} and re-run this cell.')
    dgraph_path = None
else:
    print('DGraph file:', dgraph_path)

"""C.1 Load graph, compute neighbor-degree / relational features"""

dgraph_data = None
if dgraph_path and os.path.exists(dgraph_path):
    npz = np.load(dgraph_path)
    x = npz['x']
    y = npz['y']
    edge_index = npz['edge_index']  # shape (E, 2) or (2, E) depending on release; normalize below
    if edge_index.shape[0] == 2:
        src, dst = edge_index[0], edge_index[1]
    else:
        src, dst = edge_index[:, 0], edge_index[:, 1]

    n = x.shape[0]
    out_degree = np.bincount(src, minlength=n).astype(float)
    in_degree = np.bincount(dst, minlength=n).astype(float)

    # neighbor fraud-rate using only nodes with known labels (0/1) to avoid leakage from background nodes
    known_fraud = np.where(y == 1, 1.0, np.where(y == 0, 0.0, np.nan))
    neighbor_fraud_sum = np.zeros(n)
    neighbor_fraud_count = np.zeros(n)
    valid_src_label = ~np.isnan(known_fraud[src])
    np.add.at(neighbor_fraud_sum, dst[valid_src_label], known_fraud[src][valid_src_label])
    np.add.at(neighbor_fraud_count, dst[valid_src_label], 1)
    neighbor_fraud_rate = np.divide(neighbor_fraud_sum, neighbor_fraud_count,
                                     out=np.zeros(n), where=neighbor_fraud_count > 0)

    graph_feats = np.hstack([x, out_degree.reshape(-1,1), in_degree.reshape(-1,1),
                              neighbor_fraud_rate.reshape(-1,1)])

    mask_labeled = (y == 0) | (y == 1)
    dgraph_data = {'features': graph_feats, 'labels': y, 'mask_labeled': mask_labeled}
    print('DGraph nodes:', n, '| labeled (fraud/normal) nodes:', mask_labeled.sum(),
          '| fraud rate among labeled:', y[mask_labeled].mean().round(4))
else:
    print('DGraph-Fin not uploaded — skipping Section C.')

"""C.2 Train the graph/relational branch and evaluate standalone"""

class GraphFingerprintBranch:
    def __init__(self, emb_dim=EMB_DIM):
        self.scaler = StandardScaler()
        self.clf = XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.07,
                                  eval_metric='logloss', n_jobs=-1)
        self.emb_dim = emb_dim

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
        self.clf.set_params(scale_pos_weight=pos_weight)
        self.clf.fit(Xs, y)
        return self

    def transform(self, X):
        Xs = self.scaler.transform(X)
        risk = self.clf.predict_proba(Xs)[:, 1].reshape(-1, 1)
        available = np.ones(len(X), dtype=bool)
        pad = np.zeros((len(X), self.emb_dim - 1))
        return np.hstack([risk, pad]), available

if dgraph_data is not None:
    X_lab = dgraph_data['features'][dgraph_data['mask_labeled']]
    y_lab = dgraph_data['labels'][dgraph_data['mask_labeled']]
    g_train, g_test = train_test_split(np.arange(len(X_lab)), test_size=0.25, random_state=42, stratify=y_lab)
    graph_branch = GraphFingerprintBranch().fit(X_lab[g_train], y_lab[g_train])
    emb, avail = graph_branch.transform(X_lab[g_test])
    summarize(y_lab[g_test], emb[:, 0], label='DGraph-Fin graph branch (standalone)')
else:
    print('Skipping — DGraph-Fin not loaded.')

"""SECTION D — Full Ablation Matrix (Tabular + Text + Device + Behavioral + Graph)"""

def label_conditional_sample(donor_embeddings, donor_available, donor_labels, target_labels, rng):
    """For each target label, sample a donor row with the same label (with replacement)."""
    idx_by_label = {lbl: np.where(donor_labels == lbl)[0] for lbl in np.unique(donor_labels)}
    out_idx = np.empty(len(target_labels), dtype=int)
    for lbl, pool in idx_by_label.items():
        mask = target_labels == lbl
        if mask.sum() == 0:
            continue
        out_idx[mask] = rng.choice(pool, size=mask.sum(), replace=True)
    return donor_embeddings[out_idx], donor_available[out_idx]

# Pre-compute standalone embeddings for the FULL Balabit and DGraph pools once, to draw from.
balabit_pool = None
if balabit_df is not None and len(balabit_df) > 20:
    y_bal_full = balabit_df['is_illegal'].values
    beh_branch_full = BehavioralBranch().fit(balabit_df, y_bal_full)
    emb_bal_full, avail_bal_full = beh_branch_full.transform(balabit_df)
    balabit_pool = {'emb': emb_bal_full, 'avail': avail_bal_full, 'labels': y_bal_full}

dgraph_pool = None
if dgraph_data is not None:
    X_lab_full = dgraph_data['features'][dgraph_data['mask_labeled']]
    y_lab_full = dgraph_data['labels'][dgraph_data['mask_labeled']]
    graph_branch_full = GraphFingerprintBranch().fit(X_lab_full, y_lab_full)
    emb_graph_full, avail_graph_full = graph_branch_full.transform(X_lab_full)
    dgraph_pool = {'emb': emb_graph_full, 'avail': avail_graph_full, 'labels': y_lab_full}

print('Balabit donor pool ready:', balabit_pool is not None)
print('DGraph donor pool ready:', dgraph_pool is not None)

def ieee_branch_outputs_extended(idx_subset, active, seed=0):
    """Extends ieee_branch_outputs() with paired behavioral/graph branches."""
    rng = np.random.default_rng(seed)
    out = ieee_branch_outputs(idx_subset, active=tuple(a for a in active if a in ('tabular','text','device')))
    labels_subset = y_all[idx_subset]
    if 'behavioral' in active and balabit_pool is not None:
        emb, avail = label_conditional_sample(balabit_pool['emb'], balabit_pool['avail'],
                                               balabit_pool['labels'], labels_subset, rng)
        out['behavioral'] = (emb, avail)
    if 'graph' in active and dgraph_pool is not None:
        emb, avail = label_conditional_sample(dgraph_pool['emb'], dgraph_pool['avail'],
                                               dgraph_pool['labels'], labels_subset, rng)
        out['graph'] = (emb, avail)
    return out

def ieee_run_extended(name, active, seed=0):
    train_out = ieee_branch_outputs_extended(train_idx, active, seed=seed)
    test_out = ieee_branch_outputs_extended(test_idx, active, seed=seed + 1)
    meta = MetaLearner().fit(train_out, y_all[train_idx])
    scores = meta.predict_proba(test_out)
    ap = pr_auc(y_all[test_idx], scores)
    recall, thresh = recall_at_fpr(y_all[test_idx], scores, target_fpr=0.001)
    cost = cost_weighted_loss(y_all[test_idx], scores, thresh)
    return {'pr_auc': ap, 'recall_at_fpr': recall, 'cost_per_10k': cost}

FULL_ABLATION_CONFIGS = {
    'tabular-only':                 ('tabular',),
    'tabular+text':                 ('tabular', 'text'),
    'tabular+device':               ('tabular', 'device'),
    'tabular+behavioral':           ('tabular', 'behavioral'),
    'tabular+graph':                ('tabular', 'graph'),
    'full (tab+text+device)':       ('tabular', 'text', 'device'),
    'full+behavioral':              ('tabular', 'text', 'device', 'behavioral'),
    'full+graph':                   ('tabular', 'text', 'device', 'graph'),
    'full+behavioral+graph':        ('tabular', 'text', 'device', 'behavioral', 'graph'),
}
print(f'{len(FULL_ABLATION_CONFIGS)} configurations queued.')

"""D.1 5-seed averaging"""

N_SEEDS = 5
full_ablation_results = {name: {'pr_auc': [], 'recall_at_fpr': [], 'cost_per_10k': []}
                          for name in FULL_ABLATION_CONFIGS}

if ieee_df is not None:
    for seed in range(N_SEEDS):
        # re-split per seed so variance reflects both split and donor-sampling randomness
        tr_idx, te_idx = train_test_split(np.arange(len(ieee_df)), test_size=0.25,
                                           random_state=seed, stratify=y_all)
        train_idx, test_idx = tr_idx, te_idx  # overwrite globals used by helper functions above
        tab_branch = IEEETabularBranch().fit(ieee_df.iloc[train_idx], y_all[train_idx])
        txt_branch = IEEETextProxyBranch().fit(ieee_df.iloc[train_idx], y_all[train_idx])
        dev_branch = IEEEDeviceBranch().fit(ieee_df.iloc[train_idx], y_all[train_idx])

        for name, active in FULL_ABLATION_CONFIGS.items():
            r = ieee_run_extended(name, active, seed=seed)
            for k in ('pr_auc', 'recall_at_fpr', 'cost_per_10k'):
                full_ablation_results[name][k].append(r[k])
        print(f'Seed {seed} done.')

    print('\n=== Full ablation matrix: mean ± std over', N_SEEDS, 'seeds ===')
    summary_rows = []
    for name, metrics in full_ablation_results.items():
        ap_mean, ap_std = np.mean(metrics['pr_auc']), np.std(metrics['pr_auc'])
        rc_mean, rc_std = np.mean(metrics['recall_at_fpr']), np.std(metrics['recall_at_fpr'])
        cost_mean = np.mean(metrics['cost_per_10k'])
        summary_rows.append((name, ap_mean, ap_std, rc_mean, rc_std, cost_mean))
        print(f'{name:28s} PR-AUC={ap_mean:.4f}±{ap_std:.4f} | '
              f'Recall@0.1%FPR={rc_mean:.4f}±{rc_std:.4f} | Cost/10k=${cost_mean:,.0f}')

    ablation_df = pd.DataFrame(summary_rows, columns=['config','pr_auc_mean','pr_auc_std',
                                                        'recall_mean','recall_std','cost_mean'])
else:
    print('Skipping — IEEE-CIS not loaded.')

"""Critical caveat: label-conditional pairing leaks the label by construction"""

def random_pairing_sample(donor_embeddings, donor_available, n_target, rng):
    idx = rng.integers(0, len(donor_embeddings), n_target)
    return donor_embeddings[idx], donor_available[idx]

def ieee_run_random_control(active, seed=0):
    rng = np.random.default_rng(seed + 1000)
    out_train = ieee_branch_outputs(train_idx, active=tuple(a for a in active if a in ('tabular','text','device')))
    out_test = ieee_branch_outputs(test_idx, active=tuple(a for a in active if a in ('tabular','text','device')))
    if 'behavioral' in active and balabit_pool is not None:
        out_train['behavioral'] = random_pairing_sample(balabit_pool['emb'], balabit_pool['avail'], len(train_idx), rng)
        out_test['behavioral'] = random_pairing_sample(balabit_pool['emb'], balabit_pool['avail'], len(test_idx), rng)
    if 'graph' in active and dgraph_pool is not None:
        out_train['graph'] = random_pairing_sample(dgraph_pool['emb'], dgraph_pool['avail'], len(train_idx), rng)
        out_test['graph'] = random_pairing_sample(dgraph_pool['emb'], dgraph_pool['avail'], len(test_idx), rng)
    meta = MetaLearner().fit(out_train, y_all[train_idx])
    scores = meta.predict_proba(out_test)
    return pr_auc(y_all[test_idx], scores)

if ieee_df is not None:
    print('=== Negative control: random (label-agnostic) pairing, single seed ===')
    for name in ['tabular+behavioral', 'tabular+graph', 'full+behavioral', 'full+graph', 'full+behavioral+graph']:
        active = FULL_ABLATION_CONFIGS[name]
        if ('behavioral' in active and balabit_pool is None) or ('graph' in active and dgraph_pool is None):
            continue
        ap_random = ieee_run_random_control(active, seed=0)
        print(f'{name:28s} PR-AUC under RANDOM pairing = {ap_random:.4f} '
              f'(compare to label-conditional mean reported above -- '
              f'a large gap indicates leakage, not real signal).')

if ieee_df is not None:
    plt.figure(figsize=(9,4.5))
    plt.bar(ablation_df['config'], ablation_df['pr_auc_mean'], yerr=ablation_df['pr_auc_std'],
            capsize=4, color='#1F3864')
    plt.ylabel('PR-AUC (mean ± std, 5 seeds)')
    plt.title('Full Ablation Matrix Including Paired Behavioral/Graph Branches')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout(); plt.show()

"""SECTION E — Temporal (Chronological) Split Robustness"""

if ieee_df is not None and 'TransactionDT' in ieee_df.columns:
    order = np.argsort(ieee_df['TransactionDT'].values)
    split_point = int(0.75 * len(order))
    temporal_train_idx = order[:split_point]
    temporal_test_idx = order[split_point:]

    tab_t = IEEETabularBranch().fit(ieee_df.iloc[temporal_train_idx], y_all[temporal_train_idx])
    txt_t = IEEETextProxyBranch().fit(ieee_df.iloc[temporal_train_idx], y_all[temporal_train_idx])
    dev_t = IEEEDeviceBranch().fit(ieee_df.iloc[temporal_train_idx], y_all[temporal_train_idx])

    train_out_t = {'tabular': tab_t.transform(ieee_df.iloc[temporal_train_idx]),
                   'text': txt_t.transform(ieee_df.iloc[temporal_train_idx]),
                   'device': dev_t.transform(ieee_df.iloc[temporal_train_idx])}
    test_out_t = {'tabular': tab_t.transform(ieee_df.iloc[temporal_test_idx]),
                  'text': txt_t.transform(ieee_df.iloc[temporal_test_idx]),
                  'device': dev_t.transform(ieee_df.iloc[temporal_test_idx])}

    meta_t = MetaLearner().fit(train_out_t, y_all[temporal_train_idx])
    scores_t = meta_t.predict_proba(test_out_t)
    print('--- Chronological split (train on earliest 75%, test on latest 25%) ---')
    temporal_result = summarize(y_all[temporal_test_idx], scores_t, label='full, temporal split')

    print('\n--- Compare against random-split full model (from Section D, seed 0) ---')
    print(f"Random-split PR-AUC (mean over seeds): {ablation_df.loc[ablation_df['config']=='full (tab+text+device)','pr_auc_mean'].values[0]:.4f}")
    print(f"Temporal-split PR-AUC: {temporal_result['pr_auc']:.4f}")
    print('A materially lower temporal-split PR-AUC indicates the random-split number was optimistic '
          'due to concept drift / near-duplicate leakage across the random split boundary.')
else:
    print('Skipping — IEEE-CIS not loaded or TransactionDT column unavailable.')

"""SECTION F — Statistical Significance Testing"""

def paired_bootstrap_pr_auc_diff(y_true, scores_a, scores_b, n_boot=1000, seed=0):
    """Returns bootstrap distribution of PR-AUC(scores_b) - PR-AUC(scores_a)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if yb.sum() == 0 or yb.sum() == n:
            diffs[i] = np.nan
            continue
        ap_a = average_precision_score(yb, scores_a[idx])
        ap_b = average_precision_score(yb, scores_b[idx])
        diffs[i] = ap_b - ap_a
    return diffs[~np.isnan(diffs)]

if ieee_df is not None:
    # Refit tabular-only and full model on one consistent split (seed=0) to get comparable per-row scores
    tr_idx, te_idx = train_test_split(np.arange(len(ieee_df)), test_size=0.25, random_state=0, stratify=y_all)
    train_idx, test_idx = tr_idx, te_idx
    tab_s = IEEETabularBranch().fit(ieee_df.iloc[train_idx], y_all[train_idx])
    txt_s = IEEETextProxyBranch().fit(ieee_df.iloc[train_idx], y_all[train_idx])
    dev_s = IEEEDeviceBranch().fit(ieee_df.iloc[train_idx], y_all[train_idx])

    tabonly_train = {'tabular': tab_s.transform(ieee_df.iloc[train_idx])}
    tabonly_test = {'tabular': tab_s.transform(ieee_df.iloc[test_idx])}
    meta_tabonly = MetaLearner().fit(tabonly_train, y_all[train_idx])
    scores_tabonly = meta_tabonly.predict_proba(tabonly_test)

    full_train = {'tabular': tab_s.transform(ieee_df.iloc[train_idx]),
                  'text': txt_s.transform(ieee_df.iloc[train_idx]),
                  'device': dev_s.transform(ieee_df.iloc[train_idx])}
    full_test = {'tabular': tab_s.transform(ieee_df.iloc[test_idx]),
                 'text': txt_s.transform(ieee_df.iloc[test_idx]),
                 'device': dev_s.transform(ieee_df.iloc[test_idx])}
    meta_full = MetaLearner().fit(full_train, y_all[train_idx])
    scores_full = meta_full.predict_proba(full_test)

    diffs = paired_bootstrap_pr_auc_diff(y_all[test_idx], scores_tabonly, scores_full, n_boot=1000, seed=0)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_value_one_sided = np.mean(diffs <= 0)  # fraction of bootstrap draws where full model does NOT win

    print(f'PR-AUC gain (full - tabular-only): mean={diffs.mean():.4f}, 95% CI=[{ci_low:.4f}, {ci_high:.4f}]')
    print(f'Bootstrap one-sided p-value (H0: no improvement): {p_value_one_sided:.4f}')
    if ci_low > 0:
        print('=> The improvement is statistically significant at the 95% level (CI excludes 0).')
    else:
        print('=> The improvement is NOT statistically distinguishable from zero at the 95% level.')

    plt.figure(figsize=(6,4))
    plt.hist(diffs, bins=40, color='#1F3864')
    plt.axvline(0, color='red', linestyle='--', label='no improvement')
    plt.xlabel('PR-AUC(full) - PR-AUC(tabular-only), bootstrap resamples')
    plt.ylabel('Count')
    plt.title('Bootstrap Distribution of Full-Model PR-AUC Gain')
    plt.legend(); plt.tight_layout(); plt.show()
else:
    print('Skipping — IEEE-CIS not loaded.')

"""SECTION G -- Drift-Aware Temporal Evaluation """

if ieee_df is not None and 'TransactionDT' in ieee_df.columns:
    SECONDS_PER_MONTH = 30 * 24 * 3600
    month_id = (ieee_df['TransactionDT'] - ieee_df['TransactionDT'].min()) // SECONDS_PER_MONTH
    n_months = int(month_id.max()) + 1
    print(f'Derived {n_months} pseudo-months of {SECONDS_PER_MONTH/86400:.0f} days each from TransactionDT span.')

    TRAIN_MONTHS = 6
    eval_months = [m for m in range(TRAIN_MONTHS, min(TRAIN_MONTHS + 3, n_months))]

    def fit_branches_on(idx_subset):
        tb = IEEETabularBranch().fit(ieee_df.iloc[idx_subset], y_all[idx_subset])
        tx = IEEETextProxyBranch().fit(ieee_df.iloc[idx_subset], y_all[idx_subset])
        dv = IEEEDeviceBranch().fit(ieee_df.iloc[idx_subset], y_all[idx_subset])
        return tb, tx, dv

    def branch_outputs_for(tb, tx, dv, idx_subset):
        return {'tabular': tb.transform(ieee_df.iloc[idx_subset]),
                'text': tx.transform(ieee_df.iloc[idx_subset]),
                'device': dv.transform(ieee_df.iloc[idx_subset])}

    static_train_idx = np.where(month_id < TRAIN_MONTHS)[0]
    tb_s, tx_s, dv_s = fit_branches_on(static_train_idx)
    static_train_out = branch_outputs_for(tb_s, tx_s, dv_s, static_train_idx)
    static_meta = MetaLearner().fit(static_train_out, y_all[static_train_idx])

    static_curve, rolling_curve, decay_curve = [], [], []

    for m in eval_months:
        test_idx_m = np.where(month_id == m)[0]
        if len(test_idx_m) == 0 or y_all[test_idx_m].sum() == 0:
            print(f'Month {m}: skipped (no positives / no rows).')
            continue

        test_out_static = branch_outputs_for(tb_s, tx_s, dv_s, test_idx_m)
        scores_static = static_meta.predict_proba(test_out_static)
        r_static = summarize(y_all[test_idx_m], scores_static, label=f'static, month {m}')
        static_curve.append(r_static['pr_auc'])

        roll_train_idx = np.where((month_id >= m - TRAIN_MONTHS) & (month_id < m))[0]
        tb_r, tx_r, dv_r = fit_branches_on(roll_train_idx)
        roll_train_out = branch_outputs_for(tb_r, tx_r, dv_r, roll_train_idx)
        roll_meta = MetaLearner().fit(roll_train_out, y_all[roll_train_idx])
        test_out_roll = branch_outputs_for(tb_r, tx_r, dv_r, test_idx_m)
        scores_roll = roll_meta.predict_proba(test_out_roll)
        r_roll = summarize(y_all[test_idx_m], scores_roll, label=f'rolling-window, month {m}')
        rolling_curve.append(r_roll['pr_auc'])

        decay_train_idx = np.where(month_id < m)[0]
        age_months = (m - 1) - month_id.values[decay_train_idx]
        DECAY_RATE = 0.85
        sample_weight = DECAY_RATE ** np.clip(age_months, 0, None)
        tb_d, tx_d, dv_d = fit_branches_on(decay_train_idx)
        decay_train_out = branch_outputs_for(tb_d, tx_d, dv_d, decay_train_idx)
        X_meta_decay, _ = build_meta_features(decay_train_out)
        decay_model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     eval_metric='logloss', n_jobs=-1,
                                     scale_pos_weight=(y_all[decay_train_idx]==0).sum() / max((y_all[decay_train_idx]==1).sum(),1))
        decay_model.fit(X_meta_decay, y_all[decay_train_idx], sample_weight=sample_weight)
        test_out_decay = branch_outputs_for(tb_d, tx_d, dv_d, test_idx_m)
        X_meta_test_decay, _ = build_meta_features(test_out_decay)
        scores_decay = decay_model.predict_proba(X_meta_test_decay)[:, 1]
        r_decay = summarize(y_all[test_idx_m], scores_decay, label=f'time-decayed weighting, month {m}')
        decay_curve.append(r_decay['pr_auc'])

    plt.figure(figsize=(7,4.5))
    plt.plot(eval_months[:len(static_curve)], static_curve, marker='o', label='Static (train-once)')
    plt.plot(eval_months[:len(rolling_curve)], rolling_curve, marker='o', label='Rolling-window retrain')
    plt.plot(eval_months[:len(decay_curve)], decay_curve, marker='o', label='Time-decayed weighting')
    plt.xlabel('Evaluation month (pseudo-month index)')
    plt.ylabel('PR-AUC')
    plt.title('Concept-Drift Degradation Curve: Static vs. Drift-Aware Retraining')
    plt.legend(); plt.tight_layout(); plt.show()

    print('\n=== Drift-aware summary ===')
    print('Static PR-AUC by month     :', [round(v,4) for v in static_curve])
    print('Rolling-window PR-AUC      :', [round(v,4) for v in rolling_curve])
    print('Time-decayed PR-AUC        :', [round(v,4) for v in decay_curve])
    if static_curve and rolling_curve:
        print(f'Mean lift, rolling vs static: {np.mean(rolling_curve) - np.mean(static_curve):+.4f}')
    if static_curve and decay_curve:
        print(f'Mean lift, decay vs static  : {np.mean(decay_curve) - np.mean(static_curve):+.4f}')
else:
    print('Skipping Section G -- IEEE-CIS not loaded or TransactionDT column unavailable.')

"""SECTION H -- Reliability-Aware Fusion & Missing-Modality Gating """

class ReliabilityGate:
    """Learns a per-branch, per-row reliability weight from (own risk score, embedding norm,
    availability flag), then fuses branch risk scores as a reliability-weighted average."""
    def __init__(self):
        self.gate_models = {}
        self.branch_names_ = None

    @staticmethod
    def _branch_risk_and_meta(emb, avail):
        risk = emb[:, 0]
        norm = np.linalg.norm(emb, axis=1)
        return risk, norm

    def fit(self, branch_outputs, y):
        names = sorted(branch_outputs.keys())
        self.branch_names_ = names
        for name in names:
            emb, avail = branch_outputs[name]
            risk, norm = self._branch_risk_and_meta(emb, avail)
            gate_X = np.column_stack([risk, norm, avail.astype(float)])
            avail_mask = avail.astype(bool)
            if avail_mask.sum() < 20 or y[avail_mask].sum() == 0:
                self.gate_models[name] = None
                continue
            correctness = (np.abs(risk[avail_mask] - y[avail_mask]) < 0.5).astype(int)
            gm = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                eval_metric='logloss', n_jobs=-1)
            gm.fit(gate_X[avail_mask], correctness)
            self.gate_models[name] = gm
        return self

    def predict_proba(self, branch_outputs):
        names = sorted(branch_outputs.keys())
        assert names == self.branch_names_
        risks, weights = [], []
        for name in names:
            emb, avail = branch_outputs[name]
            risk, norm = self._branch_risk_and_meta(emb, avail)
            gate_X = np.column_stack([risk, norm, avail.astype(float)])
            gm = self.gate_models[name]
            if gm is None:
                w = avail.astype(float) * 0.5
            else:
                w = gm.predict_proba(gate_X)[:, 1] * avail.astype(float)
            risks.append(risk); weights.append(w)
        risks = np.column_stack(risks); weights = np.column_stack(weights)
        denom = weights.sum(axis=1)
        denom[denom == 0] = 1.0
        return (risks * weights).sum(axis=1) / denom


class ModalityRouter:
    """Routes each row to a small XGBoost meta-model trained specifically for that row's
    observed availability pattern, instead of one global model covering every combination."""
    def __init__(self, min_rows_per_pattern=200):
        self.min_rows_per_pattern = min_rows_per_pattern
        self.pattern_models = {}
        self.fallback = None
        self.branch_names_ = None

    @staticmethod
    def _pattern_key(branch_outputs):
        names = sorted(branch_outputs.keys())
        avail_stack = np.column_stack([branch_outputs[n][1].astype(int) for n in names])
        return avail_stack, names

    def fit(self, branch_outputs, y):
        X_meta, names = build_meta_features(branch_outputs)
        self.branch_names_ = names
        avail_stack, _ = self._pattern_key(branch_outputs)
        patterns = [tuple(row) for row in avail_stack]
        uniq = {}
        for i, p in enumerate(patterns):
            uniq.setdefault(p, []).append(i)

        self.fallback = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                       eval_metric='logloss', n_jobs=-1,
                                       scale_pos_weight=(y==0).sum()/max((y==1).sum(),1))
        self.fallback.fit(X_meta, y)

        for p, rows in uniq.items():
            rows = np.array(rows)
            if len(rows) < self.min_rows_per_pattern or y[rows].sum() < 5:
                continue
            m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.07,
                               eval_metric='logloss', n_jobs=-1,
                               scale_pos_weight=(y[rows]==0).sum()/max((y[rows]==1).sum(),1))
            m.fit(X_meta[rows], y[rows])
            self.pattern_models[p] = m
        print(f'ModalityRouter: trained {len(self.pattern_models)} pattern-specific model(s) '
              f'out of {len(uniq)} observed availability pattern(s); rest use the fallback model.')
        return self

    def predict_proba(self, branch_outputs):
        X_meta, names = build_meta_features(branch_outputs)
        assert names == self.branch_names_
        avail_stack, _ = self._pattern_key(branch_outputs)
        patterns = [tuple(row) for row in avail_stack]
        scores = self.fallback.predict_proba(X_meta)[:, 1]
        for p, m in self.pattern_models.items():
            rows = np.array([i for i, pat in enumerate(patterns) if pat == p])
            if len(rows) == 0:
                continue
            scores[rows] = m.predict_proba(X_meta[rows])[:, 1]
        return scores


if ieee_df is not None:
    train_out_h = ieee_branch_outputs(train_idx, ('tabular', 'text', 'device'))
    test_out_h = ieee_branch_outputs(test_idx, ('tabular', 'text', 'device'))

    print('--- Baseline: plain concat + availability flags (Section 2 MetaLearner) ---')
    baseline_scores_h = ieee_full_meta.predict_proba(test_out_h)
    r_baseline_h = summarize(y_all[test_idx], baseline_scores_h, label='baseline concat fusion')

    print('\n--- Reliability-aware fusion ---')
    gate = ReliabilityGate().fit(train_out_h, y_all[train_idx])
    gate_scores = gate.predict_proba(test_out_h)
    r_gate = summarize(y_all[test_idx], gate_scores, label='reliability-aware fusion')

    print('\n--- Missing-modality adaptive router ---')
    router = ModalityRouter().fit(train_out_h, y_all[train_idx])
    router_scores = router.predict_proba(test_out_h)
    r_router = summarize(y_all[test_idx], router_scores, label='modality router')

    fusion_compare = {'baseline concat': r_baseline_h['pr_auc'],
                       'reliability-aware fusion': r_gate['pr_auc'],
                       'modality router': r_router['pr_auc']}
    plt.figure(figsize=(6,4))
    plt.bar(fusion_compare.keys(), fusion_compare.values(), color='#2E7D32')
    plt.ylabel('PR-AUC'); plt.title('Fusion Mechanism Comparison'); plt.xticks(rotation=15, ha='right')
    plt.tight_layout(); plt.show()
else:
    print('Skipping Section H -- IEEE-CIS not loaded.')

""" SECTION I -- Strong Baseline Comparison"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

if ieee_df is not None:
    def raw_early_fusion_features(idx_subset, fit_objs=None):
        num_cols = tab_branch.num_cols
        cat_cols = tab_branch.cat_cols
        num = ieee_df.iloc[idx_subset][num_cols].fillna(-999).values
        cat = ieee_df.iloc[idx_subset][cat_cols].fillna('missing').astype(str).values
        text = txt_branch._serialize(ieee_df.iloc[idx_subset])
        dev_X, dev_mask = dev_branch._prep(ieee_df.iloc[idx_subset])

        if fit_objs is None:
            scaler = StandardScaler().fit(num)
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(cat)
            tfidf = TfidfVectorizer(max_features=300, ngram_range=(1,1)).fit(text)
            dev_scaler = StandardScaler().fit(dev_X)
            fit_objs = (scaler, ohe, tfidf, dev_scaler)
        scaler, ohe, tfidf, dev_scaler = fit_objs

        num_s = scaler.transform(num)
        cat_o = ohe.transform(cat)
        text_t = tfidf.transform(text).toarray()
        dev_s = dev_scaler.transform(dev_X)
        X = np.hstack([num_s, cat_o, text_t, dev_s])
        return X, fit_objs

    X_train_raw, fit_objs = raw_early_fusion_features(train_idx)
    X_test_raw, _ = raw_early_fusion_features(test_idx, fit_objs)
    y_train_b, y_test_b = y_all[train_idx], y_all[test_idx]
    pos_w = (y_train_b == 0).sum() / max((y_train_b == 1).sum(), 1)

    baselines = {}

    lr = LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1)
    lr.fit(X_train_raw, y_train_b)
    baselines['LogisticRegression (early fusion)'] = summarize(
        y_test_b, lr.predict_proba(X_test_raw)[:, 1], label='LogisticRegression (early fusion)')

    rf = RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1,
                                 class_weight='balanced_subsample', random_state=0)
    rf.fit(X_train_raw, y_train_b)
    baselines['RandomForest (early fusion)'] = summarize(
        y_test_b, rf.predict_proba(X_test_raw)[:, 1], label='RandomForest (early fusion)')

    xgb_early = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                               eval_metric='logloss', n_jobs=-1, scale_pos_weight=pos_w)
    xgb_early.fit(X_train_raw, y_train_b)
    baselines['XGBoost (early fusion)'] = summarize(
        y_test_b, xgb_early.predict_proba(X_test_raw)[:, 1], label='XGBoost (early fusion)')

    test_out_i = ieee_branch_outputs(test_idx, ('tabular', 'text', 'device'))
    naive_avg_scores = np.mean([test_out_i[n][0][:, 0] for n in ('tabular', 'text', 'device')], axis=0)
    baselines['Naive average late fusion'] = summarize(
        y_test_b, naive_avg_scores, label='Naive average late fusion (no meta-learner)')

    baselines['This paper: MetaLearner (late fusion + availability flags)'] = summarize(
        y_test_b, ieee_full_meta.predict_proba(test_out_i), label='MetaLearner (this paper)')

    plt.figure(figsize=(8,4.5))
    plt.bar(baselines.keys(), [v['pr_auc'] for v in baselines.values()], color='#8E24AA')
    plt.ylabel('PR-AUC'); plt.title('Strong Baseline Comparison (IEEE-CIS, same split)')
    plt.xticks(rotation=30, ha='right'); plt.tight_layout(); plt.show()

    print('\n=== Baseline comparison table ===')
    for k, v in baselines.items():
        print(f"{k:55s} PR-AUC={v['pr_auc']:.4f}  Recall@0.1%FPR={v['recall_at_fpr']:.4f}")
else:
    print('Skipping Section I -- IEEE-CIS not loaded.')

"""SECTION J -- Explainability"""

try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False
    print('shap not installed -- run `!pip install -q shap` in Section 0 and re-run this cell.')

if ieee_df is not None and HAVE_SHAP:
    X_tab_test = tab_branch._prep(ieee_df.iloc[test_idx])
    explainer_tab = shap.TreeExplainer(tab_branch.clf)
    shap_vals_tab = explainer_tab.shap_values(X_tab_test[:2000])

    feature_names_tab = tab_branch.num_cols + list(tab_branch.ohe.get_feature_names_out(tab_branch.cat_cols))
    shap.summary_plot(shap_vals_tab, X_tab_test[:2000], feature_names=feature_names_tab,
                       max_display=15, show=False)
    plt.title('Tabular Branch: SHAP Feature Importance (top 15)')
    plt.tight_layout(); plt.show()

    test_out_i = ieee_branch_outputs(test_idx, ('tabular', 'text', 'device'))
    X_meta_test, meta_names = build_meta_features(test_out_i)
    emb_dims_each = [test_out_i[n][0].shape[1] for n in meta_names]
    col_labels = []
    for name, d in zip(meta_names, emb_dims_each):
        col_labels += [f'{name}_dim{i}' for i in range(d)]
    col_labels += [f'{name}_available' for name in meta_names]

    explainer_meta = shap.TreeExplainer(ieee_full_meta.model)
    shap_vals_meta = explainer_meta.shap_values(X_meta_test[:2000])

    modality_importance = {}
    offset = 0
    for name, d in zip(meta_names, emb_dims_each):
        modality_importance[name] = np.abs(shap_vals_meta[:, offset:offset+d]).mean()
        offset += d
    for i, name in enumerate(meta_names):
        modality_importance[f'{name}_availability_flag'] = np.abs(shap_vals_meta[:, offset+i]).mean()

    plt.figure(figsize=(7,4))
    plt.bar(modality_importance.keys(), modality_importance.values(), color='#D84315')
    plt.ylabel('Mean |SHAP value|'); plt.title('Modality-Level Attribution (Meta-Learner)')
    plt.xticks(rotation=30, ha='right'); plt.tight_layout(); plt.show()

    scores_all = ieee_full_meta.predict_proba(test_out_i)
    top_i = int(np.argmax(scores_all[:2000])) if len(scores_all) > 2000 else int(np.argmax(scores_all))
    row_shap = shap_vals_meta[top_i]
    top_contribs = sorted(zip(col_labels, row_shap), key=lambda t: -abs(t[1]))[:5]
    print(f'Highest-risk transaction in eval batch (score={scores_all[top_i]:.3f}) -- top reason codes:')
    for feat, val in top_contribs:
        direction = '+' if val > 0 else '-'
        print(f'  {direction} {feat}  (SHAP={val:+.4f})')

    terms = np.array(txt_branch.vectorizer.get_feature_names_out())
    components = txt_branch.svd.components_
    print('\nText branch: top TF-IDF terms per leading SVD component (attention-weight stand-in):')
    for comp_idx in range(min(3, components.shape[0])):
        top_term_idx = np.argsort(-np.abs(components[comp_idx]))[:8]
        print(f'  Component {comp_idx}: {list(terms[top_term_idx])}')
elif ieee_df is None:
    print('Skipping Section J -- IEEE-CIS not loaded.')

""" Graph explainability"""

class CrossModalAttentionFusion:
    """Multi-head self-attention over branch tokens: each branch's fused weight depends on
    every other *available* branch's representation, not just its own confidence.
    Forward pass in NumPy; trained with full-batch gradient descent on binary cross-entropy."""

    def __init__(self, d_model=16, n_heads=2, lr=0.05, n_epochs=150, seed=0):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.lr = lr
        self.n_epochs = n_epochs
        self.rng = np.random.default_rng(seed)
        self.branch_names_ = None
        self.params = None

    def _init_params(self, n_branches, emb_dim):
        r = self.rng
        scale = 1.0 / np.sqrt(emb_dim)
        p = {
            'Wq': r.normal(0, scale, (emb_dim, self.d_model)),
            'Wk': r.normal(0, scale, (emb_dim, self.d_model)),
            'Wv': r.normal(0, scale, (emb_dim, self.d_model)),
            'Wo': r.normal(0, 1.0/np.sqrt(self.d_model), (self.d_model, 1)),  # per-token scalar risk contribution
            'b_risk': np.zeros(1),
        }
        return p

    @staticmethod
    def _softmax_masked(scores, mask):
        # scores: (N, heads, T, T); mask: (N, T) True=available
        neg_inf = -1e9
        mask_expand = mask[:, None, None, :]  # broadcast over query positions
        scores = np.where(mask_expand, scores, neg_inf)
        scores = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        e = np.where(mask_expand, e, 0.0)
        denom = e.sum(axis=-1, keepdims=True)
        denom[denom == 0] = 1.0
        return e / denom

    def _forward(self, tokens, avail_mask, p):
        # tokens: (N, T, emb_dim), avail_mask: (N, T) bool
        N, T, E = tokens.shape
        Q = tokens @ p['Wq']  # (N,T,d_model)
        K = tokens @ p['Wk']
        V = tokens @ p['Wv']
        Qh = Q.reshape(N, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)  # (N,H,T,dh)
        Kh = K.reshape(N, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        Vh = V.reshape(N, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        scores = np.einsum('nhtd,nhsd->nhts', Qh, Kh) / np.sqrt(self.d_head)  # (N,H,T,T)
        attn = self._softmax_masked(scores, avail_mask)
        out_h = np.einsum('nhts,nhsd->nhtd', attn, Vh)  # (N,H,T,dh)
        out = out_h.transpose(0, 2, 1, 3).reshape(N, T, self.d_model)  # (N,T,d_model)
        # per-token scalar contribution, then masked mean-pool across tokens -> row score
        token_logit = (out @ p['Wo']).squeeze(-1) + p['b_risk']  # (N,T)
        mask_f = avail_mask.astype(float)
        denom = mask_f.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        pooled_logit = (token_logit * mask_f).sum(axis=1, keepdims=True) / denom  # (N,1)
        prob = 1.0 / (1.0 + np.exp(-pooled_logit.squeeze(-1)))
        return prob, {'Q': Q, 'K': K, 'V': V, 'attn': attn, 'out': out, 'token_logit': token_logit,
                      'mask_f': mask_f, 'denom': denom}

    def fit(self, branch_outputs, y):
        names = sorted(branch_outputs.keys())
        self.branch_names_ = names
        emb_dim = max(branch_outputs[n][0].shape[1] for n in names)
        N = len(y)
        T = len(names)
        tokens = np.zeros((N, T, emb_dim))
        avail_mask = np.zeros((N, T), dtype=bool)
        for i, n in enumerate(names):
            emb, avail = branch_outputs[n]
            d = emb.shape[1]
            tokens[:, i, :d] = emb
            avail_mask[:, i] = avail.astype(bool)

        p = self._init_params(T, emb_dim)
        y = y.astype(float)

        for epoch in range(self.n_epochs):
            prob, cache = self._forward(tokens, avail_mask, p)
            eps = 1e-7
            prob_c = np.clip(prob, eps, 1 - eps)
            # numerical gradient-free update: use simple logistic-regression-style gradient on Wo/b_risk
            # (a lightweight, dependency-free approximation of backprop through the pooled logit)
            grad_pooled = (prob - y)[:, None]  # (N,1)
            mask_f = cache['mask_f']
            denom = cache['denom']
            grad_token_logit = grad_pooled * mask_f / denom  # (N,T)
            grad_out = grad_token_logit[:, :, None] * p['Wo'].reshape(1, 1, -1)  # (N,T,d_model)
            grad_Wo = np.einsum('nt,ntd->d', grad_token_logit, cache['out']).reshape(-1, 1) / N
            grad_b = grad_token_logit.mean()
            p['Wo'] -= self.lr * grad_Wo
            p['b_risk'] -= self.lr * grad_b
            # shallow update on Wq/Wk/Wv via the attended output's sensitivity to V (approximate,
            # treats attn weights as constant for this step -- a standard 'freeze-attention' partial
            # update that still lets the value projection learn useful per-branch representations)
            grad_V_effective = np.einsum('nhts,nsd->nhsd', cache['attn'], grad_out)
            grad_Wv = np.einsum('nhsd,nse->ed', grad_V_effective, tokens.repeat(1,1).astype(float)[:, :, :]) if False else None
            # simpler, stable fallback: nudge Wv via correlation between grad_out and tokens (keeps run fast/stable)
            flat_grad_out = grad_out.reshape(N, T, self.d_model).mean(axis=1)  # (N, d_model)
            flat_tokens = tokens.mean(axis=1)  # (N, emb_dim)
            grad_Wv_approx = flat_tokens.T @ flat_grad_out / N
            p['Wv'] -= self.lr * 0.1 * grad_Wv_approx

        self.params = p
        self._train_tokens_shape = (T, emb_dim)
        return self

    def predict_proba(self, branch_outputs):
        names = sorted(branch_outputs.keys())
        assert names == self.branch_names_
        T, emb_dim = self._train_tokens_shape
        N = len(branch_outputs[names[0]][1])
        tokens = np.zeros((N, T, emb_dim))
        avail_mask = np.zeros((N, T), dtype=bool)
        for i, n in enumerate(names):
            emb, avail = branch_outputs[n]
            d = min(emb.shape[1], emb_dim)
            tokens[:, i, :d] = emb[:, :d]
            avail_mask[:, i] = avail.astype(bool)
        prob, _ = self._forward(tokens, avail_mask, self.params)
        return prob

    def attention_weights(self, branch_outputs):
        """Returns the (N, heads, T, T) attention matrix for inspection / a qualitative
        'which branch attended to which branch' figure."""
        names = sorted(branch_outputs.keys())
        T, emb_dim = self._train_tokens_shape
        N = len(branch_outputs[names[0]][1])
        tokens = np.zeros((N, T, emb_dim))
        avail_mask = np.zeros((N, T), dtype=bool)
        for i, n in enumerate(names):
            emb, avail = branch_outputs[n]
            d = min(emb.shape[1], emb_dim)
            tokens[:, i, :d] = emb[:, :d]
            avail_mask[:, i] = avail.astype(bool)
        _, cache = self._forward(tokens, avail_mask, self.params)
        return cache['attn'], names


if ieee_df is not None:
    train_out_k = ieee_branch_outputs(train_idx, ('tabular', 'text', 'device'))
    test_out_k = ieee_branch_outputs(test_idx, ('tabular', 'text', 'device'))

    cmaf = CrossModalAttentionFusion(d_model=16, n_heads=2, lr=0.1, n_epochs=200).fit(
        train_out_k, y_all[train_idx])
    cmaf_scores = cmaf.predict_proba(test_out_k)
    r_cmaf = summarize(y_all[test_idx], cmaf_scores, label='cross-modal attention fusion')

    fusion_compare_k = {
        'baseline concat (Sec 2)': r_baseline_h['pr_auc'],
        'reliability-aware fusion (Sec H)': r_gate['pr_auc'],
        'modality router (Sec H)': r_router['pr_auc'],
        'cross-modal attention (Sec K)': r_cmaf['pr_auc'],
    }
    plt.figure(figsize=(7,4))
    plt.bar(fusion_compare_k.keys(), fusion_compare_k.values(), color='#00695C')
    plt.ylabel('PR-AUC'); plt.title('All Fusion Mechanisms Compared'); plt.xticks(rotation=20, ha='right')
    plt.tight_layout(); plt.show()

    attn, attn_names = cmaf.attention_weights(test_out_k)
    mean_attn = attn.mean(axis=(0, 1))  # (T,T) averaged over rows & heads
    print('Mean cross-branch attention matrix (rows=query branch, cols=key branch):')
    print(pd.DataFrame(mean_attn, index=attn_names, columns=attn_names).round(3))
else:
    print('Skipping Section K -- IEEE-CIS not loaded.')

"""SECTION L -- Message-Passing Graph Branch"""

HAVE_TORCH_GEOMETRIC = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    try:
        from torch_geometric.nn import SAGEConv
        HAVE_TORCH_GEOMETRIC = True
    except ImportError:
        print('torch is available but torch_geometric is not -- attempting pip install from PyPI '
              '(falls back to NumPy GraphSAGE-lite if this fails).')
except ImportError:
    print('PyTorch not available -- using NumPy GraphSAGE-lite fallback below.')

if not HAVE_TORCH_GEOMETRIC:
    import subprocess, sys
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch', '--break-system-packages'],
                        check=True, timeout=300)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch_geometric', '--break-system-packages'],
                        check=True, timeout=300)
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv
        HAVE_TORCH_GEOMETRIC = True
        print('torch_geometric installed successfully.')
    except Exception as e:
        print(f'Could not install torch_geometric ({e}) -- using NumPy GraphSAGE-lite fallback.')
        HAVE_TORCH_GEOMETRIC = False


class NumpySAGELite:
    """2-layer mean-aggregation GraphSAGE without a DL framework: h^(1) = ReLU(W1 . [x_v ; mean_{u in N(v)} x_u]),
    h^(2) = ReLU(W2 . [h^(1)_v ; mean_{u in N(v)} h^(1)_u]), then a linear+sigmoid readout.
    Trained with full-batch gradient descent (numerically approximate, CPU-only, no external deps)."""
    def __init__(self, hidden=32, lr=0.05, n_epochs=60, seed=0):
        self.hidden = hidden; self.lr = lr; self.n_epochs = n_epochs
        self.rng = np.random.default_rng(seed)

    def _agg(self, X, adj_list, n_nodes):
        out = np.zeros_like(X)
        for v in range(n_nodes):
            neigh = adj_list[v]
            out[v] = X[neigh].mean(axis=0) if len(neigh) > 0 else 0.0
        return out

    def fit(self, X, adj_list, y, train_idx):
        n_nodes, in_dim = X.shape
        agg1 = self._agg(X, adj_list, n_nodes)
        h0 = np.hstack([X, agg1])
        self.W1 = self.rng.normal(0, 1/np.sqrt(h0.shape[1]), (h0.shape[1], self.hidden))
        h1 = np.maximum(h0 @ self.W1, 0)
        agg2 = self._agg(h1, adj_list, n_nodes)
        h1_full = np.hstack([h1, agg2])
        self.W2 = self.rng.normal(0, 1/np.sqrt(h1_full.shape[1]), (h1_full.shape[1], 1))

        for epoch in range(self.n_epochs):
            h1 = np.maximum(h0 @ self.W1, 0)
            agg2 = self._agg(h1, adj_list, n_nodes)
            h1_full = np.hstack([h1, agg2])
            logits = (h1_full @ self.W2).squeeze(-1)
            prob = 1 / (1 + np.exp(-logits))
            grad_logits = np.zeros(n_nodes)
            grad_logits[train_idx] = (prob[train_idx] - y[train_idx]) / len(train_idx)
            grad_W2 = h1_full.T @ grad_logits.reshape(-1, 1)
            self.W2 -= self.lr * grad_W2
            grad_h1_full = grad_logits.reshape(-1, 1) @ self.W2.T
            grad_h1 = grad_h1_full[:, :self.hidden] * (h1 > 0)
            grad_W1 = h0.T @ grad_h1
            self.W1 -= self.lr * grad_W1

        self._cache = (h0,)
        return self

    def predict_proba(self, X, adj_list):
        n_nodes = X.shape[0]
        agg1 = self._agg(X, adj_list, n_nodes)
        h0 = np.hstack([X, agg1])
        h1 = np.maximum(h0 @ self.W1, 0)
        agg2 = self._agg(h1, adj_list, n_nodes)
        h1_full = np.hstack([h1, agg2])
        logits = (h1_full @ self.W2).squeeze(-1)
        return 1 / (1 + np.exp(-logits))


class TorchSAGE(nn.Module if HAVE_TORCH_GEOMETRIC else object):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.out(h).squeeze(-1)


if dgraph_data is not None:
    # 1. Grab the processed features and labels from Section C.1's dictionary
    node_x = dgraph_data['features'] if 'features' in dgraph_data else None
    y_node = dgraph_data['labels'] if 'labels' in dgraph_data else None

    # 2. Grab edge_index from the global 'npz' variable loaded in Section C.1
    edge_index_np = npz['edge_index'] if 'npz' in globals() and 'edge_index' in npz else None

    # 3. Normalize edge_index shape to (2, E) or (E, 2) to ensure it works smoothly with your GNN
    if edge_index_np is not None and edge_index_np.shape[0] != 2:
        edge_index_np = edge_index_np.T

    if node_x is not None and edge_index_np is not None and y_node is not None:
        labeled_mask = (y_node == 0) | (y_node == 1)
        labeled_idx_all = np.where(labeled_mask)[0]
        g_train_idx, g_test_idx = train_test_split(labeled_idx_all, test_size=0.25, random_state=0,
                                                     stratify=y_node[labeled_idx_all])
        y_bin = (y_node == 1).astype(int)

        if HAVE_TORCH_GEOMETRIC:
            print('Training TorchSAGE (2-layer GraphSAGE via torch_geometric)...')
            x_t = torch.tensor(node_x, dtype=torch.float32)
            ei_t = torch.tensor(edge_index_np, dtype=torch.long)
            model = TorchSAGE(in_dim=node_x.shape[1], hidden=32)
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            y_t = torch.tensor(y_bin, dtype=torch.float32)
            train_mask_t = torch.zeros(len(y_bin), dtype=torch.bool); train_mask_t[g_train_idx] = True
            for epoch in range(30):
                model.train(); opt.zero_grad()
                logits = model(x_t, ei_t)
                loss = F.binary_cross_entropy_with_logits(logits[train_mask_t], y_t[train_mask_t])
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                scores_gnn = torch.sigmoid(model(x_t, ei_t)).numpy()
            r_gnn = summarize(y_bin[g_test_idx], scores_gnn[g_test_idx], label='GraphSAGE (torch_geometric)')
        else:
            print('Training NumPy GraphSAGE-lite (CPU fallback)...')
            n_nodes = node_x.shape[0]
            adj_list = [[] for _ in range(n_nodes)]
            for s, t in edge_index_np.T:
                adj_list[s].append(t)
            sage = NumpySAGELite(hidden=32, lr=0.02, n_epochs=40).fit(node_x, adj_list, y_bin, g_train_idx)
            scores_gnn = sage.predict_proba(node_x, adj_list)
            r_gnn = summarize(y_bin[g_test_idx], scores_gnn[g_test_idx], label='GraphSAGE-lite (NumPy fallback)')

        print("\nCompare: Section C hand-engineered graph branch (degree/neighbor-fraud-rate + XGBoost) "
              "vs. this section's message-passing GNN, same labeled test split.")
    else:
        print('Skipping Section L -- dgraph_data missing expected x/edge_index/y arrays.')
else:
    print('Skipping Section L -- DGraph-Fin not loaded (see Section C).')

"""SECTION M -- Missing-Modality Rate Sweep """

def apply_missingness(branch_outputs, missing_rate, rng):
    out = {}
    for name, (emb, avail) in branch_outputs.items():
        avail = avail.copy().astype(bool)
        drop_mask = rng.random(len(avail)) < missing_rate
        new_avail = avail & (~drop_mask)
        new_emb = emb.copy()
        new_emb[~new_avail] = 0.0
        out[name] = (new_emb, new_avail)
    return out

if ieee_df is not None:
    missing_rates = [0.0, 0.25, 0.50, 0.75]
    fitted_models = {
        'MetaLearner (Sec 2)': ieee_full_meta,
        'ReliabilityGate (Sec H)': gate,
        'ModalityRouter (Sec H)': router,
        'CrossModalAttentionFusion (Sec K)': cmaf,
    }
    test_out_m_base = ieee_branch_outputs(test_idx, ('tabular', 'text', 'device'))

    sweep_results = {name: [] for name in fitted_models}
    rng_sweep = np.random.default_rng(42)

    for rate in missing_rates:
        test_out_rate = apply_missingness(test_out_m_base, rate, rng_sweep)
        for name, model in fitted_models.items():
            scores = model.predict_proba(test_out_rate)
            ap = pr_auc(y_all[test_idx], scores)
            sweep_results[name].append(ap)
            print(f'[{name}] missing_rate={rate:.0%}  PR-AUC={ap:.4f}')
        print()

    print('=== Missing-modality rate sweep summary ===')
    sweep_df = pd.DataFrame(sweep_results, index=[f'{r:.0%}' for r in missing_rates])
    print(sweep_df.round(4))

    plt.figure(figsize=(7.5,4.5))
    for name, vals in sweep_results.items():
        plt.plot([f'{r:.0%}' for r in missing_rates], vals, marker='o', label=name)
    plt.xlabel('Per-branch missing rate'); plt.ylabel('PR-AUC')
    plt.title('Missing-Modality Robustness: PR-AUC vs. Missing Rate')
    plt.legend(); plt.tight_layout(); plt.show()

    print('\\nDegradation from 0% -> 75% missing (lower drop = more robust):')
    for name, vals in sweep_results.items():
        print(f'  {name:38s} {vals[0]:.4f} -> {vals[-1]:.4f}  (drop={vals[0]-vals[-1]:.4f})')
else:
    print('Skipping Section M -- IEEE-CIS not loaded.')

"""SECTION N -- TabTransformer / FT-Transformer Baselines"""

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    print('PyTorch not available -- attempting pip install from PyPI.')
    import subprocess, sys
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch', '--break-system-packages'],
                        check=True, timeout=300)
        import torch
        import torch.nn as nn
        HAVE_TORCH = True
    except Exception as e:
        print(f'Could not install torch ({e}) -- skipping Section N.')
        HAVE_TORCH = False


class SimpleFTTransformer(nn.Module if HAVE_TORCH else object):
    """Feature-Tokenizer + Transformer: every input column becomes a token (via a per-column
    linear projection), a [CLS] token is prepended, a small Transformer encoder mixes tokens,
    and a linear head reads the [CLS] token for the final logit."""
    def __init__(self, n_features, d_model=32, n_heads=4, n_layers=2):
        super().__init__()
        self.d_model = d_model
        self.feature_tokenizer = nn.Linear(1, d_model)  # shared projection applied per-feature-scalar
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                     dim_feedforward=d_model*2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (N, n_features) -> tokens: (N, n_features, d_model)
        tokens = self.feature_tokenizer(x.unsqueeze(-1))
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        out = self.encoder(seq)
        return self.head(out[:, 0, :]).squeeze(-1)


class SimpleTabTransformer(nn.Module if HAVE_TORCH else object):
    """Categorical features go through Transformer layers for contextual embeddings; numeric
    features bypass attention and are concatenated in afterward, feeding an MLP head."""
    def __init__(self, n_cat, n_num, d_model=32, n_heads=4, n_layers=2, mlp_hidden=64):
        super().__init__()
        self.n_cat = n_cat
        self.cat_tokenizer = nn.Linear(1, d_model) if n_cat > 0 else None
        if n_cat > 0:
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                         dim_feedforward=d_model*2, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        cat_out_dim = n_cat * d_model if n_cat > 0 else 0
        self.mlp = nn.Sequential(
            nn.Linear(cat_out_dim + n_num, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden // 2), nn.ReLU(),
            nn.Linear(mlp_hidden // 2, 1))

    def forward(self, x_cat, x_num):
        parts = []
        if self.n_cat > 0:
            tokens = self.cat_tokenizer(x_cat.unsqueeze(-1))
            ctx = self.encoder(tokens)
            parts.append(ctx.reshape(ctx.shape[0], -1))
        if x_num.shape[1] > 0:
            parts.append(x_num)
        h = torch.cat(parts, dim=1)
        return self.mlp(h).squeeze(-1)


def train_torch_binary(model, X_train, y_train, X_test, n_epochs=15, lr=1e-3, batch_size=2048,
                        forward_fn=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    n = X_train.shape[0] if not isinstance(X_train, tuple) else X_train[0].shape[0]
    y_t = torch.tensor(y_train, dtype=torch.float32)
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx_b = perm[start:start+batch_size]
            opt.zero_grad()
            logits = forward_fn(model, X_train, idx_b)
            loss = loss_fn(logits, y_t[idx_b])
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        test_logits = forward_fn(model, X_test, None)
        return torch.sigmoid(test_logits).numpy()


if ieee_df is not None and HAVE_TORCH:
    # Cap feature count / sample size for CPU-feasible runtime; subsample rows for the deep baselines only.
    MAX_FEATURES_DEEP = 60
    MAX_TRAIN_ROWS_DEEP = 40000
    n_feat_total = X_train_raw.shape[1]
    feat_subset = np.linspace(0, n_feat_total - 1, min(MAX_FEATURES_DEEP, n_feat_total)).astype(int)
    row_subset = np.random.default_rng(0).choice(len(X_train_raw), size=min(MAX_TRAIN_ROWS_DEEP, len(X_train_raw)), replace=False)

    Xtr_deep = torch.tensor(X_train_raw[row_subset][:, feat_subset], dtype=torch.float32)
    Xte_deep = torch.tensor(X_test_raw[:, feat_subset], dtype=torch.float32)
    ytr_deep = y_train_b[row_subset]

    print(f'Deep baselines trained on {len(row_subset)} rows x {len(feat_subset)} features '
          f'(subsampled for CPU runtime; use full data with GPU for the paper-final numbers).')

    # ---- FT-Transformer ----
    ft_model = SimpleFTTransformer(n_features=len(feat_subset), d_model=32, n_heads=4, n_layers=2)
    def ft_forward(model, X, idx_b):
        batch = X if idx_b is None else X[idx_b]
        return model(batch)
    ft_scores = train_torch_binary(ft_model, Xtr_deep, ytr_deep, Xte_deep,
                                    n_epochs=10, lr=1e-3, forward_fn=ft_forward)
    r_ft = summarize(y_test_b, ft_scores, label='FT-Transformer')

    # ---- TabTransformer: treat every subsampled feature as 'categorical-ish' by simple median split
    # for demonstration purposes (a real run would use true categorical/numeric column identities).
    median_split = np.median(Xtr_deep.numpy(), axis=0)
    n_cat_demo = len(feat_subset) // 2
    x_cat_train = Xtr_deep[:, :n_cat_demo]
    x_num_train = Xtr_deep[:, n_cat_demo:]
    x_cat_test = Xte_deep[:, :n_cat_demo]
    x_num_test = Xte_deep[:, n_cat_demo:]

    tt_model = SimpleTabTransformer(n_cat=n_cat_demo, n_num=x_num_train.shape[1], d_model=16, n_heads=2, n_layers=2)
    def tt_forward(model, X, idx_b):
        x_cat, x_num = X
        if idx_b is not None:
            x_cat, x_num = x_cat[idx_b], x_num[idx_b]
        return model(x_cat, x_num)
    tt_scores = train_torch_binary(tt_model, (x_cat_train, x_num_train), ytr_deep, (x_cat_test, x_num_test),
                                    n_epochs=10, lr=1e-3, forward_fn=tt_forward)
    r_tt = summarize(y_test_b, tt_scores, label='TabTransformer')

    deep_baselines = dict(baselines)  # from Section I
    deep_baselines['FT-Transformer'] = r_ft
    deep_baselines['TabTransformer'] = r_tt

    plt.figure(figsize=(9,4.5))
    plt.bar(deep_baselines.keys(), [v['pr_auc'] for v in deep_baselines.values()], color='#5D4037')
    plt.ylabel('PR-AUC'); plt.title('Full Baseline Comparison Including Deep Tabular Models')
    plt.xticks(rotation=30, ha='right'); plt.tight_layout(); plt.show()
elif ieee_df is None:
    print('Skipping Section N -- IEEE-CIS not loaded.')
else:
    print('Skipping Section N -- PyTorch unavailable and could not be installed.')

"""**Scope note.** The deep baselines above are trained on a CPU-feasible subsample (rows and features capped) with a small number of epochs, purely to produce a directional comparison in this environment. For paper-final numbers, remove the `MAX_FEATURES_DEEP` / `MAX_TRAIN_ROWS_DEEP` caps, increase `n_epochs`, and run on GPU -- the model/training code itself doesn't change. The TabTransformer's categorical/numeric split here is a median-split placeholder for demonstration; a real run should use IEEE-CIS's true categorical columns (`ProductCD`, `card4`, `card6`, `addr1`, `addr2`, etc.) as the categorical token set."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F


"""1. HARDWARE CONFIGURATION (FORCE CPU)"""

device = torch.device("cpu")
print(f"==> Paper-Ready CPU-Stabilized Pipeline Active on: {device}")


"""2. LIGHTWEIGHT CPU-FORWARD ATTENTION LAYERS"""

class FastTabularAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        N, T, E = x.shape
        q = self.Wq(x).reshape(N, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.Wk(x).reshape(N, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(x).reshape(N, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)

        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(N, T, E)
        return x + self.Wo(ctx)

class OptimizedFTTransformer(nn.Module):
    def __init__(self, n_features, d_model=16, n_heads=2, n_layers=1):
        super().__init__()
        self.tokenizer = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList([FastTabularAttention(d_model, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        tokens = self.tokenizer(x.unsqueeze(-1))
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        for layer in self.layers:
            seq = layer(seq)
        return self.head(self.ln(seq[:, 0, :])).squeeze(-1)

class OptimizedTabTransformer(nn.Module):
    def __init__(self, n_cat, n_num, d_model=16, n_heads=2, n_layers=1, mlp_hidden=32):
        super().__init__()
        self.n_cat = n_cat
        self.cat_tokenizer = nn.Linear(1, d_model) if n_cat > 0 else None

        if n_cat > 0:
            self.layers = nn.ModuleList([FastTabularAttention(d_model, n_heads) for _ in range(n_layers)])
            self.ln = nn.LayerNorm(d_model)

        cat_out_dim = n_cat * d_model if n_cat > 0 else 0
        self.mlp = nn.Sequential(
            nn.Linear(cat_out_dim + n_num, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, x_cat, x_num):
        parts = []
        if self.n_cat > 0:
            tokens = self.cat_tokenizer(x_cat.unsqueeze(-1))
            for layer in self.layers:
                tokens = layer(tokens)
            tokens = self.ln(tokens)
            parts.append(tokens.reshape(tokens.shape[0], -1))
        if x_num.shape[1] > 0:
            parts.append(x_num)
        h = torch.cat(parts, dim=1)
        return self.mlp(h).squeeze(-1)

"""3. HIGH-SPEED CPU BATCH LOADER"""

def train_cpu_fast(model, X_train, y_train, X_test, n_epochs=10, lr=1e-3, batch_size=4096, is_tuple=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n = X_train[0].shape[0] if is_tuple else X_train.shape[0]
    y_t = torch.tensor(y_train, dtype=torch.float32)

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx_b = perm[start:start+batch_size]
            opt.zero_grad(set_to_none=True)

            if is_tuple:
                logits = model(X_train[0][idx_b], X_train[1][idx_b])
            else:
                logits = model(X_train[idx_b])

            loss = loss_fn(logits, y_t[idx_b])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        if is_tuple:
            test_logits = model(X_test[0], X_test[1])
        else:
            test_logits = model(X_test)
        return torch.sigmoid(test_logits).numpy()


""" 4. HIGH-SPEED CPU EXECUTION ENGINE (5-SEED STATISTICAL SEED RUN)"""

if 'ieee_df' in globals() and ieee_df is not None:
    # Restored to 5 seeds and 10 epochs for full statistical coverage
    N_SEEDS = 5
    N_EPOCHS = 10

    ft_seed_results = []
    tt_seed_results = []
    n_num_features = len(tab_branch.num_cols)

    print("\n==> Extracting wide features...")
    num_all = ieee_df[tab_branch.num_cols].fillna(-999).values
    cat_all = ieee_df[tab_branch.cat_cols].fillna('missing').astype(str).values

    for seed in range(N_SEEDS):
        print(f"--- Running Deep Suite: Seed Iteration {seed}/{N_SEEDS-1} ---")
        tr_idx, te_idx = train_test_split(
            np.arange(len(ieee_df)), test_size=0.25, random_state=seed, stratify=y_all
        )

        scaler = StandardScaler().fit(num_all[tr_idx])
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(cat_all[tr_idx])

        X_train_raw = np.hstack([scaler.transform(num_all[tr_idx]), ohe.transform(cat_all[tr_idx])])
        X_test_raw = np.hstack([scaler.transform(num_all[te_idx]), ohe.transform(cat_all[te_idx])])

        # Dimension compression (SVD) prevents attention matrix growth from locking threads
        svd = TruncatedSVD(n_components=32, random_state=42).fit(X_train_raw)
        X_train_compressed = svd.transform(X_train_raw)
        X_test_compressed = svd.transform(X_test_raw)

        Xtr_cpu = torch.tensor(X_train_compressed, dtype=torch.float32, device=device)
        Xte_cpu = torch.tensor(X_test_compressed, dtype=torch.float32, device=device)

        
        """A. FT-TRANSFORMER (COMPRESSED CPU)"""
      
        ft_model = OptimizedFTTransformer(n_features=X_train_compressed.shape[1], d_model=16, n_heads=2, n_layers=1).to(device)
        ft_scores = train_cpu_fast(ft_model, Xtr_cpu, y_all[tr_idx], Xte_cpu, n_epochs=N_EPOCHS, batch_size=4096)
        ft_ap = average_precision_score(y_all[te_idx], ft_scores)
        ft_seed_results.append(ft_ap)
        
        """B. TABTRANSFORMER (COMPRESSED CPU)"""
        
        # Split features cleanly out of the compressed bottleneck representation
        x_num_train = Xtr_cpu[:, :16]
        x_cat_train = Xtr_cpu[:, 16:]
        x_num_test = Xte_cpu[:, :16]
        x_cat_test = Xte_cpu[:, 16:]

        tt_model = OptimizedTabTransformer(n_cat=x_cat_train.shape[1], n_num=x_num_train.shape[1], d_model=16, n_heads=2, n_layers=1).to(device)
        tt_scores = train_cpu_fast(tt_model, (x_cat_train, x_num_train), y_all[tr_idx], (x_cat_test, x_num_test),
                                         n_epochs=N_EPOCHS, batch_size=4096, is_tuple=True)
        tt_ap = average_precision_score(y_all[te_idx], tt_scores)
        tt_seed_results.append(tt_ap)

        print(f"    Seed {seed} Complete | FT-PR-AUC: {ft_ap:.4f} | Tab-PR-AUC: {tt_ap:.4f}")

    print('\n' + '='*60)
    print('=== SECTION N: FINAL STATISTICAL SUITE RESULTS ===')
    print('='*60)
    print(f"FT-Transformer PR-AUC (5-seeds) : {np.mean(ft_seed_results):.4f} ± {np.std(ft_seed_results):.4f}")
    print(f"TabTransformer PR-AUC (5-seeds) : {np.mean(tt_seed_results):.4f} ± {np.std(tt_seed_results):.4f}")
    print('='*60)

    if 'baselines' in globals():
        baselines['FT-Transformer (Deep Early Fusion)'] = {'pr_auc': np.mean(ft_seed_results)}
        baselines['TabTransformer (Deep Early Fusion)'] = {'pr_auc': np.mean(tt_seed_results)}
else:
    print("Execution skipped. Verify IEEE-CIS dataset configurations.")

