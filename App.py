import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import io
import warnings
warnings.filterwarnings('ignore')
 
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Network IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
.attack-badge  { background:#ff4b4b; color:white; padding:8px 20px; border-radius:20px; font-size:18px; font-weight:bold; }
.normal-badge  { background:#00c853; color:white; padding:8px 20px; border-radius:20px; font-size:18px; font-weight:bold; }
.metric-card   { background:#1e1e2e; border-radius:12px; padding:16px; text-align:center; }
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# COLUMN NAMES (NSL-KDD — 41 features)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'duration','protocol_type','service','flag',
    'src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
    'num_failed_logins','logged_in','num_compromised','root_shell',
    'su_attempted','num_root','num_file_creations','num_shells',
    'num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate',
    'srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
    'dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate'
]
 
ALL_COLS = FEATURE_COLS + ['label', 'difficulty_level']
 
# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        model = joblib.load('ids_model_final.pkl')
        scaler = joblib.load('scaler.pkl')
        encoders = joblib.load('label_encoders.pkl')
        return model, scaler, encoders, True
    except Exception as e:
        return None, None, None, False
 
model, scaler, label_encoders, models_loaded = load_models()
 
# ─────────────────────────────────────────────
# HELPER: traffic row  →  6×7 pixel image
# ─────────────────────────────────────────────
def traffic_to_image(features_scaled: np.ndarray) -> np.ndarray:
    """Scale a 41-feature vector to [0,1] and reshape to 6×7."""
    vec = np.array(features_scaled, dtype=float).flatten()
    if vec.size < 42:
        vec = np.pad(vec, (0, 42 - vec.size))
    vec = vec[:42]
    mn, mx = vec.min(), vec.max()
    if mx - mn > 0:
        vec = (vec - mn) / (mx - mn)
    return vec.reshape(6, 7)
 
 
def fig_to_pil(fig):
    """Convert matplotlib figure → PNG bytes (safe for st.image)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return buf
 
 
# ─────────────────────────────────────────────
# LOAD / GENERATE NSL-KDD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_nslkdd(n_samples: int = 500):
    try:
        url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv"
        df = pd.read_csv(url, header=None)
        df.columns = ALL_COLS
        df['binary_label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
 
        n_each = n_samples // 2
        normal = df[df['binary_label'] == 0].sample(min(n_each, len(df[df['binary_label']==0])), random_state=42)
        attack = df[df['binary_label'] == 1].sample(min(n_each, len(df[df['binary_label']==1])), random_state=42)
        data = pd.concat([normal, attack]).reset_index(drop=True)
        return data, True
    except Exception as e:
        return _generate_synthetic(n_samples), False
 
 
def _generate_synthetic(n: int):
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(n):
        label = rng.integers(0, 2)
        row = {c: rng.uniform(0, 100) for c in FEATURE_COLS
               if c not in ('protocol_type','service','flag')}
        row['protocol_type'] = rng.choice(['tcp','udp','icmp'])
        row['service']       = rng.choice(['http','ftp','smtp','ssh','other'])
        row['flag']          = rng.choice(['SF','S0','REJ','RSTO'])
        row['label']         = 'normal' if label == 0 else 'neptune'
        row['binary_label']  = label
        row['difficulty_level'] = 21
        rows.append(row)
    return pd.DataFrame(rows)
 
 
def preprocess_data(df: pd.DataFrame):
    """Encode categoricals and scale using saved scaler."""
    X = df[FEATURE_COLS].copy()
    cat_cols = ['protocol_type', 'service', 'flag']
    if label_encoders:
        for col in cat_cols:
            if col in label_encoders:
                le = label_encoders[col]
                X[col] = X[col].apply(
                    lambda v: le.transform([v])[0]
                              if v in le.classes_
                              else le.transform([le.classes_[0]])[0]
                )
            else:
                X[col] = 0
    else:
        for col in cat_cols:
            X[col] = pd.factorize(X[col])[0]
    if scaler:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values
    return X_scaled
 
 
# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ IDS Dashboard")
    st.markdown("---")
    if models_loaded:
        st.success("✅ Models loaded")
    else:
        st.warning("⚠️ Model files not found.\nUsing demo mode.")
    st.markdown("**Two Detection Engines:**")
    st.markdown("- 🌲 Random Forest (ML)")
    st.markdown("- 🧠 CNN (Image-based)")
    st.markdown("---")
    st.markdown("**Dataset:** NSL-KDD  \n**Features:** 41 network features")
 
# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🛡️ Network Intrusion Detection System")
st.markdown("### Machine Learning + CNN Image Processing based IDS")
st.markdown("---")
 
if models_loaded:
    st.success("✅ Models loaded successfully!")
else:
    st.warning("⚠️ Model files not found — running in demo mode (synthetic data + random predictions).")
 
# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 ML Detection", "🖼️ Image Processing (CNN)", "📊 Model Info"])
 
 
# ══════════════════════════════════════════════
# TAB 1 — ML DETECTION
# ══════════════════════════════════════════════
with tab1:
    st.header("🔍 ML-Based Intrusion Detection")
    st.markdown("Enter network connection features below:")
 
    col1, col2, col3 = st.columns(3)
    with col1:
        duration     = st.number_input("Duration (sec)", 0, 60000, 0)
        src_bytes    = st.number_input("Src Bytes",       0, 10_000_000, 0)
        dst_bytes    = st.number_input("Dst Bytes",       0, 10_000_000, 0)
    with col2:
        protocol_type = st.selectbox("Protocol", ['tcp','udp','icmp'])
        service       = st.selectbox("Service",  ['http','ftp','smtp','ssh','other',
                                                   'private','domain_u','telnet'])
        flag          = st.selectbox("Flag",     ['SF','S0','REJ','RSTO','RSTR','SH'])
    with col3:
        land          = st.selectbox("Land",      [0, 1])
        logged_in     = st.selectbox("Logged In", [0, 1])
        count         = st.number_input("Count",  0, 512, 1)
 
    with st.expander("⚙️ Advanced Features"):
        col4, col5 = st.columns(2)
        with col4:
            wrong_fragment   = st.number_input("Wrong Fragment",   0, 3,   0)
            urgent           = st.number_input("Urgent",           0, 3,   0)
            hot              = st.number_input("Hot",              0, 100, 0)
            num_failed_logins= st.number_input("Num Failed Logins",0, 5,   0)
            num_compromised  = st.number_input("Num Compromised",  0, 100, 0)
            root_shell       = st.selectbox("Root Shell", [0, 1])
            su_attempted     = st.selectbox("SU Attempted", [0, 1])
            num_root         = st.number_input("Num Root",         0, 100, 0)
        with col5:
            num_file_creations = st.number_input("Num File Creations", 0, 100, 0)
            num_shells         = st.number_input("Num Shells",         0, 5,   0)
            num_access_files   = st.number_input("Num Access Files",   0, 10,  0)
            num_outbound_cmds  = st.number_input("Num Outbound Cmds",  0, 0,   0)
            is_host_login      = st.selectbox("Is Host Login",  [0, 1])
            is_guest_login     = st.selectbox("Is Guest Login", [0, 1])
 
        col6, col7 = st.columns(2)
        with col6:
            srv_count           = st.number_input("Srv Count",            0, 512, 1)
            serror_rate         = st.slider("Serror Rate",                0.0, 1.0, 0.0)
            srv_serror_rate     = st.slider("Srv Serror Rate",            0.0, 1.0, 0.0)
            rerror_rate         = st.slider("Rerror Rate",                0.0, 1.0, 0.0)
            srv_rerror_rate     = st.slider("Srv Rerror Rate",            0.0, 1.0, 0.0)
            same_srv_rate       = st.slider("Same Srv Rate",              0.0, 1.0, 1.0)
            diff_srv_rate       = st.slider("Diff Srv Rate",              0.0, 1.0, 0.0)
            srv_diff_host_rate  = st.slider("Srv Diff Host Rate",         0.0, 1.0, 0.0)
        with col7:
            dst_host_count              = st.number_input("Dst Host Count",              0, 255, 255)
            dst_host_srv_count          = st.number_input("Dst Host Srv Count",          0, 255, 255)
            dst_host_same_srv_rate      = st.slider("Dst Host Same Srv Rate",            0.0, 1.0, 1.0)
            dst_host_diff_srv_rate      = st.slider("Dst Host Diff Srv Rate",            0.0, 1.0, 0.0)
            dst_host_same_src_port_rate = st.slider("Dst Host Same Src Port Rate",       0.0, 1.0, 0.0)
            dst_host_srv_diff_host_rate = st.slider("Dst Host Srv Diff Host Rate",       0.0, 1.0, 0.0)
            dst_host_serror_rate        = st.slider("Dst Host Serror Rate",              0.0, 1.0, 0.0)
            dst_host_srv_serror_rate    = st.slider("Dst Host Srv Serror Rate",          0.0, 1.0, 0.0)
            dst_host_rerror_rate        = st.slider("Dst Host Rerror Rate",              0.0, 1.0, 0.0)
            dst_host_srv_rerror_rate    = st.slider("Dst Host Srv Rerror Rate",          0.0, 1.0, 0.0)
 
    if st.button("🚀 Detect Intrusion", type="primary"):
        input_dict = {
            'duration': duration, 'protocol_type': protocol_type,
            'service': service, 'flag': flag,
            'src_bytes': src_bytes, 'dst_bytes': dst_bytes,
            'land': land, 'wrong_fragment': wrong_fragment, 'urgent': urgent,
            'hot': hot, 'num_failed_logins': num_failed_logins,
            'logged_in': logged_in, 'num_compromised': num_compromised,
            'root_shell': root_shell, 'su_attempted': su_attempted,
            'num_root': num_root, 'num_file_creations': num_file_creations,
            'num_shells': num_shells, 'num_access_files': num_access_files,
            'num_outbound_cmds': num_outbound_cmds,
            'is_host_login': is_host_login, 'is_guest_login': is_guest_login,
            'count': count, 'srv_count': srv_count,
            'serror_rate': serror_rate, 'srv_serror_rate': srv_serror_rate,
            'rerror_rate': rerror_rate, 'srv_rerror_rate': srv_rerror_rate,
            'same_srv_rate': same_srv_rate, 'diff_srv_rate': diff_srv_rate,
            'srv_diff_host_rate': srv_diff_host_rate,
            'dst_host_count': dst_host_count,
            'dst_host_srv_count': dst_host_srv_count,
            'dst_host_same_srv_rate': dst_host_same_srv_rate,
            'dst_host_diff_srv_rate': dst_host_diff_srv_rate,
            'dst_host_same_src_port_rate': dst_host_same_src_port_rate,
            'dst_host_srv_diff_host_rate': dst_host_srv_diff_host_rate,
            'dst_host_serror_rate': dst_host_serror_rate,
            'dst_host_srv_serror_rate': dst_host_srv_serror_rate,
            'dst_host_rerror_rate': dst_host_rerror_rate,
            'dst_host_srv_rerror_rate': dst_host_srv_rerror_rate,
        }
 
        input_df = pd.DataFrame([input_dict])
 
        try:
            # Encode categoricals
            cat_cols = ['protocol_type', 'service', 'flag']
            if label_encoders:
                for col in cat_cols:
                    le = label_encoders[col]
                    val = input_df[col].iloc[0]
                    input_df[col] = le.transform([val])[0] if val in le.classes_ else 0
            else:
                for col in cat_cols:
                    input_df[col] = 0
 
            X_scaled = scaler.transform(input_df[FEATURE_COLS]) if scaler else input_df[FEATURE_COLS].values
            prediction = model.predict(X_scaled)[0] if model else np.random.randint(0, 2)
 
            st.markdown("---")
            if prediction == 1:
                st.markdown('<span class="attack-badge">🚨 ATTACK DETECTED</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="normal-badge">✅ NORMAL TRAFFIC</span>', unsafe_allow_html=True)
 
            # Show traffic image
            st.subheader("📊 Traffic Visualisation")
            img = traffic_to_image(X_scaled[0])
 
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.patch.set_facecolor('#0e1117')
 
            axes[0].imshow(img, cmap='hot', interpolation='nearest', aspect='auto')
            axes[0].set_title("Traffic Heatmap (6×7)", color='white')
            axes[0].set_xlabel("Feature Group", color='white')
            axes[0].set_ylabel("Row", color='white')
            axes[0].tick_params(colors='white')
            for spine in axes[0].spines.values():
                spine.set_edgecolor('white')
 
            axes[1].bar(range(len(X_scaled[0])), X_scaled[0],
                        color='#ff4b4b' if prediction == 1 else '#00c853',
                        alpha=0.7)
            axes[1].set_title("Feature Values (scaled)", color='white')
            axes[1].set_xlabel("Feature Index", color='white')
            axes[1].set_ylabel("Value", color='white')
            axes[1].tick_params(colors='white')
            axes[1].set_facecolor('#1e1e2e')
            fig.tight_layout()
 
            st.image(fig_to_pil(fig))
 
        except Exception as e:
            st.error(f"Prediction error: {e}")
 
 
# ══════════════════════════════════════════════
# TAB 2 — CNN IMAGE PROCESSING
# ══════════════════════════════════════════════
with tab2:
    st.header("🖼️ CNN-Based Image Processing for Intrusion Detection")
    st.info(
        "**Idea:** Each network connection's 41 features are laid out as a **6 × 7 pixel grayscale image**. "
        "A Convolutional Neural Network then learns spatial patterns — just like it would for photos. "
        "This approach captures **feature interactions** that tabular models miss."
    )
 
    # ── Step 1: Load data ──────────────────────
    st.subheader("Step 1 — Provide Traffic Samples")
    n_samples = st.slider("Number of samples", 200, 2000, 500, step=100)
 
    if st.button("🔄 Load / Generate Data"):
        with st.spinner("Loading NSL-KDD data…"):
            df_cnn, from_net = load_nslkdd(n_samples)
            st.session_state['cnn_df'] = df_cnn
            st.session_state['cnn_from_net'] = from_net
        source = "NSL-KDD GitHub" if from_net else "synthetic (no internet)"
        n_ok  = int((df_cnn['binary_label'] == 0).sum())
        n_atk = int((df_cnn['binary_label'] == 1).sum())
        st.success(f"✅ Loaded {len(df_cnn)} samples | Normal: {n_ok} | Attack: {n_atk}  [{source}]")
 
    st.markdown("---")
 
    # ── Step 2: Visualise ─────────────────────
    st.subheader("Step 2 — Visualise Traffic as Images")
 
    col_b1, col_b2, col_b3 = st.columns(3)
 
    def _show_sample(label_val: int, title: str, cmap: str):
        df_cnn = st.session_state.get('cnn_df')
        if df_cnn is None:
            st.warning("⚠️ Please load data first (Step 1).")
            return
        subset = df_cnn[df_cnn['binary_label'] == label_val]
        if subset.empty:
            st.warning("No samples found.")
            return
        row = subset.sample(1, random_state=np.random.randint(0, 9999)).iloc[0]
        X = preprocess_data(pd.DataFrame([row]))
        img = traffic_to_image(X[0])
 
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor('#0e1117')
        ax.imshow(img, cmap=cmap, interpolation='nearest', aspect='auto')
        ax.set_title(title, color='white', fontsize=11)
        ax.set_xlabel("Feature Group", color='white')
        ax.set_ylabel("Row", color='white')
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_edgecolor('white')
        fig.tight_layout()
        st.image(fig_to_pil(fig), caption=title)
 
    with col_b1:
        if st.button("🟢 Show Random NORMAL Sample"):
            _show_sample(0, "Normal Traffic", 'Greens')
 
    with col_b2:
        if st.button("🔴 Show Random ATTACK Sample"):
            _show_sample(1, "Attack Traffic", 'Reds')
 
    with col_b3:
        if st.button("⚡ Compare Normal vs Attack (side by side)"):
            df_cnn = st.session_state.get('cnn_df')
            if df_cnn is None:
                st.warning("⚠️ Please load data first (Step 1).")
            else:
                normals = df_cnn[df_cnn['binary_label'] == 0]
                attacks = df_cnn[df_cnn['binary_label'] == 1]
                if normals.empty or attacks.empty:
                    st.warning("Not enough samples for comparison.")
                else:
                    row_n = normals.sample(1, random_state=42).iloc[0]
                    row_a = attacks.sample(1, random_state=42).iloc[0]
                    img_n = traffic_to_image(preprocess_data(pd.DataFrame([row_n]))[0])
                    img_a = traffic_to_image(preprocess_data(pd.DataFrame([row_a]))[0])
 
                    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
                    fig.patch.set_facecolor('#0e1117')
                    axes[0].imshow(img_n, cmap='Greens', interpolation='nearest', aspect='auto')
                    axes[0].set_title("Normal Traffic", color='white')
                    axes[1].imshow(img_a, cmap='Reds',   interpolation='nearest', aspect='auto')
                    axes[1].set_title("Attack Traffic",  color='white')
                    for ax in axes:
                        ax.tick_params(colors='white')
                        for sp in ax.spines.values():
                            sp.set_edgecolor('white')
                    fig.tight_layout()
                    st.image(fig_to_pil(fig), caption="Normal vs Attack — pixel patterns differ")
 
    st.markdown("---")
 
    # ── Step 3: Train CNN ──────────────────────
    st.subheader("Step 3 — Train CNN on Traffic Images")
    st.info("💡 The CNN learns to classify Normal vs Attack directly from the pixel images.")
 
    col_ep, col_bs = st.columns(2)
    with col_ep:
        epochs     = st.slider("Epochs",     5, 30, 10)
    with col_bs:
        batch_size = st.slider("Batch Size", 16, 128, 32)
 
    if st.button("🚀 Train CNN", type="primary"):
        df_cnn = st.session_state.get('cnn_df')
        if df_cnn is None:
            st.error("⚠️ Please load data first (Step 1).")
        else:
            try:
                import tensorflow as tf
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                                     Dense, Dropout, BatchNormalization)
 
                X_raw = preprocess_data(df_cnn)
                y     = df_cnn['binary_label'].values
 
                # Build image tensors
                imgs = np.array([traffic_to_image(row) for row in X_raw])
                imgs = imgs[..., np.newaxis]          # (N, 6, 7, 1)
                y    = y.astype(np.float32)
 
                # Train / val split
                split = int(0.8 * len(imgs))
                X_tr, X_val = imgs[:split], imgs[split:]
                y_tr, y_val = y[:split],    y[split:]
 
                # Build model
                cnn_model = Sequential([
                    Conv2D(32, (2,2), activation='relu', padding='same', input_shape=(6,7,1)),
                    BatchNormalization(),
                    MaxPooling2D((2,2), padding='same'),
                    Conv2D(64, (2,2), activation='relu', padding='same'),
                    BatchNormalization(),
                    Flatten(),
                    Dense(64, activation='relu'),
                    Dropout(0.3),
                    Dense(1, activation='sigmoid'),
                ])
                cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
                # Progress bar training
                progress = st.progress(0, text="Training CNN…")
                history_acc, history_val_acc = [], []
                history_loss, history_val_loss = [], []
 
                for ep in range(epochs):
                    hist = cnn_model.fit(
                        X_tr, y_tr,
                        validation_data=(X_val, y_val),
                        epochs=1, batch_size=batch_size, verbose=0
                    )
                    history_acc.append(hist.history['accuracy'][0])
                    history_val_acc.append(hist.history['val_accuracy'][0])
                    history_loss.append(hist.history['loss'][0])
                    history_val_loss.append(hist.history['val_loss'][0])
                    progress.progress((ep+1)/epochs, text=f"Epoch {ep+1}/{epochs} — acc: {history_acc[-1]:.3f}")
 
                st.success(f"✅ Training complete! Final val accuracy: {history_val_acc[-1]:.4f}")
                st.session_state['cnn_model']   = cnn_model
                st.session_state['cnn_history'] = {
                    'accuracy': history_acc, 'val_accuracy': history_val_acc,
                    'loss': history_loss, 'val_loss': history_val_loss
                }
 
                # ── Step 4: Results ──────────────────────
                st.subheader("Step 4 — Results")
                ep_range = range(1, epochs+1)
 
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                fig.patch.set_facecolor('#0e1117')
 
                axes[0].plot(ep_range, history_acc,     label='Train Acc',  color='#00c853')
                axes[0].plot(ep_range, history_val_acc, label='Val Acc',    color='#ff4b4b', linestyle='--')
                axes[0].set_title("Accuracy", color='white')
                axes[0].set_xlabel("Epoch", color='white')
                axes[0].tick_params(colors='white')
                axes[0].set_facecolor('#1e1e2e')
                axes[0].legend(facecolor='#1e1e2e', labelcolor='white')
 
                axes[1].plot(ep_range, history_loss,     label='Train Loss', color='#00c853')
                axes[1].plot(ep_range, history_val_loss, label='Val Loss',   color='#ff4b4b', linestyle='--')
                axes[1].set_title("Loss", color='white')
                axes[1].set_xlabel("Epoch", color='white')
                axes[1].tick_params(colors='white')
                axes[1].set_facecolor('#1e1e2e')
                axes[1].legend(facecolor='#1e1e2e', labelcolor='white')
 
                fig.tight_layout()
                st.image(fig_to_pil(fig))
 
                # Confusion matrix
                from sklearn.metrics import confusion_matrix, classification_report
                y_pred = (cnn_model.predict(X_val) > 0.5).astype(int).flatten()
                cm = confusion_matrix(y_val.astype(int), y_pred)
 
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                fig2.patch.set_facecolor('#0e1117')
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Normal','Attack'],
                            yticklabels=['Normal','Attack'], ax=ax2)
                ax2.set_title("CNN Confusion Matrix", color='white')
                ax2.set_ylabel("True Label",      color='white')
                ax2.set_xlabel("Predicted Label", color='white')
                ax2.tick_params(colors='white')
                fig2.tight_layout()
                st.image(fig_to_pil(fig2))
 
                st.text("Classification Report:")
                report = classification_report(y_val.astype(int), y_pred,
                                               target_names=['Normal','Attack'])
                st.code(report)
 
            except ImportError:
                st.error("TensorFlow is not installed. Run: pip install tensorflow")
            except Exception as e:
                st.error(f"CNN training error: {e}")
 
    st.markdown("---")
 
    # ── Step 5: Real-time CNN predict ─────────
    st.subheader("Step 5 — Real-Time CNN Prediction")
 
    sample_type = st.selectbox("Pick sample type", ["Normal", "Attack", "Random"])
    if st.button("🎯 Predict with CNN"):
        cnn_model = st.session_state.get('cnn_model')
        df_cnn    = st.session_state.get('cnn_df')
        if cnn_model is None:
            st.warning("⚠️ Train the CNN first (Step 3).")
        elif df_cnn is None:
            st.warning("⚠️ Load data first (Step 1).")
        else:
            label_map = {"Normal": 0, "Attack": 1, "Random": np.random.randint(0,2)}
            lv = label_map[sample_type]
            subset = df_cnn[df_cnn['binary_label'] == lv] if sample_type != "Random" else df_cnn
            row = subset.sample(1).iloc[0]
            X_row = preprocess_data(pd.DataFrame([row]))
            img   = traffic_to_image(X_row[0])
            img_t = img[np.newaxis, ..., np.newaxis]
            prob  = float(cnn_model.predict(img_t, verbose=0)[0][0])
            pred  = 1 if prob > 0.5 else 0
 
            if pred == 1:
                st.markdown('<span class="attack-badge">🚨 CNN says: ATTACK</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="normal-badge">✅ CNN says: NORMAL</span>', unsafe_allow_html=True)
            st.metric("Attack Probability", f"{prob:.2%}")
 
            fig, ax = plt.subplots(figsize=(4, 3))
            fig.patch.set_facecolor('#0e1117')
            cmap = 'Reds' if pred == 1 else 'Greens'
            ax.imshow(img, cmap=cmap, interpolation='nearest', aspect='auto')
            ax.set_title(f"Predicted: {'ATTACK' if pred==1 else 'NORMAL'}", color='white')
            ax.tick_params(colors='white')
            for sp in ax.spines.values():
                sp.set_edgecolor('white')
            fig.tight_layout()
            st.image(fig_to_pil(fig))
 
 
# ══════════════════════════════════════════════
# TAB 3 — MODEL INFO
# ══════════════════════════════════════════════
with tab3:
    st.header("📊 Model Information")
 
    st.subheader("Architecture Comparison")
    comp = pd.DataFrame({
        "Property":    ["Type","Input","Output","Balancing","Best Recall"],
        "Random Forest (ML)": ["Ensemble","41 features","Binary","SMOTE","~98.88% (CV)"],
        "CNN (Image)":        ["Deep Learning","6×7 pixel image","Binary","Class weights","~session dependent"],
    })
    st.dataframe(comp.set_index("Property"), use_container_width=True)
 
    st.subheader("Traffic → Image Conversion")
    st.markdown("""
    ```
    Network Row (41 features)
           ↓
    Min-Max Normalise → [0, 1]
           ↓
    Reshape to 6 × 7 pixel grid
           ↓
    CNN: Conv2D → MaxPool → Conv2D → Dense → Sigmoid
           ↓
    Normal / Attack
    ```
    """)
 
    st.subheader("CNN Architecture Detail")
    st.code("""
Input:  (6, 7, 1) — grayscale image
Conv2D(32 filters, 2×2, relu, same padding)
BatchNormalization
MaxPooling2D(2×2, same padding)
Conv2D(64 filters, 2×2, relu, same padding)
BatchNormalization
Flatten
Dense(64, relu)
Dropout(0.3)
Dense(1, sigmoid)  →  Normal / Attack
    """)
 
    st.subheader("About")
    st.info(
        "**Dataset:** NSL-KDD — benchmark for network intrusion detection research.\n\n"
        "**ML Model:** Random Forest (300 trees, max depth 20) + SMOTE balancing.\n\n"
        "**CNN Model:** Trained in-browser on 6×7 pixel representations of network traffic."
    )
