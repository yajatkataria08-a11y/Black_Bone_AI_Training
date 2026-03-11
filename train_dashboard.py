"""
Offroad Segmentation — Live Training Dashboard
Run: streamlit run train_dashboard.py
"""

import streamlit as st
import subprocess
import threading
import json
import os
import time
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DINOv2 Seg — Training Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
  --bg:       #0a0c10;
  --bg2:      #111318;
  --bg3:      #1a1d24;
  --border:   #2a2d36;
  --accent:   #00e5ff;
  --accent2:  #ff6b35;
  --accent3:  #7c3aed;
  --green:    #00ff88;
  --yellow:   #ffd700;
  --red:      #ff4560;
  --text:     #e8eaf0;
  --muted:    #6b7280;
}

html, body, [data-testid="stApp"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Syne', sans-serif;
}

[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border);
}

h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

.metric-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 24px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.4s ease, transform 0.3s ease, box-shadow 0.4s ease;
  cursor: default;
}
.metric-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--accent3));
  transform: scaleX(0.4);
  transform-origin: left;
  transition: transform 0.4s ease;
}
.metric-card::after {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse at 50% 0%, rgba(0,229,255,0.05) 0%, transparent 70%);
  opacity: 0;
  transition: opacity 0.4s ease;
}
.metric-card:hover {
  border-color: rgba(0,229,255,0.5);
  transform: translateY(-2px);
  box-shadow: 0 8px 32px rgba(0,229,255,0.08), 0 2px 8px rgba(0,0,0,0.4);
}
.metric-card:hover::before { transform: scaleX(1); }
.metric-card:hover::after  { opacity: 1; }

.metric-label {
  font-family: 'Space Mono', monospace;
  font-size: 11px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 6px;
}
.metric-value {
  font-family: 'Space Mono', monospace;
  font-size: 28px;
  font-weight: 700;
  color: var(--accent);
  line-height: 1;
}
.metric-delta {
  font-family: 'Space Mono', monospace;
  font-size: 12px;
  margin-top: 4px;
}
.delta-up   { color: var(--green); }
.delta-down { color: var(--red); }

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 16px;
  border-radius: 999px;
  font-family: 'Space Mono', monospace;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
}
.status-running { background: rgba(0,229,255,0.1); border: 1px solid var(--accent); color: var(--accent); }
.status-idle    { background: rgba(107,114,128,0.1); border: 1px solid var(--muted); color: var(--muted); }
.status-done    { background: rgba(0,255,136,0.1); border: 1px solid var(--green); color: var(--green); }
.status-error   { background: rgba(255,69,96,0.1); border: 1px solid var(--red); color: var(--red); }

.dot-pulse {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--accent);
  animation: pulse 1.4s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.4; transform: scale(0.7); }
}

.class-bar-wrap {
  margin-bottom: 8px;
}
.class-bar-label {
  display: flex;
  justify-content: space-between;
  font-family: 'Space Mono', monospace;
  font-size: 11px;
  color: var(--muted);
  margin-bottom: 3px;
}
.class-bar-bg {
  background: var(--bg3);
  border-radius: 4px;
  height: 8px;
  overflow: hidden;
}
.class-bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.6s ease;
}

.log-box {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
  font-family: 'Space Mono', monospace;
  font-size: 11px;
  color: #a0aec0;
  max-height: 320px;
  overflow-y: auto;
  line-height: 1.8;
}
.log-line-ep  { color: var(--accent); }
.log-line-ok  { color: var(--green); }
.log-line-err { color: var(--red); }
.log-line-warn{ color: var(--yellow); }

.header-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 0 28px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 28px;
}
.header-title {
  font-family: 'Syne', sans-serif;
  font-size: 26px;
  font-weight: 800;
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent3) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: -0.5px;
}
.header-sub {
  font-family: 'Space Mono', monospace;
  font-size: 11px;
  color: var(--muted);
  margin-top: 2px;
}

[data-testid="stButton"] > button {
  background: transparent !important;
  color: var(--accent) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 12px !important;
  font-weight: 700 !important;
  border: 1px solid var(--accent) !important;
  border-radius: 8px !important;
  padding: 10px 24px !important;
  text-transform: uppercase;
  letter-spacing: 1px;
  position: relative !important;
  overflow: hidden !important;
  transition: color 0.4s ease, box-shadow 0.4s ease !important;
}
[data-testid="stButton"] > button::before {
  content: '' !important;
  position: absolute !important;
  inset: 0 !important;
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent3) 100%) !important;
  transform: translateY(100%) !important;
  transition: transform 0.4s cubic-bezier(0.76, 0, 0.24, 1) !important;
  z-index: 0 !important;
}
[data-testid="stButton"] > button:hover {
  color: #000 !important;
  box-shadow: 0 4px 24px rgba(0,229,255,0.18) !important;
}
[data-testid="stButton"] > button:hover::before {
  transform: translateY(0%) !important;
}
[data-testid="stButton"] > button > * { position: relative; z-index: 1; }

[data-testid="stSelectbox"] select,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 12px !important;
  border-radius: 6px !important;
  transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
[data-testid="stSelectbox"] select:hover,
[data-testid="stNumberInput"] input:hover,
[data-testid="stTextInput"] input:hover {
  border-color: rgba(0,229,255,0.4) !important;
  box-shadow: 0 0 0 3px rgba(0,229,255,0.06) !important;
}
[data-testid="stSelectbox"] select:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(0,229,255,0.12) !important;
  outline: none !important;
}

[data-testid="stCheckbox"]:hover label {
  color: var(--accent) !important;
  transition: color 0.3s ease;
}
[data-testid="stCheckbox"] span[data-baseweb="checkbox"] {
  transition: border-color 0.3s ease, background 0.3s ease !important;
}
[data-testid="stCheckbox"]:hover span[data-baseweb="checkbox"] {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(0,229,255,0.08) !important;
}

.stSlider [data-baseweb="slider"] > div:first-child { background: var(--bg3) !important; }
.stSlider [data-baseweb="slider"] > div:nth-child(2) { background: var(--accent) !important; }
.stSlider [data-baseweb="slider"] [role="slider"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
.stSlider [data-baseweb="slider"] [role="slider"]:hover {
  transform: scale(1.35) !important;
  box-shadow: 0 0 0 8px rgba(0,229,255,0.12) !important;
}
.stProgress > div > div > div {
  background: linear-gradient(90deg, var(--accent), var(--accent3)) !important;
  transition: width 0.5s ease !important;
}

div[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
div[data-testid="stExpander"]:hover {
  border-color: rgba(0,229,255,0.3) !important;
  box-shadow: 0 2px 16px rgba(0,229,255,0.05) !important;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
  font-family: 'Space Mono', monospace !important;
  font-size: 12px !important;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--muted) !important;
  transition: color 0.3s ease !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
  color: var(--text) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Logs", "Rocks", "Landscape", "Sky"
]

CLASS_COLORS = [
    "#6b7280", "#22c55e", "#16a34a", "#fbbf24", "#a16207",
    "#9ca3af", "#92400e", "#94a3b8", "#3b82f6", "#7dd3fc"
]

METRICS_FILE  = "train_metrics.json"
LOG_FILE      = "train_log.txt"
STATUS_FILE   = "train_status.json"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Mono, monospace", color="#6b7280", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(showgrid=True, gridcolor="#1e2130", gridwidth=1,
               zeroline=False, color="#6b7280",
               showspikes=True, spikecolor="rgba(0,229,255,0.3)",
               spikethickness=1, spikedash="dot"),
    yaxis=dict(showgrid=True, gridcolor="#1e2130", gridwidth=1,
               zeroline=False, color="#6b7280",
               showspikes=True, spikecolor="rgba(0,229,255,0.3)",
               spikethickness=1, spikedash="dot"),
    hoverlabel=dict(
        bgcolor="#1a1d24",
        bordercolor="#2a2d36",
        font=dict(family="Space Mono, monospace", size=11, color="#e8eaf0")
    ),
    hovermode="x unified",
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_metrics():
    if not Path(METRICS_FILE).exists():
        return {}
    try:
        with open(METRICS_FILE) as f:
            return json.load(f)
    except:
        return {}

def load_status():
    if not Path(STATUS_FILE).exists():
        return {"state": "idle", "epoch": 0, "best_iou": 0.0, "eta_min": 0}
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except:
        return {"state": "idle", "epoch": 0, "best_iou": 0.0, "eta_min": 0}

def load_log():
    if not Path(LOG_FILE).exists():
        return []
    try:
        with open(LOG_FILE) as f:
            return f.readlines()[-60:]
    except:
        return []

def fmt_log_line(line):
    line = line.strip()
    if not line:
        return ""
    if "✅" in line or "Best saved" in line:
        return f'<div class="log-line-ok">● {line}</div>'
    if "Early stopping" in line or "Error" in line or "error" in line:
        return f'<div class="log-line-err">● {line}</div>'
    if "Ep [" in line:
        return f'<div class="log-line-ep">▶ {line}</div>'
    if "Warning" in line or "warning" in line or "throttl" in line.lower():
        return f'<div class="log-line-warn">⚠ {line}</div>'
    return f'<div>  {line}</div>'

def make_line_chart(epochs, train_vals, val_vals, title, color_train, color_val):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=train_vals, name="Train",
        line=dict(color=color_train, width=2),
        mode="lines",
        fill="tozeroy",
        fillcolor=f"rgba({int(color_train[1:3],16)},{int(color_train[3:5],16)},{int(color_train[5:7],16)},0.06)"
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=val_vals, name="Val",
        line=dict(color=color_val, width=2.5, dash="dot"),
        mode="lines+markers",
        marker=dict(size=5, color=color_val)
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=13, color="#e8eaf0"), x=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        height=260
    )
    return fig

def make_class_bar_html(class_ious):
    bars = []
    for i, (name, iou) in enumerate(zip(CLASS_NAMES, class_ious)):
        if iou is None or (isinstance(iou, float) and np.isnan(iou)):
            iou = 0.0
        pct  = max(0, min(100, iou * 100))
        col  = CLASS_COLORS[i]
        bars.append(f"""
        <div class="class-bar-wrap">
          <div class="class-bar-label">
            <span>{name}</span>
            <span style="color:#e8eaf0">{iou:.3f}</span>
          </div>
          <div class="class-bar-bg">
            <div class="class-bar-fill" style="width:{pct}%;background:{col}"></div>
          </div>
        </div>""")
    return "".join(bars)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — CONFIG
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Training Config")

    data_dir   = st.text_input("Train Dir",  r"D:\Hackathon\Startathin\Offroad_Segmentation_Training_Dataset\train")
    val_dir    = st.text_input("Val Dir",    r"D:\Hackathon\Startathin\Offroad_Segmentation_Training_Dataset\val")
    output_dir = st.text_input("Output Dir", r"D:\Hackathon\Startathin\Offroad_Segmentation_Scripts\train_stats")
    model_path = st.text_input("Model Path", r"D:\Hackathon\Startathin\Offroad_Segmentation_Scripts\segmentation_head.pth")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.number_input("Batch Size", 4, 128, 32, step=4)
        lr         = st.number_input("LR", 1e-5, 1e-1, 1e-3, step=1e-4, format="%.5f")
    with col2:
        n_epochs   = st.number_input("Epochs", 10, 200, 50)
        patience   = st.number_input("Patience", 5, 50, 15)

    backbone = st.selectbox("DINOv2 Backbone", ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"])

    st.divider()
    st.markdown("### 🔬 Augmentation")
    aug_hflip    = st.checkbox("Horizontal Flip",  True)
    aug_vflip    = st.checkbox("Vertical Flip",    True)
    aug_colorjit = st.checkbox("Color Jitter",     True)
    aug_cutmix   = st.checkbox("CutMix",           True)
    aug_mixup    = st.checkbox("MixUp",            False)
    cutmix_prob  = st.slider("CutMix Prob", 0.0, 1.0, 0.3, 0.05) if aug_cutmix else 0.0

    st.divider()
    st.markdown("### ⚡ Optimization")
    use_warmup   = st.checkbox("LR Warmup (5 epochs)", True)
    use_ema      = st.checkbox("EMA (decay=0.999)",    True)
    loss_type    = st.selectbox("Loss", ["CE + Dice", "Focal + Dice", "CE only"])
    label_smooth = st.slider("Label Smoothing", 0.0, 0.2, 0.05, 0.01)
    grad_clip    = st.slider("Grad Clip", 0.1, 5.0, 1.0, 0.1)

    st.divider()
    refresh_rate = st.slider("Auto-refresh (sec)", 2, 30, 5)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

status = load_status()
metrics = load_metrics()

state = status.get("state", "idle")
status_html = {
    "running": '<span class="status-pill status-running"><span class="dot-pulse"></span>TRAINING</span>',
    "idle":    '<span class="status-pill status-idle">◌ IDLE</span>',
    "done":    '<span class="status-pill status-done">✓ COMPLETE</span>',
    "error":   '<span class="status-pill status-error">✕ ERROR</span>',
}.get(state, '<span class="status-pill status-idle">◌ IDLE</span>')

st.markdown(f"""
<div class="header-bar">
  <div>
    <div class="header-title">🧠 DINOv2 Offroad Segmentation</div>
    <div class="header-sub">Live Training Dashboard · {backbone} · {n_epochs} epochs</div>
  </div>
  <div>{status_html}</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LAUNCH / STOP CONTROLS
# ─────────────────────────────────────────────────────────────────────────────

ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1,1,1,3])

with ctrl_col1:
    launch = st.button("▶  Launch Training")
with ctrl_col2:
    stop = st.button("⏹  Stop")
with ctrl_col3:
    clear = st.button("🗑  Clear Logs")

if launch:
    cfg = dict(
        data_dir=data_dir, val_dir=val_dir, output_dir=output_dir,
        model_path=model_path, batch_size=int(batch_size), n_epochs=int(n_epochs),
        lr=float(lr), patience=int(patience), backbone=backbone,
        aug_hflip=aug_hflip, aug_vflip=aug_vflip, aug_colorjit=aug_colorjit,
        aug_cutmix=aug_cutmix, cutmix_prob=cutmix_prob, aug_mixup=aug_mixup,
        use_warmup=use_warmup, use_ema=use_ema, loss_type=loss_type,
        label_smooth=float(label_smooth), grad_clip=float(grad_clip),
    )
    with open("train_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    with open(STATUS_FILE, "w") as f:
        json.dump({"state": "running", "epoch": 0, "best_iou": 0.0, "eta_min": 0}, f)
    # Clear old metrics
    for fname in [METRICS_FILE, LOG_FILE]:
        try:
            if Path(fname).exists():
                os.remove(fname)
        except PermissionError:
            pass
    st.success("✅ Config written to train_config.json — run: `python train_v7.py`")

if stop:
    with open(STATUS_FILE, "w") as f:
        json.dump({"state": "idle", "epoch": 0, "best_iou": 0.0, "eta_min": 0}, f)
    st.warning("Stop signal sent (set train_status.json state to idle).")

if clear:
    for fname in [METRICS_FILE, LOG_FILE, STATUS_FILE]:
        try:
            if Path(fname).exists():
                os.remove(fname)
        except PermissionError:
            pass  # Windows: file locked by training process — skip
    st.rerun()

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# LIVE METRICS — TOP ROW
# ─────────────────────────────────────────────────────────────────────────────

ep      = status.get("epoch", 0)
best    = status.get("best_iou", 0.0)
eta     = status.get("eta_min", 0)

vi_hist = metrics.get("vi", [])
vl_hist = metrics.get("vl", [])
ti_hist = metrics.get("ti", [])
vd_hist = metrics.get("vd", [])
va_hist = metrics.get("va", [])
tl_hist = metrics.get("tl", [])
lr_hist = metrics.get("lr", [])
cls_hist = metrics.get("cls_iou", [])  # list of lists

cur_vi  = vi_hist[-1] if vi_hist else 0.0
cur_vl  = vl_hist[-1] if vl_hist else 0.0
cur_ti  = ti_hist[-1] if ti_hist else 0.0
cur_vd  = vd_hist[-1] if vd_hist else 0.0
cur_va  = va_hist[-1] if va_hist else 0.0
cur_lr  = lr_hist[-1] if lr_hist else lr

d_vi = cur_vi - vi_hist[-2] if len(vi_hist) >= 2 else 0
d_vl = cur_vl - vl_hist[-2] if len(vl_hist) >= 2 else 0

def metric_card(label, value, delta=None, fmt=".4f", color="var(--accent)"):
    delta_html = ""
    if delta is not None:
        sign  = "+" if delta >= 0 else ""
        cls   = "delta-up" if delta >= 0 else "delta-down"
        arrow = "↑" if delta >= 0 else "↓"
        delta_html = f'<div class="metric-delta {cls}">{arrow} {sign}{delta:{fmt}}</div>'
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value" style="color:{color}">{value:{fmt}}</div>
      {delta_html}
    </div>"""

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.markdown(metric_card("Epoch", ep, fmt="d", color="var(--accent)"), unsafe_allow_html=True)
with m2:
    st.markdown(metric_card("Val mIoU", cur_vi, delta=d_vi, color="var(--accent)"), unsafe_allow_html=True)
with m3:
    st.markdown(metric_card("Val Loss", cur_vl, delta=d_vl, color="var(--accent2)"), unsafe_allow_html=True)
with m4:
    st.markdown(metric_card("Val Dice", cur_vd, color="var(--green)"), unsafe_allow_html=True)
with m5:
    st.markdown(metric_card("Val Acc", cur_va, color="var(--yellow)"), unsafe_allow_html=True)
with m6:
    st.markdown(metric_card("Best IoU", best, color="var(--accent3)"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Progress bar
if ep > 0 and n_epochs > 0:
    prog = ep / int(n_epochs)
    st.markdown(f"**Epoch {ep}/{n_epochs}** · ETA: {eta:.1f} min · LR: `{cur_lr:.6f}`")
    st.progress(prog)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["📈 Metrics", "🎯 Per-Class IoU", "📋 Log", "📊 Analysis"])

with tab1:
    epochs = list(range(1, len(vi_hist)+1))

    if len(vi_hist) >= 2:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = make_line_chart(epochs, tl_hist, vl_hist, "Loss", "#ff6b35", "#ff4560")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = make_line_chart(epochs, ti_hist, vi_hist, "mIoU", "#00e5ff", "#7c3aed")
            st.plotly_chart(fig, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            fig = make_line_chart(epochs, [0]*len(vd_hist), vd_hist, "Dice Score", "#00ff88", "#00ff88")
            st.plotly_chart(fig, use_container_width=True)
        with col_d:
            if lr_hist:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=epochs[:len(lr_hist)], y=lr_hist,
                    line=dict(color="#ffd700", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(255,215,0,0.05)"
                ))
                fig.update_layout(**PLOTLY_LAYOUT,
                    title=dict(text="Learning Rate", font=dict(size=13, color="#e8eaf0"), x=0),
                    height=260
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:80px 0;color:#6b7280;font-family:'Space Mono',monospace;font-size:13px;">
          <div style="font-size:40px;margin-bottom:16px">📡</div>
          Waiting for training data...<br>
          <span style="font-size:11px">Start training to see live metrics</span>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    if cls_hist:
        latest_cls = cls_hist[-1]
        best_ep_idx = int(np.argmax(vi_hist)) if vi_hist else -1
        best_cls = cls_hist[best_ep_idx] if best_ep_idx >= 0 and best_ep_idx < len(cls_hist) else latest_cls

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown(f"**Latest Epoch ({ep}) — Per-Class IoU**")
            st.markdown(make_class_bar_html(latest_cls), unsafe_allow_html=True)

        with col_right:
            st.markdown(f"**Best Epoch ({best_ep_idx+1}) — Per-Class IoU**")
            st.markdown(make_class_bar_html(best_cls), unsafe_allow_html=True)

        # Radar chart
        if len(latest_cls) == 10:
            clean = [max(0, v) if not np.isnan(v) else 0 for v in latest_cls]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=clean + [clean[0]],
                theta=CLASS_NAMES + [CLASS_NAMES[0]],
                fill='toself',
                fillcolor='rgba(0,229,255,0.08)',
                line=dict(color='#00e5ff', width=2),
                name='Latest'
            ))
            if best_cls != latest_cls:
                clean_b = [max(0, v) if not np.isnan(v) else 0 for v in best_cls]
                fig.add_trace(go.Scatterpolar(
                    r=clean_b + [clean_b[0]],
                    theta=CLASS_NAMES + [CLASS_NAMES[0]],
                    fill='toself',
                    fillcolor='rgba(124,58,237,0.06)',
                    line=dict(color='#7c3aed', width=2, dash='dot'),
                    name='Best'
                ))
            fig.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0,1], color='#6b7280', gridcolor='#1e2130'),
                    angularaxis=dict(color='#6b7280', gridcolor='#1e2130')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Space Mono', color='#6b7280', size=10),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                height=380,
                margin=dict(l=30, r=30, t=30, b=30)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Evolution heatmap
        if len(cls_hist) >= 3:
            st.markdown("**Per-Class IoU Evolution**")
            arr = np.array([[max(0, v) if not np.isnan(v) else 0 for v in row] for row in cls_hist])
            fig = go.Figure(go.Heatmap(
                z=arr.T,
                x=[f"Ep{i+1}" for i in range(len(cls_hist))],
                y=CLASS_NAMES,
                colorscale=[[0,"#0a0c10"],[0.3,"#1a1d24"],[0.7,"#00e5ff"],[1.0,"#7c3aed"]],
                showscale=True
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Space Mono', color='#6b7280', size=10),
                height=300,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:80px 0;color:#6b7280;font-family:'Space Mono',monospace;font-size:13px;">
          <div style="font-size:40px;margin-bottom:16px">🎯</div>
          No per-class data yet
        </div>
        """, unsafe_allow_html=True)

with tab3:
    log_lines = load_log()
    if log_lines:
        html = "".join(fmt_log_line(l) for l in log_lines)
        st.markdown(f'<div class="log-box">{html}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="log-box" style="text-align:center;padding:40px 0;color:#6b7280">
          No logs yet. Start training and run: python train_v7.py
        </div>
        """, unsafe_allow_html=True)

with tab4:
    if vi_hist and len(vi_hist) >= 3:
        col1, col2 = st.columns(2)

        with col1:
            # Train vs Val gap (overfitting monitor)
            gaps = [abs(t - v) for t, v in zip(ti_hist, vi_hist)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(gaps)+1)), y=gaps,
                fill='tozeroy', line=dict(color='#ff6b35', width=2),
                fillcolor='rgba(255,107,53,0.08)', name='IoU Gap'
            ))
            fig.update_layout(**PLOTLY_LAYOUT,
                title=dict(text="Overfitting Monitor (Train−Val IoU Gap)", font=dict(size=13, color="#e8eaf0"), x=0),
                height=260)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Improvement per epoch
            if len(vi_hist) >= 2:
                deltas = [vi_hist[i] - vi_hist[i-1] for i in range(1, len(vi_hist))]
                colors = ['#00ff88' if d >= 0 else '#ff4560' for d in deltas]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(range(2, len(vi_hist)+1)), y=deltas,
                    marker_color=colors, name='ΔmIoU'
                ))
                fig.update_layout(**PLOTLY_LAYOUT,
                    title=dict(text="Val mIoU Improvement per Epoch", font=dict(size=13, color="#e8eaf0"), x=0),
                    height=260)
                st.plotly_chart(fig, use_container_width=True)

        # Summary table
        if vi_hist:
            df = pd.DataFrame({
                "Epoch":    list(range(1, len(vi_hist)+1)),
                "Tr Loss":  [f"{v:.4f}" for v in tl_hist],
                "Val Loss": [f"{v:.4f}" for v in vl_hist],
                "Tr mIoU":  [f"{v:.4f}" for v in ti_hist],
                "Val mIoU": [f"{v:.4f}" for v in vi_hist],
                "Val Dice": [f"{v:.4f}" for v in vd_hist],
                "Val Acc":  [f"{v:.4f}" for v in va_hist],
            })
            best_row = int(np.argmax(vi_hist))
            st.markdown(f"**Epoch History** · Best epoch: **{best_row+1}** (Val mIoU = {max(vi_hist):.4f})")
            st.dataframe(df.tail(20), use_container_width=True, height=320)
    else:
        st.markdown("""
        <div style="text-align:center;padding:80px 0;color:#6b7280;font-family:'Space Mono',monospace;font-size:13px;">
          <div style="font-size:40px;margin-bottom:16px">📊</div>
          Analysis available after 3+ epochs
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH
# ─────────────────────────────────────────────────────────────────────────────

if state == "running":
    time.sleep(refresh_rate)
    st.rerun()