"""
NeuroTrack — Step 2: Feature Engineering
=========================================
Converts raw session columns into a 62-feature numeric matrix
that the RandomForest models can learn from.

Features cover:
  - Duration (raw, log, sqrt, bucket flags)
  - Time of day / day of week
  - Process name signals
  - URL domain signals (github, gemini, gmail, meet, whatsapp…)
  - Window title signals (file extensions, keywords, AI, meetings…)
  - Exec path signals
  - Window class signals

Run:
    python ml/step2_features.py
Output:
    data/features.csv       (numeric matrix, 62 columns)
    data/labels.csv         (4 label columns)
"""

import pandas as pd
import numpy as np
import os

# ── PATHS ─────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data")

print("NeuroTrack | Step 2: Feature Engineering")
print("=" * 50)

# ── LOAD TAGGED DATA ──────────────────────────────────────────────
src = os.path.join(DATA, "tagged_sessions.csv")
df  = pd.read_csv(src)

for col in ["window_title", "process_name", "exec_path", "window_class", "url", "source"]:
    df[col] = df[col].fillna("").astype(str).str.strip()

df["duration_seconds"] = df["duration_seconds"].fillna(0).astype(float)
df["start_time"]       = pd.to_datetime(df["start_time"], utc=True, errors="coerce")

print(f"Loaded {len(df)} tagged sessions")


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# Each feature group is labelled with its index range so you can
# quickly match Python ↔ JavaScript implementations.
# ══════════════════════════════════════════════════════════════════

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame of 62 numeric features.
    All values are float32, ready for sklearn.
    """
    t   = df["window_title"].str.lower()
    p   = df["process_name"].str.lower()
    pa  = df["exec_path"].str.lower()
    wc  = df["window_class"].str.lower()
    url = df["url"].str.lower()
    d   = df["duration_seconds"].fillna(0).astype(float)

    st   = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    hour = st.dt.hour.fillna(14).astype(int)
    dow  = st.dt.dayofweek.fillna(2).astype(int)   # 0 = Monday

    F = pd.DataFrame()

    # ── [0–2] Duration core ───────────────────────────────────────
    F["dur"]      = d
    F["dur_log"]  = np.log1p(d)          # compress large values
    F["dur_sqrt"] = np.sqrt(d)

    # ── [3–7] Duration bucket flags ───────────────────────────────
    # Helps the model learn category-specific thresholds
    F["is_micro"]  = (d <= 3).astype(int)            # OS flicker
    F["is_short"]  = ((d > 3)   & (d <= 30)).astype(int)   # quick glance
    F["is_medium"] = ((d > 30)  & (d <= 300)).astype(int)  # normal task
    F["is_long"]   = ((d > 300) & (d <= 3600)).astype(int) # deep work
    F["is_vlong"]  = (d > 3600).astype(int)           # marathon

    # ── [8–13] Time signals ───────────────────────────────────────
    F["hour"]         = hour
    F["is_night"]     = ((hour >= 22) | (hour < 6)).astype(int)
    F["is_morning"]   = ((hour >= 6)  & (hour < 12)).astype(int)
    F["is_afternoon"] = ((hour >= 12) & (hour < 18)).astype(int)
    F["is_evening"]   = ((hour >= 18) & (hour < 22)).astype(int)
    F["is_weekend"]   = (dow >= 5).astype(int)

    # ── [14–19] Process name signals ──────────────────────────────
    F["p_chrome"]    = p.str.contains("chrome",   regex=False).astype(int)
    F["p_rustrover"] = p.str.contains("rustrover",regex=False).astype(int)
    F["p_recall"]    = p.str.contains("recall",   regex=False).astype(int)
    F["p_league"]    = p.str.contains("league|leagueclient", regex=True).astype(int)
    F["p_explorer"]  = p.str.contains("explorer", regex=False).astype(int)
    F["p_idle"]      = p.str.contains("idle|system", regex=True).astype(int)

    # ── [20–28] URL domain signals ────────────────────────────────
    # These are the most precise signals — a URL cannot be faked by the title
    F["url_github"]          = url.str.contains("github.com", regex=False).astype(int)
    F["url_github_admin"]    = url.str.contains(
        "github.com.*settings|github.com.*forks|github.com.*dependencies",
        regex=True
    ).astype(int)
    F["url_gemini"]          = url.str.contains("gemini.google.com", regex=False).astype(int)
    F["url_chatgpt"]         = url.str.contains("chatgpt.com",        regex=False).astype(int)
    F["url_gmail"]           = url.str.contains("mail.google.com",    regex=False).astype(int)
    F["url_meet"]            = url.str.contains("meet.google.com",    regex=False).astype(int)
    F["url_whatsapp"]        = url.str.contains("web.whatsapp.com",   regex=False).astype(int)
    F["url_google_search"]   = url.str.contains("google.com/search",  regex=False).astype(int)
    F["url_has_any"]         = (url != "").astype(int)

    # ── [29–30] Title length signals ──────────────────────────────
    F["t_len"]   = t.str.len().fillna(0).astype(int)
    F["t_words"] = t.str.split().str.len().fillna(0).astype(int)

    # ── [31–46] Title keyword signals ─────────────────────────────
    F["t_has_code_file"] = t.str.contains(
        r"\.rs|\.svelte|schema|lib\.|db\.|console \[|v_sessions|raw_events",
        regex=True
    ).astype(int)
    F["t_has_ai"]    = t.str.contains(
        "chatgpt|gemini|claude|gpt|copilot", regex=True
    ).astype(int)
    F["t_has_topic"] = t.str.contains(
        "setup|intern|offer|branding|marketing|monetization|timeslate|"
        "low.cost|machine learning|ml",
        regex=True
    ).astype(int)
    # Compound: AI + topic together = research (strongest signal)
    F["t_ai_and_topic"]  = (F["t_has_ai"] & F["t_has_topic"]).astype(int)
    F["t_has_github"]    = t.str.contains(
        "github|gitlab|yashrocky|classify-app", regex=True
    ).astype(int)
    F["t_has_meet"]      = t.str.contains(r"\bmeet\b", regex=True).astype(int)
    F["t_has_meet_ctx"]  = t.str.contains(
        "project discussion|kmi-|sharing screen|wants to", regex=True
    ).astype(int)
    F["t_has_gmail"]     = t.str.contains("gmail|inbox",     regex=True).astype(int)
    F["t_has_whatsapp"]  = t.str.contains("whatsapp",        regex=False).astype(int)
    F["t_has_league"]    = t.str.contains("league of legends", regex=False).astype(int)
    F["t_has_newtab"]    = t.str.contains(
        "new tab|new incognito|untitled", regex=True
    ).astype(int)
    F["t_has_recall"]    = t.str.contains(
        "recall.nexus|recall_nexus", regex=True
    ).astype(int)
    F["t_has_connecting"]  = t.str.contains("connecting to", regex=False).astype(int)
    F["t_has_invitation"]  = t.str.contains("invitation|updated invitation", regex=True).astype(int)
    F["t_has_cashback"]    = t.str.contains("cashback|recap", regex=True).astype(int)
    F["t_has_image_gen"]   = t.str.contains("image generator|dall", regex=True).astype(int)

    # ── [47–52] Exec path signals ─────────────────────────────────
    F["pa_rustrover"] = pa.str.contains("rustrover", regex=False).astype(int)
    F["pa_riot"]      = pa.str.contains("riot",      regex=False).astype(int)
    F["pa_chrome"]    = pa.str.contains("chrome",    regex=False).astype(int)
    F["pa_recall"]    = pa.str.contains("recall|nexus", regex=True).astype(int)
    F["pa_focus"]     = pa.str.contains("focus",     regex=False).astype(int)
    F["pa_windows"]   = pa.str.contains("windows|explorer", regex=True).astype(int)

    # ── [53] Source ───────────────────────────────────────────────
    F["src_chrome"] = (df["source"] == "chrome").astype(int)

    # ── [54–58] Window class signals ──────────────────────────────
    # Window class is set by the OS / app — very reliable
    F["wc_java"]     = wc.str.contains("sunawtframe|sunawtdialog", regex=True).astype(int)  # RustRover (Java)
    F["wc_tauri"]    = wc.str.contains("tauri",         regex=False).astype(int)  # recall-nexus app
    F["wc_chrome"]   = wc.str.contains("chrome_widget", regex=False).astype(int)  # Chrome browser
    F["wc_riot"]     = wc.str.contains("rclient|riotwindow", regex=True).astype(int)  # League client
    F["wc_explorer"] = wc.str.contains(
        "toplevelfwindow|xamlexplorer|tray|foreground", regex=True
    ).astype(int)  # Windows Explorer / taskbar

    # ── [59–61] Entertainment signals ─────────────────────────────
    F["url_entertainment"] = url.str.contains(
        "youtube.com|youtu.be|netflix.com|primevideo.com|twitch.tv|spotify.com",
        regex=True
    ).astype(int)
    F["t_has_entertainment"] = t.str.contains(
        "youtube|netflix|prime video|twitch|spotify|movie|video|watch",
        regex=True
    ).astype(int)
    F["p_entertainment"] = p.str.contains(
        "netflix|spotify",
        regex=True
    ).astype(int)

    return F.fillna(0).astype(np.float32)


# ── BUILD & SAVE ──────────────────────────────────────────────────
X = build_feature_matrix(df)

# 4 label columns
LABEL_COLS = ["activity_type", "project", "cognitive_state", "productivity"]
y = df[LABEL_COLS].copy()

print(f"\nFeature matrix : {X.shape[0]} rows × {X.shape[1]} features")
print(f"Feature names  : {list(X.columns)}")

feat_path  = os.path.join(DATA, "features.csv")
label_path = os.path.join(DATA, "labels.csv")

X.to_csv(feat_path,  index=False)
y.to_csv(label_path, index=False)

print(f"\n✓ Saved features → {feat_path}")
print(f"✓ Saved labels   → {label_path}")
