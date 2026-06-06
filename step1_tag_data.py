"""
NeuroTrack — Step 1: Auto-Tag Sessions
=======================================
Reads raw tracker data and assigns 4 labels to every session:
  Layer 1 — activity_type   (what kind of work)
  Layer 2 — project         (which project)
  Layer 3 — cognitive_state (how focused)
  Layer 4 — productivity    (score 0–100)

Run:
    python ml/step1_tag_data.py
Output:
    data/tagged_sessions.csv
"""

import pandas as pd
import numpy as np
import re
import json
import os

# ── PATHS ─────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, "data")
UPLOAD = os.path.join(DATA, "raw")       # put your CSVs here

os.makedirs(DATA, exist_ok=True)

print("NeuroTrack | Step 1: Tagging sessions")
print("=" * 50)

# ── LOAD ──────────────────────────────────────────────────────────
print("Loading raw data...")

registry  = pd.read_csv(os.path.join(UPLOAD, "registry.csv"))
vsessions = pd.read_csv(os.path.join(UPLOAD, "v_sessions.csv"))
chrome    = pd.read_csv(os.path.join(UPLOAD, "chrome_sessions_separated.csv"))
raw       = pd.read_csv(os.path.join(UPLOAD, "raw_events.csv"))

# Parse timestamps
for df in [vsessions, chrome, raw]:
    for col in ["start_time", "end_time", "timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

# Extract URLs from raw event metadata
url_map = {}
for _, r in raw.iterrows():
    if pd.notna(r.get("metadata")):
        try:
            d = json.loads(r["metadata"])
            if "url" in d:
                url_map[r["registry_id"]] = d["url"]
        except Exception:
            pass

# Merge native sessions with registry
sessions         = vsessions.merge(registry, on="registry_id", how="left")
sessions["source"] = "native"
chrome["source"]   = "chrome"

COLS = ["start_time", "end_time", "duration_seconds",
        "process_name", "window_title", "exec_path",
        "window_class", "source", "registry_id"]

for col in COLS:
    for df in [sessions, chrome]:
        if col not in df.columns:
            df[col] = ""

all_df = pd.concat([sessions[COLS], chrome[COLS]], ignore_index=True)

# Clean columns
for col in ["window_title", "process_name", "exec_path", "window_class"]:
    all_df[col] = all_df[col].fillna("").astype(str).str.strip()

all_df["duration_seconds"] = all_df["duration_seconds"].fillna(0).astype(float)
all_df["url"]              = all_df["registry_id"].map(url_map).fillna("")
all_df["start_time"]       = pd.to_datetime(all_df["start_time"], utc=True, errors="coerce")

print(f"  Total sessions loaded : {len(all_df)}")
print(f"  Sessions with URL     : {(all_df['url'] != '').sum()}")


# ══════════════════════════════════════════════════════════════════
# LAYER 1 — ACTIVITY TYPE
# What kind of activity is happening in this session?
# ══════════════════════════════════════════════════════════════════

# Compile regex patterns once (fast reuse)
AI_BARE  = re.compile(r"chatgpt|gemini|claude|gpt|copilot|perplexity|bard")
AI_TOPIC = re.compile(
    r"setup|environment|low.cost|intern|offer|branding|marketing|"
    r"monetization|timeslate|launch|how to|tutorial|docs|overview|"
    r"learn|course|arxiv|paper|research|article|blog|machine learning|"
    r"ml|classifier"
)

def classify_activity(row) -> str:
    """
    Rule-based classifier using all available signals:
    window_title, process_name, exec_path, url, duration_seconds
    """
    t   = row["window_title"].lower()
    p   = row["process_name"].lower()
    pa  = row["exec_path"].lower()
    url = row["url"].lower()
    d   = row["duration_seconds"]

    # ── System noise ─────────────────────────────────────────────
    if p in ("system_idle", "system") or d > 7200:
        return "idle"

    if p == "explorer" and t in (
        "task switching", "system tray overflow window.", "",
        "inactivity threshold crossed", "input resumed"
    ):
        return "system_ui"

    # ── Gaming ───────────────────────────────────────────────────
    if any(x in p for x in ("leagueclientux", "league of legends")):
        return "gaming"
    if "riot games" in pa:
        return "gaming"

    # ── Entertainment ─────────────────────────────────────────────
    if any(x in url for x in ("youtube.com", "youtu.be", "netflix.com", "primevideo.com", "twitch.tv", "spotify.com")):
        return "entertainment"
    if any(x in t for x in ("youtube", "netflix", "prime video", "twitch", "spotify", "watch movie", "watching movie", "movie")):
        return "entertainment"

    # ── Deep development (sustained IDE coding) ───────────────────
    if p in ("rustrover64", "rustrover") or "rustrover" in pa:
        return "deep_dev"
    if p == "recall-nexus" or ("recall" in pa and "nexus" in pa):
        return "deep_dev"
    if any(x in t for x in (
        ".rs –", "db.rs", "lib.rs", "+page.svelte", "schema.rs",
        "src-tauri", "console [recall", "v_sessions", "raw_events ["
    )):
        return "deep_dev"
    if any(x in t for x in ("connecting to 'recall", "connecting to 'qa")):
        return "deep_dev"

    # ── GitHub / repo activity ────────────────────────────────────
    if "github.com" in url:
        if any(x in url for x in ("/settings", "/delete", "/forks",
                                   "/network", "/dependencies")):
            return "repo_admin"
        return "repo_review"
    if any(x in t for x in (
        "your repositories", "new repository", "classify-app",
        "forks ·", "dependencies ·", "page not found · github"
    )):
        return "repo_review"

    # ── AI-assisted research (AI + topic = active learning) ───────
    if AI_BARE.search(t) and AI_TOPIC.search(t):
        return "ai_research"
    if "gemini.google.com" in url and len(t) > 30:
        return "ai_research"

    # ── Bare AI tool (assistant open, no topic context) ───────────
    if AI_BARE.search(t) or "image generator" in t:
        return "ai_tools"

    # ── Async communications ──────────────────────────────────────
    if "web.whatsapp.com" in url or "whatsapp" in t:
        return "async_comms"
    if "mail.google.com" in url or any(x in t for x in ("gmail", "inbox")):
        return "async_comms"
    if "general" in t and "chrome" in t:
        return "async_comms"   # Slack general channel in browser

    # ── Synchronous meetings ──────────────────────────────────────
    if "meet.google.com" in url:
        return "sync_meeting"
    if "meet.google.com is sharing" in t or "meet.google.com wants to" in t:
        return "sync_meeting"
    if "meet" in t and any(x in t for x in (
        "project discussion", "kmi-", "sharing"
    )):
        return "sync_meeting"

    # ── Calendar / scheduling ─────────────────────────────────────
    if "updated invitation" in t:
        return "calendar_admin"

    # ── Pure navigation noise ─────────────────────────────────────
    if t in (
        "new tab - google chrome",
        "new incognito tab - google chrome",
        "untitled - google chrome",
    ):
        return "navigation"

    # ── Google search intent ──────────────────────────────────────
    if "google.com/search" in url:
        return "web_search"

    return "other"


# ══════════════════════════════════════════════════════════════════
# LAYER 2 — PROJECT CONTEXT
# Which project or life area does this session belong to?
# ══════════════════════════════════════════════════════════════════

def classify_project(row) -> str:
    t   = row["window_title"].lower()
    url = row["url"].lower()
    pa  = row["exec_path"].lower()

    if any(x in t for x in (
        "recall-nexus", "recall_nexus", "db.rs", "lib.rs",
        "schema.rs", "+page.svelte", "v_sessions", "raw_events",
        "qa environment", "focus"
    )) or ("recall" in pa and "nexus" in pa):
        return "proj_recall_nexus"

    if (any(x in t for x in ("classify-app", "classify_app"))
            or "classify-app-usage-logs" in url):
        return "proj_classify_app"

    if any(x in t for x in (
        "timeslate", "branding", "marketing", "monetization", "launch"
    )):
        return "proj_timeslate"

    if any(x in t for x in ("intern", "offer letter", "machine learning intern")):
        return "proj_job_hunt"

    if any(x in t for x in ("low-cost", "low cost", "dev environment setup")):
        return "proj_learning"

    if any(x in t for x in ("whatsapp", "cashback", "your cashback")):
        return "personal"

    if any(x in t for x in ("league of legends", "league", "youtube", "netflix", "prime video", "twitch", "spotify", "movie")):
        return "leisure"

    if any(x in t for x in (
        "project discussion", "meeting", "meet -", "kmi-",
        "invitation", "schedule"
    )):
        return "proj_meetings"

    if any(x in t for x in ("github", "rustrover", "new repository")):
        return "proj_dev_generic"

    return "unassigned"


# ══════════════════════════════════════════════════════════════════
# LAYER 3 — COGNITIVE STATE
# How mentally engaged is the user during this session?
# ══════════════════════════════════════════════════════════════════

def classify_cognitive(row) -> str:
    d        = row["duration_seconds"]
    activity = row["activity_type"]

    # Maintenance: OS-level, no cognitive load
    if activity in ("idle", "system_ui", "navigation"):
        return "maintenance"

    # Context switching: too short to do real work
    if d < 5:
        return "context_switch"

    # Flow state: long, deep, focused sessions
    if d >= 120 and activity in ("deep_dev", "ai_research", "sync_meeting"):
        return "flow_state"
    if d >= 60 and activity == "deep_dev":
        return "flow_state"

    # Distracted: gaming / entertainment (always off-task in a work tracker)
    if activity in ("gaming", "entertainment"):
        return "distracted"

    # Shallow work: quick lookups, short browsing
    if d < 30 and activity in ("ai_tools", "ai_research", "web_search", "repo_review"):
        return "shallow_work"

    return "engaged_work"


# ══════════════════════════════════════════════════════════════════
# LAYER 4 — PRODUCTIVITY SCORE (0–100)
# Weighted formula: base_score × project_bonus × state_multiplier
# + duration_bonus (diminishing returns after 5 min)
# ══════════════════════════════════════════════════════════════════

ACTIVITY_BASE = {
    "deep_dev": 50, "ai_research": 40, "sync_meeting": 30,
    "async_comms": 20, "repo_review": 25, "ai_tools": 20,
    "repo_admin": 15, "calendar_admin": 10, "web_search": 10,
    "navigation": 2,  "system_ui": 1, "idle": 0,
    "gaming": -10, "entertainment": -10, "other": 5,
}
PROJECT_BONUS = {
    "proj_recall_nexus": 20, "proj_classify_app": 20,
    "proj_timeslate": 15,    "proj_learning": 15,
    "proj_job_hunt": 10,     "proj_dev_generic": 10,
    "proj_meetings": 5,      "personal": 0, "leisure": -10,
}
STATE_MULT = {
    "flow_state": 1.4, "engaged_work": 1.0, "shallow_work": 0.6,
    "context_switch": 0.4, "maintenance": 0.2, "distracted": 0.1,
}

def calc_productivity(row) -> int:
    base    = ACTIVITY_BASE.get(row["activity_type"], 5)
    pb      = PROJECT_BONUS.get(row["project"], 0)
    sm      = STATE_MULT.get(row["cognitive_state"], 1.0)
    dur_b   = min(15, np.log1p(row["duration_seconds"] / 60) * 5)
    score   = (base + pb + dur_b) * sm
    return int(max(0, min(100, round(score))))


# ── APPLY ALL 4 LAYERS ────────────────────────────────────────────
print("\nApplying classification layers...")

all_df["activity_type"]   = all_df.apply(classify_activity, axis=1)
all_df["project"]         = all_df.apply(classify_project, axis=1)
all_df["cognitive_state"] = all_df.apply(classify_cognitive, axis=1)
all_df["productivity"]    = all_df.apply(calc_productivity, axis=1)

# ── RESULTS SUMMARY ───────────────────────────────────────────────
print("\n── Layer 1: Activity Type ──────────────────────────")
print(all_df["activity_type"].value_counts().to_string())

print("\n── Layer 2: Project ────────────────────────────────")
print(all_df["project"].value_counts().to_string())

print("\n── Layer 3: Cognitive State ────────────────────────")
print(all_df["cognitive_state"].value_counts().to_string())

print("\n── Layer 4: Productivity (avg by activity) ─────────")
print(all_df.groupby("activity_type")["productivity"]
      .mean().round(1).sort_values(ascending=False).to_string())

# ── SAVE ─────────────────────────────────────────────────────────
out_path = os.path.join(DATA, "tagged_sessions.csv")
all_df.to_csv(out_path, index=False)
print(f"\n✓ Saved → {out_path}  ({len(all_df)} rows)")
