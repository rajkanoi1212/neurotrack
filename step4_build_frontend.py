"""
NeuroTrack — Step 4: Build Frontend
=====================================
Injects the trained model JSON + session history into the
HTML template to create a single self-contained website
you can open directly in any browser.

Run:
    python ml/step4_build_frontend.py
Output:
    frontend/index.html   ← open this in your browser / VS Code Live Server
"""

import json
import os
import pandas as pd
import re

# ── PATHS ─────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA     = os.path.join(BASE, "data")
FRONTEND = os.path.join(BASE, "frontend")

print("NeuroTrack | Step 4: Build Frontend")
print("=" * 50)

# ── LOAD MODEL JSON ───────────────────────────────────────────────
model_path = os.path.join(DATA, "model.json")
MODEL_JSON = open(model_path).read()
print(f"Model loaded  : {os.path.getsize(model_path)//1024} KB")

# ── LOAD TAGGED SESSIONS ──────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, "tagged_sessions.csv"))
df["start_time"]       = df["start_time"].astype(str).str[:16].str.replace("T", " ").str.replace("+00:00", "")
df["duration_seconds"] = df["duration_seconds"].fillna(0).astype(int)
df["window_title"]     = df["window_title"].fillna("").str[:72]
df["url"]              = df["url"].fillna("")

# History (exclude idle for cleaner display)
hist = df[~df["activity_type"].isin(["idle", "system_ui"])].head(80)
HISTORY_JSON = hist[[
    "start_time", "duration_seconds", "process_name",
    "window_title", "activity_type", "project",
    "cognitive_state", "productivity", "source", "url"
]].to_json(orient="records")

# Stats per layer
a_stats  = (df.groupby("activity_type")
              .agg(sessions=("activity_type","count"),
                   minutes=("duration_seconds", lambda x: round(x.sum()/60,1)),
                   avg_prod=("productivity","mean"))
              .sort_values("minutes", ascending=False)
              .reset_index())
ASTATS_JSON = a_stats.to_json(orient="records")

p_stats  = (df.groupby("project")
              .agg(sessions=("project","count"),
                   minutes=("duration_seconds", lambda x: round(x.sum()/60,1)),
                   avg_prod=("productivity","mean"))
              .sort_values("minutes", ascending=False)
              .reset_index())
PSTATS_JSON = p_stats.to_json(orient="records")

print(f"History rows  : {len(hist)}")
print(f"Activity types: {df['activity_type'].nunique()}")
print(f"Projects      : {df['project'].nunique()}")

# ── READ TEMPLATE ─────────────────────────────────────────────────
template_path = os.path.join(FRONTEND, "template.html")
html = open(template_path, encoding="utf-8").read()

# ── INJECT DATA ───────────────────────────────────────────────────
html = (html
    .replace("/*__MODEL__*/",   MODEL_JSON)
    .replace("/*__HISTORY__*/", HISTORY_JSON)
    .replace("/*__ASTATS__*/",  ASTATS_JSON)
    .replace("/*__PSTATS__*/",  PSTATS_JSON))

# ── WRITE ─────────────────────────────────────────────────────────
out_path = os.path.join(FRONTEND, "index.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

size = os.path.getsize(out_path)
print(f"\n✓ Built → {out_path}  ({size//1024} KB)")
print("\nHow to open:")
print("  VS Code       → right-click index.html → Open with Live Server")
print("  Browser       → File → Open File → frontend/index.html")
print("  Terminal      → python -m http.server 8080  (then visit localhost:8080/frontend/)")
