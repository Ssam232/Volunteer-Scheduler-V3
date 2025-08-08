import streamlit as st
import pandas as pd
import io, re, difflib
import Scheduler2  # scheduling core module
from Scheduler2 import build_schedule, MAX_PER_SHIFT, load_preferences

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(page_title="PEMRAP Volunteer Scheduler V3", layout="wide")
st.title("📅 PEMRAP Volunteer Scheduler V3")

# ── Initialize session state ─────────────────────────────────────────────────
for key, default in (
    ("df_raw", None),
    ("sched_df", None),
    ("breakdown_df", None),
    ("group_report", None),
    ("pairs_text", ""),
    ("trigger_run", False),
):
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helpers ─────────────────────────────────────────────────────────────────
def extract_names_for_ui(df: pd.DataFrame) -> list[str]:
    """Build a clean Name list for the Group UI from the raw sheet."""
    cols = {c.lower(): c for c in df.columns}
    first = next((cols[k] for k in cols if "first" in k and "name" in k), None)
    last  = next((cols[k] for k in cols if "last"  in k and "name" in k), None)
    if first and last:
        names = (df[first].astype(str).str.strip() + " " +
                 df[last].astype(str).str.strip()).tolist()
    elif "name" in cols:
        names = df[cols["name"]].astype(str).tolist()
    else:
        names = (df.iloc[:, 0].astype(str).str.strip() + " " +
                 df.iloc[:, 1].astype(str).str.strip()).tolist()
    names = {re.sub(r"\s+", " ", n).strip() for n in names if str(n).strip()}
    return sorted(names)

def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).lower()

def style_group_report(df: pd.DataFrame):
    """Darker row colors for dark backgrounds (results table)."""
    def _row_style(row):
        status = str(row.get("Status", ""))
        if "Grouped ✓" in status:
            bg, fg = "#0f3d1e", "#b9f6ca"   # dark green bg, mint text
        elif "Not grouped" in status or "Not in schedule" in status:
            bg, fg = "#3d0f12", "#ff8a80"   # dark red bg, soft red text
        elif "Skipped" in status:
            bg, fg = "#3d2a00", "#ffd54f"   # dark amber bg, amber text
        else:
            bg, fg = "#263238", "#eceff1"   # slate bg, light text
        return [f"background-color: {bg}; color: {fg};"] * len(row)
    return df.style.apply(_row_style, axis=1).set_properties(**{"white-space": "nowrap"})

def build_placement_maps(sched_df: pd.DataFrame):
    """
    Return two dicts keyed by normalized name:
      placed_by_key[name] -> 'Friday 10:00-14:00' (friendly slot)
      forced_by_key[name] -> True/False from 'Fallback'
    """
    tmp = sched_df.copy()
    if "Day" not in tmp.columns or "Shift" not in tmp.columns:
        ts_tmp = tmp["Time Slot"].astype(str)
        parts_tmp = ts_tmp.str.split(r"\s+", n=1, expand=True)
        tmp["Day"] = parts_tmp[0].str.strip().str.title()
        tmp["Shift"] = parts_tmp[1].str.replace(r"[–—−]", "-", regex=True).str.strip()
    tmp["_key"] = tmp["Name"].map(norm_name)

    placed_by_key: dict[str, str] = {}
    forced_by_key: dict[str, bool] = {}

    for _, r in tmp.iterrows():
        key = r["_key"]
        day = str(r.get("Day", "")).strip()
        shf = str(r.get("Shift", "")).strip()
        ts_label = str(r.get("Time Slot", "")).strip()
        label = f"{day} {shf}".strip() if (day and shf) else ts_label
        if key not in placed_by_key:
            placed_by_key[key] = label
            forced_by_key[key] = bool(r.get("Fallback", False))
    return placed_by_key, forced_by_key

def _replace_tokens(text: str, replacements: list[tuple[str, str]]) -> str:
    """Replace exact tokens between commas across lines."""
    lines = (text or "").splitlines()
    rep_map = {raw: sugg for raw, sugg in replacements if sugg}
    new_lines = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        new_parts = [rep_map.get(p, p) for p in parts if p]
        new_lines.append(", ".join(new_parts))
    return "\n".join(new_lines)

# ---- WHAT-IF EXPLAINER ------------------------------------------------------
def explain_non_grouping(a_can: str, b_can: str, sched_df: pd.DataFrame, df_raw: pd.DataFrame) -> str:
    """
    Heuristic post-solve explainer:
    - If no shared availability → say so.
    - Else check each shared slot for capacity/mentor coverage feasibility w.r.t. current schedule.
    - If some slot looks feasible, say it would worsen the objective (more forced assignments / fewer first choices).
    """
    try:
        volunteers, roles, shifts, weights, prefs_map = load_preferences(df_raw)
    except Exception:
        return "Could not analyze availability details from the survey."

    prefs_a = set(prefs_map.get(a_can, []))
    prefs_b = set(prefs_map.get(b_can, []))
    shared = [s for s in shifts if s in prefs_a and s in prefs_b]
    if not shared:
        return "They don't share any common availability in the survey."

    # Current slot -> list of (name, role, fallback)
    by_slot: dict[str, list[tuple[str, str, bool]]] = {}
    for _, r in sched_df.iterrows():
        slot = str(r["Time Slot"])
        by_slot.setdefault(slot, []).append((r["Name"], str(r.get("Role", "")), bool(r["Fallback"])))

    capacity_block, mentor_block = True, True
    for s in shared:
        entries = by_slot.get(s, [])
        # capacity check if adding both
        already = sum(1 for n, _, _ in entries if n in (a_can, b_can))
        needed = 2 - already
        if len(entries) + max(0, needed) > MAX_PER_SHIFT:
            continue  # capacity blocked here
        capacity_block = False

        # mentor coverage if any mentee present
        roles_here = [rl for _, rl, _ in entries]
        roles_aug = roles_here + [roles.get(a_can, ""), roles.get(b_can, "")]
        any_mentee = any(r == "mentee" for r in roles_aug)
        any_mentor = any(r == "mentor" for r in roles_aug)
        if any_mentee and not any_mentor:
            continue  # mentor rule blocked here
        mentor_block = False

        # Feasible under rules, solver likely avoided due to objective tradeoffs
        return (f"Could only group them by placing both on **{s}** without breaking rules, "
                f"but that would worsen the objective (more forced assignments / fewer first choices).")

    if capacity_block and mentor_block:
        return ("All shared slots were at capacity and would also violate mentor coverage "
                "(adding a mentee without a mentor).")
    if capacity_block:
        return "All shared slots were full (3 of 3)."
    if mentor_block:
        return "Shared slots would break mentor coverage (mentee present without a mentor)."
    return "Grouping would worsen the objective under current constraints."

def run_scheduler(df_raw: pd.DataFrame, vol_lookup: dict):
    """Run the solver using current pairs_text and store results in session_state."""
    # Parse pairs for this run
    pairs_text = (st.session_state.get("pairs_text") or "").strip()
    valid_pairs_run, raw_pairs_run = [], []
    if pairs_text:
        for line in pairs_text.splitlines():
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) != 2:
                continue
            a_raw, b_raw = parts
            raw_pairs_run.append((a_raw, b_raw))
            a_can = vol_lookup.get(norm_name(a_raw))
            b_can = vol_lookup.get(norm_name(b_raw))
            if a_can and b_can:
                valid_pairs_run.append((a_can, b_can))

    # Apply and solve
    Scheduler2.GROUP_PAIRS = valid_pairs_run
    sched_df, _, breakdown_df = build_schedule(df_raw)

    # Parse Day/Shift for grid
    if "Time Slot" in sched_df.columns:
        ts = sched_df["Time Slot"].astype(str)
        parts = ts.str.split(r"\s+", n=1, expand=True)
        sched_df["Day"] = parts[0].str.strip().str.title().fillna("")
        sched_df["Shift"] = parts[1].str.replace(r"[–—−]", "-", regex=True).str.strip().fillna("")
    else:
        sched_df["Day"], sched_df["Shift"] = "", ""

    # Save schedule + breakdown
    st.session_state.sched_df = sched_df
    st.session_state.breakdown_df = breakdown_df

    # Build group report (with forced indication + what-if)
    placed_by_key, forced_by_key = build_placement_maps(sched_df)
    report_rows = []
    for a_raw, b_raw in raw_pairs_run:
        a_can = vol_lookup.get(norm_name(a_raw))
        b_can = vol_lookup.get(norm_name(b_raw))

        if not a_can or not b_can:
            missing = []
            if not a_can: missing.append(a_raw)
            if not b_can: missing.append(b_raw)
            nice = ", ".join(missing)
            report_rows.append({
                "Pair": f"{a_raw} & {b_raw}",
                "Status": "Skipped",
                "Details": f"We couldn't find {nice} in the uploaded list. Use the suggestions above to fix the spelling."
            })
            continue

        sa = placed_by_key.get(norm_name(a_can))
        sb = placed_by_key.get(norm_name(b_can))

        if sa and sb and sa == sb:
            fa = forced_by_key.get(norm_name(a_can), False)
            fb_ = forced_by_key.get(norm_name(b_can), False)
            forced_names = [n for n, f in ((a_can, fa), (b_can, fb_)) if f]
            details = f"Scheduled together on **{sa}**."
            if forced_names:
                details += " (forced for " + ", ".join(forced_names) + ")"
            report_rows.append({
                "Pair": f"{a_can} & {b_can}",
                "Status": "Grouped ✓",
                "Details": details
            })
            continue

        if sa and sb and sa != sb:
            reason = explain_non_grouping(a_can, b_can, sched_df, df_raw)
            report_rows.append({
                "Pair": f"{a_can} & {b_can}",
                "Status": "Not grouped ✗",
                "Details": reason
            })
            continue

        # Not placed
        not_placed = []
        if not sa: not_placed.append(a_can)
        if not sb: not_placed.append(b_can)
        who = f"{not_placed[0]} and {not_placed[1]}" if len(not_placed) == 2 else not_placed
