import streamlit as st
import pandas as pd
import io, re, difflib, unicodedata, html
import importlib.util, glob, os, sys

# ── Robust import for Scheduler2 on case-sensitive systems / odd filenames ──
try:
    import Scheduler2  # preferred
    from Scheduler2 import build_schedule, MAX_PER_SHIFT, load_preferences
except ModuleNotFoundError:
    try:
        import scheduler2 as Scheduler2  # alt case
        from scheduler2 import build_schedule, MAX_PER_SHIFT, load_preferences
    except ModuleNotFoundError:
        # Last resort: locate a nearby Scheduler2*.py
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = glob.glob(os.path.join(here, "Scheduler2*.py")) + \
                     glob.glob(os.path.join(here, "scheduler2*.py"))
        if not candidates:
            raise  # nothing to import
        spec = importlib.util.spec_from_file_location("Scheduler2", candidates[0])
        _mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(_mod)
        sys.modules["Scheduler2"] = _mod
        Scheduler2 = _mod
        build_schedule = _mod.build_schedule
        MAX_PER_SHIFT = getattr(_mod, "MAX_PER_SHIFT", 3)
        load_preferences = _mod.load_preferences

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
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))

def norm_name(s: str) -> str:
    # accent/case/whitespace-insensitive
    s = _strip_accents(str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s.casefold()

def extract_names_for_ui(df: pd.DataFrame) -> list[str]:
    """Build a clean Name list for the Group UI from the raw sheet."""
    cols = {c.lower(): c for c in df.columns}
    first = next((cols[k] for k in cols if "first" in k and "name" in k), None)
    last  = next((cols[k] for k in cols if "last"  in k and "name" in k), None)
    if first and last:
        s = (df[first].fillna("").astype(str).str.strip() + " " +
             df[last].fillna("").astype(str).str.strip())
    elif "name" in cols:
        s = df[cols["name"]].fillna("").astype(str).str.strip()
    else:
        s = (df.iloc[:, 0].fillna("").astype(str).str.strip() + " " +
             df.iloc[:, 1].fillna("").astype(str).str.strip())
    names = {re.sub(r"\s+", " ", n).strip() for n in s.tolist() if str(n).strip()}
    return sorted(names)

def style_group_report(df: pd.DataFrame):
    """Colorized result rows, rendered via st.dataframe(Styler)."""
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
    """Return maps of placement and forced flags by normalized name."""
    tmp = sched_df.copy()
    if "Day" not in tmp.columns or "Shift" not in tmp.columns:
        ts_tmp = tmp["Time Slot"].astype(str)
        parts_tmp = ts_tmp.str.split(r"\s+", n=1, expand=True)
        tmp["Day"] = parts_tmp[0].fillna("").astype(str).str.strip().str.title()
        tmp["Shift"] = parts_tmp[1].fillna("").astype(str).str.replace(r"[–—−]", "-", regex=True).str.strip()
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

def parse_time_to_minutes(s: str) -> int | float:
    """Robust shift sort key from 'HH:MM' or 'H:MMam/pm' etc. Returns minutes from 00:00."""
    s = (s or "").strip().lower()
    s = re.sub(r"[–—−]", "-", s)
    s = re.sub(r"\s+", "", s)
    start = s.split("-", 1)[0]
    m = re.match(r"^(\d{1,2}):?(\d{2})(am|pm)?$", start)
    if not m:
        m = re.match(r"^(\d{1,2})(am|pm)$", start)
        if m:
            h = int(m.group(1)) % 12
            if m.group(2) == "pm":
                h += 12
            return h * 60
        return float("inf")
    h = int(m.group(1))
    mm = int(m.group(2))
    ampm = m.group(3)
    if ampm:
        h = h % 12
        if ampm == "pm":
            h += 12
    return h * 60 + mm

# ---- WHAT-IF EXPLAINER ------------------------------------------------------
def explain_non_grouping(a_can: str, b_can: str, sched_df: pd.DataFrame, df_raw: pd.DataFrame) -> str:
    try:
        _, roles, shifts, _, prefs_map = load_preferences(df_raw)
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
        slot = str(r.get("Time Slot", ""))
        by_slot.setdefault(slot, []).append(
            (str(r.get("Name", "")), str(r.get("Role", "")), bool(r.get("Fallback", False)))
        )

    capacity_block, mentor_block = True, True
    for s in shared:
        entries = by_slot.get(s, [])
        already = sum(1 for n, _, _ in entries if n in (a_can, b_can))
        needed = max(0, 2 - already)
        if len(entries) + needed > MAX_PER_SHIFT:
            continue
        capacity_block = False

        roles_here = [rl for _, rl, _ in entries]
        roles_aug = roles_here + [roles.get(a_can, ""), roles.get(b_can, "")]
        any_mentee = any(r == "mentee" for r in roles_aug)
        any_mentor = any(r == "mentor" for r in roles_aug)
        if any_mentee and not any_mentor:
            continue
        mentor_block = False

        return (f"Could only group them by placing both on **{s}** without breaking rules, "
                "but that would worsen the objective (more forced assignments / fewer first choices).")

    if capacity_block and mentor_block:
        return ("All shared slots were at capacity and would also violate mentor coverage "
                "(adding a mentee without a mentor).")
    if capacity_block:
        return "All shared slots were full."
    if mentor_block:
        return "Shared slots would break mentor coverage (mentee present without a mentor)."
    return "Grouping would worsen the objective under current constraints."

def run_scheduler(df_raw: pd.DataFrame, vol_lookup: dict):
    """Run the solver using current pairs_text and store results in session_state."""
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
            if a_can and b_can and a_can != b_can:
                valid_pairs_run.append((a_can, b_can))

    # Apply to module (Streamlit and solver share the same module object)
    Scheduler2.GROUP_PAIRS = list(dict.fromkeys(valid_pairs_run))  # dedupe

    # Solve
    try:
        sched_df, _, breakdown_df = build_schedule(df_raw)
    except Exception as e:
        st.error(f"Scheduling failed: {e}")
        return

    # Parse Day/Shift for grid
    if "Time Slot" in sched_df.columns:
        ts = sched_df["Time Slot"].astype(str)
        parts = ts.str.split(r"\s+", n=1, expand=True)
        sched_df["Day"] = parts[0].fillna("").astype(str).str.strip().str.title()
        sched_df["Shift"] = parts[1].fillna("").astype(str).str.replace(r"[–—−]", "-", regex=True).str.strip()
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
            report_rows.append({"Pair": f"{a_can} & {b_can}", "Status": "Grouped ✓", "Details": details})
            continue

        if sa and sb and sa != sb:
            reason = explain_non_grouping(a_can, b_can, sched_df, df_raw)
            report_rows.append({"Pair": f"{a_can} & {b_can}", "Status": "Not grouped ✗", "Details": reason})
            continue

        not_placed = []
        if not sa: not_placed.append(a_can)
        if not sb: not_placed.append(b_can)
        who = f"{not_placed[0]} and {not_placed[1]}" if len(not_placed) == 2 else (not_placed[0] if not_placed else "")
        reason = explain_non_grouping(a_can, b_can, sched_df, df_raw)
        report_rows.append({
            "Pair": f"{a_can} & {b_can}",
            "Status": "Not in schedule",
            "Details": reason if reason else f"We couldn't place {who} given availability, capacity, and mentor/mentee rules."
        })

    st.session_state.group_report = pd.DataFrame(report_rows)

# ── Data hygiene ─────────────────────────────────────────────────────────────
def validate_names(df_raw: pd.DataFrame):
    cols = {c.lower(): c for c in df_raw.columns}
    first = next((cols[k] for k in cols if "first" in k and "name" in k), None)
    last  = next((cols[k] for k in cols if "last"  in k and "name" in k), None)
    if first and last:
        series = (df_raw[first].fillna("").astype(str).str.strip() + " " +
                  df_raw[last].fillna("").astype(str).str.strip())
    elif "name" in cols:
        series = df_raw[cols["name"]].fillna("").astype(str).str.strip()
    else:
        series = (df_raw.iloc[:,0].fillna("").astype(str).str.strip() + " " +
                  df_raw.iloc[:,1].fillna("").astype(str).str.strip())

    key_to_names = {}
    for raw in series.dropna().tolist():
        key = norm_name(raw)
        key_to_names.setdefault(key, set()).add(raw)

    collisions = {k: sorted(v) for k, v in key_to_names.items() if len(v) > 1}
    if collisions:
        st.warning("⚠️ Some names normalize to the same person (possible duplicates): " +
                   "; ".join([", ".join(v) for v in collisions.values()]))

def validate_preference_strings(df_raw: pd.DataFrame):
    pref_cols = [c for c in df_raw.columns if str(c).lower().startswith(("pref","availability"))]
    if not pref_cols:
        return
    values = []
    for c in pref_cols:
        values += [str(v) for v in df_raw[c].dropna().tolist()]
    values = list({re.sub(r"\s+", " ", v).strip() for v in values})
    values = [re.sub(r"[–—−]", "-", v) for v in values]
    pat = re.compile(r"^[A-Za-z]+ \d{1,2}:\d{2}-\d{1,2}:\d{2}$")
    bad = [v for v in values if not pat.match(v)]
    if bad:
        sample = ", ".join(bad[:8]) + ("…" if len(bad) > 8 else "")
        st.warning("⚠️ Some availability strings look off (expect 'Day HH:MM-HH:MM'). "
                   f"Examples: {sample}")

# ── File upload ──────────────────────────────────────────────────────────────
uploader = st.file_uploader("Upload survey XLSX", type="xlsx")

# Reset the app when the file is removed
if not uploader:
    st.session_state.df_raw = None
    st.session_state.sched_df = None
    st.session_state.breakdown_df = None
    st.session_state.group_report = None
    st.session_state.pairs_text = ""
    st.session_state.trigger_run = False

if uploader:
    # Cache raw DataFrame
    if st.session_state.df_raw is None:
        try:
            st.session_state.df_raw = pd.read_excel(uploader)
        except Exception as e:
            st.error(f"Failed to read Excel: {e}")
            st.stop()
    df_raw = st.session_state.df_raw

    # Data hygiene checks
    validate_names(df_raw)
    validate_preference_strings(df_raw)

    # Build volunteer list for grouping UI
    ui_names = extract_names_for_ui(df_raw)
    vol_lookup = {norm_name(n): n for n in ui_names}

    # ── Group-Together Exceptions UI (with clickable typo fixes) ─────────────
    st.subheader("Group-Together Exceptions")
    st.write("Enter each pair on its own line: First Last, First Last")
    st.write("_Leave blank and run if no exceptions._")

    pairs_input = st.text_area(
        "Pairs (one per line)",
        key="pairs_text",
        placeholder="Alice Smith, Bob Jones",
        height=100,
    ).strip()

    # Build suggestions (clickable)
    suggestions: list[tuple[str, str | None]] = []  # (raw_name, suggested or None)
    if pairs_input:
        for line in pairs_input.splitlines():
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) != 2:
                continue
            a_raw, b_raw = parts

            def canon_or_suggest(raw: str) -> tuple[str | None, str | None]:
                canon = vol_lookup.get(norm_name(raw))
                if canon:
                    return canon, None
                keys = list(vol_lookup.keys())
                match = difflib.get_close_matches(norm_name(raw), keys, n=1, cutoff=0.6)
                return None, (vol_lookup[match[0]] if match else None)

            a_can, a_sugg = canon_or_suggest(a_raw)
            b_can, b_sugg = canon_or_suggest(b_raw)
            if not a_can: suggestions.append((a_raw, a_sugg))
            if not b_can: suggestions.append((b_raw, b_sugg))

    def _apply_and_rerun(replacements: list[tuple[str, str]]):
        # de-duplicate replacements for safety
        uniq = []
        seen = set()
        for raw, sugg in replacements:
            key = (raw, sugg)
            if key not in seen and sugg:
                uniq.append((raw, sugg))
                seen.add(key)
        text = st.session_state.get("pairs_text", "") or ""
        st.session_state["pairs_text"] = _replace_tokens(text, uniq)
        st.session_state["trigger_run"] = True
        st.rerun()

    if suggestions:
        st.markdown("**Name fixes:** click to apply")
        cols = st.columns(2)
        fixable = [(raw, sugg) for (raw, sugg) in suggestions if sugg]
        for i, (raw, sugg) in enumerate(suggestions):
            if sugg:
                cols[i % 2].button(
                    f'Replace "{raw}" → "{sugg}"',
                    key=f"fix_{i}",
                    on_click=_apply_and_rerun,
                    args=([(raw, sugg)],),
                )
            else:
                cols[i % 2].markdown(f"⚠️ **{html.escape(raw)}** — no close match found")
        if fixable:
            st.button(
                "Apply all fixes",
                key="apply_all_fixes",
                on_click=_apply_and_rerun,
                args=(fixable,),
            )

    # Manual run button
    if st.button("Run Scheduler"):
        run_scheduler(df_raw, vol_lookup)

    # Auto-run if a fix was applied
    if st.session_state.trigger_run:
        st.session_state.trigger_run = False
        run_scheduler(df_raw, vol_lookup)

# ── Display Results ──────────────────────────────────────────────────────────
if st.session_state.sched_df is not None:
    sched_df = st.session_state.sched_df.copy()
    breakdown_df = st.session_state.breakdown_df
    has_forced = bool(sched_df.get("Fallback", pd.Series([], dtype=bool)).any())
    has_mit = sched_df.get("Role", pd.Series([], dtype=str)).astype(str).str.lower().eq("mit").any()

    # Order days Mon→Sun
    desired_days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    if "Day" not in sched_df.columns:
        sched_df["Day"] = ""
    days = [d for d in desired_days if d in set(sched_df["Day"].dropna().astype(str))]

    # Sort shifts robustly
    if "Shift" not in sched_df.columns:
        sched_df["Shift"] = ""
    raw_shifts = [str(s).strip() for s in sched_df["Shift"].dropna().unique()]
    shifts = sorted(raw_shifts, key=parse_time_to_minutes)

    # Build grid safely
    grid = {sh: {d: [] for d in days} for sh in shifts}
    for _, row in sched_df.iterrows():
        d = str(row.get("Day", "")).strip()
        sh = str(row.get("Shift", "")).strip()
        if sh in grid and d in grid[sh]:
            nm = html.escape(str(row.get("Name", "")))
            role = str(row.get("Role", "")).lower()
            fb = bool(row.get("Fallback", False))
            grid[sh][d].append((nm, role, fb))

    # Grouping results (colorized) before schedule preview
    if st.session_state.group_report is not None and not st.session_state.group_report.empty:
        st.subheader("Group-Together Results")
        styled = style_group_report(st.session_state.group_report)
        st.dataframe(styled, use_container_width=True)

    # ── HTML Grid (light theme) ───────────────────────────────────────────────
    html_grid = "<table style='border-collapse: collapse; width:100%;'>"
    html_grid += "<tr><th style='border:1px solid #ddd; padding:8px;'></th>" + \
                 "".join(f"<th style='border:1px solid #ddd; padding:8px;'>{d}</th>" for d in days) + \
                 "</tr>"
    for sh in shifts:
        for i in range(MAX_PER_SHIFT):
            html_grid += "<tr>"
            if i == 0:
                html_grid += (
                    f"<td rowspan='{MAX_PER_SHIFT}' style='border:1px solid #ddd; "
                    f"padding:8px; vertical-align:middle;'>{sh}</td>"
                )
            for d in days:
                cell = ""
                entries = grid.get(sh, {}).get(d, [])
                if i < len(entries):
                    name, role, fb = entries[i]
                    if role == "mentor":
                        cell = f"<strong>{name}</strong>"
                    elif role == "mentee":
                        cell = ("<span style='background:#ADD8E6; padding:2px 4px; "
                                "border-radius:3px'>" + f"{name}</span>")
                    elif role == "mit":
                        cell = f"<em>{name}</em>"
                    else:
                        cell = name
                    if fb:
                        cell += " *"
                html_grid += (
                    f"<td style='border:1px solid #ddd; padding:8px; vertical-align:top;'>{cell}</td>"
                )
            html_grid += "</tr>"
    html_grid += "</table>"

    st.markdown("### Schedule Preview", unsafe_allow_html=True)
    st.markdown(
        "Mentors are **bold**, mentees highlighted in light blue, * denotes forced assignments, and _Mentor in training_ is italicized.",
        unsafe_allow_html=True,
    )
    st.markdown(html_grid, unsafe_allow_html=True)

    # Legend (conditional MIT/Forced) + Current volunteer
    legend = (
        "<div style='margin: 12px 0;'>"
        "<span style='display:inline-block;margin-right:16px;'><strong>Mentor</strong></span>"
        "<span style='display:inline-block;margin-right:16px;background:#ADD8E6;"
        "padding:2px 6px;border-radius:4px;'>Mentee</span>"
    )
    if has_mit:
        legend += "<span style='display:inline-block;margin-right:16px;'><em>Mentor in training</em></span>"
    if has_forced:
        legend += "<span style='display:inline-block;margin-right:16px;'>* Forced</span>"
    legend += "<span style='display:inline-block;margin-right:16px;'>Current volunteer</span>"
    legend += "</div>"
    st.markdown(legend, unsafe_allow_html=True)

    # Preference Breakdown
    st.subheader("Preference Breakdown")
    st.dataframe(breakdown_df, use_container_width=True)

    # Forced Assignments
    st.subheader("Forced Assignments")
    fb = sched_df[sched_df.get("Fallback", False) == True] if "Fallback" in sched_df.columns else pd.DataFrame()
    if not fb.empty:
        for name in fb["Name"]:
            st.write(f"- {name}")
    else:
        st.write("_None_")

    # ── Excel export (xlsxwriter if available, fallback otherwise) ────────────
    def to_excel_bytes():
        buf = io.BytesIO()
        try:
            import xlsxwriter  # noqa: F401
            engine = "xlsxwriter"
        except Exception:
            engine = "openpyxl"
