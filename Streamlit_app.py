import streamlit as st
import pandas as pd
import io, re, difflib
import Scheduler2  # scheduling core module
from Scheduler2 import build_schedule, MAX_PER_SHIFT

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
        names = (df[first].astype(str).str.strip() + " " + df[last].astype(str).str.strip()).tolist()
    elif "name" in cols:
        names = df[cols["name"]].astype(str).tolist()
    else:
        names = (df.iloc[:, 0].astype(str).str.strip() + " " + df.iloc[:, 1].astype(str).str.strip()).tolist()
    names = {re.sub(r"\s+", " ", n).strip() for n in names if str(n).strip()}
    return sorted(names)

def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).lower()

def style_group_report(df: pd.DataFrame):
    """Darker row colors for dark backgrounds."""
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

    # Build group report (with forced indication)
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
            report_rows.append({
                "Pair": f"{a_can} & {b_can}",
                "Status": "Not grouped ✗",
                "Details": (
                    f"Both were scheduled but on different shifts — **{a_can} → {sa}**, "
                    f"**{b_can} → {sb}**. Try freeing space or adjusting their preferences."
                )
            })
            continue

        not_placed = []
        if not sa: not_placed.append(a_can)
        if not sb: not_placed.append(b_can)
        who = f"{not_placed[0]} and {not_placed[1]}" if len(not_placed) == 2 else not_placed[0]
        report_rows.append({
            "Pair": f"{a_can} & {b_can}",
            "Status": "Not in schedule",
            "Details": f"We couldn't place {who} given availability, capacity, and mentor/mentee rules."
        })

    st.session_state.group_report = pd.DataFrame(report_rows)

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

    # Build volunteer list for grouping UI (robust to column variations)
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

    # Clickable suggestions — and auto-run after each fix
    def _apply_and_rerun(replacements: list[tuple[str, str]]):
        text = st.session_state.get("pairs_text", "") or ""
        st.session_state["pairs_text"] = _replace_tokens(text, replacements)
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
                cols[i % 2].markdown(f"⚠️ **{raw}** — no close match found")
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
    sched_df = st.session_state.sched_df
    breakdown_df = st.session_state.breakdown_df

    # Order days Mon→Sun
    desired_days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    days = [d for d in desired_days if d in sched_df["Day"].unique()]

    # Sort shifts chronologically (robust)
    raw_shifts = [str(s).strip() for s in sched_df["Shift"].dropna().unique()]
    def shift_key(s):
        try:
            start = s.split("-")[0].replace(":", "").replace(" ", "")
            return int(start)
        except Exception:
            return float("inf")
    shifts = sorted(raw_shifts, key=shift_key)

    # Build grid safely
    grid = {sh: {d: [] for d in days} for sh in shifts}
    for _, row in sched_df.iterrows():
        d, sh = str(row.get("Day", "")).strip(), str(row.get("Shift", "")).strip()
        if sh in grid and d in grid[sh]:
            grid[sh][d].append((row["Name"], row.get("Role", ""), bool(row["Fallback"])))

    # Grouping results (colorized) before schedule preview
    if st.session_state.group_report is not None and not st.session_state.group_report.empty:
        st.subheader("Group-Together Results")
        styled = style_group_report(st.session_state.group_report)
        st.table(styled)

    # ── HTML Grid ──────────────────────────────────────────────────────────────
    html = "<table style='border-collapse: collapse; width:100%;'>"
    html += "<tr><th></th>" + "".join(f"<th>{d}</th>" for d in days) + "</tr>"
    for sh in shifts:
        for i in range(MAX_PER_SHIFT):
            html += "<tr>"
            if i == 0:
                html += (
                    f"<td rowspan='{MAX_PER_SHIFT}' style='vertical-align:middle;border:1px solid #ddd;padding:8px;'>{sh}</td>"
                )
            for d in days:
                cell = ""
                entries = grid.get(sh, {}).get(d, [])
                if i < len(entries):
                    name, role, fb = entries[i]
                    if role == "mentor":
                        cell = f"<strong>{name}</strong>"
                    elif role == "mentee":
                        cell = (
                            "<span style='background:#add8e6;padding:2px 4px;border-radius:3px'>"
                            f"{name}</span>"
                        )
                    else:
                        cell = name
                    if fb:
                        cell += " *"
                html += (
                    f"<td style='border:1px solid #ddd;padding:8px;vertical-align:top;'>{cell}</td>"
                )
            html += "</tr>"
    html += "</table>"

    st.markdown("### Schedule Preview", unsafe_allow_html=True)
    st.markdown(
        "Mentors are **bold**, mentees highlighted in light blue, and * denotes forced assignments.",
        unsafe_allow_html=True,
    )
    st.markdown(html, unsafe_allow_html=True)

    # Preference Breakdown
    st.subheader("Preference Breakdown")
    st.dataframe(breakdown_df, use_container_width=True)

    # Forced Assignments
    st.subheader("Forced Assignments")
    fb = sched_df[sched_df["Fallback"]]
    if not fb.empty:
        for name in fb["Name"]:
            st.write(f"- {name}")
    else:
        st.write("_None_")

    # ── Excel export with formatted grid ───────────────────────────────────────
    def to_excel_bytes():
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            wb = writer.book
            # Formats
            border = wb.add_format({"border": 1})
            mentor_fmt = wb.add_format({"border": 1, "bold": True})
            mentee_fmt = wb.add_format({"border": 1, "bg_color": "#ADD8E6"})
            vol_fmt = wb.add_format({"border": 1})

            # Local day/shift lists derived from the current schedule
            days_local = [d for d in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"] if d in sched_df["Day"].unique()]
            raw_shifts_local = [str(s).strip() for s in sched_df["Shift"].dropna().unique()]
            def _key(s):
                try:
                    return int(s.split("-")[0].replace(":", "").replace(" ", ""))
                except Exception:
                    return float("inf")
            shifts_local = sorted(raw_shifts_local, key=_key)

            # Build grid dict safely for export
            grid_x = {sh: {d: [] for d in days_local} for sh in shifts_local}
            for _, r in sched_df.iterrows():
                d = str(r.get("Day", "")).strip()
                sh = str(r.get("Shift", "")).strip()
                if sh in grid_x and d in grid_x[sh]:
                    grid_x[sh][d].append((r["Name"], r.get("Role", ""), bool(r["Fallback"])))

            # Grid sheet
            ws = wb.add_worksheet("Grid")
            ws.write_blank(0, 0, None, border)
            for c, day in enumerate(days_local, start=1):
                ws.write(0, c, day, border)
            row_idx = 1
            for sh in shifts_local:
                ws.merge_range(row_idx, 0, row_idx + MAX_PER_SHIFT - 1, 0, sh, border)
                for i in range(MAX_PER_SHIFT):
                    for c, day in enumerate(days_local, start=1):
                        ppl = grid_x.get(sh, {}).get(day, [])
                        if i < len(ppl):
                            nm, rl, fb_flag = ppl[i]
                            fmt = mentor_fmt if rl == "mentor" else mentee_fmt if rl == "mentee" else vol_fmt
                            ws.write(row_idx + i, c, nm + (" *" if fb_flag else ""), fmt)
                        else:
                            ws.write_blank(row_idx + i, c, None, border)
                    ws.set_row(row_idx + i, 30)
                row_idx += MAX_PER_SHIFT
            ws.set_column(0, 0, 16)
            ws.set_column(1, len(days_local), 22)

            # Preferences & Fallback sheets
            breakdown_df.to_excel(writer, sheet_name="Preferences", index=False)
            fb_df = sched_df[sched_df["Fallback"]][["Time Slot","Name","Role"]]
            fb_df.to_excel(writer, sheet_name="Fallback", index=False)
        return buf.getvalue()

    st.download_button(
        "⬇️ Download schedule.xlsx",
        data=to_excel_bytes(),
        file_name="schedule.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
