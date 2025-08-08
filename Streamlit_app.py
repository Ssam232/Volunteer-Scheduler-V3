import streamlit as st
import pandas as pd
import io, re, difflib
import Scheduler2  # scheduling core module
from Scheduler2 import build_schedule, MAX_PER_SHIFT

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(page_title="PEMRAP Volunteer Scheduler V3", layout="wide")
st.title("📅 PEMRAP Volunteer Scheduler V3")

# ── Initialize session state ─────────────────────────────────────────────────
for key in ("df_raw", "sched_df", "breakdown_df", "group_report", "pairs_text"):
    if key not in st.session_state:
        st.session_state[key] = None if key != "pairs_text" else ""

# ── Helpers ─────────────────────────────────────────────────────────────────
def extract_names_for_ui(df: pd.DataFrame) -> list[str]:
    """Build a clean Name list for the Group UI from the raw sheet.
    Tries First/Last -> Name -> first two columns. Case/space insensitive.
    """
    cols = {c.lower(): c for c in df.columns}
    first = next((cols[k] for k in cols if 'first' in k and 'name' in k), None)
    last  = next((cols[k] for k in cols if 'last'  in k and 'name' in k), None)
    if first and last:
        names = (df[first].astype(str).str.strip() + ' ' + df[last].astype(str).str.strip()).tolist()
    elif 'name' in cols:
        names = df[cols['name']].astype(str).tolist()
    else:
        names = (df.iloc[:,0].astype(str).str.strip() + ' ' + df.iloc[:,1].astype(str).str.strip()).tolist()
    names = {re.sub(r'\s+', ' ', n).strip() for n in names if str(n).strip()}
    return sorted(names)

def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).lower()

# ── File upload ──────────────────────────────────────────────────────────────
uploader = st.file_uploader("Upload survey XLSX", type="xlsx")
# Reset the app when the file is removed
if not uploader:
    st.session_state.df_raw = None
    st.session_state.sched_df = None
    st.session_state.breakdown_df = None
    st.session_state.group_report = None

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

    valid_pairs: list[tuple[str, str]] = []
    raw_pairs: list[tuple[str, str]] = []
    suggestions: list[tuple[str, str | None]] = []  # (raw_name, suggested or None)

    if pairs_input:
        for line in pairs_input.splitlines():
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) != 2:
                continue
            a_raw, b_raw = parts
            raw_pairs.append((a_raw, b_raw))

            def canon_or_suggest(raw: str) -> tuple[str | None, str | None]:
                canon = vol_lookup.get(norm_name(raw))
                if canon:
                    return canon, None
                keys = list(vol_lookup.keys())
                match = difflib.get_close_matches(norm_name(raw), keys, n=1, cutoff=0.6)
                return None, (vol_lookup[match[0]] if match else None)

            a_can, a_sugg = canon_or_suggest(a_raw)
            b_can, b_sugg = canon_or_suggest(b_raw)

            if a_can and b_can:
                valid_pairs.append((a_can, b_can))
            else:
                if not a_can:
                    suggestions.append((a_raw, a_sugg))
                if not b_can:
                    suggestions.append((b_raw, b_sugg))

    # Render clickable suggestions (non-blocking) using callbacks
    def _replace_tokens(text: str, replacements: list[tuple[str, str]]) -> str:
        # Parse pairs text line-by-line, replace exact tokens between commas
        lines = (text or "").splitlines()
        new_lines = []
        rep_map = {raw: sugg for raw, sugg in replacements if sugg}
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            new_parts = [rep_map.get(p, p) for p in parts if p]
            new_lines.append(', '.join(new_parts))
        return '
'.join(new_lines)

    def _replace_one_cb(raw: str, sugg: str):
        text = st.session_state.get("pairs_text", "") or ""
        st.session_state["pairs_text"] = _replace_tokens(text, [(raw, sugg)])
        st.rerun()

    def _replace_all_cb(suggs: list[tuple[str, str | None]]):
        text = st.session_state.get("pairs_text", "") or ""
        repl = [(raw, sugg) for raw, sugg in suggs if sugg]
        st.session_state["pairs_text"] = _replace_tokens(text, repl)
        st.rerun()

    if suggestions:
        st.markdown("**Name fixes:** click to apply")
        cols = st.columns(2)
        any_fixable = any(s for _, s in suggestions)
        for i, (raw, sugg) in enumerate(suggestions):
            if sugg:
                cols[i % 2].button(
                    f"Replace “{raw}” → “{sugg}”",
                    key=f"fix_{i}",
                    on_click=_replace_one_cb,
                    args=(raw, sugg),
                )
            else:
                cols[i % 2].markdown(f"⚠️ **{raw}** — no close match found")
        if any_fixable:
            st.button(
                "Apply all fixes",
                key="apply_all_fixes",
                on_click=_replace_all_cb,
                args=(suggestions,),
            )

    # Apply grouping rules (only validated canonical pairs so far)
    Scheduler2.GROUP_PAIRS = valid_pairs

    Scheduler2.GROUP_PAIRS = valid_pairs

    # ── Run Scheduler ─────────────────────────────────────────────────────────
    if st.button("Run Scheduler"):
        # Re-parse pairs at run time to capture any last edits
        pairs_text = (st.session_state.get("pairs_text") or "").strip()
        valid_pairs_run: list[tuple[str, str]] = []
        raw_pairs_run: list[tuple[str, str]] = []
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
        Scheduler2.GROUP_PAIRS = valid_pairs_run

        # Solve
        sched_df, _, breakdown_df = build_schedule(df_raw)

        # Parse Day/Shift for grid
        if "Time Slot" in sched_df.columns:
            ts = sched_df["Time Slot"].astype(str)
            parts = ts.str.split(r"\s+", n=1, expand=True)
            sched_df["Day"] = parts[0].str.strip().str.title().fillna("")
            sched_df["Shift"] = (
                parts[1].str.replace(r"[–—−]", "-", regex=True).str.strip().fillna("")
            )
        else:
            sched_df["Day"], sched_df["Shift"] = "", ""

        # Save to session state
        st.session_state.sched_df = sched_df
        st.session_state.breakdown_df = breakdown_df

        # Build a grouping-results report for the pairs the user typed
        slot_by_name = (
            sched_df[["Name", "Time Slot"]]
            .dropna()
            .groupby("Name")["Time Slot"].first()
            .to_dict()
        )
        report_rows = []
        for a_raw, b_raw in raw_pairs_run:
            a_can = vol_lookup.get(norm_name(a_raw))
            b_can = vol_lookup.get(norm_name(b_raw))
            if not a_can or not b_can:
                missing = []
                if not a_can:
                    missing.append(a_raw)
                if not b_can:
                    missing.append(b_raw)
                report_rows.append({
                    "Pair": f"{a_raw} & {b_raw}",
                    "Status": "Skipped",
                    "Reason": f"Name not recognized: {', '.join(missing)}"
                })
                continue
            sa = slot_by_name.get(a_can)
            sb = slot_by_name.get(b_can)
            if sa and sb:
                if sa == sb:
                    report_rows.append({
                        "Pair": f"{a_can} & {b_can}",
                        "Status": "Grouped ✓",
                        "Reason": f"Assigned to {sa}"
                    })
                else:
                    report_rows.append({
                        "Pair": f"{a_can} & {b_can}",
                        "Status": "Not grouped ✗",
                        "Reason": f"{a_can} at {sa}, {b_can} at {sb} (check capacity/coverage)"
                    })
            else:
                missing = []
                if not sa:
                    missing.append(a_can)
                if not sb:
                    missing.append(b_can)
                report_rows.append({
                    "Pair": f"{a_can} & {b_can}",
                    "Status": "Not in schedule",
                    "Reason": f"{', '.join(missing)} not placed under constraints"
                })
        st.session_state.group_report = pd.DataFrame(report_rows)

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

    # Grouping results table (now shown BEFORE the schedule preview)
    if st.session_state.group_report is not None and not st.session_state.group_report.empty:
        st.subheader("Group-Together Results")
        st.dataframe(st.session_state.group_report, use_container_width=True)

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
                    if role == 'mentor':
                        cell = f"<strong>{name}</strong>"
                    elif role == 'mentee':
                        cell = (
                            "<span style='background:#add8e6;padding:2px 4px;border-radius:3px'>"
                            f"{name}</span>"
                        )
                    else:
                        cell = name
                    if fb:
                        cell += ' *'
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

    # Grouping results table
    if st.session_state.group_report is not None and not st.session_state.group_report.empty:
        st.subheader("Group-Together Results")
        st.dataframe(st.session_state.group_report, use_container_width=True)

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
                            fmt = mentor_fmt if rl == 'mentor' else mentee_fmt if rl == 'mentee' else vol_fmt
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

# Note: Clear the upload to reset the app
