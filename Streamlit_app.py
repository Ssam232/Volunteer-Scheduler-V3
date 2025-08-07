import streamlit as st
import pandas as pd
import io, re, difflib
import Scheduler2  # scheduling core module
from Scheduler2 import build_schedule, MAX_PER_SHIFT, load_preferences

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(page_title="PEMRAP Volunteer Scheduler V3", layout="wide")
st.title("📅 PEMRAP Volunteer Scheduler V3")

# ── Initialize session state ─────────────────────────────────────────────────
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "sched_df" not in st.session_state:
    st.session_state.sched_df = None
    st.session_state.breakdown_df = None

# ── File upload ──────────────────────────────────────────────────────────────
uploader = st.file_uploader("Upload survey XLSX", type="xlsx")
# Reset the app when the file is removed
if not uploader:
    st.session_state.df_raw = None
    st.session_state.sched_df = None
    st.session_state.breakdown_df = None

if uploader:
    # Cache raw DataFrame
    if st.session_state.df_raw is None:
        try:
            st.session_state.df_raw = pd.read_excel(uploader)
        except Exception as e:
            st.error(f"Failed to read Excel: {e}")
            st.stop()
    df_raw = st.session_state.df_raw

    # Extract volunteers for grouping
    try:
        volunteers, _, _, _, _ = load_preferences(df_raw)
    except Exception as e:
        st.error(f"Error parsing preferences: {e}")
        volunteers = []

    # ── Group-Together Exceptions UI ─────────────────────────────────────────
    st.subheader("Group-Together Exceptions")
    st.write("Enter each pair on its own line: First Last,First Last")

    def _norm_name(s: str) -> str:
        return re.sub(r"\s+", " ", str(s).strip()).lower()

    pairs_input = st.text_area(
        "Pairs (one per line)", placeholder="Alice Smith,Bob Jones", height=100
    ).strip()

    # Build lookup: case/whitespace-insensitive
    vol_lookup = {_norm_name(n): n for n in volunteers}

    valid_pairs = []
    typos = []
    if pairs_input:
        for line in pairs_input.splitlines():
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) != 2:
                continue
            a_raw, b_raw = parts
            a = vol_lookup.get(_norm_name(a_raw))
            b = vol_lookup.get(_norm_name(b_raw))
            if a and b:
                valid_pairs.append((a, b))
            else:
                for raw in (a_raw, b_raw):
                    if vol_lookup.get(_norm_name(raw)) is None:
                        keys = list(vol_lookup.keys())
                        sugg_key = difflib.get_close_matches(_norm_name(raw), keys, n=1, cutoff=0.6)
                        if sugg_key:
                            typos.append(f"{raw} (did you mean: {vol_lookup[sugg_key[0]]})")
                        else:
                            typos.append(raw)

    # Show typos but do NOT block scheduling
    err_box = st.empty()
    if typos:
        err_box.error(f"Typo detected in group-together names: {typos}")
    else:
        err_box.empty()

    # Apply grouping rules
    Scheduler2.GROUP_PAIRS = valid_pairs

    # ── Run Scheduler ─────────────────────────────────────────────────────────
    if st.button("Run Scheduler"):
        sched_df, _, breakdown_df = build_schedule(df_raw)
        # Parse Day/Shift
        if "Time Slot" in sched_df.columns:
            ts = sched_df["Time Slot"].astype(str)
            parts = ts.str.split(r"\s+", n=1, expand=True)
            sched_df["Day"] = parts[0].str.strip().str.title().fillna("")
            sched_df["Shift"] = (
                parts[1].str.replace(r"[–—−]", "-", regex=True).str.strip().fillna("")
            )
        else:
            sched_df["Day"] = ""
            sched_df["Shift"] = ""
        # Save to session state
        st.session_state.sched_df = sched_df
        st.session_state.breakdown_df = breakdown_df

# ── Display Results ──────────────────────────────────────────────────────────
if st.session_state.sched_df is not None:
    sched_df = st.session_state.sched_df
    breakdown_df = st.session_state.breakdown_df

    # Order days Mon→Sun
    desired_days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    days = [d for d in desired_days if d in sched_df["Day"].unique()]

    # Sort shifts chronologically, robust to malformed entries
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
    # Italic note below the schedule
    st.markdown("_This note is italicized. Replace with your own text._", unsafe_allow_html=True)

    # ── Preference Breakdown ───────────────────────────────────────────────────
    st.subheader("Preference Breakdown")
    st.dataframe(breakdown_df, use_container_width=True)

    # ── Forced Assignments ────────────────────────────────────────────────────
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

            # Preferences sheet
            breakdown_df.to_excel(writer, sheet_name="Preferences", index=False)
            # Fallback sheet
            fb_df = sched_df[sched_df["Fallback"]][["Time Slot","Name","Role"]]
            fb_df.to_excel(writer, sheet_name="Fallback", index=False)
        return buf.getvalue()

    st.download_button(
        "⬇️ Download schedule.xlsx",
        data=to_excel_bytes(),
        file_name="schedule.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# Note: Refresh (F5) to reset

