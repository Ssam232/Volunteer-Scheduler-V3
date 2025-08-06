import streamlit as st
import pandas as pd
import io
import Scheduler2  # scheduling core module
from Scheduler2 import build_schedule, MAX_PER_SHIFT

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(page_title="PEMRAP Volunteer Scheduler", layout="wide")
st.title("📅 PEMRAP Volunteer Scheduler V3")

# ── Initialize session state ─────────────────────────────────────────────────
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "sched_df" not in st.session_state:
    st.session_state.sched_df = None
    st.session_state.breakdown_df = None

# ── File upload ──────────────────────────────────────────────────────────────
uploader = st.file_uploader("Upload Survey Results .xlsx", type="xlsx")
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
        volunteers, _, _, _, _ = Scheduler2.load_preferences(df_raw)
    except Exception as e:
        st.error(f"Error parsing preferences: {e}")
        volunteers = []

    # ── Group-Together Exceptions UI ─────────────────────────────────────────
    st.subheader("Group-Together Exceptions")
    st.write("Enter each pair on its own line: First Last, First Last")
    st.markdown("_Leave blank and run if no exceptions._")
    pairs_input = st.text_area(
        "Pairs (one per line)", placeholder="Alice Smith,Bob Jones", height=100
    ).strip()

    # Parse and validate pairs
    pairs = []
    for line in pairs_input.splitlines():
        parts = [n.strip() for n in line.split(",")]
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    error_ph = st.empty()
    invalid = [name for a, b in pairs for name in (a, b) if name not in volunteers]
    if invalid:
        error_ph.error(f"Typo detected in group-together names: {invalid}")
    else:
        error_ph.empty()
    valid_pairs = [(a, b) for a, b in pairs if a in volunteers and b in volunteers]
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

    # Sort shifts chronologically
    raw_shifts = sched_df["Shift"].dropna().unique().tolist()
    def shift_key(s):
        try:
            return int(s.split("-")[0].replace(':', ''))
        except:
            return s
    shifts = sorted(raw_shifts, key=shift_key)

    # Build grid
    grid = {sh: {d: [] for d in days} for sh in shifts}
    for _, row in sched_df.iterrows():
        d, sh = row["Day"], row["Shift"]
        if d and sh and d in grid.get(sh, {}):
            grid[sh][d].append((row["Name"], row.get("Role", ""), row["Fallback"]))

    # ── HTML Grid ──────────────────────────────────────────────────────────────
    html = "<table style='border-collapse: collapse; width:100%;'>"
    html += "<tr><th></th>" + "".join(f"<th>{d}</th>" for d in days) + "</tr>"
    for sh in shifts:
        for i in range(MAX_PER_SHIFT):
            html += "<tr>"
            if i == 0:
                html += f"<td rowspan='{MAX_PER_SHIFT}' style='vertical-align:middle;border:1px solid #ddd;padding:8px;'>{sh}</td>"
            for d in days:
                cell = ""
                entries = grid[sh][d]
                if i < len(entries):
                    name, role, fb = entries[i]
                    if role == 'mentor':
                        cell = f"<strong>{name}</strong>"
                    elif role == 'mentee':
                        cell = f"<span style='background:#add8e6;padding:2px 4px;border-radius:3px'>{name}</span>"
                    else:
                        cell = name
                    if fb:
                        cell += ' *'
                html += f"<td style='border:1px solid #ddd;padding:8px;vertical-align:top;'>{cell}</td>"
            html += "</tr>"
    html += "</table>"

    st.markdown("### Schedule Preview", unsafe_allow_html=True)
    st.markdown(
        "Mentors are **bold**, mentees highlighted in light blue, and * denotes forced assignments.",
        unsafe_allow_html=True
    )
    st.markdown(html, unsafe_allow_html=True)

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

        # ── Excel export ──────────────────────────────────────────────────────────
    def to_excel_bytes():
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            # Grid sheet
            days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            ws = writer.book.add_worksheet("Grid")
            border = writer.book.add_format({"border":1})
            mentor_fmt = writer.book.add_format({"border":1,"bold":True})
            mentee_fmt = writer.book.add_format({"border":1,"bg_color":"#ADD8E6"})
            vol_fmt = writer.book.add_format({"border":1})
            # Write header row
            ws.write_blank(0, 0, None, border)
            for c, day in enumerate(days, start=1):
                ws.write(0, c, day, border)
            row_idx = 1
            # Populate grid rows
            for sh in shifts:
                ws.merge_range(row_idx, 0, row_idx + MAX_PER_SHIFT - 1, 0, sh, border)
                for i in range(MAX_PER_SHIFT):
                    for c, day in enumerate(days, start=1):
                        ppl = grid[sh][day]
                        if i < len(ppl):
                            nm, rl, fb_flag = ppl[i]
                            fmt = mentor_fmt if rl == 'mentor' else mentee_fmt if rl == 'mentee' else vol_fmt
                            ws.write(row_idx + i, c, nm + (" *" if fb_flag else ""), fmt)
                        else:
                            ws.write_blank(row_idx + i, c, None, border)
                    ws.set_row(row_idx + i, 30)
                row_idx += MAX_PER_SHIFT
            ws.set_column(0, 0, 16)
            ws.set_column(1, len(days), 22)

            # Preferences breakdown sheet
            writer.book.add_worksheet("Preferences")
            breakdown_df.to_excel(writer, sheet_name="Preferences", index=False)

            # Fallback sheet
            writer.book.add_worksheet("Fallback")
            fb = sched_df[sched_df["Fallback"]][["Time Slot","Name","Role"]]
            fb.to_excel(writer, sheet_name="Fallback", index=False)
        return buf.getvalue()

    st.download_button(
        "⬇️ Download schedule.xlsx",
        data=to_excel_bytes(),
        file_name="schedule.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Note: Refresh (F5) to reset
