# Scheduler2.py
import re
import pandas as pd
from ortools.sat.python import cp_model

# ----------------------------------
# Configuration
# ----------------------------------
MAX_PER_SHIFT = 3
LEX_WEIGHTS = {1: 100000, 2: 10000, 3: 1000, 4: 100, 5: 10}  # Phase-2 scoring
FALLBACK_WEIGHT = 0
GROUP_PAIRS = []  # set by Streamlit: list of (nameA, nameB)

# ----------------------------------
# Helpers
# ----------------------------------
def _normalize_role(s: str) -> str:
    """
    Canonical roles (tolerant):
      - contains 'mentor in training' or standalone 'mit' -> 'mit'
      - contains 'mentor' (and NOT 'in training')        -> 'mentor'
      - contains 'mentee' / 'trainee' / 'new volunteer'  -> 'mentee'
      - otherwise                                         -> 'volunteer'
    """
    r = re.sub(r"\s+", " ", str(s).strip().lower())
    r = r.replace("-", " ")
    if "mentor in training" in r or re.search(r"\bmit\b", r):
        return "mit"
    if "mentor" in r and "in training" not in r:
        return "mentor"
    if any(k in r for k in ["mentee", "trainee", "new volunteer", "newvolunteer"]):
        return "mentee"
    return "volunteer"

DAY_MAP = {
    "mon": "Monday", "monday": "Monday",
    "tue": "Tuesday", "tues": "Tuesday", "tuesday": "Tuesday",
    "wed": "Wednesday", "weds": "Wednesday", "wednesday": "Wednesday",
    "thu": "Thursday", "thur": "Thursday", "thurs": "Thursday", "thursday": "Thursday",
    "fri": "Friday", "friday": "Friday",
    "sat": "Saturday", "saturday": "Saturday",
    "sun": "Sunday", "sunday": "Sunday",
}

def _fmt_24h(minutes: int) -> str:
    h = (minutes // 60) % 24
    m = minutes % 60
    return f"{h:02d}:{m:02d}"

def _parse_time_component(t: str):
    """
    Parse a single time like '10', '10:30', '2pm', '2:30 PM', '14:00'.
    Returns (minutes, had_meridian, was_12hr_candidate)
    """
    raw = str(t)
    s = re.sub(r"\s+", "", raw.lower()).replace(".", "")
    m = re.match(r"^(\d{1,2})(?::?(\d{2}))?(am|pm)?$", s)
    if not m:
        return float("nan"), False, False
    hh = int(m.group(1))
    mm = int(m.group(2) or 0)
    mer = m.group(3)
    if mer:
        hh = hh % 12
        if mer == "pm":
            hh += 12
        return hh * 60 + mm, True, True
    # No meridian → treat as 24h if valid; also track if it *could* be 12h
    could_be_12h = 1 <= hh <= 12
    if 0 <= hh <= 23 and 0 <= mm <= 59:
        return hh * 60 + mm, False, could_be_12h
    return float("nan"), False, could_be_12h

def _parse_time_range_to_24h(range_text: str) -> str | None:
    """
    Parse messy ranges like:
      '10-2pm', '10am-2', '10:00-14:00', '10am - 2:30pm', '10.00AM-2.00 PM'
    → 'HH:MM-HH:MM' (24h).
    Inference rules:
      • If exactly one side has am/pm, infer the *other* side’s meridian to match.
      • If both lack am/pm, assume 24h.
      • No overnight handling (if start >= end we keep it; that’s the data).
    """
    txt = str(range_text or "")
    txt = txt.replace("–", "-").replace("—", "-").replace("−", "-")
    txt = re.sub(r"\bto\b", "-", txt, flags=re.IGNORECASE)
    parts = [p for p in txt.split("-") if p.strip()]
    if len(parts) != 2:
        return None

    left_raw, right_raw = parts[0].strip(), parts[1].strip()

    L, L_has, L_cand12 = _parse_time_component(left_raw)
    R, R_has, R_cand12 = _parse_time_component(right_raw)

    if L != L or R != R:  # NaN
        return None

    # If right has meridian but left doesn't → try to align left to same half-day if it could be 12h
    if R_has and not L_has and L_cand12:
        mer = re.search(r"(am|pm)", right_raw.lower())
        if mer:
            mer = mer.group(1)
            m = re.match(r"^\s*(\d{1,2})(?::?(\d{2}))?\s*$", left_raw)
            if m:
                hh = int(m.group(1)); mm = int(m.group(2) or 0)
                hh = hh % 12
                if mer == "pm":
                    hh += 12
                L = hh * 60 + mm

    # If left has meridian but right doesn't → align right if it could be 12h
    if L_has and not R_has and R_cand12:
        mer = re.search(r"(am|pm)", left_raw.lower())
        if mer:
            mer = mer.group(1)
            m = re.match(r"^\s*(\d{1,2})(?::?(\d{2}))?\s*$", right_raw)
            if m:
                hh = int(m.group(1)); mm = int(m.group(2) or 0)
                hh = hh % 12
                if mer == "pm":
                    hh += 12
                R = hh * 60 + mm

    return f"{_fmt_24h(int(L))}-{_fmt_24h(int(R))}"

def _norm_slot(s: str) -> str:
    """
    Normalize 'Day time-range' to 'Day HH:MM-HH:MM' (24h).
    """
    s = str(s or "").strip()
    if not s:
        return ""
    s = re.sub(r"[–—−]", "-", s)          # normalize em/en dashes
    s = re.sub(r"\s+", " ", s).strip()    # collapse spaces
    parts = s.split(" ", 1)
    if len(parts) < 2:
        return s
    day_raw, rest = parts[0], parts[1]
    day = DAY_MAP.get(day_raw.lower(), day_raw.title())
    rest = rest.strip()
    rng = _parse_time_range_to_24h(rest)
    if not rng:
        rest = rest.replace(" ", "")
        rest = re.sub(r"-+", "-", rest)
        rng = rest
    return f"{day} {rng}"

def _clean_pairs(volunteers, pairs):
    vset = set(volunteers)
    out, seen = [], set()
    for a, b in (pairs or []):
        if not a or not b or a == b:
            continue
        if a in vset and b in vset:
            key = tuple(sorted((a, b)))
            if key not in seen:
                seen.add(key)
                out.append((a, b))
    return out

def _find_col(cols_lower: dict, *candidates: str):
    for cand in candidates:
        for k, real in cols_lower.items():
            if cand in k:
                return real
    return None

def _rank_from_header(colname: str):
    m = re.search(r"\b(1st|2nd|3rd|4th|5th|[1-5])\b", colname.lower())
    if not m:
        return None
    token = m.group(1)
    if token.isdigit():
        return int(token)
    return {"1st":1, "2nd":2, "3rd":3, "4th":4, "5th":5}.get(token)

# ----------------------------------
# Load and parse preferences
# ----------------------------------
def load_preferences(df: pd.DataFrame):
    if df is None or df.empty:
        raise ValueError("The uploaded sheet is empty.")

    df = df.copy()
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    # Names
    first_col = _find_col(cols_lower, "first name", "firstname", "first")
    last_col  = _find_col(cols_lower, "last name", "lastname", "last", "surname")
    name_col  = _find_col(cols_lower, "name")
    if first_col and last_col:
        df["Name"] = df[first_col].fillna("").astype(str).str.strip() + " " + df[last_col].fillna("").astype(str).str.strip()
    elif name_col:
        df["Name"] = df[name_col].fillna("").astype(str).str.strip()
    else:
        if df.shape[1] >= 2:
            df["Name"] = df.iloc[:, 0].fillna("").astype(str).str.strip() + " " + df.iloc[:, 1].fillna("").astype(str).str.strip()
        else:
            raise ValueError("Could not find name columns (need 'First/Last name' or 'Name').")

    # Role (lenient)
    role_col = (_find_col(cols_lower, "role") or
                _find_col(cols_lower, "position") or
                _find_col(cols_lower, "type") or
                _find_col(cols_lower, "volunteer type"))

    volunteers = sorted([str(x).strip() for x in df["Name"].tolist() if str(x).strip()])
    if not volunteers:
        raise ValueError("No volunteers found after parsing names.")

    # Pref columns
    pref_cols = []
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ("choice", "availability", "avail", "pref", "preference")) or _rank_from_header(cl):
            pref_cols.append(c)
    if not pref_cols:
        raise ValueError("No preference/availability columns detected.")

    ranked, unranked = [], []
    for c in pref_cols:
        r = _rank_from_header(str(c))
        (ranked if r else unranked).append((r, c))
    ranked.sort(key=lambda t: t[0] or 999)
    ordered_pref_cols = [c for _, c in ranked] + [c for _, c in unranked]

    roles, prefs_map = {}, {}
    for _, row in df.iterrows():
        name = str(row["Name"]).strip()
        if not name:
            continue
        raw_role = row.get(role_col, "") if role_col else ""
        roles[name] = _normalize_role(raw_role)

        prefs = []
        for c in ordered_pref_cols:
            val = row.get(c, None)
            if pd.notna(val):
                ns = _norm_slot(val)  # normalized to Day HH:MM-HH:MM (24h) if possible
                if ns:
                    prefs.append(ns)
        prefs_map[name] = prefs

    shifts_set = {slot for prefs in prefs_map.values() for slot in prefs if slot}
    if not shifts_set:
        raise ValueError("No valid shift strings found in preferences.")
    shifts = sorted(shifts_set)

    weights = {}
    for name, prefs in prefs_map.items():
        for rank, slot in enumerate(prefs, start=1):
            weights[(name, slot)] = LEX_WEIGHTS.get(rank, 0)
    for name in volunteers:
        for slot in shifts:
            weights.setdefault((name, slot), FALLBACK_WEIGHT)

    return volunteers, roles, shifts, weights, prefs_map

# ----------------------------------
# Two-phase lexicographic solve
# ----------------------------------
def solve_schedule(volunteers, roles, shifts, weights):
    volunteers = sorted(volunteers)
    shifts = sorted(shifts)

    mentees = [v for v in volunteers if roles.get(v) == "mentee"]
    mentors = [v for v in volunteers if roles.get(v) == "mentor"]

    clean_pairs = _clean_pairs(volunteers, GROUP_PAIRS)

    # Phase 1
    model1 = cp_model.CpModel()
    x1 = {(v, s): model1.NewBoolVar(f"x1_{v}_{s}") for v in volunteers for s in shifts}

    for v in volunteers:
        model1.Add(sum(x1[(v, s)] for s in shifts) == 1)
    for s in shifts:
        model1.Add(sum(x1[(v, s)] for v in volunteers) <= MAX_PER_SHIFT)
    for a, b in clean_pairs:
        for s in shifts:
            model1.Add(x1[(a, s)] == x1[(b, s)])

    if mentees:
        z1 = {s: model1.NewBoolVar(f"z1_mentor_present_{s}") for s in shifts}
        for s in shifts:
            model1.Add(sum(x1[(v, s)] for v in mentees) <= MAX_PER_SHIFT * z1[s])
            if mentors:
                model1.Add(sum(x1[(v, s)] for v in mentors) >= z1[s])
            else:
                model1.Add(z1[s] == 0)

    model1.Maximize(
        sum((1 if weights[(v, s)] != FALLBACK_WEIGHT else 0) * x1[(v, s)]
            for v in volunteers for s in shifts)
    )

    solver1 = cp_model.CpSolver()
    solver1.parameters.random_seed = 42
    solver1.parameters.num_search_workers = 1
    solver1.parameters.max_time_in_seconds = 30
    status1 = solver1.Solve(model1)
    if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule.")
    best_nonfb = int(solver1.ObjectiveValue())

    # Phase 2
    model2 = cp_model.CpModel()
    x2 = {(v, s): model2.NewBoolVar(f"x2_{v}_{s}") for v in volunteers for s in shifts}

    for v in volunteers:
        model2.Add(sum(x2[(v, s)] for s in shifts) == 1)
    for s in shifts:
        model2.Add(sum(x2[(v, s)] for v in volunteers) <= MAX_PER_SHIFT)
    for a, b in clean_pairs:
        for s in shifts:
            model2.Add(x2[(a, s)] == x2[(b, s)])

    if mentees:
        z2 = {s: model2.NewBoolVar(f"z2_mentor_present_{s}") for s in shifts}
        for s in shifts:
            model2.Add(sum(x2[(v, s)] for v in mentees) <= MAX_PER_SHIFT * z2[s])
            if mentors:
                model2.Add(sum(x2[(v, s)] for v in mentors) >= z2[s])
            else:
                model2.Add(z2[s] == 0)

    model2.Add(
        sum((1 if weights[(v, s)] != FALLBACK_WEIGHT else 0) * x2[(v, s)]
            for v in volunteers for s in shifts) == best_nonfb
    )

    model2.Maximize(sum(weights[(v, s)] * x2[(v, s)] for v in volunteers for s in shifts))

    solver2 = cp_model.CpSolver()
    solver2.parameters.random_seed = 42
    solver2.parameters.num_search_workers = 1
    solver2.parameters.max_time_in_seconds = 30
    status2 = solver2.Solve(model2)
    if status2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule in phase 2.")

    schedule = {s: [] for s in shifts}
    assigned = set()
    for (v, s), var in x2.items():
        if solver2.Value(var):
            schedule[s].append({
                "Name": v,
                "Role": roles.get(v, "volunteer"),
                "Fallback": (weights[(v, s)] == FALLBACK_WEIGHT),
            })
            assigned.add(v)
    return schedule, assigned

# ----------------------------------
# Relaxed solve
# ----------------------------------
def solve_relaxed(volunteers, roles, shifts, weights):
    volunteers = sorted(volunteers)
    shifts = sorted(shifts)
    model = cp_model.CpModel()
    x = {(v, s): model.NewBoolVar(f"x_{v}_{s}") for v in volunteers for s in shifts}

    for v in volunteers:
        model.Add(sum(x[(v, s)] for s in shifts) <= 1)
    for s in shifts:
        model.Add(sum(x[(v, s)] for v in volunteers) <= MAX_PER_SHIFT)

    clean_pairs = _clean_pairs(volunteers, GROUP_PAIRS)
    for a, b in clean_pairs:
        for s in shifts:
            model.Add(x[(a, s)] == x[(b, s)])

    mentees = [v for v in volunteers if roles.get(v) == "mentee"]
    mentors = [v for v in volunteers if roles.get(v) == "mentor"]

    if mentees:
        z = {s: model.NewBoolVar(f"z_relax_mentor_present_{s}") for s in shifts}
        for s in shifts:
            model.Add(sum(x[(v, s)] for v in mentees) <= MAX_PER_SHIFT * z[s])
            if mentors:
                model.Add(sum(x[(v, s)] for v in mentors) >= z[s])
            else:
                model.Add(z[s] == 0)

    model.Maximize(sum(x[(v, s)] for v in volunteers for s in shifts))

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 42
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 30
    solver.Solve(model)

    schedule = {s: [] for s in shifts}
    assigned = set()
    for (v, s), var in x.items():
        if solver.Value(var):
            schedule[s].append({
                "Name": v,
                "Role": roles.get(v, "volunteer"),
                "Fallback": (weights[(v, s)] == FALLBACK_WEIGHT),
            })
            assigned.add(v)
    return schedule, assigned

# ----------------------------------
# DataFrame builders / summaries
# ----------------------------------
def prepare_schedule_df(schedule: dict) -> pd.DataFrame:
    rows = []
    for slot, items in schedule.items():
        for a in items:
            rows.append({
                "Time Slot": slot,  # already 'Day HH:MM-HH:MM' (24h)
                "Name": a.get("Name", ""),
                "Role": a.get("Role", "volunteer"),
                "Fallback": bool(a.get("Fallback", False)),
            })
    return pd.DataFrame(rows)

def compute_breakdown(schedule: dict, prefs_map: dict[str, list[str]]) -> pd.DataFrame:
    total = sum(len(items) for items in schedule.values())
    ord_names = ["1st", "2nd", "3rd", "4th", "5th"]
    counts = {k: 0 for k in ord_names}
    counts["Fallback"] = 0
    for slot, items in schedule.items():
        for a in items:
            name = a.get("Name", "")
            prefs = prefs_map.get(name, [])
            if slot in prefs:
                idx = prefs.index(slot)
                if 0 <= idx < len(ord_names):
                    counts[ord_names[idx]] += 1
                else:
                    counts["Fallback"] += 1
            else:
                counts["Fallback"] += 1
    rows = []
    for k in ord_names + ["Fallback"]:
        v = counts[k]
        pct = (v / total * 100) if total else 0.0
        rows.append({"Preference": k, "Count": v, "Percentage": f"{pct:.1f}%"})
    return pd.DataFrame(rows)

# ----------------------------------
# Entrypoint
# ----------------------------------
def build_schedule(df: pd.DataFrame):
    """
    Returns:
      sched_df:      [Time Slot, Name, Role, Fallback]
      unassigned_df: [Name]
      breakdown_df:  [Preference, Count, Percentage]
    """
    volunteers, roles, shifts, weights, prefs_map = load_preferences(df)
    try:
        schedule, assigned = solve_schedule(volunteers, roles, shifts, weights)
    except RuntimeError:
        schedule, assigned = solve_relaxed(volunteers, roles, shifts, weights)
    sched_df = prepare_schedule_df(schedule)
    unassigned = [v for v in volunteers if v not in assigned]
    unassigned_df = pd.DataFrame({"Name": unassigned})
    breakdown_df = compute_breakdown(schedule, prefs_map)
    return sched_df, unassigned_df, breakdown_df
