# Scheduler2.py (stable)
import re
import pandas as pd
from ortools.sat.python import cp_model

# ----------------------------------
# Configuration
# ----------------------------------
MAX_PER_SHIFT = 3

# Phase-2 weights (after minimizing fallbacks)
LEX_WEIGHTS = {1: 100_000, 2: 10_000, 3: 1_000, 4: 100, 5: 10}
FALLBACK_WEIGHT = 0

# Streamlit sets this before calling build_schedule(); pairs to keep together
GROUP_PAIRS = []  # list of (nameA, nameB)


# ----------------------------------
# Helpers
# ----------------------------------
def _normalize_role(s):
    """
    Canonicalize free-text roles:
      mentor                               -> 'mentor'
      mentee / trainee / new volunteer     -> 'mentee'
      mentor in training / mit             -> 'mit'   (does NOT count as mentor)
      anything else                        -> 'volunteer'
    """
    r = re.sub(r"\s+", " ", str(s).strip().lower())
    r = r.replace("-", " ")
    if r == "mentor":
        return "mentor"
    if (r in {"mentee", "trainee", "new volunteer", "newvolunteer", "new volunteer (mentee)"} 
        or r.startswith("new volunteer")):
        return "mentee"
    if r in {"mentor in training", "mit", "mentor in training (mit)"}:
        return "mit"
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

def _norm_slot(s):
    """
    Conservative normalization for 'Day HH:MM-HH:MM':
    - normalize unicode dashes to '-'
    - collapse spaces
    - title-case day using a small map (Mon/wed/thurs → Monday/Wednesday/Thursday)
    This avoids over-merging legitimately different strings.
    """
    s = str(s or "").strip()
    if not s:
        return ""
    s = re.sub(r"[–—−]", "-", s)           # normalize dashes
    s = re.sub(r"\s+", " ", s).strip()     # collapse spaces
    parts = s.split(" ", 1)
    if len(parts) < 2:
        return s
    day_raw, rest = parts[0], parts[1]
    day = DAY_MAP.get(day_raw.lower(), day_raw.title())
    rest = rest.replace(" ", "")
    rest = re.sub(r"-+", "-", rest)  # ensure single hyphen
    return f"{day} {rest}"

def _clean_pairs(volunteers, pairs):
    """Remove self/unknown/duplicate pairs; keep input order."""
    vset = set(volunteers)
    out = []
    seen = set()
    for a, b in (pairs or []):
        if not a or not b or a == b:
            continue
        if a in vset and b in vset:
            key = tuple(sorted((a, b)))
            if key not in seen:
                seen.add(key)
                out.append((a, b))
    return out


# ----------------------------------
# Load and parse preferences
# ----------------------------------
def load_preferences(df: pd.DataFrame):
    """
    Returns:
      volunteers: list[str]
      roles: dict[name -> role]
      shifts: list[str]
      weights: dict[(name, shift) -> int]
      prefs_map: dict[name -> list[str]]
    Raises:
      ValueError with a friendly message if required data is missing.
    """
    if df is None or df.empty:
        raise ValueError("The uploaded sheet is empty.")

    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    # Build Name column robustly
    first_cols = [cols_lower[k] for k in cols_lower if "first" in k and "name" in k]
    last_cols  = [cols_lower[k] for k in cols_lower if "last"  in k and "name" in k]
    if first_cols and last_cols:
        df["Name"] = (
            df[first_cols[0]].fillna("").astype(str).str.strip()
            + " "
            + df[last_cols[0]].fillna("").astype(str).str.strip()
        )
    elif "name" in cols_lower:
        df["Name"] = df[cols_lower["name"]].fillna("").astype(str).str.strip()
    else:
        if df.shape[1] >= 2:
            df["Name"] = (
                df.iloc[:, 0].fillna("").astype(str).str.strip()
                + " "
                + df.iloc[:, 1].fillna("").astype(str).str.strip()
            )
        else:
            raise ValueError("Could not find name columns (need 'First/Last name' or 'Name').")

    # Role column: accept synonyms, else default to volunteer
    role_col = (cols_lower.get("role") or cols_lower.get("position") or cols_lower.get("title"))

    # Preference columns (keep sheet order)
    pref_cols = [c for c in df.columns if any(k in str(c).lower() for k in ("choice", "availability", "pref", "avail"))]
    if not pref_cols:
        raise ValueError("No preference/availability columns detected (need columns containing 'choice' or 'availability').")

    # Volunteers (sorted for determinism)
    volunteers = sorted([str(x).strip() for x in df["Name"].tolist() if str(x).strip()])
    if not volunteers:
        raise ValueError("No volunteers found after parsing names.")

    roles = {}
    prefs_map = {}

    for _, row in df.iterrows():
        name = str(row["Name"]).strip()
        if not name:
            continue
        raw_role = row.get(role_col, "") if role_col else ""
        roles[name] = _normalize_role(raw_role)

        prefs = []
        for c in pref_cols:
            val = row.get(c, None)
            if pd.notna(val):
                s = _norm_slot(val)
                if s:
                    prefs.append(s)
        prefs_map[name] = prefs

    shifts_set = {slot for prefs in prefs_map.values() for slot in prefs if slot}
    if not shifts_set:
        raise ValueError("No valid shift strings found in preferences.")
    shifts = sorted(shifts_set)

    # Build weights
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

    # -------- Phase 1: maximize non-fallback count --------
    model1 = cp_model.CpModel()
    x1 = {(v, s): model1.NewBoolVar(f"x1_{v}_{s}") for v in volunteers for s in shifts}

    for v in volunteers:
        model1.Add(sum(x1[(v, s)] for s in shifts) == 1)
    for s in shifts:
        model1.Add(sum(x1[(v, s)] for v in volunteers) <= MAX_PER_SHIFT)
    for a, b in clean_pairs:
        for s in shifts:
            model1.Add(x1[(a, s)] == x1[(b, s)])

    # Mentor coverage only if both roles exist
    if mentees and mentors:
        y1 = {s: model1.NewBoolVar(f"y1_{s}") for s in shifts}
        for s in shifts:
            model1.Add(sum(x1[(v, s)] for v in mentees) >= 1).OnlyEnforceIf(y1[s])
            model1.Add(sum(x1[(v, s)] for v in mentees) == 0).OnlyEnforceIf(y1[s].Not())
            model1.Add(sum(x1[(v, s)] for v in mentors) >= 1).OnlyEnforceIf(y1[s])
    # If there are mentees but no mentors, we DO NOT ban mentees; we skip the rule (legacy behavior).

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
        raise RuntimeError("No feasible schedule found in phase 1.")
    best_nonfb = int(solver1.ObjectiveValue())

    # -------- Phase 2: maximize preference score w/ same non-fallback count --------
    model2 = cp_model.CpModel()
    x2 = {(v, s): model2.NewBoolVar(f"x2_{v}_{s}") for v in volunteers for s in shifts}

    for v in volunteers:
        model2.Add(sum(x2[(v, s)] for s in shifts) == 1)
    for s in shifts:
        model2.Add(sum(x2[(v, s)] for v in volunteers) <= MAX_PER_SHIFT)
    for a, b in clean_pairs:
        for s in shifts:
            model2.Add(x2[(a, s)] == x2[(b, s)])
    if mentees and mentors:
        y2 = {s: model2.NewBoolVar(f"y2_{s}") for s in shifts}
        for s in shifts:
            model2.Add(sum(x2[(v, s)] for v in mentees) >= 1).OnlyEnforceIf(y2[s])
            model2.Add(sum(x2[(v, s)] for v in mentees) == 0).OnlyEnforceIf(y2[s].Not())
            model2.Add(sum(x2[(v, s)] for v in mentors) >= 1).OnlyEnforceIf(y2[s])

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
        raise RuntimeError("No feasible schedule found in phase 2.")

    # Extract final schedule
    role_order = {"mentor": 0, "mentee": 1, "mit": 2, "volunteer": 3}
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

    # Stable ordering in each shift
    for s in schedule:
        schedule[s].sort(key=lambda a: (role_order.get(a.get("Role", "volunteer"), 99), a.get("Name", "")))

    return schedule, assigned


# ----------------------------------
# Relaxed solve (≤1 per person, keep rules that make sense)
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
    if mentees and mentors:
        y = {s: model.NewBoolVar(f"y_{s}") for s in shifts}
        for s in shifts:
            model.Add(sum(x[(v, s)] for v in mentees) >= 1).OnlyEnforceIf(y[s])
            model.Add(sum(x[(v, s)] for v in mentees) == 0).OnlyEnforceIf(y[s].Not())
            model.Add(sum(x[(v, s)] for v in mentors) >= 1).OnlyEnforceIf(y[s])

    model.Maximize(sum(x[(v, s)] for v in volunteers for s in shifts))

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 42
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 15
    solver.Solve(model)

    role_order = {"mentor": 0, "mentee": 1, "mit": 2, "volunteer": 3}
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

    for s in schedule:
        schedule[s].sort(key=lambda a: (role_order.get(a.get("Role", "volunteer"), 99), a.get("Name", "")))

    return schedule, assigned


# ----------------------------------
# DataFrame builders / summaries
# ----------------------------------
def prepare_schedule_df(schedule):
    rows = []
    for slot, items in schedule.items():
        for a in items:
            rows.append({
                "Time Slot": slot,
                "Name": a.get("Name", ""),
                "Role": a.get("Role", "volunteer"),
                "Fallback": bool(a.get("Fallback", False)),
            })
    return pd.DataFrame(rows)

def compute_breakdown(schedule, prefs_map):
    total = sum(len(items) for items in schedule.values())
    ord_names = ["1st", "2nd", "3rd", "4th", "5th"]
    counts = {k: 0 for k in ord_names}
    counts["Fallback"] = 0

    for slot, items in schedule.items():
        for a in items:
            name = a.get("Name", "")
