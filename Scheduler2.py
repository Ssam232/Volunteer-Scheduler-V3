import re
import pandas as pd
from ortools.sat.python import cp_model

# ----------------------------------
# Configuration
# ----------------------------------
MAX_PER_SHIFT = 3
LEX_WEIGHTS = {1: 100000, 2: 10000, 3: 1000, 4: 100, 5: 10}
FALLBACK_WEIGHT = 0
GROUP_PAIRS = []  # list of ("NameA","NameB") pairs to lock together

# ----------------------------------
# Helpers
# ----------------------------------
def _normalize_role(s: str) -> str:
    """
    Map free-text roles to canonical tags:
      - 'mentor' -> 'mentor'
      - 'mentee' / 'trainee' -> 'mentee'
      - 'mentor in training' / 'mit' -> 'mit' (NEW: does NOT count as mentor)
      - everything else -> 'volunteer'
    """
    r = re.sub(r"\s+", " ", str(s).strip().lower())
    r = r.replace("-", " ")
    if r == "mentor":
        return "mentor"
    if r in {"mentee", "trainee"}:
        return "mentee"
    if r in {"mentor in training", "mit", "mentor in training (mit)"}:
        return "mit"
    return "volunteer"

# ----------------------------------
# Load and parse preferences
# ----------------------------------
def load_preferences(df: pd.DataFrame):
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    # Build Name column
    first_cols = [cols_lower[k] for k in cols_lower if 'first' in k and 'name' in k]
    last_cols  = [cols_lower[k] for k in cols_lower if 'last'  in k and 'name' in k]
    if first_cols and last_cols:
        df['Name'] = df[first_cols[0]].astype(str) + ' ' + df[last_cols[0]].astype(str)
    elif 'name' in cols_lower:
        df['Name'] = df[cols_lower['name']].astype(str)
    else:
        df['Name'] = df.iloc[:, 0].astype(str) + ' ' + df.iloc[:, 1].astype(str)

    # Role column
    if 'role' not in cols_lower:
        raise ValueError("Missing 'Role' column.")
    role_col = cols_lower['role']

    # Volunteers list sorted for reproducibility
    volunteers = sorted(df['Name'].astype(str).tolist())
    roles = {}
    prefs_map = {}

    # Identify preference columns
    pref_cols = [c for c in df.columns if 'choice' in c.lower() or 'availability' in c.lower()]
    if not pref_cols:
        raise ValueError("No preference columns detected.")

    for _, row in df.iterrows():
        name = row['Name']
        roles[name] = _normalize_role(row.get(role_col, ''))  # <-- updated to use safe parser
        prefs_map[name] = [str(row[c]) for c in pref_cols if pd.notna(row[c])]

    # Distinct sorted shifts
    shifts = sorted({slot for prefs in prefs_map.values() for slot in prefs})

    # Build weight matrix
    weights = {}
    for name, prefs in prefs_map.items():
        for rank, slot in enumerate(prefs, start=1):
            weights[(name, slot)] = LEX_WEIGHTS.get(rank, 0)
    for name in volunteers:
        for slot in shifts:
            weights.setdefault((name, slot), FALLBACK_WEIGHT)

    return volunteers, roles, shifts, weights, prefs_map

# ----------------------------------
# Two-phase solve (lexicographic)
#   Phase 1: maximize #non-fallback assignments
#   Phase 2: maximize preference weight given Phase 1 optimum
# ----------------------------------
def solve_schedule(volunteers, roles, shifts, weights):
    volunteers = sorted(volunteers)
    shifts = sorted(shifts)

    # Precompute mentees/mentors
    # NOTE: 'mit' is intentionally NOT treated as mentor
    mentees = [v for v in volunteers if roles[v] == 'mentee']
    mentors = [v for v in volunteers if roles[v] == 'mentor']

    # ----- Phase 1 -----
    model1 = cp_model.CpModel()
    x1 = {(v, s): model1.NewBoolVar(f"x1_{v}_{s}") for v in volunteers for s in shifts}

    # One shift per volunteer
    for v in volunteers:
        model1.Add(sum(x1[(v, s)] for s in shifts) == 1)

    # Capacity per shift
    for s in shifts:
        model1.Add(sum(x1[(v, s)] for v in volunteers) <= MAX_PER_SHIFT)

    # Group-pairs constraint
    for a, b in GROUP_PAIRS:
        if a in volunteers and b in volunteers:
            for s in shifts:
                model1.Add(x1[(a, s)] == x1[(b, s)])

    # Mentor coverage: if any mentee assigned, require ≥1 mentor (MIT does not count)
    if mentees and mentors:
        y1 = {s: model1.NewBoolVar(f"y1_{s}") for s in shifts}
        for s in shifts:
            model1.Add(sum(x1[(v, s)] for v in mentees) >= 1).OnlyEnforceIf(y1[s])
            model1.Add(sum(x1[(v, s)] for v in mentees) == 0).OnlyEnforceIf(y1[s].Not())
            model1.Add(sum(x1[(v, s)] for v in mentors) >= 1).OnlyEnforceIf(y1[s])

    # Objective: maximize non-fallback count
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

    # ----- Phase 2 -----
    model2 = cp_model.CpModel()
    x2 = {(v, s): model2.NewBoolVar(f"x2_{v}_{s}") for v in volunteers for s in shifts}

    for v in volunteers:
        model2.Add(sum(x2[(v, s)] for s in shifts) == 1)
    for s in shifts:
        model2.Add(sum(x2[(v, s)] for v in volunteers) <= MAX_PER_SHIFT)
    for a, b in GROUP_PAIRS:
        if a in volunteers and b in volunteers:
            for s in shifts:
                model2.Add(x2[(a, s)] == x2[(b, s)])
    if mentees and mentors:
        y2 = {s: model2.NewBoolVar(f"y2_{s}") for s in shifts}
        for s in shifts:
            model2.Add(sum(x2[(v, s)] for v in mentees) >= 1).OnlyEnforceIf(y2[s])
            model2.Add(sum(x2[(v, s)] for v in mentees) == 0).OnlyEnforceIf(y2[s].Not())
            model2.Add(sum(x2[(v, s)] for v in mentors) >= 1).OnlyEnforceIf(y2[s])

    # Preserve Phase 1 result (lexicographic)
    model2.Add(
        sum((1 if weights[(v, s)] != FALLBACK_WEIGHT else 0) * x2[(v, s)]
            for v in volunteers for s in shifts) == best_nonfb
    )

    # Objective: maximize weighted preferences
    model2.Maximize(
        sum(weights[(v, s)] * x2[(v, s)] for v in volunteers for s in shifts)
    )

    solver2 = cp_model.CpSolver()
    solver2.parameters.random_seed = 42
    solver2.parameters.num_search_workers = 1
    solver2.parameters.max_time_in_seconds = 30
    solver2.Solve(model2)

    # Extract schedule
    schedule = {s: [] for s in shifts}
    assigned = set()
    for (v, s), var in x2.items():
        if solver2.Value(var):
            schedule[s].append({
                'Name': v,
                'Role': roles[v],  # 'mit' will appear as-is; UI can style or ignore
                'Fallback': weights[(v, s)] == FALLBACK_WEIGHT
            })
            assigned.add(v)
    return schedule, assigned

# ----------------------------------
# Relaxed solve for infeasible inputs
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

    # Group pairs
    for a, b in GROUP_PAIRS:
        if a in volunteers and b in volunteers:
            for s in shifts:
                model.Add(x[(a, s)] == x[(b, s)])

    # Mentor coverage (MIT does NOT count as mentor)
    mentees = [v for v in volunteers if roles[v] == 'mentee']
    mentors = [v for v in volunteers if roles[v] == 'mentor']
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
    solver.parameters.max_time_in_seconds = 30
    solver.Solve(model)

    schedule = {s: [] for s in shifts}
    assigned = set()
    for (v, s), var in x.items():
        if solver.Value(var):
            schedule[s].append({
                'Name': v,
                'Role': roles[v],
                'Fallback': weights[(v, s)] == FALLBACK_WEIGHT
            })
            assigned.add(v)
    return schedule, assigned

# ----------------------------------
# DataFrame builders
# ----------------------------------
def prepare_schedule_df(schedule):
    rows = []
    for slot, items in schedule.items():
        for a in items:
            rows.append({
                'Time Slot': slot,
                'Name': a['Name'],
                'Role': a['Role'],
                'Fallback': a['Fallback']
            })
    return pd.DataFrame(rows)

# ----------------------------------
# Preference breakdown
# ----------------------------------
def compute_breakdown(schedule, prefs_map):
    total = sum(len(items) for items in schedule.values())
    ord_names = ['1st', '2nd', '3rd', '4th', '5th']
    counts = {name: 0 for name in ord_names}
    counts['Fallback'] = 0
    for slot, items in schedule.items():
        for a in items:
            name = a['Name']
            prefs = prefs_map[name]
            if slot in prefs:
                idx = prefs.index(slot)
                if idx < len(ord_names):
                    counts[ord_names[idx]] += 1
                else:
                    counts['Fallback'] += 1
            else:
                counts['Fallback'] += 1
    rows = []
    for k, v in counts.items():
        pct = (v / total * 100) if total else 0
        rows.append({
            'Preference': k,
            'Count': v,
            'Percentage': f"{pct:.1f}%"
        })
    return pd.DataFrame(rows)

# ----------------------------------
# Entrypoint
# ----------------------------------
def build_schedule(df: pd.DataFrame):
    vols, roles, shifts, weights, prefs_map = load_preferences(df)
    try:
        sched, assigned = solve_schedule(vols, roles, shifts, weights)
    except RuntimeError:import re
import pandas as pd
from ortools.sat.python import cp_model

# ----------------------------------
# Configuration
# ----------------------------------
MAX_PER_SHIFT = 3
LEX_WEIGHTS = {1: 100000, 2: 10000, 3: 1000, 4: 100, 5: 10}
FALLBACK_WEIGHT = 0
GROUP_PAIRS = []  # list of ("NameA","NameB") pairs to lock together

# ----------------------------------
# Helpers
# ----------------------------------
def _normalize_role(s: str) -> str:
    """
    Map free-text roles to canonical tags:
      - 'mentor' -> 'mentor'
      - 'mentee' / 'trainee' / 'new volunteer' (and variants) -> 'mentee'
      - 'mentor in training' / 'mit' -> 'mit' (does NOT count as mentor)
      - everything else -> 'volunteer'
    """
    r = re.sub(r"\s+", " ", str(s).strip().lower())
    r = r.replace("-", " ")
    if r == "mentor":
        return "mentor"
    # mentee synonyms (include "new volunteer" variants)
    if r in {"mentee", "trainee", "new volunteer", "newvolunteer", "new volunteer (mentee)"} or r.startswith("new volunteer"):
        return "mentee"
    if r in {"mentor in training", "mit", "mentor in training (mit)"}:
        return "mit"
    return "volunteer"

# Day normalization for slots
DAY_MAP = {
    "mon": "Monday", "monday": "Monday",
    "tue": "Tuesday", "tues": "Tuesday", "tuesday": "Tuesday",
    "wed": "Wednesday", "weds": "Wednesday", "wednesday": "Wednesday",
    "thu": "Thursday", "thur": "Thursday", "thurs": "Thursday", "thursday": "Thursday",
    "fri": "Friday", "friday": "Friday",
    "sat": "Saturday", "saturday": "Saturday",
    "sun": "Sunday", "sunday": "Sunday",
}

def _norm_slot(s: str) -> str:
    """
    Conservative normalization for 'Day HH:MM-HH:MM':
    - normalize unicode dashes to '-'
    - collapse spaces
    - title-case day via DAY_MAP
    (Avoids over-merging distinct times; only de-gunks format.)
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
    rest = re.sub(r"-+", "-", rest)        # ensure single hyphen
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
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    # Build Name column
    first_cols = [cols_lower[k] for k in cols_lower if 'first' in k and 'name' in k]
    last_cols  = [cols_lower[k] for k in cols_lower if 'last'  in k and 'name' in k]
    if first_cols and last_cols:
        df['Name'] = df[first_cols[0]].astype(str).str.strip() + ' ' + df[last_cols[0]].astype(str).str.strip()
    elif 'name' in cols_lower:
        df['Name'] = df[cols_lower['name']].astype(str).str.strip()
    else:
        df['Name'] = df.iloc[:, 0].astype(str).str.strip() + ' ' + df.iloc[:, 1].astype(str).str.strip()

    # Role column
    if 'role' not in cols_lower:
        raise ValueError("Missing 'Role' column.")
    role_col = cols_lower['role']

    # Volunteers list sorted for reproducibility
    volunteers = sorted(df['Name'].astype(str).tolist())
    roles = {}
    prefs_map = {}

    # Identify preference columns
    pref_cols = [c for c in df.columns if ('choice' in c.lower() or 'availability' in c.lower())]
    if not pref_cols:
        raise ValueError("No preference columns detected.")

    for _, row in df.iterrows():
        name = str(row['Name'])
        roles[name] = _normalize_role(row.get(role_col, ''))
        # normalize each preference string (spacing/dashes/day)
        prefs_map[name] = [ _norm_slot(row[c]) for c in pref_cols if pd.notna(row[c]) and _norm_slot(row[c]) ]

    # Distinct sorted shifts (normalized)
    shifts = sorted({slot for prefs in prefs_map.values() for slot in prefs})

    # Build weight matrix
    weights = {}
    for name, prefs in prefs_map.items():
        for rank, slot in enumerate(prefs, start=1):
            weights[(name, slot)] = LEX_WEIGHTS.get(rank, 0)
    for name in volunteers:
        for slot in shifts:
            weights.setdefault((name, slot), FALLBACK_WEIGHT)

    return volunteers, roles, shifts, weights, prefs_map

# ----------------------------------
# Two-phase solve (lexicographic)
#   Phase 1: maximize #non-fallback assignments
#   Phase 2: maximize preference weight given Phase 1 optimum
# ----------------------------------
def solve_schedule(volunteers, roles, shifts, weights):
    volunteers = sorted(volunteers)
    shifts = sorted(shifts)

    # Precompute mentees/mentors
    # NOTE: 'mit' is intentionally NOT treated as mentor
    mentees = [v for v in volunteers if roles[v] == 'mentee']
    mentors = [v for v in volunteers if roles[v] == 'mentor']

    # Cleaned group-pairs (no self/unknown/dupes)
    clean_pairs = _clean_pairs(volunteers, GROUP_PAIRS)

    # ----- Phase 1 -----
    model1 = cp_model.CpModel()
    x1 = {(v, s): model1.NewBoolVar(f"x1_{v}_{s}") for v in volunteers for s in shifts}

    # One shift per volunteer
    for v in volunteers:
        model1.Add(sum(x1[(v, s)] for s in shifts) == 1)

    # Capacity per shift
    for s in shifts:
        model1.Add(sum(x1[(v, s)] for v in volunteers) <= MAX_PER_SHIFT)

    # Group-pairs constraint
    for a, b in clean_pairs:
        for s in shifts:
            model1.Add(x1[(a, s)] == x1[(b, s)])

    # Mentor coverage: if any mentee assigned, require ≥1 mentor (MIT does not count)
    if mentees and mentors:
        y1 = {s: model1.NewBoolVar(f"y1_{s}") for s in shifts}
        for s in shifts:
            model1.Add(sum(x1[(v, s)] for v in mentees) >= 1).OnlyEnforceIf(y1[s])
            model1.Add(sum(x1[(v, s)] for v in mentees) == 0).OnlyEnforceIf(y1[s].Not())
            model1.Add(sum(x1[(v, s)] for v in mentors) >= 1).OnlyEnforceIf(y1[s])

    # Objective: maximize non-fallback count
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

    # ----- Phase 2 -----
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

    # Preserve Phase 1 result (lexicographic)
    model2.Add(
        sum((1 if weights[(v, s)] != FALLBACK_WEIGHT else 0) * x2[(v, s)]
            for v in volunteers for s in shifts) == best_nonfb
    )

    # Objective: maximize weighted preferences
    model2.Maximize(
        sum(weights[(v, s)] * x2[(v, s)] for v in volunteers for s in shifts)
    )

    solver2 = cp_model.CpSolver()
    solver2.parameters.random_seed = 42
    solver2.parameters.num_search_workers = 1
    solver2.parameters.max_time_in_seconds = 30
    solver2.Solve(model2)

    # Extract schedule (DO NOT sort within shift — keep your original behavior)
    schedule = {s: [] for s in shifts}
    assigned = set()
    for (v, s), var in x2.items():
        if solver2.Value(var):
            schedule[s].append({
                'Name': v,
                'Role': roles[v],  # 'mit' will appear as-is; UI can style or ignore
                'Fallback': weights[(v, s)] == FALLBACK_WEIGHT
            })
            assigned.add(v)
    return schedule, assigned

# ----------------------------------
# Relaxed solve for infeasible inputs
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

    # Group pairs (cleaned)
    clean_pairs = _clean_pairs(volunteers, GROUP_PAIRS)
    for a, b in clean_pairs:
        for s in shifts:
            model.Add(x[(a, s)] == x[(b, s)])

    # Mentor coverage (MIT does NOT count as mentor)
    mentees = [v for v in volunteers if roles[v] == 'mentee']
    mentors = [v for v in volunteers if roles[v] == 'mentor']
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
    solver.parameters.max_time_in_seconds = 30
    solver.Solve(model)

    schedule = {s: [] for s in shifts}
    assigned = set()
    for (v, s), var in x.items():
        if solver.Value(var):
            schedule[s].append({
                'Name': v,
                'Role': roles[v],
                'Fallback': weights[(v, s)] == FALLBACK_WEIGHT
            })
            assigned.add(v)
    return schedule, assigned

# ----------------------------------
# DataFrame builders
# ----------------------------------
def prepare_schedule_df(schedule):
    rows = []
    for slot, items in schedule.items():
        for a in items:
            rows.append({
                'Time Slot': slot,
                'Name': a['Name'],
                'Role': a['Role'],
                'Fallback': a['Fallback']
            })
    return pd.DataFrame(rows)

# ----------------------------------
# Preference breakdown
# ----------------------------------
def compute_breakdown(schedule, prefs_map):
    total = sum(len(items) for items in schedule.values())
    ord_names = ['1st', '2nd', '3rd', '4th', '5th']
    counts = {name: 0 for name in ord_names}
    counts['Fallback'] = 0
    for slot, items in schedule.items():
        for a in items:
            name = a['Name']
            prefs = prefs_map.get(name, [])  # defensive: .get
            if slot in prefs:
                idx = prefs.index(slot)
                if idx < len(ord_names):
                    counts[ord_names[idx]] += 1
                else:
                    counts['Fallback'] += 1
            else:
                counts['Fallback'] += 1
    rows = []
    for k, v in counts.items():
        pct = (v / total * 100) if total else 0
        rows.append({
            'Preference': k,
            'Count': v,
            'Percentage': f"{pct:.1f}%"
        })
    return pd.DataFrame(rows)

# ----------------------------------
# Entrypoint
# ----------------------------------
def build_schedule(df: pd.DataFrame):
    vols, roles, shifts, weights, prefs_map = load_preferences(df)
    try:
        sched, assigned = solve_schedule(vols, roles, shifts, weights)
    except RuntimeError:
        sched, assigned = solve_relaxed(vols, roles, shifts, weights)
    sched_df = prepare_schedule_df(sched)
    unassigned = [v for v in vols if v not in assigned]
    unassigned_df = pd.DataFrame({'Name': unassigned})
    breakdown_df = compute_breakdown(sched, prefs_map)
    return sched_df, unassigned_df, breakdown_df

        sched, assigned = solve_relaxed(vols, roles, shifts, weights)
    sched_df = prepare_schedule_df(sched)
    unassigned = [v for v in vols if v not in assigned]
    unassigned_df = pd.DataFrame({'Name': unassigned})
    breakdown_df = compute_breakdown(sched, prefs_map)
    return sched_df, unassigned_df, breakdown_df

