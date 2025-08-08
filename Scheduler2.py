import re
import pandas as pd
from ortools.sat.python import cp_model

# ----------------------------------
# Configuration
# ----------------------------------
MAX_PER_SHIFT = 3
# Strict lexicographic weighting for phase 2 (after minimizing fallbacks)
LEX_WEIGHTS = {1: 100000, 2: 10000, 3: 1000, 4: 100, 5: 10}
# Anything not explicitly listed in top-5 is counted as fallback (weight==0)
FALLBACK_WEIGHT = 0

# Streamlit will set this before calling build_schedule()
GROUP_PAIRS: list[tuple[str, str]] = []

# ----------------------------------
# Role normalization
# ----------------------------------
def _normalize_role(s: str) -> str:
    """
    Canonicalize free-text roles:
      mentor                               -> 'mentor'
      mentee / trainee / new volunteer(*)  -> 'mentee'
      mentor in training / mit             -> 'mit'   (does NOT count as mentor)
      anything else                        -> 'volunteer'

    (*) 'new volunteer' variants map to mentee so they follow mentee rules.
    """
    r = re.sub(r"\s+", " ", str(s).strip().lower())
    r = r.replace("-", " ")

    if r == "mentor":
        return "mentor"

    # mentee synonyms (incl. new volunteer variants)
    if (
        r in {
            "mentee",
            "trainee",
            "new volunteer",
            "newvolunteer",
            "new volunteer (mentee)",
        }
        or r.startswith("new volunteer")
    ):
        return "mentee"

    # mentor-in-training (does not satisfy mentor coverage)
    if r in {"mentor in training", "mit", "mentor in training (mit)"}:
        return "mit"

    return "volunteer"

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
    """
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    # Build Name column robustly
    first_cols = [cols_lower[k] for k in cols_lower if "first" in k and "name" in k]
    last_cols  = [cols_lower[k] for k in cols_lower if "last"  in k and "name" in k]
    if first_cols and last_cols:
        df["Name"] = (
            df[first_cols[0]].astype(str).str.strip()
            + " "
            + df[last_cols[0]].astype(str).str.strip()
        )
    elif "name" in cols_lower:
        df["Name"] = df[cols_lower["name"]].astype(str).str.strip()
    else:
        # Fallback: assume first two columns are first/last
        df["Name"] = (
            df.iloc[:, 0].astype(str).str.strip()
            + " "
            + df.iloc[:, 1].astype(str).str.strip()
        )

    # Role column
    if "role" not in cols_lower:
        raise ValueError("Missing 'Role' column.")
    role_col = cols_lower["role"]

    # Identify preference/availability columns; keep sheet order
    pref_cols = [c for c in df.columns if ("choice" in c.lower() or "availability" in c.lower())]
    if not pref_cols:
        raise ValueError("No preference/availability columns detected (need columns containing 'choice' or 'availability').")

    # Volunteers sorted for reproducibility (deterministic solver)
    volunteers = sorted(df["Name"].astype(str).tolist())

    # Role map + preference lists
    roles: dict[str, str] = {}
    prefs_map: dict[str, list[str]] = {}

    for _, row in df.iterrows():
        name = str(row["Name"]).strip()
        roles[name] = _normalize_role(row.get(role_col, ""))
        # Collect non-empty prefs in the order they appear
        prefs = []
        for c in pref_cols:
            val = row[c]
            if pd.notna(val):
                s = str(val).strip()
                if s:
                    prefs.append(s)
        prefs_map[name] = prefs

    # Distinct shifts (as they appear across all prefs), then sorted for determinism
    shifts = sorted({slot for prefs in prefs_map.values() for slot in prefs})

    # Preference weights: rank -> big number; unlisted -> FALLBACK (0)
    weights: dict[tuple[str, str], int] = {}
    for name, prefs in prefs_map.items():
        for rank, slot in enumerate(prefs, start=1):
            weights[(name, slot)] = LEX_WEIGHTS.get(rank, 0)
    # ensure every (name, shift) pair exists in weights (fallback == 0)
    for name in volunteers:
        for slot in shifts:
            weights.setdefault((name, slot), FALLBACK_WEIGHT)

    return volunteers, roles, shifts, weights, prefs_map

# ----------------------------------
# Two-phase lexicographic solve
#   Phase 1: maximize # non-fallback assignments
#   Phase 2: maximize preference weights subject to Phase 1 optimum
# ----------------------------------
def solve_schedule(volunteers, roles, shifts, weights):
    volunteers = sorted(volunteers)
    shifts = sorted(shifts)

    # Roles for constraints
    # NOTE: 'mit' is intentionally NOT in the mentors list
    mentees = [v for v in volunteers if roles.get(v) == "mentee"]
    mentors = [v for v in volunteers if roles.get(v) == "mentor"]

    # ----------------- Phase 1 -----------------
    model1 = cp_model.CpModel()
    x1 = {(v, s): model1.NewBoolVar(f"x1_{v}_{s}") for v in volunteers for s in shifts}

    # Exactly one shift per volunteer
    for v in volunteers:
        model1.Add(sum(x1[(v, s)] for s in shifts) == 1)

    # Capacity per shift
    for s in shifts:
        model1.Add(sum(x1[(v, s)] for v in volunteers) <= MAX_PER_SHIFT)

    # Group-together constraints
    for a, b in GROUP_PAIRS:
        if a in volunteers and b in volunteers:
            for s in shifts:
                model1.Add(x1[(a, s)] == x1[(b, s)])

    # Mentee→Mentor coverage (only if both sets exist)
    if mentees and mentors:
        y1 = {s: model1.NewBoolVar(f"y1_{s}") for s in shifts}
        for s in shifts:
            # y1[s] == 1 ↔ any mentee assigned
            model1.Add(sum(x1[(v, s)] for v in mentees) >= 1).OnlyEnforceIf(y1[s])
            model1.Add(sum(x1[(v, s)] for v in mentees) == 0).OnlyEnforceIf(y1[s].Not())
            # If a mentee is present, require ≥1 mentor (MIT doesn't count)
            model1.Add(sum(x1[(v, s)] for v in mentors) >= 1).OnlyEnforceIf(y1[s])

    # Objective 1: maximize count of non-fallback assignments
    model1.Maximize(
        sum(
            (1 if weights[(v, s)] != FALLBACK_WEIGHT else 0) * x1[(v, s)]
            for v in volunteers
            for s in shifts
        )
    )

    solver1 = cp_model.CpSolver()
    # Deterministic settings to eliminate run-to-run variation
    solver1.parameters.random_seed = 42
    solver1.parameters.num_search_workers = 1
    solver1.parameters.max_time_in_seconds = 30
    status1 = solver1.Solve(model1)
    if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule found in phase 1.")
    best_nonfb = int(solver1.ObjectiveValue())

    # ----------------- Phase 2 -----------------
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

    # Lock in Phase 1 optimum: same maximum number of non-fallback assignments
    model2.Add(
        sum(
            (1 if weights[(v, s)] != FALLBACK_WEIGHT else 0) * x2[(v, s)]
            for v in volunteers
            for s in shifts
        )
        == best_nonfb
    )

    # Objective 2: maximize preference weights (1st >> 2nd >> ... >> fallback)
    model2.Maximize(sum(weights[(v, s)] * x2[(v, s)] for v in volunteers for s in shifts))

    solver2 = cp_model.CpSolver()
    solver2.parameters.random_seed = 42
    solver2.parameters.num_search_workers = 1
    solver2.parameters.max_time_in_seconds = 30
    status2 = solver2.Solve(model2)
    if status2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule found in phase 2.")

    # Extract final schedule
    schedule = {s: [] for s in shifts}
    assigned = set()
    for (v, s), var in x2.items():
        if solver2.Value(var):
            schedule[s].append(
                {
                    "Name": v,
                    "Role": roles.get(v, "volunteer"),
                    "Fallback": weights[(v, s)] == FALLBACK_WEIGHT,
                }
            )
            assigned.add(v)
    return schedule, assigned

# ----------------------------------
# Relaxed solve for infeasible inputs
# (assign ≤1 shift per person; still obeys capacity, group pairs, and coverage)
# ----------------------------------
def solve_relaxed(volunteers, roles, shifts, weights):
    volunteers = sorted(volunteers)
    shifts = sorted(shifts)

    model = cp_model.CpModel()
    x = {(v, s): model.NewBoolVar(f"x_{v}_{s}") for v in volunteers for s in shifts}

    # At most one shift per volunteer
    for v in volunteers:
        model.Add(sum(x[(v, s)] for s in shifts) <= 1)

    # Capacity per shift
    for s in shifts:
        model.Add(sum(x[(v, s)] for v in volunteers) <= MAX_PER_SHIFT)

    # Group pairs
    for a, b in GROUP_PAIRS:
        if a in volunteers and b in volunteers:
            for s in shifts:
                model.Add(x[(a, s)] == x[(b, s)])

    # Mentor coverage if applicable
    mentees = [v for v in volunteers if roles.get(v) == "mentee"]
    mentors = [v for v in volunteers if roles.get(v) == "mentor"]
    if mentees and mentors:
        y = {s: model.NewBoolVar(f"y_{s}") for s in shifts}
        for s in shifts:
            model.Add(sum(x[(v, s)] for v in mentees) >= 1).OnlyEnforceIf(y[s])
            model.Add(sum(x[(v, s)] for v in mentees) == 0).OnlyEnforceIf(y[s].Not())
            model.Add(sum(x[(v, s)] for v in mentors) >= 1).OnlyEnforceIf(y[s])

    # Maximize number of assignments (ignores preference ranking in relaxed mode)
    model.Maximize(sum(x[(v, s)] for v in volunteers for s in shifts))

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 42
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 15
    solver.Solve(model)

    schedule = {s: [] for s in shifts}
    assigned = set()
    for (v, s), var in x.items():
        if solver.Value(var):
            schedule[s].append(
                {
                    "Name": v,
                    "Role": roles.get(v, "volunteer"),
                    "Fallback": weights[(v, s)] == FALLBACK_WEIGHT,
                }
            )
            assigned.add(v)
    return schedule, assigned

# ----------------------------------
# DataFrame builders
# ----------------------------------
def prepare_schedule_df(schedule: dict) -> pd.DataFrame:
    rows = []
    for slot, items in schedule.items():
        for a in items:
            rows.append(
                {
                    "Time Slot": slot,
                    "Name": a["Name"],
                    "Role": a.get("Role", "volunteer"),
                    "Fallback": bool(a.get("Fallback", False)),
                }
            )
    return pd.DataFrame(rows)

def compute_breakdown(schedule: dict, prefs_map: dict[str, list[str]]) -> pd.DataFrame:
    total = sum(len(items) for items in schedule.values())
    ord_names = ["1st", "2nd", "3rd", "4th", "5th"]
    counts = {name: 0 for name in ord_names}
    counts["Fallback"] = 0

    for slot, items in schedule.items():
        for a in items:
            name = a["Name"]
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
    Orchestrates parsing and solving.
    Returns:
      sched_df:      DataFrame with columns [Time Slot, Name, Role, Fallback]
      unassigned_df: DataFrame of any people not assigned (usually empty with strict solve)
      breakdown_df:  Preference breakdown summary
    """
    volunteers, roles, shifts, weights, prefs_map = load_preferences(df)
    try:
        schedule, assigned = solve_schedule(volunteers, roles, shifts, weights)
    except RuntimeError:
        # Fall back to a relaxed solve (still obeys core rules)
        schedule, assigned = solve_relaxed(volunteers, roles, shifts, weights)

    sched_df = prepare_schedule_df(schedule)
    unassigned = [v for v in volunteers if v not in assigned]
    unassigned_df = pd.DataFrame({"Name": unassigned})
    breakdown_df = compute_breakdown(schedule, prefs_map)
    return sched_df, unassigned_df, breakdown_df
