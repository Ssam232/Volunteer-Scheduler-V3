# Scheduler2.py
import re
import pandas as pd
from ortools.sat.python import cp_model

# ----------------------------------
# Configuration
# ----------------------------------
MAX_PER_SHIFT = 3
# Preference weights used only as a final tie-break after tiered fill
LEX_WEIGHTS = {1: 100000, 2: 10000, 3: 1000, 4: 100, 5: 10}

# Filled by Streamlit: list of (nameA, nameB) to schedule together
GROUP_PAIRS = []

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
    Parse a single time: '10', '10:30', '2pm', '2:30 PM', '14:00'.
    Returns (minutes, had_meridian, could_be_12h)
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
      • If exactly one side has am/pm, infer the other side’s meridian to match (when plausible).
      • If both lack am/pm, assume 24h.
      • No overnight handling (if start >= end we keep as-is).
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

    # Align meridian if only one side has it and the other could plausibly be 12h
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

def _map_simple_windows(rest: str) -> str | None:
    """
    Map bare survey windows (no am/pm) to canonical 24h blocks.
    Accepts '10-2', '10:00-2', '10-2:00' (spaces/dashes irrelevant).
    """
    s = re.sub(r"\s+", "", str(rest))
    s = s.replace("–","-").replace("—","-").replace("−","-")
    s = re.sub(r":?00", "", s)  # strip ':00'
    m = {
        "10-2": "10:00-14:00",
        "2-6":  "14:00-18:00",
        "6-10": "18:00-22:00",
    }
    return m.get(s)

def _norm_slot(s: str) -> str:
    """
    Normalize 'Day time-range' to 'Day HH:MM-HH:MM' (24h).
    """
    s = str(s or "").strip()
    if not s:
        return ""
    s = re.sub(r"[–—−]", "-", s)          # normalize dashes
    s = re.sub(r"\s+", " ", s).strip()    # collapse spaces
    parts = s.split(" ", 1)
    if len(parts) < 2:
        return s
    day_raw, rest = parts[0], parts[1].strip()
    day = DAY_MAP.get(day_raw.lower(), day_raw.title())

    # Special-case common windows from your survey
    mapped = _map_simple_windows(rest)
    if mapped:
        return f"{day} {mapped}"

    # General parser
    rng = _parse_time_range_to_24h(rest)
    if not rng:
        rest = rest.replace(" ", "")
        rest = re.sub(r"-+", "-", rest)
        rng = rest
    return f"{day} {rng}"

def _clean_pairs(volunteers, pairs):
    """Remove self/unknown/duplicate pairs; keep input order."""
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
    """Find a column by candidate substring(s), case-insensitive. Returns real column or None."""
    for cand in candidates:
        for k, real in cols_lower.items():
            if cand in k:
                return real
    return None

def _rank_from_header(colname: str):
    """
    Extract rank (1..5) from header like:
      '1st choice', '2nd availability', 'Pref 3', 'Preference 4', 'Avail 5'
    """
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
    """
    Returns:
      volunteers: list[str]
      roles: dict[name -> role]
      shifts: list[str] (normalized to 'Day HH:MM-HH:MM' 24h)
      weights: dict[(name, shift) -> int]     (available pairs only)
      prefs_map: dict[name -> list[str]]
      A: dict[(name, shift) -> 0/1]           (availability mask)
    Raises:
      ValueError if required data is missing.
    """
    if df is None or df.empty:
        raise ValueError("The uploaded sheet is empty.")

    df = df.copy()
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    # Names: (First+Last) or single Name, else first two columns
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

    # Role: accept role/position/type/volunteer type; if missing, default to volunteer
    role_col = (_find_col(cols_lower, "role") or
                _find_col(cols_lower, "position") or
                _find_col(cols_lower, "type") or
                _find_col(cols_lower, "volunteer type"))

    # Volunteers sorted for determinism
    volunteers = sorted([str(x).strip() for x in df["Name"].tolist() if str(x).strip()])
    if not volunteers:
        raise ValueError("No volunteers found after parsing names.")

    # Preference columns (tolerant: "choice" or "availability")
    pref_cols = []
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ("choice", "availability", "avail", "pref", "preference")) or _rank_from_header(cl):
            pref_cols.append(c)
    if not pref_cols:
        raise ValueError("No preference/availability columns detected (need headers with 'choice', 'availability', 'pref', or a 1..5 rank).")

    # Keep ranked columns in order; then unranked in sheet order
    ranked, unranked = [], []
    for c in pref_cols:
        r = _rank_from_header(str(c))
        (ranked if r else unranked).append((r, c))
    ranked.sort(key=lambda t: t[0] or 999)
    ordered_pref_cols = [c for _, c in ranked] + [c for _, c in unranked]

    # Roles + Preferences
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
                ns = _norm_slot(val)  # normalize to Day HH:MM-HH:MM (24h) if possible
                if ns:
                    prefs.append(ns)
        prefs_map[name] = prefs

    # Distinct normalized shifts
    shifts_set = {slot for prefs in prefs_map.values() for slot in prefs if slot}
    if not shifts_set:
        raise ValueError("No valid shift strings found in preferences.")
    shifts = sorted(shifts_set)

    # Build weights (for available pairs only) and availability mask A
    weights = {}
    A = {}
    for name, prefs in prefs_map.items():
        pref_set = set(prefs)
        for slot in shifts:
            if slot in pref_set:
                rank = prefs.index(slot) + 1
                weights[(name, slot)] = LEX_WEIGHTS.get(rank, 0)
                A[(name, slot)] = 1
            else:
                A[(name, slot)] = 0  # outside availability

    return volunteers, roles, shifts, weights, prefs_map, A

# ----------------------------------
# Core constraints (availability-only, no fallbacks)
# ----------------------------------
def _add_core_constraints(model, x, volunteers, shifts, roles, clean_pairs, A):
    # ≤ 1 assignment per volunteer (unassigned allowed)
    for v in volunteers:
        model.Add(sum(x[(v, s)] for s in shifts) <= 1)

    # Capacity per shift
    for s in shifts:
        model.Add(sum(x[(v, s)] for v in volunteers) <= MAX_PER_SHIFT)

    # Forbid assigning outside availability
    for v in volunteers:
        for s in shifts:
            if A[(v, s)] == 0:
                model.Add(x[(v, s)] == 0)

    # Group pairs together (same shift or both unassigned)
    for a, b in clean_pairs:
        for s in shifts:
            model.Add(x[(a, s)] == x[(b, s)])

    # Mentee→Mentor coverage (MIT doesn't count as mentor)
    mentees = [v for v in volunteers if roles.get(v) == "mentee"]
    mentors = [v for v in volunteers if roles.get(v) == "mentor"]
    if mentees:
        z = {s: model.NewBoolVar(f"z_mentor_present_{s}") for s in shifts}
        for s in shifts:
            model.Add(sum(x[(v, s)] for v in mentees) <= MAX_PER_SHIFT * z[s])
            if mentors:
                model.Add(sum(x[(v, s)] for v in mentors) >= z[s])
            else:
                model.Add(z[s] == 0)

# ----------------------------------
# Tiered "fill" phases then preference tie-break
#   For l in 1..MAX_PER_SHIFT:
#     maximize number of shifts with occupancy ≥ l (within availability only).
#   Then maximize preference weights subject to those locked best-tier counts.
# ----------------------------------
def _solve_tier(volunteers, roles, shifts, A, lock_best_by_level: dict[int, int], level_to_optimize: int):
    model = cp_model.CpModel()
    x = {(v, s): model.NewBoolVar(f"x_{v}_{s}") for v in volunteers for s in shifts}

    clean_pairs = _clean_pairs(volunteers, GROUP_PAIRS)
    _add_core_constraints(model, x, volunteers, shifts, roles, clean_pairs, A)

    # Build tier indicators b_l[s]
    maxL = MAX_PER_SHIFT
    b = {l: {s: model.NewBoolVar(f"b{l}_{s}") for s in shifts} for l in range(1, maxL + 1)}

    for s in shifts:
        occ = sum(x[(v, s)] for v in volunteers)  # integer sum
        for l in range(1, maxL + 1):
            # If b_l[s] = 1 → occ >= l (objective pushes b up)
            model.Add(occ >= l * b[l][s])

    # Lock previously optimized levels
    for l, best in lock_best_by_level.items():
        model.Add(sum(b[l][s] for s in shifts) == best)

    # Optimize current level
    model.Maximize(sum(b[level_to_optimize][s] for s in shifts))

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 42
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 30
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule in tiered fill.")
    best = int(solver.ObjectiveValue())
    return best

def _solve_preferences(volunteers, roles, shifts, A, weights, lock_best_by_level: dict[int, int]):
    model = cp_model.CpModel()
    x = {(v, s): model.NewBoolVar(f"x_{v}_{s}") for v in volunteers for s in shifts}

    clean_pairs = _clean_pairs(volunteers, GROUP_PAIRS)
    _add_core_constraints(model, x, volunteers, shifts, roles, clean_pairs, A)

    # Rebuild b_l[s] and lock all levels to their best
    maxL = MAX_PER_SHIFT
    b = {l: {s: model.NewBoolVar(f"b{l}_{s}") for s in shifts} for l in range(1, maxL + 1)}
    for s in shifts:
        occ = sum(x[(v, s)] for v in volunteers)
        for l in range(1, maxL + 1):
            model.Add(occ >= l * b[l][s])
    for l, best in lock_best_by_level.items():
        model.Add(sum(b[l][s] for s in shifts) == best)

    # Objective: maximize preference weights (within availability + locked fill)
    model.Maximize(sum(weights.get((v, s), 0) * x[(v, s)] for v in volunteers for s in shifts))

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 42
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 30
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule in preference phase.")

    # Extract schedule
    schedule = {s: [] for s in shifts}
    assigned = set()
    for (v, s), var in x.items():
        if solver.Value(var):
            schedule[s].append({
                "Name": v,
                "Role": roles.get(v, "volunteer"),
                "Fallback": False,  # availability-only
            })
            assigned.add(v)
    return schedule, assigned

def solve_schedule(volunteers, roles, shifts, weights, A):
    volunteers = sorted(volunteers)
    shifts = sorted(shifts)

    # Tiered fill: l = 1..MAX_PER_SHIFT
    lock_best: dict[int, int] = {}
    for level in range(1, MAX_PER_SHIFT + 1):
        best = _solve_tier(volunteers, roles, shifts, A, lock_best, level)
        lock_best[level] = best

    # Final: preference tie-break under locked best tier counts
    schedule, assigned = _solve_preferences(volunteers, roles, shifts, A, weights, lock_best)
    return schedule, assigned

# ----------------------------------
# Relaxed solve (safety net): maximize total assigned within availability
# ----------------------------------
def solve_relaxed(volunteers, roles, shifts, A, weights):
    volunteers = sorted(volunteers)
    shifts = sorted(shifts)
    model = cp_model.CpModel()
    x = {(v, s): model.NewBoolVar(f"x_{v}_{s}") for v in volunteers for s in shifts}

    # ≤ 1 per person and capacity per shift
    for v in volunteers:
        model.Add(sum(x[(v, s)] for s in shifts) <= 1)
    for s in shifts:
        model.Add(sum(x[(v, s)] for v in volunteers) <= MAX_PER_SHIFT)

    # Availability only
    for v in volunteers:
        for s in shifts:
            if A[(v, s)] == 0:
                model.Add(x[(v, s)] == 0)

    # Group pairs
    clean_pairs = _clean_pairs(volunteers, GROUP_PAIRS)
    for a, b in clean_pairs:
        for s in shifts:
            model.Add(x[(a, s)] == x[(b, s)])

    # Mentee coverage
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

    # Objective: maximize total assigned; light tie-break by preference
    model.Maximize(
        1_000_000 * sum(x[(v, s)] for v in volunteers for s in shifts) +
        sum(weights.get((v, s), 0) * x[(v, s)] for v in volunteers for s in shifts)
    )

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
                "Fallback": False,
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
                "Time Slot": slot,            # normalized 'Day HH:MM-HH:MM' (24h)
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
                # Should not happen (availability-only), but keep defensive
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
      sched_df:      [Time Slot, Name, Role, Fallback=False]  (Time Slot is 24h)
      unassigned_df: [Name]
      breakdown_df:  [Preference, Count, Percentage]
    """
    volunteers, roles, shifts, weights, prefs_map, A = load_preferences(df)
    try:
        schedule, assigned = solve_schedule(volunteers, roles, shifts, weights, A)
    except RuntimeError:
        # Safety net if tiered locking ever becomes infeasible
        schedule, assigned = solve_relaxed(volunteers, roles, shifts, A, weights)
    sched_df = prepare_schedule_df(schedule)
    unassigned = [v for v in volunteers if v not in assigned]
    unassigned_df = pd.DataFrame({"Name": unassigned})
    breakdown_df = compute_breakdown(schedule, prefs_map)
    return sched_df, unassigned_df, breakdown_df
