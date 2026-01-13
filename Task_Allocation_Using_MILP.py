from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import pandas as pd

try:
    import pulp
except Exception:
    print("This script requires 'pulp'. Install with: pip install pulp", file=sys.stderr)
    raise


# ----------------------- Required columns -----------------------
REQ_TASK_COLS = ["TaskID", "DurationDays", "ManHoursPerDay", "SkillRequired", "Predecessors"]
REQ_SITE_COLS = ["SiteID", "ManHoursPerDay", "SkillLevel"]


# ----------------------- Helpers -----------------------
def _flex_get_col(df: pd.DataFrame, canonical: str, alts: List[str]) -> str:
    """
    Ensure a required column is present. If not found, try flexible alternatives and rename.
    Returns the final column name (canonical).
    """
    if canonical in df.columns:
        return canonical
    lower_map = {c.lower(): c for c in df.columns}
    if canonical.lower() in lower_map:
        df.rename(columns={lower_map[canonical.lower()]: canonical}, inplace=True)
        return canonical
    for a in alts:
        if a in df.columns:
            df.rename(columns={a: canonical}, inplace=True)
            return canonical
        if a.lower() in lower_map:
            df.rename(columns={lower_map[a.lower()]: canonical}, inplace=True)
            return canonical
    raise ValueError(f"Missing required column '{canonical}' (tried alternatives {alts})")


def parse_preds(cell) -> List[str]:
    """
    Robustly parse a Predecessors cell:
      - Accepts comma, semicolon, or newline separators
      - Trims spaces and surrounding quotes
      - Returns [] for blanks/NaN
    Examples of valid cells:
      "T2"
      "T3,T4,T5"
      "T7;T8"
      "T10\\nT11"  (Excel/Sheets line breaks)
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip().strip('"').strip("'")
    if not s:
        return []
    # Normalize separators: CR/LF/semicolon -> comma
    s = s.replace("\r", "\n")
    s = s.replace(";", ",")
    s = s.replace("\n", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts


def topo_order(tasks: pd.DataFrame) -> List[str]:
    """
    Topological sort using Kahn's algorithm.
    Raises ValueError if a cycle is detected.
    """
    succ: Dict[str, List[str]] = defaultdict(list)
    indeg: Dict[str, int] = {tid: 0 for tid in tasks["TaskID"].astype(str).tolist()}

    for _, r in tasks.iterrows():
        tid = str(r["TaskID"])
        for p in parse_preds(r["Predecessors"]):
            succ[p].append(tid)
            indeg[tid] = indeg.get(tid, 0) + 1
            indeg.setdefault(p, indeg.get(p, 0))

    q = deque([t for t, d in indeg.items() if d == 0])
    order: List[str] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) != len(indeg):
        raise ValueError("Cycle detected in Predecessors; please fix the dependencies.")
    return order


# ------------------- Load & Validate -----------------
def load_tasks_sites(tasks_path: str, sites_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tasks = pd.read_csv(tasks_path).fillna("")
    sites = pd.read_csv(sites_path).fillna("")

    # Allow some flexible header spellings for robustness
    _flex_get_col(tasks, "TaskID", ["task_id", "task", "Task"])
    _flex_get_col(tasks, "DurationDays", ["duration", "Duration", "Duration_Days"])
    _flex_get_col(tasks, "ManHoursPerDay", ["manhoursperday", "ManHours", "DailyEffort", "DailyMH"])
    _flex_get_col(tasks, "SkillRequired", ["skillrequired", "Skill", "RequiredSkill"])
    _flex_get_col(tasks, "Predecessors", ["predecessors", "Predecessor", "predecessor", "Deps", "DependsOn"])

    _flex_get_col(sites, "SiteID", ["site_id", "Site"])
    _flex_get_col(sites, "ManHoursPerDay", ["capacity", "CapacityPerDay", "DailyCapacity", "mh_per_day"])
    _flex_get_col(sites, "SkillLevel", ["skilllevel", "Skill", "Level"])

    # Normalize dtypes
    tasks = tasks.copy()
    tasks["TaskID"] = tasks["TaskID"].astype(str)
    tasks["DurationDays"] = tasks["DurationDays"].astype(int)
    tasks["ManHoursPerDay"] = tasks["ManHoursPerDay"].astype(float)
    tasks["SkillRequired"] = tasks["SkillRequired"].astype(int)
    tasks["Predecessors"] = tasks["Predecessors"].astype(str)

    sites = sites.copy()
    sites["SiteID"] = sites["SiteID"].astype(str)
    sites["ManHoursPerDay"] = sites["ManHoursPerDay"].astype(float)
    sites["SkillLevel"] = sites["SkillLevel"].astype(int)

    # Validate required columns exist
    missing_t = [c for c in REQ_TASK_COLS if c not in tasks.columns]
    missing_s = [c for c in REQ_SITE_COLS if c not in sites.columns]
    if missing_t:
        raise ValueError(f"tasks.csv missing columns: {missing_t}")
    if missing_s:
        raise ValueError(f"sites.csv missing columns: {missing_s}")

    # Validate that all predecessors reference existing TaskIDs
    task_ids = set(tasks["TaskID"].astype(str))
    unknown: set[str] = set()
    for _, r in tasks.iterrows():
        for p in parse_preds(r["Predecessors"]):
            if p not in task_ids:
                unknown.add(p)
    if unknown:
        raise ValueError(
            f"Unknown predecessor TaskID(s) referenced: {sorted(unknown)}. "
            "Every predecessor must appear in the TaskID column."
        )

    return tasks, sites


# ------------------- Initial (pre-optimisation) assignment -----------------
def initial_assignment(tasks: pd.DataFrame, sites: pd.DataFrame, overtime_pct: float = 0.0) -> pd.DataFrame:
    """
    Simple 'before optimisation' assignment:
      - For each task, pick the FIRST site (by input order) whose capacity per day
        can accommodate the task's ManHoursPerDay (with optional overtime).
      - No global optimisation, just a greedy, constraint-respecting choice.
      - Objective (skill mismatch) is computed afterward.
    """
    cap = {row.SiteID: row.ManHoursPerDay * (1 + overtime_pct) for _, row in sites.iterrows()}
    site_skill = {row.SiteID: int(row.SkillLevel) for _, row in sites.iterrows()}

    rows = []
    for _, t in tasks.iterrows():
        tid = str(t.TaskID)
        t_mh = float(t.ManHoursPerDay)
        t_skill = int(t.SkillRequired)

        chosen_sid = None
        for _, s in sites.iterrows():
            sid = str(s.SiteID)
            if t_mh <= cap[sid]:
                chosen_sid = sid
                break

        if chosen_sid is None:
            raise ValueError(f"Task {tid} infeasible at all sites in initial (pre-optimisation) assignment.")

        mismatch = abs(t_skill - site_skill[chosen_sid])

        rows.append(
            {
                "TaskID": tid,
                "SiteID": chosen_sid,
                "SkillRequired": t_skill,
                "SiteSkill": site_skill[chosen_sid],
                "SkillMismatch": mismatch,
                "TaskManHoursPerDay": t_mh,
                "SiteCapacityPerDay": float(cap[chosen_sid]),
                "OvertimePctUsed": overtime_pct,
            }
        )

    return pd.DataFrame(rows).sort_values(["TaskID"]).reset_index(drop=True)


# ------------------- MILP Assignment (parametric overtime) -----------------
def assign(tasks: pd.DataFrame, sites: pd.DataFrame, overtime_pct: float = 0.0) -> pd.DataFrame:
    """
    Build and solve a small MILP:
      - Decision x_(task,site) ∈ {0,1}
      - Minimize total |SkillRequired - SkillLevel|
      - Each task assigned exactly once
      - Pair feasible only if task.ManHoursPerDay <= site capacity * (1 + overtime_pct)
    """
    cap = {row.SiteID: row.ManHoursPerDay * (1 + overtime_pct) for _, row in sites.iterrows()}
    site_skill = {row.SiteID: int(row.SkillLevel) for _, row in sites.iterrows()}

    feasible: List[Tuple[str, str]] = []
    cost: Dict[Tuple[str, str], float] = {}

    for _, t in tasks.iterrows():
        tid = str(t.TaskID)
        t_mh = float(t.ManHoursPerDay)
        t_skill = int(t.SkillRequired)
        for _, s in sites.iterrows():
            sid = str(s.SiteID)
            if t_mh <= cap[sid]:
                pair = (tid, sid)
                feasible.append(pair)
                cost[pair] = abs(t_skill - int(s.SkillLevel))

    if not feasible:
        raise ValueError("No feasible task→site pairs. Check capacities or increase --overtime.")

    model = pulp.LpProblem("TaskAssignment_SkillMismatch", pulp.LpMinimize)
    x = {p: pulp.LpVariable(f"x_{p[0]}_{p[1]}", 0, 1, cat=pulp.LpBinary) for p in feasible}

    # Objective: minimize total mismatch
    model += pulp.lpSum(cost[p] * x[p] for p in feasible)

    # Each task exactly once
    for tid in tasks["TaskID"].astype(str):
        cand = [p for p in feasible if p[0] == tid]
        if not cand:
            raise ValueError(f"Task {tid} infeasible at all sites under current settings.")
        model += pulp.lpSum(x[p] for p in cand) == 1

    # Solve
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        print(f"Warning: solver status = {status}", file=sys.stderr)

    # Extract
    rows = []
    for (t, s), var in x.items():
        val = var.value()
        if val is not None and val > 0.5:
            trow = tasks.loc[tasks["TaskID"].astype(str) == t].iloc[0]
            rows.append(
                {
                    "TaskID": t,
                    "SiteID": s,
                    "SkillRequired": int(trow.SkillRequired),
                    "SiteSkill": site_skill[s],
                    "SkillMismatch": abs(int(trow.SkillRequired) - site_skill[s]),
                    "TaskManHoursPerDay": float(trow.ManHoursPerDay),
                    "SiteCapacityPerDay": float(cap[s]),
                    "OvertimePctUsed": overtime_pct,
                }
            )
    return pd.DataFrame(rows).sort_values(["TaskID"]).reset_index(drop=True)


# ------------------- ASAP Schedule (unchanged) -------------------
def asap(tasks: pd.DataFrame) -> pd.DataFrame:
    """
    Precedence-only ASAP (no resource leveling). For each task:
      StartDay = max(FinishDay of predecessors)
      FinishDay = StartDay + DurationDays
    """
    order = topo_order(tasks)
    duration = {str(r.TaskID): int(r.DurationDays) for _, r in tasks.iterrows()}
    preds = {str(r.TaskID): parse_preds(r.Predecessors) for _, r in tasks.iterrows()}

    start: Dict[str, int] = {}
    finish: Dict[str, int] = {}

    for t in order:
        es = 0
        for p in preds[t]:
            es = max(es, finish[p])
        start[t] = es
        finish[t] = es + duration[t]

    out = pd.DataFrame(
        [{"TaskID": t, "StartDay": start[t], "FinishDay": finish[t], "DurationDays": duration[t]} for t in order]
    )
    return out.sort_values(["StartDay", "TaskID"]).reset_index(drop=True)


# ------------------- Comparison helpers -------------------
def total_mismatch(assign_df: pd.DataFrame) -> int:
    return int(assign_df["SkillMismatch"].sum())


def mean_mismatch(assign_df: pd.DataFrame) -> float:
    """
    Mean value of the objective function (average SkillMismatch per task).
    """
    return float(assign_df["SkillMismatch"].mean())


def compare_mismatch(baseline: pd.DataFrame, overtime: pd.DataFrame) -> pd.DataFrame:
    # Rename columns for clarity then outer merge on TaskID
    b = baseline.rename(columns={"SiteID": "BaselineSite", "SkillMismatch": "BaselineMismatch"})[
        ["TaskID", "BaselineSite", "BaselineMismatch"]
    ]
    o = overtime.rename(columns={"SiteID": "OvertimeSite", "SkillMismatch": "OvertimeMismatch"})[
        ["TaskID", "OvertimeSite", "OvertimeMismatch"]
    ]
    cmp_df = pd.merge(b, o, on="TaskID", how="outer")
    cmp_df["DeltaMismatch"] = cmp_df["OvertimeMismatch"] - cmp_df["BaselineMismatch"]
    cmp_df["AssignmentChanged"] = (cmp_df["BaselineSite"] != cmp_df["OvertimeSite"])
    return cmp_df.sort_values("TaskID").reset_index(drop=True)


# ----------------------- CLI main --------------------
def main():
    ap = argparse.ArgumentParser(description="Task allocation with before/after optimisation and overtime what-if.")
    ap.add_argument("--tasks", required=True, help="Path to tasks.csv")
    ap.add_argument("--sites", required=True, help="Path to sites.csv")
    ap.add_argument("--overtime", type=float, default=0.0, help="Planned overtime percentage for what-if, e.g., 0.2")
    ap.add_argument("--outdir", default=".", help="Where to write outputs")
    args = ap.parse_args()

    tasks, sites = load_tasks_sites(args.tasks, args.sites)

    # 1) BEFORE optimisation (naive assignment, no overtime)
    init_df = initial_assignment(tasks, sites, overtime_pct=0.0)
    init_total = total_mismatch(init_df)
    init_mean = mean_mismatch(init_df)

    # 2) AFTER optimisation (MILP, no overtime)
    base_df = assign(tasks, sites, overtime_pct=0.0)
    base_total = total_mismatch(base_df)
    base_mean = mean_mismatch(base_df)

    # 3) AFTER optimisation with overtime (what-if)
    if args.overtime > 0:
        ot_df = assign(tasks, sites, overtime_pct=args.overtime)
        ot_total = total_mismatch(ot_df)
        ot_mean = mean_mismatch(ot_df)
    else:
        ot_df = base_df.copy()
        ot_total = base_total
        ot_mean = base_mean

    # 4) ASAP schedule (precedence only, independent of resources)
    sched_df = asap(tasks)

    # 5) Comparison between optimised no-overtime and optimised overtime
    cmp_df = compare_mismatch(base_df, ot_df)
    delta_total_ot = ot_total - base_total

    # Console summary
    print("=== No-overtime scenario (before vs after optimisation) ===")
    print(f"Before optimisation (naive)  - total mismatch: {init_total}  (mean {init_mean:.3f})")
    print(
        f"After optimisation (MILP)    - total mismatch: {base_total}  "
        f"(mean {base_mean:.3f}, delta {base_total - init_total:+d})"
    )

    if args.overtime > 0:
        print("\n=== With overtime what-if (after optimisation) ===")
        print(f"Overtime = {args.overtime*100:.0f}%")
        print(f"Optimised total mismatch (OT): {ot_total}  (mean {ot_mean:.3f}, delta vs no-OT {delta_total_ot:+d})")
        changed = int(cmp_df["AssignmentChanged"].sum())
        print(f"Tasks that changed site due to overtime: {changed}/{len(cmp_df)}")

    # 6) Write files
    os.makedirs(args.outdir, exist_ok=True)
    init_path = os.path.join(args.outdir, "assignments_initial_before_optimisation.csv")
    base_path = os.path.join(args.outdir, "assignments_baseline_optimised_no_overtime.csv")
    ot_path = os.path.join(args.outdir, "assignments_with_overtime_optimised.csv")
    cmp_path = os.path.join(args.outdir, "mismatch_comparison_optimised_noOT_vs_OT.csv")
    sched_path = os.path.join(args.outdir, "schedule_asap.csv")

    init_df.to_csv(init_path, index=False)
    base_df.to_csv(base_path, index=False)
    ot_df.to_csv(ot_path, index=False)

    # Append totals (for optimised solutions) to the comparison file for quick reading
    totals_row = pd.DataFrame([{
        "TaskID": "TOTALS",
        "BaselineSite": "",
        "BaselineMismatch": base_total,
        "OvertimeSite": "",
        "OvertimeMismatch": ot_total,
        "DeltaMismatch": delta_total_ot,
        "AssignmentChanged": ""
    }])
    pd.concat([cmp_df, totals_row], ignore_index=True).to_csv(cmp_path, index=False)
    sched_df.to_csv(sched_path, index=False)

    print("\nSaved:")
    print(f"  {init_path}   (before optimisation)")
    print(f"  {base_path}   (after optimisation, no overtime)")
    print(f"  {ot_path}     (after optimisation, with overtime)")
    print(f"  {cmp_path}    (comparison of optimised no-OT vs OT)")
    print(f"  {sched_path}  (ASAP schedule)")


if __name__ == "__main__":
    main()
