#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rl_cp_allocation.py

Hybrid Task Allocation:
  1) "RL-style" epsilon-greedy heuristic to produce an initial assignment (acts like the DQN stage).
  2) OR-Tools CP-SAT to enforce hard constraints (capacity, exactly-one assignment) and optimize skill mismatch.

Inputs (CSV):
  tasks: TaskID, DurationDays, ManHoursPerDay, SkillRequired, Predecessors(optional)
  sites: SiteID, ManHoursPerDay (capacity per day OR MaxCapacity), SkillLevel

Notes:
- To stay runnable in common environments, the RL stage is implemented as a lightweight epsilon-greedy
  bandit/QL-like policy (no TensorFlow dependency). CP-SAT performs the real optimization.
- If you want an actual DQN, you can plug one in and pass its proposed assignment as "hints" to CP-SAT.

How to run:

python rl_cp_allocation.py --tasks tasks.csv --sites sites.csv --outdir out

python rl_cp_allocation.py --tasks tasks.csv --sites sites.csv --overtime 0.2 --outdir out


"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

import pandas as pd

try:
    from ortools.sat.python import cp_model
except Exception as e:
    print("This script requires OR-Tools. Install with: pip install ortools", file=sys.stderr)
    raise


# ----------------------- Flexible column helpers -----------------------
def _flex_get_col(df: pd.DataFrame, canonical: str, alts: List[str]) -> str:
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
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip().strip('"').strip("'")
    if not s:
        return []
    s = s.replace("\r", "\n").replace(";", ",").replace("\n", ",")
    return [p.strip() for p in s.split(",") if p.strip()]


def topo_order(tasks: pd.DataFrame) -> List[str]:
    succ: Dict[str, List[str]] = defaultdict(list)
    indeg: Dict[str, int] = {tid: 0 for tid in tasks["TaskID"].astype(str).tolist()}

    for _, r in tasks.iterrows():
        tid = str(r["TaskID"])
        for p in parse_preds(r.get("Predecessors", "")):
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
        raise ValueError("Cycle detected in Predecessors; please fix task dependencies.")
    return order


def asap(tasks: pd.DataFrame) -> pd.DataFrame:
    if "Predecessors" not in tasks.columns:
        # If missing, treat as no deps
        tasks = tasks.copy()
        tasks["Predecessors"] = ""
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

    return pd.DataFrame(
        [{"TaskID": t, "StartDay": start[t], "FinishDay": finish[t], "DurationDays": duration[t]} for t in order]
    ).sort_values(["StartDay", "TaskID"]).reset_index(drop=True)


# ----------------------- Data model -----------------------
@dataclass(frozen=True)
class Task:
    tid: str
    skill: int
    workload: float  # total effort units used for capacity constraint (see below)


@dataclass(frozen=True)
class Dev:
    sid: str
    skill: int
    capacity: float  # total capacity units used for capacity constraint


def load_tasks_sites(tasks_path: str, sites_path: str, *, capacity_mode: str = "daily") -> Tuple[pd.DataFrame, pd.DataFrame]:
    tasks = pd.read_csv(tasks_path).fillna("")
    sites = pd.read_csv(sites_path).fillna("")

    # tasks
    _flex_get_col(tasks, "TaskID", ["task_id", "task", "Task", "ID", "Id", "id"])
    _flex_get_col(tasks, "SkillRequired", ["skillrequired", "Skill", "RequiredSkill", "Skill Level", "SkillRequiredLevel"])
    if "Workload" not in tasks.columns:
        _flex_get_col(tasks, "DurationDays", ["duration", "Duration", "Duration_Days", "Days"])
        _flex_get_col(tasks, "ManHoursPerDay", ["manhoursperday", "ManHours", "DailyEffort", "DailyMH", "MH/Day"])
    else:
        _flex_get_col(tasks, "Workload", ["TotalWorkload", "WorkloadHours", "Effort"])

    # predecessors optional
    if "Predecessors" not in tasks.columns:
        # keep optional but support common alternates
        try:
            _flex_get_col(tasks, "Predecessors", ["predecessors", "Predecessor", "Deps", "DependsOn", "Dependencies"])
        except Exception:
            tasks["Predecessors"] = ""

    # sites
    _flex_get_col(sites, "SiteID", ["site_id", "Site", "DeveloperID", "DevID", "ID", "Id", "id"])
    _flex_get_col(sites, "SkillLevel", ["skilllevel", "Skill", "Level", "Skill level"])
    if "MaxCapacity" not in sites.columns:
        _flex_get_col(sites, "ManHoursPerDay", ["capacity", "CapacityPerDay", "DailyCapacity", "mh_per_day", "DailyManHours"])
    else:
        _flex_get_col(sites, "MaxCapacity", ["Capacity", "Max Workload", "TotalCapacity"])

    # Normalize dtypes
    tasks = tasks.copy()
    tasks["TaskID"] = tasks["TaskID"].astype(str)
    tasks["SkillRequired"] = tasks["SkillRequired"].astype(int)

    if "Workload" in tasks.columns:
        tasks["Workload"] = tasks["Workload"].astype(float)
    else:
        tasks["DurationDays"] = tasks["DurationDays"].astype(int)
        tasks["ManHoursPerDay"] = tasks["ManHoursPerDay"].astype(float)

    tasks["Predecessors"] = tasks["Predecessors"].astype(str)

    sites = sites.copy()
    sites["SiteID"] = sites["SiteID"].astype(str)
    sites["SkillLevel"] = sites["SkillLevel"].astype(int)
    if "MaxCapacity" in sites.columns:
        sites["MaxCapacity"] = sites["MaxCapacity"].astype(float)
    else:
        sites["ManHoursPerDay"] = sites["ManHoursPerDay"].astype(float)

    # Validate predecessors reference existing tasks
    task_ids = set(tasks["TaskID"].astype(str))
    unknown = set()
    for _, r in tasks.iterrows():
        for p in parse_preds(r["Predecessors"]):
            if p and p not in task_ids:
                unknown.add(p)
    if unknown:
        raise ValueError(f"Unknown predecessor TaskID(s): {sorted(unknown)}")

    # capacity_mode (daily vs total) only affects how we compute Task.workload and Dev.capacity later
    if capacity_mode not in ("daily", "total"):
        raise ValueError("--capacity-mode must be 'daily' or 'total'")

    return tasks, sites


def build_objects(tasks_df: pd.DataFrame, sites_df: pd.DataFrame, *, overtime_pct: float = 0.0, capacity_mode: str = "daily") -> Tuple[List[Task], List[Dev]]:
    # Workload definition:
    # - If tasks have explicit Workload: use that as total effort.
    # - Else: use DurationDays * ManHoursPerDay as total effort (makes CP meaningful).
    if "Workload" in tasks_df.columns:
        t_work = tasks_df["Workload"].astype(float).tolist()
    else:
        t_work = (tasks_df["DurationDays"].astype(float) * tasks_df["ManHoursPerDay"].astype(float)).tolist()

    tasks = [
        Task(
            tid=str(tasks_df.loc[i, "TaskID"]),
            skill=int(tasks_df.loc[i, "SkillRequired"]),
            workload=float(t_work[i]),
        )
        for i in range(len(tasks_df))
    ]

    # Capacity definition:
    # - If MaxCapacity exists: treat as total capacity over planning horizon.
    # - Else: treat ManHoursPerDay as daily capacity. If capacity_mode='total', we approximate a horizon
    #   by summing task durations' max (ASAP makespan) and multiply daily capacity by that horizon.
    if "MaxCapacity" in sites_df.columns:
        caps = sites_df["MaxCapacity"].astype(float).tolist()
    else:
        daily = sites_df["ManHoursPerDay"].astype(float).tolist()
        if capacity_mode == "daily":
            # Interpret "capacity" as total capacity proxy per day; to compare with total workloads,
            # we convert it to total capacity using a conservative horizon: sum of task durations.
            horizon = float(tasks_df["DurationDays"].astype(float).sum()) if "DurationDays" in tasks_df.columns else float(len(tasks))
            caps = [d * horizon for d in daily]
        else:
            # total with an ASAP-based horizon if possible
            try:
                sched = asap(tasks_df)
                horizon = float(sched["FinishDay"].max()) if len(sched) else 1.0
            except Exception:
                horizon = float(tasks_df["DurationDays"].astype(float).sum()) if "DurationDays" in tasks_df.columns else float(len(tasks))
            caps = [d * horizon for d in daily]

    devs = [
        Dev(
            sid=str(sites_df.loc[i, "SiteID"]),
            skill=int(sites_df.loc[i, "SkillLevel"]),
            capacity=float(caps[i]) * (1.0 + overtime_pct),
        )
        for i in range(len(sites_df))
    ]
    return tasks, devs


# ----------------------- Stage 1: "RL-style" heuristic -----------------------
def rl_style_assignment(tasks: List[Task], devs: List[Dev], *, epsilon: float = 0.10, seed: int = 7) -> Dict[str, str]:
    """
    Lightweight epsilon-greedy policy as a practical stand-in for the notebook's DQN step.
    It builds an assignment sequentially, trying to minimize (skill mismatch + overload penalty),
    while occasionally exploring random feasible devs.

    Returns mapping TaskID -> SiteID.
    """
    rng = random.Random(seed)
    remaining = {d.sid: d.capacity for d in devs}
    dev_skill = {d.sid: d.skill for d in devs}

    assignment: Dict[str, str] = {}
    for t in tasks:
        # Feasible devs by remaining capacity (soft in this stage; we try to respect it)
        feasible = [d.sid for d in devs if remaining[d.sid] >= t.workload]
        if not feasible:
            # If none feasible, we still pick the least-bad dev (will be fixed by CP or trigger infeasibility)
            feasible = [d.sid for d in devs]

        explore = rng.random() < epsilon
        if explore:
            chosen = rng.choice(feasible)
        else:
            def score(sid: str) -> float:
                mismatch = abs(t.skill - dev_skill[sid])
                overload = max(0.0, t.workload - remaining[sid])
                return mismatch + 1000.0 * (overload > 0) + overload  # heavy penalty if overload
            chosen = min(feasible, key=score)

        assignment[t.tid] = chosen
        remaining[chosen] -= t.workload

    return assignment


# ----------------------- Stage 2: CP-SAT feasibility + optimization -----------------------
def cp_optimize_assignment(
    tasks: List[Task],
    devs: List[Dev],
    *,
    hint_assignment: Optional[Dict[str, str]] = None,
    time_limit_s: int = 10,
) -> Tuple[pd.DataFrame, str]:
    """
    CP-SAT:
      - x[t,d] in {0,1}
      - Each task assigned exactly once
      - Sum(workload[t] * x[t,d]) <= capacity[d]
      - Minimize total skill mismatch

    Optionally accepts hint_assignment (TaskID->SiteID) as a warm start.
    """
    model = cp_model.CpModel()

    T = list(range(len(tasks)))
    D = list(range(len(devs)))

    # Index maps
    t_index = {tasks[i].tid: i for i in T}
    d_index = {devs[j].sid: j for j in D}

    x: Dict[Tuple[int, int], cp_model.IntVar] = {}
    for i in T:
        for j in D:
            x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")

    # Each task exactly one dev
    for i in T:
        model.Add(sum(x[(i, j)] for j in D) == 1)

    # Capacity for each dev
    for j in D:
        model.Add(sum(int(round(tasks[i].workload * 1000)) * x[(i, j)] for i in T) <= int(round(devs[j].capacity * 1000)))

    # Objective: minimize total mismatch
    obj_terms = []
    for i in T:
        for j in D:
            mismatch = abs(tasks[i].skill - devs[j].skill)
            obj_terms.append(mismatch * x[(i, j)])
    model.Minimize(sum(obj_terms))

    # Warm start / hints
    if hint_assignment:
        for tid, sid in hint_assignment.items():
            if tid in t_index and sid in d_index:
                model.AddHint(x[(t_index[tid], d_index[sid])], 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status_name not in ("OPTIMAL", "FEASIBLE"):
        raise ValueError(f"CP-SAT failed: {status_name}. Likely infeasible capacities; try --overtime or check inputs.")

    rows = []
    for i in T:
        chosen_j = None
        for j in D:
            if solver.Value(x[(i, j)]) == 1:
                chosen_j = j
                break
        assert chosen_j is not None
        rows.append(
            {
                "TaskID": tasks[i].tid,
                "SiteID": devs[chosen_j].sid,
                "SkillRequired": tasks[i].skill,
                "SiteSkill": devs[chosen_j].skill,
                "SkillMismatch": abs(tasks[i].skill - devs[chosen_j].skill),
                "TaskWorkload": tasks[i].workload,
                "SiteCapacity": devs[chosen_j].capacity,
            }
        )

    return pd.DataFrame(rows).sort_values(["TaskID"]).reset_index(drop=True), status_name


def total_mismatch(df: pd.DataFrame) -> int:
    return int(df["SkillMismatch"].sum())


# ----------------------- CLI -----------------------
def main():
    ap = argparse.ArgumentParser(description="Hybrid RL-style + CP-SAT task allocation.")
    ap.add_argument("--tasks", required=True, help="Path to tasks.csv")
    ap.add_argument("--sites", required=True, help="Path to sites.csv")
    ap.add_argument("--overtime", type=float, default=0.0, help="Overtime percent (e.g., 0.2 for +20% capacity).")
    ap.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate for RL-style stage.")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for RL-style stage.")
    ap.add_argument("--cp-time", type=int, default=10, help="CP-SAT time limit (seconds).")
    ap.add_argument("--capacity-mode", choices=["daily", "total"], default="total",
                    help="How to convert daily ManHoursPerDay into total capacity if MaxCapacity is not provided.")
    ap.add_argument("--outdir", default=".", help="Output directory for CSVs.")
    args = ap.parse_args()

    tasks_df, sites_df = load_tasks_sites(args.tasks, args.sites, capacity_mode=args.capacity_mode)

    # Build objects (no overtime for RL baseline)
    tasks0, devs0 = build_objects(tasks_df, sites_df, overtime_pct=0.0, capacity_mode=args.capacity_mode)

    # Stage 1: RL-style heuristic (before optimization)
    rl_assign = rl_style_assignment(tasks0, devs0, epsilon=args.epsilon, seed=args.seed)
    rl_df, _ = cp_optimize_assignment(tasks0, devs0, hint_assignment=rl_assign, time_limit_s=max(1, args.cp_time))
    # Note: We run CP once even for the "before" report to ensure feasibility; otherwise RL could violate hard caps.
    # To mirror "before optimization" from your other scripts, we also output the raw RL choice (may be infeasible).

    # Raw RL report (no hard enforcement)
    raw_rows = []
    dev_map0 = {d.sid: d for d in devs0}
    for t in tasks0:
        sid = rl_assign.get(t.tid)
        d = dev_map0[sid]
        raw_rows.append({
            "TaskID": t.tid,
            "SiteID": sid,
            "SkillRequired": t.skill,
            "SiteSkill": d.skill,
            "SkillMismatch": abs(t.skill - d.skill),
            "TaskWorkload": t.workload,
            "SiteCapacity": d.capacity,
            "Decision": "ProposedByRL"
        })
    raw_rl_df = pd.DataFrame(raw_rows).sort_values(["TaskID"]).reset_index(drop=True)

    # Stage 2: CP-SAT optimized (no overtime)
    cp_df, cp_status = cp_optimize_assignment(tasks0, devs0, hint_assignment=rl_assign, time_limit_s=args.cp_time)

    # Stage 3: CP-SAT optimized with overtime (what-if)
    if args.overtime > 0:
        tasks_ot, devs_ot = build_objects(tasks_df, sites_df, overtime_pct=args.overtime, capacity_mode=args.capacity_mode)
        # reuse RL hint (still valid ids)
        cp_ot_df, cp_ot_status = cp_optimize_assignment(tasks_ot, devs_ot, hint_assignment=rl_assign, time_limit_s=args.cp_time)
    else:
        cp_ot_df, cp_ot_status = cp_df.copy(), cp_status

    # Schedule (ASAP, precedence only)
    sched_df = asap(tasks_df)

    # Write outputs
    os.makedirs(args.outdir, exist_ok=True)
    p_raw = os.path.join(args.outdir, "assignments_stage1_rl_proposed.csv")
    p_cp = os.path.join(args.outdir, "assignments_stage2_cp_optimised_no_overtime.csv")
    p_ot = os.path.join(args.outdir, "assignments_stage3_cp_optimised_with_overtime.csv")
    p_sched = os.path.join(args.outdir, "schedule_asap.csv")

    raw_rl_df.to_csv(p_raw, index=False)
    cp_df.to_csv(p_cp, index=False)
    cp_ot_df.to_csv(p_ot, index=False)
    sched_df.to_csv(p_sched, index=False)

    # Console summary
    print("=== Hybrid RL-style + CP-SAT Task Allocation ===")
    print(f"Tasks: {len(tasks0)}  Sites: {len(devs0)}")
    print(f"Stage 1 (RL-style proposal) saved: {p_raw}")
    print(f"Stage 2 (CP-SAT optimized, no OT) status={cp_status} total_mismatch={total_mismatch(cp_df)} saved: {p_cp}")
    if args.overtime > 0:
        print(f"Stage 3 (CP-SAT optimized, OT={args.overtime*100:.0f}%) status={cp_ot_status} total_mismatch={total_mismatch(cp_ot_df)} saved: {p_ot}")
    else:
        print(f"Stage 3 (CP-SAT optimized, OT=0%) reused stage 2 saved: {p_ot}")
    print(f"ASAP schedule saved: {p_sched}")


if __name__ == "__main__":
    import random
    main()
