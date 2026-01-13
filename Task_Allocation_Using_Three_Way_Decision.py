import pandas as pd
from typing import List, Dict, Tuple

# Thresholds for decision
MAX_MISMATCH_THRESHOLD = 2  # Accept if mismatch <= 2
DEFER_MISMATCH_THRESHOLD = 4  # Defer if mismatch <= 4
OVERTIME_FACTOR = 0.2  # 20% planned overtime

def parse_preds(cell) -> List[str]:
    if pd.isna(cell):
        return []
    s = str(cell).replace(";", ",").replace("\n", ",")
    return [p.strip() for p in s.split(",") if p.strip()]

def twd_assignment(tasks: pd.DataFrame, sites: pd.DataFrame, use_overtime: bool = False) -> pd.DataFrame:
    accepted, deferred, rejected = [], [], []

    for _, t in tasks.iterrows():
        task_id = t["TaskID"]
        required_skill = int(t["SkillRequired"])
        mh_per_day = float(t["ManHoursPerDay"])

        best_match = None
        best_mismatch = float("inf")

        for _, s in sites.iterrows():
            site_id = s["SiteID"]
            site_skill = int(s["SkillLevel"])
            site_capacity = float(s["ManHoursPerDay"])

            if use_overtime:
                site_capacity *= (1 + OVERTIME_FACTOR)

            if mh_per_day <= site_capacity:
                mismatch = abs(required_skill - site_skill)
                if mismatch < best_mismatch:
                    best_mismatch = mismatch
                    best_match = (task_id, site_id, mismatch)

        if best_match:
            if best_mismatch <= MAX_MISMATCH_THRESHOLD:
                accepted.append(best_match)
            elif best_mismatch <= DEFER_MISMATCH_THRESHOLD:
                deferred.append((task_id, best_match[1], best_mismatch))
            else:
                rejected.append((task_id, None, best_mismatch))
        else:
            rejected.append((task_id, None, None))

    results = []
    for task_id, site_id, mismatch in accepted:
        results.append({"TaskID": task_id, "SiteID": site_id, "SkillMismatch": mismatch, "Decision": "Accept"})
    for task_id, site_id, mismatch in deferred:
        results.append({"TaskID": task_id, "SiteID": site_id, "SkillMismatch": mismatch, "Decision": "Defer"})
    for task_id, site_id, mismatch in rejected:
        results.append({"TaskID": task_id, "SiteID": site_id, "SkillMismatch": mismatch, "Decision": "Reject"})

    return pd.DataFrame(results)

if __name__ == "__main__":
    tasks_df = pd.read_csv("STLISMS.csv")
    sites_df = pd.read_csv("sites.csv")

    # Step 1: Before Optimization (Baseline Feasibility Only)
    baseline_df = twd_assignment(tasks_df, sites_df, use_overtime=False)
    baseline_df.to_csv("assignments_before_optimization.csv", index=False)

    # Step 2: After Optimization (same as best-case assignment for TWD logic)
    optimized_df = twd_assignment(tasks_df, sites_df, use_overtime=False)
    optimized_df.to_csv("assignments_after_optimization.csv", index=False)

    # Step 3: After Adding Overtime Constraint
    overtime_df = twd_assignment(tasks_df, sites_df, use_overtime=True)
    overtime_df.to_csv("assignments_with_overtime.csv", index=False)

    print("Assignment reports generated:")
    print("- assignments_before_optimization.csv")
    print("- assignments_after_optimization.csv")
    print("- assignments_with_overtime.csv")