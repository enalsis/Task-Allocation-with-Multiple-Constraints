## How to Run the Experiments

This repository provides **three executable Python scripts**, one for each task allocation method.
All methods take **task and site CSV files** as input and support an optional **planned overtime** parameter.



### 1️⃣ Run MILP-Based Task Allocation

This command runs the **Mixed-Integer Linear Programming (MILP)** model to compute a globally optimised task allocation under hard constraints.

```bash
python "Mixed-Integer Linear Programming (MILP).py" \
  --tasks tasks_fintech.csv \
  --sites sites_fintech.csv \
  --overtime 0.2
```

**Arguments:**

* `--tasks` : CSV file containing task definitions
* `--sites` : CSV file containing site/developer definitions
* `--overtime` : Planned overtime ratio (e.g., `0.2` = 20%)

---

### 2️⃣ Run Threshold-Driven Rule-Based Allocation (3WD)

This command executes the **threshold-driven heuristic (3WD)** allocation approach, which classifies tasks into *Accept*, *Defer*, or *Reject* categories.

```bash
python "Threshold-driven rule-based allocation (3WD).py" \
  --tasks tasks_fintech.csv \
  --sites sites_fintech.csv \
  --method threeway \
  --overtime 0.2 \
  --outdir results
```

**Arguments:**

* `--tasks` : Task dataset (CSV)
* `--sites` : Site/developer dataset (CSV)
* `--method` : Allocation strategy (`threeway` for Accept/Defer/Reject)
* `--overtime` : Planned overtime ratio
* `--outdir` : Directory where results will be saved

---

### 3️⃣ Run Hybrid Reinforcement Learning + Constraint Programming (RL–CP)

This command runs the **hybrid RL–CP model**, where reinforcement learning generates adaptive allocation preferences and constraint programming enforces feasibility.

```bash
python "Hybrid Reinforcement Learning + Constraint Programming (RL–CP).py" \
  --tasks tasks_fintech.csv \
  --sites sites_fintech.csv \
  --overtime 0.2 \
  --outdir out
```

**Arguments:**

* `--tasks` : Task dataset (CSV)
* `--sites` : Site/developer dataset (CSV)
* `--overtime` : Planned overtime ratio
* `--outdir` : Output directory for allocations and metrics

---

## Notes

* All methods use **identical input data** to ensure fair comparison
* CSV files must follow the task and site formats described in the paper
* Planned overtime can be set to `0.0` to disable overtime
* Output files include allocation results and evaluation metrics

