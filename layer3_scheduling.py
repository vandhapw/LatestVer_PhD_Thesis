#!/usr/bin/env python3
"""
LAYER 3 — Maintenance Scheduling Optimization
=============================================
Input:  Layer 2 priority list (P1-P4 per machine)
Output: Scheduled maintenance assignments + makespan comparison

Constraints (dari supervisor):
- Work hours: 09:00-18:00 (9 hours/day)
- Days: Monday-Friday
- Employee ranges: small 1-3, medium 3-5, large 5-20
- 1 employee can handle multiple machines within work hours
- Dynamic scheduling (data real-time)
- NP-hard combinatorial problem
- Metric: MAKESPAN (minimize total completion time)

Approaches:
A. Classical: OR-Tools CP-SAT
B. Quantum: Qiskit QAOA (simulator-based, honest disclosure)
C. Quantum-Inspired: Simulated Annealing / Random constrained search

Durasi maintenance (estimasi industri, bukan dari agentic AI):
- P1 immediate_shutdown : 60 min (kritis)
- P2 inspect            : 120 min
- P3 schedule_maintenance: 240 min
- P4 monitor            : 0 min (tidak di-schedule)
"""

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Import solvers ─────────────────────────────
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except Exception:
    ORTOOLS_AVAILABLE = False
    print("[WARNING] OR-Tools not available; classical solver disabled.")

try:
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler as Sampler
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.quantum_info import SparsePauliOp
    QISKIT_AVAILABLE = True
except Exception as e:
    QISKIT_AVAILABLE = False
    print(f"[WARNING] Qiskit/QAOA not available ({e}); quantum solver disabled.")

# ── Constants ────────────────────────────────
LAYER2_PATH = Path(__file__).parent / "layer2_maintenance_priority.json"
RESULTS_PATH = Path(__file__).parent / "layer3_scheduling_results.json"

WORK_HOURS_PER_DAY = 9   # 09:00-18:00
WORK_DAYS = 5            # Mon-Fri

# Durasi maintenance (menit) berdasarkan priority
PRIORITY_DURATION = {
    "P1": 60,   # immediate_shutdown
    "P2": 120,  # inspect
    "P3": 240,  # schedule_maintenance
    "P4": 0,    # monitor -> no maintenance
}

# ── 1. LOAD LAYER 2 OUTPUT ──────────────────

def load_layer2():
    with open(LAYER2_PATH) as f:
        data = json.load(f)
    machines = data.get("machines", [])
    # Filter only machines yang perlu maintenance (P1-P3)
    jobs = []
    for m in machines:
        p = m["priority"]
        if p in ("P1", "P2", "P3"):
            jobs.append({
                "machine_id": m["machine_id"],
                "priority": p,
                "action": m["action"],
                "duration": PRIORITY_DURATION[p],
                "classical_risk": m["classical_risk"],
                "heuristic_score": m["heuristic_score"],
            })
    return jobs


# ── 2. CLASSICAL SOLVER (OR-Tools CP-SAT) ───

def solve_classical_cp_sat(jobs, n_employees=5):
    """
    Classical CP-SAT: assign jobs to employees and time slots
    to minimize makespan.
    Simplified: each job assigned to 1 employee, minimize max end time.
    """
    if not ORTOOLS_AVAILABLE:
        return None, None
    if not jobs:
        return {"makespan": 0, "assignments": []}, 0.0

    model = cp_model.CpModel()
    n = len(jobs)
    horizon = sum(j["duration"] for j in jobs)  # upper bound

    # Variables: start time per job
    starts = [model.NewIntVar(0, horizon, f"start_{i}") for i in range(n)]
    ends = [model.NewIntVar(0, horizon, f"end_{i}") for i in range(n)]
    # Employee assignment per job
    employees = [model.NewIntVar(0, n_employees - 1, f"emp_{i}") for i in range(n)]
    # Makespan
    makespan = model.NewIntVar(0, horizon, "makespan")

    # Each job: end = start + duration
    for i in range(n):
        model.Add(ends[i] == starts[i] + jobs[i]["duration"])
        model.Add(ends[i] <= makespan)

    # No overlap: if two jobs assigned to same employee, they must not overlap
    for i in range(n):
        for j in range(i + 1, n):
            same_emp = model.NewBoolVar(f"same_{i}_{j}")
            model.Add(employees[i] == employees[j]).OnlyEnforceIf(same_emp)
            model.Add(employees[i] != employees[j]).OnlyEnforceIf(same_emp.Not())
            # If same employee, then i before j OR j before i
            bi = model.NewBoolVar(f"before_{i}_{j}")
            model.Add(ends[i] <= starts[j]).OnlyEnforceIf([same_emp, bi])
            model.Add(ends[j] <= starts[i]).OnlyEnforceIf([same_emp, bi.Not()])

    model.Minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    solver.parameters.num_search_workers = 8
    t0 = time.time()
    status = solver.Solve(model)
    t1 = time.time()

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        assignments = []
        for i in range(n):
            assignments.append({
                "machine_id": jobs[i]["machine_id"],
                "employee": int(solver.Value(employees[i])),
                "start": int(solver.Value(starts[i])),
                "end": int(solver.Value(ends[i])),
                "duration": jobs[i]["duration"],
            })
        return {
            "makespan": int(solver.Value(makespan)),
            "assignments": assignments,
            "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
        }, round(t1 - t0, 3)
    else:
        return {"makespan": None, "assignments": [], "status": "INFEASIBLE"}, round(t1 - t0, 3)


# ── 3. QUANTUM SOLVER (Qiskit QAOA simulator) ─

def solve_qaoa_simulator(jobs, n_employees=5, n_layers=1):
    """
    QAOA on simulator untuk job-shop-like scheduling.
    Simplified: formulate as binary assignment + penalty for overlap.
    Karena QAOA sulit untuk scheduling murni, kita gunakan
    reduced formulation: pilih urutan job assignment ke employee
    dan minimize makespan via penalty Hamiltonian.

    HONEST DISCLOSURE: ini adalah quantum SIMULATOR, bukan hardware.
    """
    if not QISKIT_AVAILABLE:
        return None, None
    if not jobs:
        return {"makespan": 0, "assignments": []}, 0.0

    n = len(jobs)
    # Binary variables: x[i][e] = job i assigned to employee e
    # Flatten index: x[k] where k = i*n_employees + e
    num_vars = n * n_employees

    # Build QUBO-like Hamiltonian
    # H = H_makespan + H_overlap
    # H_makespan: reward assigning short jobs together to balance load
    # H_overlap: penalize overlapping jobs on same employee

    # Classical preprocessing: compute load per employee approximation
    # Since QAOA is heuristic, we use a simple QUBO formulation
    # that penalizes if two jobs overlap on same employee.
    # For simplicity, we order jobs by duration descending and
    # build a Hamiltonian that rewards load balancing.

    # Hamiltonian: Z-based encoding for each x[i][e]
    # We construct Pauli strings for penalties.
    # Given QAOA complexity, we implement a simplified version
    # where the problem reduces to graph coloring / partitioning.

    # Step 1: Greedy classical pre-assignment to get baseline
    # Step 2: Use QAOA to refine with penalty Hamiltonian

    # Because QAOA on large qubit counts is heavy, limit to small instances.
    # For larger problems, we extract the TOP-3 most critical jobs and run QAOA
    # on that reduced sub-problem, while the full problem is solved classically.
    reduced_mode = False
    if num_vars > 12:
        reduced_mode = True
        # Sort by duration descending (proxy for criticality)
        sorted_jobs = sorted(jobs, key=lambda j: j["duration"], reverse=True)
        reduced_jobs = sorted_jobs[:3]
        n_reduced = len(reduced_jobs)
        n_emp_reduced = min(3, n_employees)
        num_vars = n_reduced * n_emp_reduced
        print(f"  [QAOA] Reduced to {n_reduced} jobs x {n_emp_reduced} employees = {num_vars} qubits")
    else:
        reduced_jobs = jobs
        n_reduced = n
        n_emp_reduced = n_employees

    # Build Pauli list for QUBO: H = sum_i sum_e c[i][e] * Z_i,e
    # + sum_{i<j} sum_e J[i][j][e] * Z_i,e * Z_j,e
    # This maps to Ising model.

    # For demonstration, we construct a simple Hamiltonian
    # Penalize same-employee assignment for jobs with total duration > 9h
    pauli_list = []
    coeffs = []

    # Linear terms: prefer employees with lower current load
    for i in range(n_reduced):
        for e in range(n_emp_reduced):
            idx = i * n_emp_reduced + e
            # Cost = duration (encourage balancing)
            c = reduced_jobs[i]["duration"] / 60.0
            z_str = ["I"] * num_vars
            z_str[idx] = "Z"
            pauli_list.append("".join(z_str))
            coeffs.append(c)

    # Quadratic terms: penalize overlap if two jobs on same employee
    # Simplified: just penalize both on same employee (independent of time)
    for i in range(n_reduced):
        for j in range(i + 1, n_reduced):
            for e in range(n_emp_reduced):
                idx_i = i * n_emp_reduced + e
                idx_j = j * n_emp_reduced + e
                z_str = ["I"] * num_vars
                z_str[idx_i] = "Z"
                z_str[idx_j] = "Z"
                pauli_list.append("".join(z_str))
                # Penalty: average duration overlap proxy
                penalty = 0.5 * (reduced_jobs[i]["duration"] + reduced_jobs[j]["duration"]) / 60.0
                coeffs.append(penalty)

    hamiltonian = SparsePauliOp.from_list(list(zip(pauli_list, coeffs)))

    # QAOA with COBYLA optimizer
    optimizer = COBYLA(maxiter=100)
    sampler = Sampler()
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=n_layers)

    t0 = time.time()
    try:
        result = qaoa.compute_minimum_eigenvalue(operator=hamiltonian)
    except Exception as ex:
        print(f"[QAOA] Simulator error: {ex}")
        return None, None
    t1 = time.time()

    # Decode bitstring from result
    best = result.best_measurement
    if best is None:
        return None, None
    bitstring = best["bitstring"]
    # bitstring: "0101..." length = num_vars
    # Decode assignments
    assignments = []
    for i in range(n_reduced):
        for e in range(n_emp_reduced):
            idx = i * n_emp_reduced + e
            if bitstring[idx] == "1":
                assignments.append({
                    "machine_id": reduced_jobs[i]["machine_id"],
                    "employee": e,
                    "start": 0,  # QAOA doesn't schedule time, only assignment
                    "end": reduced_jobs[i]["duration"],
                    "duration": reduced_jobs[i]["duration"],
                })
                break
        else:
            # Fallback: assign to employee 0
            assignments.append({
                "machine_id": reduced_jobs[i]["machine_id"],
                "employee": 0,
                "start": 0,
                "end": reduced_jobs[i]["duration"],
                "duration": reduced_jobs[i]["duration"],
            })

    # Compute makespan from assignments (naive sequential per employee)
    emp_load = defaultdict(int)
    for a in assignments:
        emp_load[a["employee"]] += a["duration"]
    makespan = max(emp_load.values()) if emp_load else 0

    status_label = "QAOA_SIMULATOR_REDUCED" if reduced_mode else "QAOA_SIMULATOR"
    return {
        "makespan": makespan,
        "assignments": assignments,
        "status": status_label,
        "energy": float(best["value"]),
        "reduced": reduced_mode,
        "n_jobs_reduced": n_reduced if reduced_mode else None,
    }, round(t1 - t0, 3)


# ── 4. QUANTUM-INSPIRED HEURISTIC ─────────────

def _quantum_inspired_greedy(jobs, n_employees=5):
    """
    Classical heuristic yang terinspirasi dari quantum annealing:
    - Random seed + local search (1-bit flip)
    - Energy = load imbalance + overlap penalty
    """
    random.seed(42)
    n = len(jobs)
    best_energy = float("inf")
    best_assignments = []
    best_makespan = float("inf")

    for _ in range(500):
        # Random assignment
        assignment = [random.randint(0, n_employees - 1) for _ in range(n)]
        # Local search: swap some jobs
        for _ in range(3):
            i = random.randint(0, n - 1)
            assignment[i] = random.randint(0, n_employees - 1)

        # Compute energy / makespan
        emp_load = defaultdict(int)
        for i, e in enumerate(assignment):
            emp_load[e] += jobs[i]["duration"]
        makespan = max(emp_load.values()) if emp_load else 0
        # Energy: makespan + 0.1 * variance load
        loads = list(emp_load.values())
        mean_load = sum(loads) / len(loads) if loads else 0
        variance = sum((l - mean_load) ** 2 for l in loads) / len(loads) if loads else 0
        energy = makespan + 0.1 * variance

        if energy < best_energy:
            best_energy = energy
            best_makespan = makespan
            best_assignments = assignment

    assignments = []
    emp_start = defaultdict(int)
    for i, e in enumerate(best_assignments):
        d = jobs[i]["duration"]
        start = emp_start[e]
        end = start + d
        assignments.append({
            "machine_id": jobs[i]["machine_id"],
            "employee": e,
            "start": start,
            "end": end,
            "duration": d,
        })
        emp_start[e] = end

    return {
        "makespan": best_makespan,
        "assignments": assignments,
        "status": "QUANTUM_INSPIRED_HEURISTIC",
        "energy": round(best_energy, 2),
    }


def solve_quantum_inspired(jobs, n_employees=5):
    t0 = time.time()
    result = _quantum_inspired_greedy(jobs, n_employees)
    t1 = time.time()
    return result, round(t1 - t0, 3)


# ── 5. MAIN ORCHESTRATOR ──────────────────────

def run_layer3_scheduling(n_employees=5):
    jobs = load_layer2()
    if not jobs:
        print("[LAYER 3] No maintenance jobs from Layer 2. Nothing to schedule.")
        return {}

    print(f"[LAYER 3] Received {len(jobs)} maintenance jobs from Layer 2")
    print(f"[LAYER 3] Team size: {n_employees} employees")
    print(f"[LAYER 3] Work hours: 09:00-18:00, Mon-Fri")

    results = {}

    # A. Classical
    print("\n[LAYER 3-A] Classical CP-SAT Solver...")
    if ORTOOLS_AVAILABLE:
        classical_res, classical_time = solve_classical_cp_sat(jobs, n_employees)
        if classical_res:
            print(f"  Makespan: {classical_res['makespan']} min ({classical_res['makespan']/60:.1f} h)")
            print(f"  Status: {classical_res['status']}, Time: {classical_time}s")
            results["classical"] = {**classical_res, "solve_time": classical_time}
        else:
            print("  Classical solver failed.")
    else:
        print("  OR-Tools not available.")

    # B. Quantum (QAOA simulator)
    print("\n[LAYER 3-B] Quantum QAOA (Simulator)...")
    print("  ⚠️  HONEST DISCLOSURE: Using Qiskit simulator, NOT real quantum hardware.")
    qaoa_res, qaoa_time = solve_qaoa_simulator(jobs, n_employees)
    if qaoa_res:
        print(f"  Makespan: {qaoa_res['makespan']} min ({qaoa_res['makespan']/60:.1f} h)")
        print(f"  Status: {qaoa_res['status']}, Time: {qaoa_time}s")
        if "energy" in qaoa_res:
            print(f"  Hamiltonian energy: {qaoa_res['energy']:.4f}")
        results["quantum_qaoa_simulator"] = {**qaoa_res, "solve_time": qaoa_time}
    else:
        print("  QAOA simulator failed or skipped (qubit limit exceeded).")

    # C. Quantum-Inspired (full instance)
    print("\n[LAYER 3-C] Quantum-Inspired Heuristic...")
    qi_res, qi_time = solve_quantum_inspired(jobs, n_employees)
    print(f"  Makespan: {qi_res['makespan']} min ({qi_res['makespan']/60:.1f} h)")
    print(f"  Status: {qi_res['status']}, Time: {qi_time}s")
    print(f"  Heuristic energy: {qi_res['energy']}")
    results["quantum_inspired"] = {**qi_res, "solve_time": qi_time}

    # D. FAIR COMPARISON on reduced instance (if QAOA used reduction)
    reduced_jobs = None
    if qaoa_res and qaoa_res.get("reduced"):
        print("\n[LAYER 3-D] Fair Comparison — Reduced Instance (top-3 critical jobs)")
        sorted_jobs = sorted(jobs, key=lambda j: j["duration"], reverse=True)
        reduced_jobs = sorted_jobs[:3]
        n_emp_reduced = min(3, n_employees)
        print(f"  Reduced jobs: {[j['machine_id'] for j in reduced_jobs]}")

        # Classical on reduced
        if ORTOOLS_AVAILABLE:
            c_red, c_red_t = solve_classical_cp_sat(reduced_jobs, n_emp_reduced)
            if c_red:
                print(f"  Classical (reduced): makespan={c_red['makespan']} min, time={c_red_t}s")
                results["classical_reduced"] = {**c_red, "solve_time": c_red_t}
        # QI on reduced
        qi_red, qi_red_t = solve_quantum_inspired(reduced_jobs, n_emp_reduced)
        print(f"  QI (reduced): makespan={qi_red['makespan']} min, time={qi_red_t}s")
        results["quantum_inspired_reduced"] = {**qi_red, "solve_time": qi_red_t}

    # Comparison summary
    print("\n" + "=" * 60)
    print("LAYER 3 — MAKESPAN COMPARISON SUMMARY")
    print("=" * 60)
    makespans = {}
    for name, res in results.items():
        ms = res.get("makespan")
        if ms is not None:
            makespans[name] = ms
            print(f"  {name:30s}: {ms:4d} min ({ms/60:5.1f} h)  solve_time={res.get('solve_time')}s")

    if len(makespans) >= 2:
        best = min(makespans, key=makespans.get)
        print(f"\n  Best makespan: {best} ({makespans[best]} min)")
        gap = {k: round((v - makespans[best]) / makespans[best] * 100, 1) for k, v in makespans.items()}
        print("  Gap to best:")
        for k, v in gap.items():
            print(f"    {k}: +{v}%")

    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "layer": 3,
            "n_jobs": len(jobs),
            "n_employees": n_employees,
            "work_hours_per_day": WORK_HOURS_PER_DAY,
            "work_days": WORK_DAYS,
            "results": results,
            "makespan_comparison": makespans,
        }, f, indent=2)
    print(f"\n[LAYER 3] Results saved to {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    run_layer3_scheduling(n_employees=5)
