"""
Task 4: Run algorithms (PG, FB, ASM) on pre-conditioned problems and compare.

This script runs FB (penalized), PG and ASM (constrained) on:
- Original problems
- Pre-conditioned problems (diagonal, SVD)

We collect metrics (iterations, objective, residuals, time), compare performance
and approximation quality, and save figures and data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import json
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

from problems.sine_approximation import SineApproximationProblem
from problems.preconditioned_problem import PreconditionedSineProblem
from algorithms.forward_backward import ForwardBackward
from algorithms.projected_gradient import ProjectedGradient
from algorithms.active_set_method import ActiveSetMethod


DegreesType = List[int]


@dataclass
class RunResult:
    algorithm: str
    problem_type: str  # original | diagonal | svd
    degree: int
    lambda_param: float | None
    iterations: int
    final_objective: float
    residual: float
    l1_norm: float
    elapsed_ms: float
    solution: np.ndarray

    def to_jsonable(self) -> Dict[str, Any]:
        d = asdict(self)
        d["solution"] = self.solution.tolist()
        return d


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return out, elapsed_ms


def run_algorithms_on_problem(problem, lambda_values: List[float]) -> Dict[str, Dict[str, Any]]:
    """Run FB (for each lambda), PG, ASM on the given problem."""
    results: Dict[str, Dict[str, Any]] = {}

    # PG (constrained)
    try:
        pg = ProjectedGradient(problem)
        (pg_res, pg_ms) = timed(pg.solve, max_iterations=100000, tolerance=1e-6, verbose=False)
        results["PG"] = {**pg_res, "elapsed_ms": pg_ms}
    except Exception as e:
        results["PG"] = {"error": str(e)}

    # ASM (constrained)
    try:
        asm = ActiveSetMethod(problem)
        (asm_res, asm_ms) = timed(asm.solve, max_iterations=5000, tolerance=1e-6, verbose=False)
        results["ASM"] = {**asm_res, "elapsed_ms": asm_ms}
    except Exception as e:
        results["ASM"] = {"error": str(e)}

    # FB (penalized) for several lambda values
    fb_outs = []
    for lam in lambda_values:
        try:
            fb = ForwardBackward(problem, lambda_param=float(lam))
            (fb_res, fb_ms) = timed(fb.solve, max_iterations=100000, tolerance=1e-6, verbose=False)
            fb_outs.append({**fb_res, "elapsed_ms": fb_ms, "lambda_param": float(lam)})
        except Exception as e:
            fb_outs.append({"error": str(e), "lambda_param": float(lam)})
    results["FB"] = fb_outs

    return results


def build_problems(degree: int) -> Dict[str, Any]:
    original = SineApproximationProblem(degree=degree, num_samples=100)
    diagonal = PreconditionedSineProblem(original, "diagonal")
    svd = PreconditionedSineProblem(original, "svd")
    return {"original": original, "diagonal": diagonal, "svd": svd}


def aggregate_results(degree: int, problems: Dict[str, Any], lambda_values: List[float]) -> List[RunResult]:
    rows: List[RunResult] = []
    for problem_type, problem in problems.items():
        results = run_algorithms_on_problem(problem, lambda_values)

        # PG
        if "error" not in results["PG"]:
            sol = results["PG"]["solution"]
            # If preconditioned, transform back for norms
            if problem_type in ("diagonal", "svd"):
                sol = problems[problem_type].transform_solution(np.array(sol))
            rows.append(
                RunResult(
                    algorithm="PG",
                    problem_type=problem_type,
                    degree=degree,
                    lambda_param=None,
                    iterations=int(results["PG"]["iterations"]),
                    final_objective=float(results["PG"]["final_objective"]),
                    residual=float(results["PG"].get("final_projected_gradient", results["PG"].get("final_proximal_residual", 0.0))),
                    l1_norm=float(np.sum(np.abs(sol))),
                    elapsed_ms=float(results["PG"]["elapsed_ms"]),
                    solution=np.array(sol),
                )
            )

        # ASM
        if "error" not in results["ASM"]:
            sol = results["ASM"]["solution"]
            if problem_type in ("diagonal", "svd"):
                sol = problems[problem_type].transform_solution(np.array(sol))
            rows.append(
                RunResult(
                    algorithm="ASM",
                    problem_type=problem_type,
                    degree=degree,
                    lambda_param=None,
                    iterations=int(results["ASM"]["iterations"]),
                    final_objective=float(results["ASM"]["final_objective"]),
                    residual=float(results["ASM"].get("final_projected_gradient", results["ASM"].get("final_proximal_residual", 0.0))),
                    l1_norm=float(np.sum(np.abs(sol))),
                    elapsed_ms=float(results["ASM"]["elapsed_ms"]),
                    solution=np.array(sol),
                )
            )

        # FB list
        for fb_res in results["FB"]:
            if "error" in fb_res:
                continue
            sol = fb_res["solution"]
            if problem_type in ("diagonal", "svd"):
                sol = problems[problem_type].transform_solution(np.array(sol))
            rows.append(
                RunResult(
                    algorithm="FB",
                    problem_type=problem_type,
                    degree=degree,
                    lambda_param=float(fb_res.get("lambda_param", 0.0)),
                    iterations=int(fb_res["iterations"]),
                    final_objective=float(fb_res["final_objective"]),
                    residual=float(fb_res.get("final_proximal_residual", fb_res.get("final_projected_gradient", 0.0))),
                    l1_norm=float(np.sum(np.abs(sol))),
                    elapsed_ms=float(fb_res["elapsed_ms"]),
                    solution=np.array(sol),
                )
            )
    return rows


def plot_performance(rows: List[RunResult], degrees: DegreesType) -> None:
    # Bar charts for iterations and objective by algorithm and problem_type per degree
    os.makedirs('results/plots', exist_ok=True)

    algs = ["PG", "ASM", "FB"]
    methods = ["original", "diagonal", "svd"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Iterations (PG/ASM best single, FB take median over lambdas)
    ax = axes[0, 0]
    width = 0.25
    x = np.arange(len(degrees))
    for i, method in enumerate(methods):
        vals = []
        for d in degrees:
            iters_for_alg = []
            for alg in algs:
                if alg == "FB":
                    fb_iters = [r.iterations for r in rows if r.algorithm == "FB" and r.degree == d and r.problem_type == method]
                    val = float(np.median(fb_iters)) if fb_iters else 0.0
                else:
                    rr = [r for r in rows if r.algorithm == alg and r.degree == d and r.problem_type == method]
                    val = float(rr[0].iterations) if rr else 0.0
                iters_for_alg.append(val)
            # sum across algs as composite score for the method at degree
            vals.append(sum(iters_for_alg))
        ax.bar(x + i*width - width, vals, width, label=method)
    ax.set_title('Composite Iterations by Method (sum over algorithms)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in degrees])
    ax.set_xlabel('Degree')
    ax.set_ylabel('Iterations (sum across PG, ASM, median(FB))')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Objective (sum across algorithms median for FB)
    ax = axes[0, 1]
    for i, method in enumerate(methods):
        vals = []
        for d in degrees:
            objs_for_alg = []
            for alg in algs:
                if alg == "FB":
                    fb_objs = [r.final_objective for r in rows if r.algorithm == "FB" and r.degree == d and r.problem_type == method]
                    val = float(np.median(fb_objs)) if fb_objs else 0.0
                else:
                    rr = [r for r in rows if r.algorithm == alg and r.degree == d and r.problem_type == method]
                    val = float(rr[0].final_objective) if rr else 0.0
                objs_for_alg.append(val)
            vals.append(sum(objs_for_alg))
        ax.bar(x + i*width - width, vals, width, label=method)
    ax.set_title('Composite Objective by Method (sum over algorithms)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in degrees])
    ax.set_xlabel('Degree')
    ax.set_ylabel('Objective (sum)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Time comparison (ms)
    ax = axes[1, 0]
    for i, method in enumerate(methods):
        vals = []
        for d in degrees:
            times_for_alg = []
            for alg in algs:
                if alg == "FB":
                    fb_times = [r.elapsed_ms for r in rows if r.algorithm == "FB" and r.degree == d and r.problem_type == method]
                    val = float(np.median(fb_times)) if fb_times else 0.0
                else:
                    rr = [r for r in rows if r.algorithm == alg and r.degree == d and r.problem_type == method]
                    val = float(rr[0].elapsed_ms) if rr else 0.0
                times_for_alg.append(val)
            vals.append(sum(times_for_alg))
        ax.bar(x + i*width - width, vals, width, label=method)
    ax.set_title('Composite Time by Method (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in degrees])
    ax.set_xlabel('Degree')
    ax.set_ylabel('Time (ms, sum across algs)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residual comparison (sum across algorithms)
    ax = axes[1, 1]
    for i, method in enumerate(methods):
        vals = []
        for d in degrees:
            res_for_alg = []
            for alg in algs:
                if alg == "FB":
                    fb_resid = [r.residual for r in rows if r.algorithm == "FB" and r.degree == d and r.problem_type == method]
                    val = float(np.median(fb_resid)) if fb_resid else 0.0
                else:
                    rr = [r for r in rows if r.algorithm == alg and r.degree == d and r.problem_type == method]
                    val = float(rr[0].residual) if rr else 0.0
                res_for_alg.append(val)
            vals.append(sum(res_for_alg))
        ax.bar(x + i*width - width, vals, width, label=method)
    ax.set_title('Composite Optimality Residual by Method (sum)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in degrees])
    ax.set_xlabel('Degree')
    ax.set_ylabel('Residual (sum across algs)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/plots/task4_preconditioned_performance.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_approximation_quality(rows: List[RunResult], degree: int, problems: Dict[str, Any]) -> None:
    os.makedirs('results/plots', exist_ok=True)
    t_vals = problems["original"].sample_points
    true_vals = problems["original"].target_values

    # choose best per algorithm and method (lowest objective)
    def best_row(alg: str, method: str) -> RunResult | None:
        candidates = [r for r in rows if r.algorithm == alg and r.problem_type == method and r.degree == degree]
        if not candidates:
            return None
        return min(candidates, key=lambda r: r.final_objective)

    methods = ["original", "diagonal", "svd"]
    algs = ["PG", "ASM", "FB"]

    fig, axes = plt.subplots(len(algs), len(methods), figsize=(16, 10), sharex=True, sharey=True)
    for i, alg in enumerate(algs):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            r = best_row(alg, method)
            if r is None:
                ax.set_axis_off()
                continue
            # evaluate polynomial in original space
            x = r.solution
            y_hat = problems["original"].evaluate_polynomial(x, t_vals)
            rmse = float(np.sqrt(np.mean((y_hat - true_vals) ** 2)))
            ax.plot(t_vals, true_vals, 'k--', label='true')
            ax.plot(t_vals, y_hat, label=f'{alg}-{method}, RMSE={rmse:.3f}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            if i == len(algs)-1:
                ax.set_xlabel('t')
            if j == 0:
                ax.set_ylabel('sin(t) vs approx')

    plt.tight_layout()
    plt.savefig('results/plots/task4_approximation_quality.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> bool:
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/data', exist_ok=True)

    degrees: DegreesType = [8, 10, 12, 15]
    lambda_values: List[float] = [0.1, 0.5, 1.0, 2.0]

    print("=" * 80)
    print("TASK 4: RUN ALGORITHMS ON PRE-CONDITIONED PROBLEMS")
    print("=" * 80)

    all_rows: List[RunResult] = []
    per_degree_summaries: Dict[int, Dict[str, Any]] = {}

    for deg in degrees:
        print(f"Running degree {deg}...")
        problems = build_problems(deg)
        rows = aggregate_results(deg, problems, lambda_values)
        all_rows.extend(rows)

        # quick per-degree summary
        per_degree_summaries[deg] = {
            'original_condition': float(problems['original'].condition_number),
            'diagonal_condition': float(problems['diagonal'].condition_number),
            'svd_condition': float(problems['svd'].condition_number),
        }

    # Plot performance and approximation (use degree 12 for approximation)
    plot_performance(all_rows, degrees)
    problems_12 = build_problems(12)
    plot_approximation_quality(all_rows, 12, problems_12)

    # Save data
    data = {
        'rows': [r.to_jsonable() for r in all_rows],
        'degrees': degrees,
        'lambda_values': lambda_values,
        'per_degree_summaries': per_degree_summaries,
    }

    with open('results/data/task4_results.pkl', 'wb') as f:
        pickle.dump({'rows': all_rows, 'meta': data}, f)

    # json-safe conversion
    def to_jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_jsonable(v) for v in obj]
        return obj

    with open('results/data/task4_results.json', 'w') as f:
        json.dump(to_jsonable(data), f, indent=2)

    print("Generated:")
    print("  - results/plots/task4_preconditioned_performance.png")
    print("  - results/plots/task4_approximation_quality.png")
    print("  - results/data/task4_results.pkl")
    print("  - results/data/task4_results.json")

    return True


if __name__ == '__main__':
    main()


