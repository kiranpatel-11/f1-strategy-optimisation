import numpy as np

from appendix_A_core_model import TOTAL_BUDGET, make_teams, run_one_season

# ============================================================
# Appendix B: objective evaluation and search
# ============================================================


def estimate_mean_points(teamA_alloc, n_seasons, seed=123):
    """
    Estimate expected Team A season points with a 95% confidence interval.
    """
    rng = np.random.default_rng(seed)
    teams = make_teams(teamA_alloc)

    pts = np.zeros(n_seasons, dtype=float)
    for k in range(n_seasons):
        pts[k], _ = run_one_season(teams, rng)

    mean = pts.mean()
    sd = pts.std(ddof=1)
    se = sd / np.sqrt(n_seasons)

    return {
        "mean": mean,
        "se": se,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
    }


def estimate_winprob(teamA_alloc, n_seasons, seed=123):
    """
    Estimate Team A constructors' win probability with a 95% confidence interval.
    """
    rng = np.random.default_rng(seed)
    teams = make_teams(teamA_alloc)

    wins = 0
    for _ in range(n_seasons):
        _, win = run_one_season(teams, rng)
        wins += 1 if win else 0

    p = wins / n_seasons
    se = np.sqrt(p * (1.0 - p) / n_seasons)

    return {
        "p": p,
        "se": se,
        "ci_low": p - 1.96 * se,
        "ci_high": p + 1.96 * se,
    }


def evaluate_allocation(teamA_alloc, n_seasons, seed=123):
    """
    Evaluate one allocation under the two main objectives:
      - J_mean: expected season points
      - J_win: probability of winning the constructors' championship
    """
    rng = np.random.default_rng(seed)
    teams = make_teams(teamA_alloc)

    pts = np.zeros(n_seasons, dtype=float)
    wins = 0

    for k in range(n_seasons):
        pts[k], win = run_one_season(teams, rng)
        wins += 1 if win else 0

    mean_points = pts.mean()
    sd_points = pts.std(ddof=1)
    p_win = wins / n_seasons
    q10 = np.quantile(pts, 0.10)
    q90 = np.quantile(pts, 0.90)

    return {
        "mean_points": mean_points,
        "sd_points": sd_points,
        "p_win_constructors": p_win,
        "q10": q10,
        "q90": q90,
    }


def generate_allocations(step, total_budget=TOTAL_BUDGET):
    """
    Enumerate all allocations (m, c, e, r) on a step grid
    such that m + c + e + r = total_budget.
    """
    units_total = int(round(total_budget / step))
    allocations = []

    for m_u in range(units_total + 1):
        for r_u in range(units_total - m_u + 1):
            for c_u in range(units_total - m_u - r_u + 1):
                e_u = units_total - m_u - r_u - c_u

                allocations.append({
                    "marketing": m_u * step,
                    "chassis": c_u * step,
                    "engine": e_u * step,
                    "reliability": r_u * step,
                })

    return allocations


def paired_winprob_diff(baseline_alloc, test_alloc, n_seasons, seed=123):
    """
    Paired comparison of two allocations on the binary win indicator,
    using common random numbers.
    Returns estimated difference in win probability:
        P(win | baseline) - P(win | test).
    """
    base_rng = np.random.default_rng(seed)
    season_seeds = base_rng.integers(0, 2**31 - 1, size=n_seasons)

    diffs = np.zeros(n_seasons, dtype=float)

    for k in range(n_seasons):
        s = int(season_seeds[k])

        rng1 = np.random.default_rng(s)
        _, win1 = run_one_season(make_teams(baseline_alloc), rng1)

        rng2 = np.random.default_rng(s)
        _, win2 = run_one_season(make_teams(test_alloc), rng2)

        diffs[k] = (1.0 if win1 else 0.0) - (1.0 if win2 else 0.0)

    mean = diffs.mean()
    sd = diffs.std(ddof=1)
    se = sd / np.sqrt(n_seasons)

    return {
        "mean": mean,
        "se": se,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
    }


def two_stage_search(
    step_screen=0.5,
    n_screen=120,
    top_k_screen=30,
    n_refine=8000,
    seed=123,
):
    """
    Two-stage search used in the project:
      1) coarse screening over all allocations,
      2) accurate re-evaluation of shortlisted candidates.
    """
    candidates = generate_allocations(step_screen)

    screen_results = []
    for i, alloc in enumerate(candidates):
        metrics = evaluate_allocation(alloc, n_screen, seed=seed + i)
        screen_results.append({**alloc, **metrics})

    top_mean = sorted(
        screen_results, key=lambda row: row["mean_points"], reverse=True
    )[:top_k_screen]
    top_win = sorted(
        screen_results, key=lambda row: row["p_win_constructors"], reverse=True
    )[:top_k_screen]

    shortlisted = []
    seen = set()
    for row in top_mean + top_win:
        key = (
            row["marketing"],
            row["chassis"],
            row["engine"],
            row["reliability"],
        )
        if key not in seen:
            shortlisted.append({
                "marketing": row["marketing"],
                "chassis": row["chassis"],
                "engine": row["engine"],
                "reliability": row["reliability"],
            })
            seen.add(key)

    refined_results = []
    for i, alloc in enumerate(shortlisted):
        metrics = evaluate_allocation(alloc, n_refine, seed=10_000 + seed + i)
        refined_results.append({**alloc, **metrics})

    return screen_results, refined_results


def local_refinement_rows(
    baseline_alloc,
    m_values,
    fixed_r=2.5,
    fixed_c=0.0,
    n_mean=5000,
    n_paired=3000,
    seed=123,
):
    """
    Local refinement around a win-focused allocation.
    Marketing is varied while reliability and chassis are fixed;
    engine is adjusted to preserve the total budget.
    """
    rows = []

    for m in m_values:
        test_alloc = {
            "marketing": float(m),
            "chassis": float(fixed_c),
            "reliability": float(fixed_r),
            "engine": float(TOTAL_BUDGET - m - fixed_r - fixed_c),
        }

        pts_stats = estimate_mean_points(test_alloc, n_mean, seed)
        win_stats = estimate_winprob(test_alloc, n_mean, seed)
        win_diff = paired_winprob_diff(baseline_alloc, test_alloc, n_paired, seed)

        rows.append({
            "marketing": test_alloc["marketing"],
            "chassis": test_alloc["chassis"],
            "engine": test_alloc["engine"],
            "reliability": test_alloc["reliability"],
            "mean_points": pts_stats["mean"],
            "mean_ci_low": pts_stats["ci_low"],
            "mean_ci_high": pts_stats["ci_high"],
            "p_win": win_stats["p"],
            "win_ci_low": win_stats["ci_low"],
            "win_ci_high": win_stats["ci_high"],
            "paired_diff_mean": win_diff["mean"],
            "paired_diff_ci_low": win_diff["ci_low"],
            "paired_diff_ci_high": win_diff["ci_high"],
        })

    return rows
