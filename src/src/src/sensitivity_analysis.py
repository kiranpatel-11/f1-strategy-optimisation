import numpy as np

from appendix_A_core_model import TOTAL_BUDGET, make_teams, p_dnf_from_reliability, run_one_season
from appendix_B_search import estimate_mean_points

# ============================================================
# Appendix C: robustness and opponent sensitivity
# ============================================================


def perturb_one_at_a_time(base_alloc, key, frac_change):
    """
    Perturb one component by a given percentage and preserve the total budget
    by rebalancing chassis and engine.

    This is the rule used in the robustness analysis.
    """
    alloc = base_alloc.copy()

    old = alloc[key]
    new = max(0.0, old * (1.0 + frac_change))
    alloc[key] = new

    delta = new - old

    if delta >= 0:
        c_try = alloc["chassis"] - delta / 2.0
        e_try = alloc["engine"] - delta / 2.0

        if c_try >= 0 and e_try >= 0:
            alloc["chassis"] = c_try
            alloc["engine"] = e_try
        else:
            e_only = alloc["engine"] - delta
            if e_only < 0:
                return None
            alloc["engine"] = e_only
    else:
        alloc["chassis"] += (-delta) / 2.0
        alloc["engine"] += (-delta) / 2.0

    total = (
        alloc["marketing"]
        + alloc["chassis"]
        + alloc["engine"]
        + alloc["reliability"]
    )
    correction = (TOTAL_BUDGET - total) / 2.0
    alloc["chassis"] += correction
    alloc["engine"] += correction

    if alloc["chassis"] < -1e-9 or alloc["engine"] < -1e-9:
        return None

    alloc["chassis"] = max(0.0, alloc["chassis"])
    alloc["engine"] = max(0.0, alloc["engine"])

    return alloc


def paired_points_diff(baseline_alloc, test_alloc, n_seasons, seed=123):
    """
    Paired comparison of two allocations on Team A season points.
    Returns baseline - test.
    """
    base_rng = np.random.default_rng(seed)
    season_seeds = base_rng.integers(0, 2**31 - 1, size=n_seasons)

    diffs = np.zeros(n_seasons, dtype=float)

    for k in range(n_seasons):
        s = int(season_seeds[k])

        rng1 = np.random.default_rng(s)
        pts1, _ = run_one_season(make_teams(baseline_alloc), rng1)

        rng2 = np.random.default_rng(s)
        pts2, _ = run_one_season(make_teams(test_alloc), rng2)

        diffs[k] = pts1 - pts2

    mean = diffs.mean()
    sd = diffs.std(ddof=1)
    se = sd / np.sqrt(n_seasons)

    return {
        "mean": mean,
        "se": se,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
    }


def paired_winprob_diff(baseline_alloc, test_alloc, n_seasons, seed=123):
    """
    Paired comparison of two allocations on the binary constructors-win indicator.
    Returns baseline - test.
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


def robustness_scan(
    baseline_alloc,
    keys=("marketing", "reliability"),
    fracs=(-0.10, -0.05, 0.05, 0.10),
    n_mean=5000,
    n_paired=5000,
    seed=123,
):
    """
    One-at-a-time robustness analysis around a baseline allocation.
    """
    rows = []

    for key in keys:
        for frac in fracs:
            test_alloc = perturb_one_at_a_time(baseline_alloc, key, frac)

            if test_alloc is None:
                rows.append({
                    "key": key,
                    "frac": frac,
                    "feasible": False,
                })
                continue

            stats = estimate_mean_points(test_alloc, n_mean, seed)
            pts_diff = paired_points_diff(baseline_alloc, test_alloc, n_paired, seed)

            rows.append({
                "key": key,
                "frac": frac,
                "feasible": True,
                "marketing": test_alloc["marketing"],
                "chassis": test_alloc["chassis"],
                "engine": test_alloc["engine"],
                "reliability": test_alloc["reliability"],
                "p_dnf": p_dnf_from_reliability(test_alloc["reliability"]),
                "mean_points": stats["mean"],
                "mean_ci_low": stats["ci_low"],
                "mean_ci_high": stats["ci_high"],
                "paired_points_diff": pts_diff["mean"],
                "paired_points_ci_low": pts_diff["ci_low"],
                "paired_points_ci_high": pts_diff["ci_high"],
            })

    return rows


opponent_styles = {
    "Balanced (2.0,2.0,2.0,2.0)": {
        "marketing": 2.0,
        "chassis": 2.0,
        "engine": 2.0,
        "reliability": 2.0,
    },
    "Marketing-heavy (3.2,1.6,1.6,1.6)": {
        "marketing": 3.2,
        "chassis": 1.6,
        "engine": 1.6,
        "reliability": 1.6,
    },
    "Chassis-heavy (1.6,3.2,1.6,1.6)": {
        "marketing": 1.6,
        "chassis": 3.2,
        "engine": 1.6,
        "reliability": 1.6,
    },
    "Engine-heavy (1.6,1.6,3.2,1.6)": {
        "marketing": 1.6,
        "chassis": 1.6,
        "engine": 3.2,
        "reliability": 1.6,
    },
    "Reliability-heavy (1.6,1.6,1.6,3.2)": {
        "marketing": 1.6,
        "chassis": 1.6,
        "engine": 1.6,
        "reliability": 3.2,
    },
}

def simulate_many(teamA_alloc, opp_alloc, n_seasons, seed=123):
    """
    Evaluate one Team A strategy against one opponent style.
    """
    rng = np.random.default_rng(seed)
    teams = make_teams(teamA_alloc, opp_alloc=opp_alloc)

    pts = np.zeros(n_seasons, dtype=float)
    wins = 0

    for k in range(n_seasons):
        pts[k], win = run_one_season(teams, rng)
        wins += 1 if win else 0

    mean = pts.mean()
    sd = pts.std(ddof=1)
    se = sd / np.sqrt(n_seasons)
    p = wins / n_seasons
    se_p = np.sqrt(p * (1.0 - p) / n_seasons)

    return {
        "mean_points": mean,
        "ci_points_low": mean - 1.96 * se,
        "ci_points_high": mean + 1.96 * se,
        "p_win": p,
        "ci_pwin_low": p - 1.96 * se_p,
        "ci_pwin_high": p + 1.96 * se_p,
    }


def opponent_sensitivity(teamA_strategies, styles=None, n_seasons=10000, seed=123):
    """
    Compare multiple Team A strategies across a set of representative opponent styles.
    The default styles are one balanced profile together with marketing-heavy,
    chassis-heavy, engine-heavy and reliability-heavy profiles.
    """
    if styles is None:
        styles = opponent_styles

    rows = []

    for strategy_name, teamA_alloc in teamA_strategies.items():
        for style_name, opp_alloc in styles.items():
            out = simulate_many(teamA_alloc, opp_alloc, n_seasons, seed)
            rows.append({
                "teamA_strategy": strategy_name,
                "opponent_style": style_name,
                **out,
            })

    return rows
