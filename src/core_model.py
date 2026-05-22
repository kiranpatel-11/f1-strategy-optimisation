import math
import numpy as np

# ============================================================
# Appendix A: core season-level model
# Distilled from the final working scripts.
# ============================================================

h = 3.2
TOTAL_BUDGET = 8.0
SEATS_PER_TEAM = 2
POINTS_16 = [10, 8, 6, 5, 4, 3, 2, 1] + [0] * 8

OPP_BASE = {"marketing": 2.0, "chassis": 2.0, "engine": 2.0, "reliability": 2.0}

drivers = [
    ("Ricardo Raul (BRA)", 5.0),
    ("Fabrizio Panduri (ITA)", 4.4),
    ("Luigi Disanti (ITA)", 4.2),
    ("Philipp Krafft (GER)", 3.8),
    ("Joris Gootjes (NED)", 3.7),
    ("Alfredo Alcarria (SPA)", 3.6),
    ("Akiko Tomiko (JAP)", 3.5),
    ("Allan Johnson (GBR)", 3.4),
    ("Jiang Lin (CHI)", 2.8),
    ("Germain Grisel (FRA)", 2.4),
    ("Fabian Metzger (SWI)", 1.8),
    ("Simao Faria (POR)", 1.5),
    ("Alfonso Llave (SPA)", 1.2),
    ("Michel Fabron (FRA)", 0.9),
    ("Armand Brice (FRA)", 0.3),
    ("Miko Lautela (FIN)", 0.0),
]

circuits = [
    {"name": "Spain", "type": "balanced"},
    {"name": "Belgium", "type": "chassis_engine"},
    {"name": "USA", "type": "engine"},
    {"name": "Japan", "type": "driver_chassis"},
    {"name": "Canada", "type": "balanced"},
    {"name": "Hungary", "type": "chassis"},
    {"name": "Monaco", "type": "driver"},
    {"name": "Italy", "type": "engine"},
    {"name": "France", "type": "balanced"},
    {"name": "Australia", "type": "balanced"},
]


# ============================================================
# Draft-signing model
# ============================================================

def two_team_signing_probability(m_A, m_B, h_param=h):
    """
    Analytic two-team signing probability used in the validation stage.
    """
    A_term = max(0.0, float(m_A)) ** h_param
    B_term = max(0.0, float(m_B)) ** h_param
    total = A_term + B_term
    if total == 0.0:
        return 0.5
    return A_term / total


def contest_probs(marketing_dict, eligible_teams, h_param=h):
    """
    Compute contest probabilities among eligible teams:
        p_i = m_i^h / sum_j m_j^h
    """
    m = np.array([float(marketing_dict[t]) for t in eligible_teams], dtype=float)
    m = np.clip(m, 0.0, None)

    weights = m ** h_param
    total = weights.sum()

    if total == 0.0:
        return np.ones(len(eligible_teams)) / len(eligible_teams)

    return weights / total


def sign_drivers_sequential(marketing_dict, drivers_list, seats_per_team, h_param, rng):
    """
    Sequential driver draft:
    strongest available driver signs first, then the next, and so on.
    """
    teams_list = list(marketing_dict.keys())
    seats_left = {t: seats_per_team for t in teams_list}
    roster = {t: [] for t in teams_list}

    drivers_sorted = sorted(drivers_list, key=lambda x: x[1], reverse=True)

    for driver_name, ability in drivers_sorted:
        eligible = [t for t in teams_list if seats_left[t] > 0]
        if not eligible:
            break

        probs = contest_probs(marketing_dict, eligible, h_param)
        chosen_team = rng.choice(eligible, p=probs)

        roster[chosen_team].append((driver_name, ability))
        seats_left[chosen_team] -= 1

    return roster


# ============================================================
# Performance and reliability model
# ============================================================

def circuit_coefficients(circuit_type):
    if circuit_type == "balanced":
        return 1.0, 1.0, 1.0
    if circuit_type == "driver":
        return 2.0, 1.0, 1.0
    if circuit_type == "chassis":
        return 1.0, 2.0, 1.0
    if circuit_type == "engine":
        return 1.0, 1.0, 2.0
    if circuit_type == "chassis_engine":
        return 1.0, 2.0, 2.0
    if circuit_type == "driver_chassis":
        return 2.0, 2.0, 1.0
    return 1.0, 1.0, 1.0


def performance_score(driving, chassis, engine, circuit_type):
    """
    Brief-based performance law with Setup fixed at zero.
    """
    c_d, c_c, c_e = circuit_coefficients(circuit_type)
    numerator = c_d * driving + c_c * chassis + c_e * engine
    denominator = c_d + c_c + c_e
    return 3.0 * numerator / denominator


def p_dnf_from_reliability(r):
    """
    Mechanical-failure law from the competition brief.
    """
    val = 1.0 - math.erf(0.7 * r - 1.8)
    p = (val * val) / 5.0
    return max(0.0, min(1.0, p))


# ============================================================
# Race and season simulation
# ============================================================

def run_one_race(roster, teams_dict, circuit, rng):
    """
    Simulate one race:
      1) compute performance scores,
      2) rank entries,
      3) apply DNFs,
      4) award points to finishers.
    """
    entries = []

    for team_name, signed_drivers in roster.items():
        chassis = teams_dict[team_name]["chassis"]
        engine = teams_dict[team_name]["engine"]
        reliability = teams_dict[team_name]["reliability"]

        for driver_name, ability in signed_drivers:
            score = performance_score(
                driving=ability,
                chassis=chassis,
                engine=engine,
                circuit_type=circuit["type"],
            )
            entries.append({
                "team": team_name,
                "score": score,
                "reliability": reliability,
            })

    entries.sort(key=lambda x: x["score"], reverse=True)

    finishers = []
    for entry in entries:
        if rng.random() >= p_dnf_from_reliability(entry["reliability"]):
            finishers.append(entry)

    race_points = {t: 0 for t in teams_dict}
    for pos, entry in enumerate(finishers):
        pts = POINTS_16[pos] if pos < len(POINTS_16) else 0
        race_points[entry["team"]] += pts

    return race_points


def make_teams(teamA_alloc, opp_alloc=None):
    """
    Build the eight-team championship field.
    Team A uses teamA_alloc; all opponents use opp_alloc.
    """
    if opp_alloc is None:
        opp_alloc = OPP_BASE

    def check_budget(alloc):
        total = (
            alloc["marketing"]
            + alloc["chassis"]
            + alloc["engine"]
            + alloc["reliability"]
        )
        if abs(total - TOTAL_BUDGET) > 1e-9:
            raise ValueError(f"Allocation must sum to {TOTAL_BUDGET}. Got {total}")

    check_budget(teamA_alloc)
    check_budget(opp_alloc)

    return {
        "Team A (my team)": teamA_alloc.copy(),
        "Team B": opp_alloc.copy(),
        "Team C": opp_alloc.copy(),
        "Team D": opp_alloc.copy(),
        "Team E": opp_alloc.copy(),
        "Team F": opp_alloc.copy(),
        "Team G": opp_alloc.copy(),
        "Team H": opp_alloc.copy(),
    }


def run_one_season(teams_dict, rng):
    """
    Simulate one season and return:
      - Team A season points
      - whether Team A wins the constructors' championship
    """
    marketing_dict = {t: teams_dict[t]["marketing"] for t in teams_dict}
    roster = sign_drivers_sequential(marketing_dict, drivers, SEATS_PER_TEAM, h, rng)

    team_points = {t: 0 for t in teams_dict}
    for circuit in circuits:
        race_pts = run_one_race(roster, teams_dict, circuit, rng)
        for t in team_points:
            team_points[t] += race_pts[t]

    winner = max(team_points.items(), key=lambda x: x[1])[0]
    return team_points["Team A (my team)"], (winner == "Team A (my team)")
