# Formula 1 Budget Allocation Model

Python-based Monte Carlo simulation and optimisation project exploring strategic budget allocation in a Formula 1-style championship environment.

## Project Overview

This project investigates how a racing team should allocate a fixed budget across:

- Marketing
- Chassis development
- Engine development
- Reliability

to maximise season performance under uncertainty.

The model simulates:

- Probabilistic driver recruitment
- Circuit-dependent car performance
- Mechanical failures (DNFs)
- Full 10-race championship seasons
- Championship win probabilities

using large-scale Monte Carlo simulation.

## Key Features

- Sequential stochastic driver-signing model
- Monte Carlo season simulation
- Mean-points vs championship-win optimisation
- Sensitivity and robustness analysis
- Opponent-style scenario testing
- Confidence intervals and paired simulations
- Data visualisation using Matplotlib

## Example Results

### Trade-off between consistency and championship upside

![Trade-off](figures/fig4_tradeoff_scatter.png)

### Reliability investment vs DNF probability

![Reliability](figures/fig2_reliability_dnf_curve.png)

### Robustness analysis

![Robustness](figures/fig8_meanopt_robustness.png)

## Repository Structure

```text
src/
    core_model.py
    search_optimisation.py
    sensitivity_analysis.py

figures/
    *.png

report/
    MATH3001_PATEL_REPORT.pdf
