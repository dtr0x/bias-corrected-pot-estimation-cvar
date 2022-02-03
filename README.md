## Bias-Corrected Peaks-Over-Threshold Estimation of the CVaR (UAI 2021)

https://proceedings.mlr.press/v161/troop21a.html

This code will generate all data used in simulations presented in the paper and produce plots. Parameters can be adjusted to produce results for other scenarios.

## Requirements:
* Python v3.xx
* Modules:
  * numpy
  * scipy
  * matplotlib
  * multiprocessing

## Replicating results
Executing **run_sim.py** will generate all samples from all distributions given in the paper, and compute estimated CVaRs (alpha=0.998) using **UPOT**, **BPOT**, and **SA**. All samples and CVaR values will be stored in the **data** folder.
The following files can be used to reproduce plots in the paper:
 * **dist_plots.py** Produces RMSE and absolute bias plots from the generated CVaR data
 * **coverage_prob.py** Produces plots of coverage probability from the generated CVaR data
 * **asymp_var.py** Comparison of UPOT and SA asymptotic variance for the Frechet distribution


