## Code for paper "Bias-Corrected Peaks-Over-Threshold Estimation of the CVaR."
This code will generate all data used in simulations presented in the paper and produce plots. Parameters can be adjusted to produce results for other scenarios.

## Requirements:
* Python v3.xx
* Modules:
  * numpy
  * scipy
  * matplotlib
  * multiprocessing

## Replicating results
Executing **gen_data.py** will generate all samples from all distributions given in the paper, and compute estimated CVaRs (alpha=0.999) using **UPOT**, **BPOT**, and **SA**. Predetermined sample fractions to estimate **rho** for the Burr distributions have been generated using **rho_search.py** and saved in **rho_samp_theta.npy**. All values are stored in the **data** folder. The exact plots shown in the paper can then be obtained by running **make_plots.py**, with the output stored in the **plots** folder.

Since generating all samples and estimated quantities is quite memory-intensive (requiring the multiprocessing module), all data used in experiments can alternatively be downloaded here:

where the contents of the .zip file should be extracted into the **data** folder.
