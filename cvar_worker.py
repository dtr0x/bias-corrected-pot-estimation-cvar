import argparse
import numpy as np
from run_sim import get_cvars

# pre-compute CVaRs in batches
def run_worker(alph, sampsizes, row_start, row_end):
    data = np.load('data/samples.npy')[:, row_start:row_end, :]
    cvars = get_cvars(data, alph, sampsizes)
    return cvars

if __name__ == '__main__':
    # get arguments from command line for asset type and year range
    parser = argparse.ArgumentParser()
    parser.add_argument('-row_start', type=int, required=True)
    parser.add_argument('-row_end', type=int, required=True)
    args = parser.parse_args()
    row_start = args.row_start
    row_end = args.row_end

    # CVaR level
    alph = 0.998
    # sample sizes to test CVaR estimation
    sampsizes = np.linspace(5000, 50000, 10).astype(int)
    # compute CVaR data
    cvars = run_worker(alph, sampsizes, row_start, row_end)

    np.save('data/cvars_{}_{}.npy'.format(row_start, row_end), cvars)
