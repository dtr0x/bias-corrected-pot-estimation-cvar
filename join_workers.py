import numpy as np
import os

if __name__ == '__main__':
    arrs = [np.load('data/' + c) for c in os.listdir('data') if 'cvars_' in c]
    cvars = np.concatenate(arrs, axis=2)
    np.save('data/cvars.npy', cvars)
