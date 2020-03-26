from burr import Burr
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    B = Burr(2, 1)

    x = 6

    t = np.linspace(2, 500, 10000)

    Ut_vals = B.U(t)
    Utx_vals = B.U(x*t)

    a_vals = t * B.Uprime(t)

    sig_vals = B.sigma(t)

    lim_vals = np.array(len(t) * [B.evtLim(x)])

    y = (Utx_vals - Ut_vals)/a_vals

    plt.plot(t, y)
    plt.plot(t, lim_vals)
    plt.plot(t, (Utx_vals - Ut_vals)/sig_vals)
    plt.legend(labels=["Y", "Lim", "Y_sig"])
    plt.show()
    plt.clf()
