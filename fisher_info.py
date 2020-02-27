import numpy as np
from eva import rgpd

def d2xi(y, xi, sig):
    x1 = y*xi*(y*xi*(3+xi)+2*sig)/(y*xi+sig)**2
    x2 = -2*np.log(1+y*xi/sig)
    return (x1 + x2)/xi**3

def d2sig(y, xi, sig):
    x1 = -1/sig**2
    x2 = (1+xi)/(y*xi+sig)**2
    return (x1 + x2)/xi

def dxisig(y, xi, sig):
    return y*(sig-y)/sig/(y*xi+sig)**2

def Ed2xi(xi):
    return -2/(1+xi)/(1+2*xi)

def Ed2sig(xi, sig):
    return -1/sig**2/(1+2*xi)

def Edxisig(xi, sig):
    return -1/sig/(1+xi)/(1+2*xi)

if __name__ == '__main__':
    np.random.seed(7)
    xi = 0.25
    sig = 1.5
    n = 10000000
    data = rgpd(n, xi, sig)

    d2xi_vals = d2xi(data, xi, sig)
    d2sig_vals = d2sig(data, xi, sig)
    dxisig_vals = dxisig(data, xi, sig)

    print("D2(Xi) Sample Mean: {:3f}, D2(Xi) ExpVal: {:3f}".format( \
            np.mean(d2xi_vals), Ed2xi(xi)))

    print("D2(Sigma) Sample Mean: {:3f}, D2(Sigma) ExpVal: {:3f}".format( \
            np.mean(d2sig_vals), Ed2sig(xi, sig)))

    print("D(Xi,Sigma) Sample Mean: {:3f}, D(Xi,Sigma) ExpVal: {:3f}".format( \
            np.mean(dxisig_vals), Edxisig(xi, sig)))
