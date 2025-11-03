import numpy as np

def rho0(x):
    return np.where(x <= 0, 0.1, 10.0)

def u0(x):
    return np.where(x <= 0, 2.0, 1.0)

def analytical_solution(x, t):
    x = np.array(x)
    t = np.array(t)
    
    # Characteristic variables
    xp = x - t  # along dx/dt = +1
    xm = x + t  # along dx/dt = -1
    
    # Riemann invariants
    wplus = rho0(xp) + u0(xp)   # w+ = rho + u (constant along +1 characteristic)
    wminus = rho0(xm) - u0(xm)  # w- = rho - u (constant along -1 characteristic)
    
    # Recover physical variables
    rho = 0.5 * (wplus + wminus)
    u = 0.5 * (wplus - wminus)
    
    return rho, u