import numpy as np
import matplotlib.pyplot as plt

def compute_params_CG(scale_factor):
    MASS_Ni = 58.69
    EPS_Ni = 2.5  
    SIGMA_Ni = 2.28
    
    sigma_for_nickel = SIGMA_Ni
    SIGMA = sigma_for_nickel * scale_factor
    
    ATOMIC_UNIT_MASS = MASS_Ni * (scale_factor ** 3)
    EPSILON = EPS_Ni * (scale_factor ** 3)
    
    A = 3.52 * scale_factor
    return SIGMA, A, EPSILON, ATOMIC_UNIT_MASS