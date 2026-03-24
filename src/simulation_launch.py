def compute_params_CG(scale_factor):
    sigma_for_gold = 4.08 * (3.6 / 2.33) ** (-1)
    SIGMA = sigma_for_gold  # Scaling for sigma

    ATOMIC_UNIT_MASS = 196.196 * 3  # Scaling for masses
    EPSILON = 0.4  # * (scale_factor**3) # Scaling for masses

    A = SIGMA * (3.6 / 2.33) * scale_factor
    return SIGMA, A, EPSILON, ATOMIC_UNIT_MASS