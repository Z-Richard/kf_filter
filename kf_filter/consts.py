# Define constants
g = 9.81  # gravity, m/s^2
omega_earth = 7.292e-5  # s^-1
radius_earth = 6.37e6  # m
beta = 2 * omega_earth / radius_earth  # s^-1 m^-1

# NOTE: Nondimensionalized dispersion relations for equatorial waves
kelvin = lambda op, omega, k, n : op(omega, k)
er = lambda op, omega, k, n : op(omega, -k / (2 * n + 1 + k ** 2))
eig = lambda op, omega, k, n : op(omega ** 2, k * omega + 1)
ig = lambda op, omega, k, n : op(omega ** 2, k ** 2 + (2 * n + 1))

wave_func = {
    'kelvin': kelvin,
    'er': er,
    'eig': eig,
    'mrg': eig,  # same as eig...
    'ig': ig
}

wave_title = {
    'kelvin': 'Kelvin wave',
    'er': 'Equatorial Rossby wave',
    'eig': 'Eastward inertia-gravity wave',
    'mrg': 'Mixed-Rossby gravity wave',
    'ig': 'Westward inertia-gravity wave',
    'td': 'Tropical depression',
    'mjo': 'Madden-Julian Oscillation'
}

wave_types = list(wave_func.keys())
