"""
Archer (1992) Pitzer model for NaCl(aq) water activity.

Reference: D.G. Archer, J. Phys. Chem. Ref. Data 21(4), 793-829, 1992
DOI: 10.1063/1.555987

Implements the modified Pitzer ion-interaction model (Eq. 7) with
ionic-strength dependent third virial coefficient (Eq. 5).

Parameters beta0(T), beta1(T), C0(T), C1(T) are extracted at each of
Archer's tabulated temperatures by least-squares fitting of Table 9
osmotic coefficients, using Table 7 Debye-Hueckel A_phi values.
Smooth T-interpolation via PCHIP preserves monotonicity.

Valid range: 273-573 K, 0-6 mol/kg NaCl, p ~ 0.1 MPa (vapor pressure).

Usage:
    from archer_water_activity import water_activity_archer
    aw = water_activity_archer(T_K=373.15, m_NaCl=2.0)
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

# =============================================================================
# Pitzer equation constants for NaCl (1:1 electrolyte)
# =============================================================================
NU_M = 1        # stoichiometric number of cations (Na+)
NU_X = 1        # stoichiometric number of anions (Cl-)
NU = NU_M + NU_X  # = 2
Z_M = 1         # charge of cation
Z_X = -1        # charge of anion
ALPHA = 2.0     # kg^{1/2} mol^{-1/2} (second virial)
ALPHA2 = 2.5    # kg^{1/2} mol^{-1/2} (third virial, Archer's modification)
B_DH = 1.2      # kg^{1/2} mol^{-1/2} (Debye-Hueckel denominator)
M_WATER = 0.018015  # kg/mol (molar mass of water)

# =============================================================================
# Table 7: Debye-Hueckel A_phi at reference pressures (Archer 1992, p.818)
# Computed from Hill's EOS for water + Archer & Wang dielectric constant
# At p = 0.1 MPa for T <= 373 K; at saturation pressure for T > 373 K
# =============================================================================
_APHI_T = np.array([273.15, 298.15, 323.15, 373.15, 473.15, 573.15])
_APHI_V = np.array([0.3763, 0.3914, 0.4102, 0.4597, 0.6168, 0.9563])

_aphi_interp = PchipInterpolator(_APHI_T, _APHI_V)


def debye_hueckel_Aphi(T_K):
    """Debye-Hueckel A_phi coefficient interpolated from Archer Table 7.

    Valid 273-573 K.  Extrapolation outside this range uses endpoint slopes.
    """
    return float(_aphi_interp(np.clip(T_K, 273.15, 573.15)))


# =============================================================================
# Table 9: Calculated osmotic coefficients (Archer 1992, p.819)
# phi at p = 0.1 MPa (or saturation pressure at high T)
# =============================================================================
_TABLE9_T = [273.15, 298.15, 323.15, 373.15, 473.15, 573.15]
_TABLE9_M = [0.1, 0.5, 1.0, 3.0, 6.0]
_TABLE9_PHI = [
    # T = 273.15 K
    [0.9316, 0.9108, 0.9157, 1.0107, 1.2466],
    # T = 298.15 K
    [0.9322, 0.9218, 0.9371, 1.0486, 1.2694],
    # T = 323.15 K
    [0.9302, 0.9234, 0.9431, 1.0609, 1.2648],
    # T = 373.15 K
    [0.9219, 0.9139, 0.9340, 1.0459, 1.2111],
    # T = 473.15 K
    [0.8893, 0.8606, 0.8667, 0.9294, 1.0163],
    # T = 573.15 K
    [0.8105, 0.7302, 0.7110, 0.7125, 0.7444],
]

# Table 10: Calculated stoichiometric activity coefficient gamma_+/-
_TABLE10_GAMMA = [
    # T = 273.15 K
    [0.7796, 0.6706, 0.6340, 0.6585, 0.8990],
    # T = 298.15 K
    [0.7774, 0.6808, 0.6573, 0.7177, 0.9903],
    # T = 323.15 K
    [0.7697, 0.6755, 0.6569, 0.7299, 0.9933],
    # T = 373.15 K
    [0.7459, 0.6443, 0.6228, 0.6801, 0.8727],
    # T = 473.15 K
    [0.6666, 0.5269, 0.4818, 0.4551, 0.4864],
    # T = 573.15 K
    [0.5129, 0.3258, 0.2631, 0.1912, 0.1632],
]


# =============================================================================
# Pitzer osmotic coefficient equation (Archer 1992, Eq. 7)
# =============================================================================
def osmotic_coeff_pitzer(m, beta0, beta1, C0, C1, Aphi):
    """Pitzer osmotic coefficient for 1:1 electrolyte (NaCl).

    Archer Eq. 7 (modified with ionic-strength dependent C_MX):
      phi - 1 = -A_phi * sqrt(I) / (1 + b*sqrt(I))
               + m * (beta0 + beta1 * exp(-alpha*sqrt(I)))
               + 2*m^2 * (C0 + C1 * exp(-alpha2*sqrt(I)))

    For NaCl: I = m, nu_M = nu_X = 1, z_M = 1, nu = 2.
    The factor 2 in the C term = 4*nu_M^2*nu_X*z_M / nu = 4*1*1*1/2 = 2.
    """
    if m <= 0:
        return 1.0
    sqrtm = np.sqrt(m)
    # Debye-Hueckel term
    dh = -Aphi * sqrtm / (1.0 + B_DH * sqrtm)
    # Second virial coefficient term
    bterm = m * (beta0 + beta1 * np.exp(-ALPHA * sqrtm))
    # Third virial coefficient term (Archer's modification: I-dependent C)
    cterm = 2.0 * m**2 * (C0 + C1 * np.exp(-ALPHA2 * sqrtm))
    return 1.0 + dh + bterm + cterm


# =============================================================================
# Extract Pitzer parameters at each tabulated temperature
# Solve: for each T, find [beta0, beta1, C0, C1] that reproduce Table 9
# =============================================================================
def _fit_pitzer_params_at_T(phi_values, m_values, Aphi):
    """Least-squares fit of [beta0, beta1, C0, C1] at one temperature.

    At each molality m_j, the equation is linear in the 4 parameters:
      phi_j - 1 + Aphi*sqrt(m_j)/(1+b*sqrt(m_j))
        = m_j*beta0 + m_j*exp(-alpha*sqrt(m_j))*beta1
        + 2*m_j^2*C0 + 2*m_j^2*exp(-alpha2*sqrt(m_j))*C1
    """
    n = len(m_values)
    A = np.zeros((n, 4))
    y = np.zeros(n)
    for j, m in enumerate(m_values):
        sqrtm = np.sqrt(m)
        # LHS: move the DH term to the right
        y[j] = phi_values[j] - 1.0 + Aphi * sqrtm / (1.0 + B_DH * sqrtm)
        # Design matrix columns: beta0, beta1, C0, C1
        A[j, 0] = m
        A[j, 1] = m * np.exp(-ALPHA * sqrtm)
        A[j, 2] = 2.0 * m**2
        A[j, 3] = 2.0 * m**2 * np.exp(-ALPHA2 * sqrtm)
    # Least squares (5 equations, 4 unknowns)
    result, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
    return result  # [beta0, beta1, C0, C1]


def _build_parameter_interpolators():
    """Build PCHIP interpolators for each Pitzer parameter vs temperature."""
    n_T = len(_TABLE9_T)
    params = np.zeros((n_T, 4))  # [beta0, beta1, C0, C1] at each T

    for i in range(n_T):
        T = _TABLE9_T[i]
        Aphi = debye_hueckel_Aphi(T)
        phi_vals = _TABLE9_PHI[i]
        params[i, :] = _fit_pitzer_params_at_T(phi_vals, _TABLE9_M, Aphi)

    T_arr = np.array(_TABLE9_T)
    interps = []
    for k in range(4):
        interps.append(PchipInterpolator(T_arr, params[:, k]))
    return interps, params


# Build interpolators at module load
_PARAM_INTERPS, _PARAM_TABLE = _build_parameter_interpolators()

_PARAM_NAMES = ['beta0', 'beta1', 'C0', 'C1']


def get_pitzer_params(T_K):
    """Get Pitzer parameters [beta0, beta1, C0, C1] at temperature T_K.

    Interpolated from Archer (1992) Table 9 calibration.
    Valid 273-573 K.
    """
    T_clamp = np.clip(T_K, 273.15, 573.15)
    return [float(interp(T_clamp)) for interp in _PARAM_INTERPS]


# =============================================================================
# Public API
# =============================================================================
def osmotic_coefficient_archer(T_K, m_NaCl):
    """Osmotic coefficient of NaCl(aq) from Archer (1992) Pitzer model.

    Args:
        T_K: Temperature in Kelvin (273-573 K)
        m_NaCl: NaCl molality in mol/kg H2O (0-6)

    Returns:
        Osmotic coefficient phi (dimensionless)
    """
    if m_NaCl <= 0:
        return 1.0
    Aphi = debye_hueckel_Aphi(T_K)
    beta0, beta1, C0, C1 = get_pitzer_params(T_K)
    return osmotic_coeff_pitzer(m_NaCl, beta0, beta1, C0, C1, Aphi)


def water_activity_archer(T_K, m_NaCl):
    """Water activity of NaCl(aq) from Archer (1992) Pitzer model.

    Uses: a_w = exp(-nu * m * M_water * phi)
    where phi is the osmotic coefficient (Archer Eq. 7 + Table 9 calibration).

    Args:
        T_K: Temperature in Kelvin (273-573 K)
        m_NaCl: NaCl molality in mol/kg H2O (0-6)

    Returns:
        Water activity a_w (dimensionless, 0 < a_w <= 1)
    """
    if m_NaCl <= 0:
        return 1.0
    phi = osmotic_coefficient_archer(T_K, m_NaCl)
    return np.exp(-NU * m_NaCl * M_WATER * phi)


def vapor_pressure_ratio(T_K, m_NaCl):
    """Ratio of brine to pure water vapor pressure from Archer (1992).

    For an ideal solution: P_vap,brine / P_vap,pure = a_w
    (exact in the limit of non-volatile solute).

    Args:
        T_K: Temperature in Kelvin (273-573 K)
        m_NaCl: NaCl molality in mol/kg H2O (0-6)

    Returns:
        P_vap,brine / P_vap,pure (dimensionless)
    """
    return water_activity_archer(T_K, m_NaCl)


# =============================================================================
# Conversion utilities
# =============================================================================
def csw_to_molality(csw):
    """Convert NaCl weight percent to molality.

    Args:
        csw: Weight percent NaCl (0-26)

    Returns:
        Molality in mol NaCl per kg H2O
    """
    if csw <= 0:
        return 0.0
    MW_NaCl = 58.443  # g/mol
    return (csw / MW_NaCl) / ((100.0 - csw) / 1000.0)


def molality_to_csw(m):
    """Convert molality to NaCl weight percent.

    Args:
        m: Molality in mol NaCl per kg H2O

    Returns:
        Weight percent NaCl
    """
    if m <= 0:
        return 0.0
    MW_NaCl = 58.443  # g/mol
    g_NaCl = m * MW_NaCl  # grams NaCl per kg water
    return 100.0 * g_NaCl / (1000.0 + g_NaCl)


# =============================================================================
# Validation and reporting
# =============================================================================
def validate_against_table9():
    """Validate implementation against Archer's Table 9 osmotic coefficients."""
    print("Validation against Archer (1992) Table 9")
    print("=" * 78)
    print(f"{'T (K)':>8}  {'m':>6}  {'phi_Archer':>12}  {'phi_calc':>12}  "
          f"{'diff':>10}  {'a_w':>8}")
    print("-" * 78)

    max_err = 0.0
    for i, T in enumerate(_TABLE9_T):
        for j, m in enumerate(_TABLE9_M):
            phi_ref = _TABLE9_PHI[i][j]
            phi_calc = osmotic_coefficient_archer(T, m)
            diff = phi_calc - phi_ref
            aw = water_activity_archer(T, m)
            max_err = max(max_err, abs(diff))
            print(f"{T:8.2f}  {m:6.1f}  {phi_ref:12.4f}  {phi_calc:12.4f}  "
                  f"{diff:10.4f}  {aw:8.4f}")
        print()

    print(f"Maximum |phi_calc - phi_Archer| = {max_err:.6f}")
    print()

    # Show fitted Pitzer parameters at each temperature
    print("Fitted Pitzer parameters at each tabulated temperature:")
    print("-" * 78)
    print(f"{'T (K)':>8}  {'A_phi':>10}  {'beta0':>12}  {'beta1':>12}  "
          f"{'C0':>12}  {'C1':>12}")
    print("-" * 78)
    for i, T in enumerate(_TABLE9_T):
        Aphi = debye_hueckel_Aphi(T)
        b0, b1, c0, c1 = _PARAM_TABLE[i]
        print(f"{T:8.2f}  {Aphi:10.4f}  {b0:12.6f}  {b1:12.6f}  "
              f"{c0:12.6f}  {c1:12.6f}")
    print()


def print_water_activity_grid():
    """Print water activity grid for representative conditions."""
    print("Water activity a_w(T, m) from Archer (1992)")
    print("=" * 78)

    temperatures = [283.15, 323.15, 373.15, 423.15, 473.15, 523.15]
    molalities = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    print(f"{'T (K)':>8} {'T (C)':>7}", end="")
    for m in molalities:
        print(f"  m={m:.1f}", end="")
    print()
    print("-" * 78)

    for T in temperatures:
        print(f"{T:8.2f} {T-273.15:7.1f}", end="")
        for m in molalities:
            aw = water_activity_archer(T, m)
            print(f"  {aw:.4f}", end="")
        print()
    print()

    # Also show Pvap ratio = a_w for comparison with S&W Katz correlation
    print("Comparison with S&W Katz correlation: a_w = 1 - 0.02865*csw^1.44")
    print("-" * 78)
    print(f"{'csw (wt%)':>10} {'m (mol/kg)':>10} {'a_w Archer 25C':>15} "
          f"{'a_w Archer 100C':>15} {'a_w Katz':>10}")
    print("-" * 78)

    for csw in [1, 3, 5, 10, 15, 20, 25]:
        m = csw_to_molality(csw)
        aw_25 = water_activity_archer(298.15, min(m, 6.0))
        aw_100 = water_activity_archer(373.15, min(m, 6.0))
        aw_katz = 1.0 - 0.02865 * csw**1.44
        note = " *" if m > 6.0 else ""
        print(f"{csw:10d} {m:10.3f} {aw_25:15.4f} {aw_100:15.4f} "
              f"{aw_katz:10.4f}{note}")
    print("* extrapolated beyond 6 mol/kg validation range")
    print()


def main():
    validate_against_table9()
    print()
    print_water_activity_grid()

    # Show interpolation at intermediate temperatures
    print("Interpolated phi at intermediate temperatures (m=1.0 mol/kg):")
    print("-" * 50)
    print(f"{'T (K)':>8}  {'T (C)':>7}  {'phi':>8}  {'a_w':>8}")
    print("-" * 50)
    for T_C in range(0, 301, 10):
        T_K = T_C + 273.15
        phi = osmotic_coefficient_archer(T_K, 1.0)
        aw = water_activity_archer(T_K, 1.0)
        print(f"{T_K:8.2f}  {T_C:7.1f}  {phi:8.4f}  {aw:8.4f}")


if __name__ == '__main__':
    main()
