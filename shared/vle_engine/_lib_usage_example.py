#!/usr/bin/env python3
"""
VLE Engine Usage Examples
=========================
Demonstrates the Soreide-Whitson VLE engine for hydrogen-water-brine systems.

Examples include:
- Binary systems (single gas + water/brine)
- Multi-component gas mixtures
- Fresh water and brine cases
- Metric and field (oilfield) units
- Practical engineering units (scf/stb, Sm³/m³, etc.)

"""

from _lib_vle_engine import (
    # Core VLE classes
    SWBinaryVLE,
    H2WaterVLE,

    # Multi-component API
    calc_gas_brine_equilibrium,

    # Unit parsing helpers
    parse_temperature,
    parse_pressure,
    parse_salinity,

    # Unit conversions
    celsius_to_kelvin,
    fahrenheit_to_kelvin,
    bar_to_pascal,
    psia_to_pascal,
    wt_pct_to_molality,
    ppm_to_molality,
    kelvin_to_celsius,
    kelvin_to_fahrenheit,
    pascal_to_bar,
    pascal_to_psia,
    molality_to_wt_pct,
    molality_to_ppm,
    y_h2o_to_stb_mmscf,
    y_h2o_to_lb_mmscf,

    # Component data
    COMPONENTS,
    GAS_SPECIES,

    # Correlations (for advanced use)
    kij_aq_h2,
    get_kij_na,
    sw_equation_8_ks,
)


# =============================================================================
# ENGINEERING UNIT CONVERSIONS
# =============================================================================
# These convert VLE results (mole fractions) to practical engineering units

# --- Constants for unit conversions ---
# Field units (US oilfield)
SCF_PER_LBMOL = 379.5       # scf/lbmol at 60°F, 14.7 psia
LB_WATER_PER_STB = 350.2    # lb water per stock tank barrel at 60°F
MW_WATER = 18.015           # g/mol or lb/lbmol
LBMOL_WATER_PER_STB = LB_WATER_PER_STB / MW_WATER  # ≈ 19.44 lbmol/stb

# Metric units (standard conditions: 15°C, 1 atm)
SM3_PER_KMOL = 23.645       # Sm³/kmol at 15°C, 101.325 kPa
KG_WATER_PER_M3 = 999.1     # kg/m³ water at 15°C
KMOL_WATER_PER_M3 = KG_WATER_PER_M3 / MW_WATER  # ≈ 55.46 kmol/m³

# Normal conditions (0°C, 1 atm) - European convention
NM3_PER_KMOL = 22.414       # Nm³/kmol at 0°C, 101.325 kPa


def x_gas_to_scf_per_stb(x_gas: float) -> float:
    """
    Convert dissolved gas mole fraction to scf/stb.

    Dissolved gas volume at standard conditions (60°F, 14.7 psia)
    per stock tank barrel of liquid.

    Args:
        x_gas: Mole fraction of dissolved gas in liquid phase

    Returns:
        Dissolved gas in scf per stb of liquid
    """
    if x_gas <= 0:
        return 0.0
    # For dilute solutions: moles gas per mole liquid ≈ x_gas
    # scf/stb = (mol gas / mol liq) × (lbmol liq / stb) × (scf / lbmol gas)
    return x_gas * LBMOL_WATER_PER_STB * SCF_PER_LBMOL


def x_gas_to_sm3_per_m3(x_gas: float) -> float:
    """
    Convert dissolved gas mole fraction to Sm³/m³.

    Dissolved gas volume at standard conditions (15°C, 1 atm)
    per cubic meter of liquid.

    Args:
        x_gas: Mole fraction of dissolved gas in liquid phase

    Returns:
        Dissolved gas in Sm³ per m³ of liquid
    """
    if x_gas <= 0:
        return 0.0
    # Sm³/m³ = (mol gas / mol liq) × (kmol liq / m³) × (Sm³ / kmol gas)
    return x_gas * KMOL_WATER_PER_M3 * SM3_PER_KMOL


def x_gas_to_nm3_per_m3(x_gas: float) -> float:
    """
    Convert dissolved gas mole fraction to Nm³/m³.

    Dissolved gas volume at normal conditions (0°C, 1 atm)
    per cubic meter of liquid.

    Args:
        x_gas: Mole fraction of dissolved gas in liquid phase

    Returns:
        Dissolved gas in Nm³ per m³ of liquid
    """
    if x_gas <= 0:
        return 0.0
    return x_gas * KMOL_WATER_PER_M3 * NM3_PER_KMOL


def y_h2o_to_mg_per_sm3(y_H2O: float) -> float:
    """
    Convert water mole fraction in gas to mg/Sm³.

    Water content in milligrams per standard cubic meter of dry gas
    (at 15°C, 1 atm).

    Args:
        y_H2O: Water mole fraction in gas phase

    Returns:
        Water content in mg per Sm³ of dry gas
    """
    if y_H2O <= 0:
        return 0.0
    if y_H2O >= 1:
        return float('inf')
    # mg/Sm³ = (mol H2O / mol dry gas) × (mol dry gas / Sm³) × (g / mol H2O) × 1000 mg/g
    # mol/Sm³ = 1000/SM3_PER_KMOL (since SM3_PER_KMOL is Sm³/kmol)
    mol_per_sm3 = 1000 / SM3_PER_KMOL  # ≈ 42.29 mol/Sm³
    return (y_H2O / (1 - y_H2O)) * mol_per_sm3 * MW_WATER * 1000


def y_h2o_to_g_per_sm3(y_H2O: float) -> float:
    """
    Convert water mole fraction in gas to g/Sm³.

    Args:
        y_H2O: Water mole fraction in gas phase

    Returns:
        Water content in grams per Sm³ of dry gas
    """
    return y_h2o_to_mg_per_sm3(y_H2O) / 1000


def stb_mmscf_to_m3_per_million_sm3(stb_mmscf: float) -> float:
    """
    Convert stb/mmscf to m³/million Sm³.

    Args:
        stb_mmscf: Water content in stb per mmscf

    Returns:
        Water content in m³ per million Sm³
    """
    # 1 stb = 0.15899 m³
    # 1 mmscf = 28316.8 Sm³ (approximate, depends on reference conditions)
    # But more directly: stb/mmscf × 0.15899/28.3168 = stb/mmscf × 5.615e-3 × 1e6/35.3147e6
    # Simpler: 1 stb = 0.15899 m³, 1 scf = 0.02832 Sm³
    # stb/mmscf → m³/MSm³ = stb/mmscf × 0.15899 / 28.3168 × 1000 = stb/mmscf × 5.615
    return stb_mmscf * 5.615


def print_header(title: str, char: str = "="):
    """Print a formatted section header."""
    width = 78
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


# =============================================================================
# EXAMPLE 1: Binary H2-Water System (Metric Units)
# =============================================================================
def example_binary_h2_metric():
    """Binary H2-water VLE with metric units (bar, °C, molal)."""
    print_header("EXAMPLE 1: Binary H2-Water (Metric Units)")

    print("\nConditions:")
    print("  Gas: Pure H2")
    print("  Temperature: 50°C, 100°C, 150°C")
    print("  Pressure: 100 bar, 200 bar")
    print("  Salinity: Fresh water (0 molal)")

    # Create VLE calculator for H2
    vle = SWBinaryVLE('H2', salinity_molal=0.0)

    print(f"\n{'T (°C)':<8} {'P (bar)':<8} {'x_H2':<12} {'Sm³/m³':<10} {'y_H2O':<12} {'mg/Sm³':<10}")
    print("-" * 65)

    for T_C in [50, 100, 150]:
        for P_bar in [100, 200]:
            T_K = celsius_to_kelvin(T_C)
            P_Pa = bar_to_pascal(P_bar)

            x_H2 = vle.calc_gas_solubility(T_K, P_Pa)
            y_H2O = vle.calc_water_content(T_K, P_Pa)

            # Convert to engineering units
            dissolved_sm3 = x_gas_to_sm3_per_m3(x_H2)
            water_mg = y_h2o_to_mg_per_sm3(y_H2O)

            print(f"{T_C:<8} {P_bar:<8} {x_H2:<12.6f} {dissolved_sm3:<10.2f} {y_H2O:<12.6f} {water_mg:<10.1f}")

    print("\nUnits:")
    print("  x_H2: mole fraction of dissolved H2 in liquid")
    print("  Sm³/m³: standard cubic meters of dissolved gas per m³ liquid (at 15°C, 1 atm)")
    print("  y_H2O: mole fraction of water in gas phase")
    print("  mg/Sm³: milligrams of water per Sm³ of dry gas")


# =============================================================================
# EXAMPLE 2: Binary H2-Brine System (Field Units)
# =============================================================================
def example_binary_h2_field():
    """Binary H2-brine VLE with oilfield units (psia, °F, ppm)."""
    print_header("EXAMPLE 2: Binary H2-Brine (Field Units)")

    print("\nConditions:")
    print("  Gas: Pure H2")
    print("  Temperature: 150°F, 200°F, 250°F")
    print("  Pressure: 2000 psia, 3000 psia")
    print("  Salinity: 50,000 ppm NaCl (~0.86 molal)")

    # Convert salinity
    salinity_ppm = 50000
    salinity_molal = ppm_to_molality(salinity_ppm)
    print(f"\n  Salinity conversion: {salinity_ppm:,} ppm = {salinity_molal:.3f} molal")

    # Create VLE calculator for H2 with brine
    vle = SWBinaryVLE('H2', salinity_molal=salinity_molal)

    print(f"\n{'T (°F)':<8} {'P (psia)':<9} {'x_H2':<11} {'scf/stb':<10} {'y_H2O':<11} {'stb/mmscf':<11} {'lb/mmscf':<10}")
    print("-" * 78)

    for T_F in [150, 200, 250]:
        for P_psia in [2000, 3000]:
            T_K = fahrenheit_to_kelvin(T_F)
            P_Pa = psia_to_pascal(P_psia)

            x_H2 = vle.calc_gas_solubility(T_K, P_Pa)
            y_H2O = vle.calc_water_content(T_K, P_Pa)

            # Convert to engineering units
            dissolved_scf = x_gas_to_scf_per_stb(x_H2)
            water_stb = y_h2o_to_stb_mmscf(y_H2O)
            water_lb = y_h2o_to_lb_mmscf(y_H2O)

            print(f"{T_F:<8} {P_psia:<9} {x_H2:<11.6f} {dissolved_scf:<10.1f} {y_H2O:<11.6f} {water_stb:<11.3f} {water_lb:<10.0f}")

    print("\nUnits:")
    print("  x_H2: mole fraction of dissolved H2 in brine")
    print("  scf/stb: standard cubic feet of dissolved gas per stock tank barrel of brine")
    print("  y_H2O: mole fraction of water in gas phase")
    print("  stb/mmscf: stock tank barrels of water per million scf of dry gas")
    print("  lb/mmscf: pounds of water per million scf of dry gas")


# =============================================================================
# EXAMPLE 3: Multiple Gas Species Comparison (Metric)
# =============================================================================
def example_multigas_comparison():
    """Compare solubility of different gases at same conditions."""
    print_header("EXAMPLE 3: Gas Solubility Comparison (Metric)")

    T_C = 75
    P_bar = 150
    salinity_molal = 0.0

    print(f"\nConditions:")
    print(f"  Temperature: {T_C}°C")
    print(f"  Pressure: {P_bar} bar")
    print(f"  Salinity: Fresh water")

    T_K = celsius_to_kelvin(T_C)
    P_Pa = bar_to_pascal(P_bar)

    gases = ['H2', 'CH4', 'CO2', 'N2', 'H2S']

    print(f"\n{'Gas':<6} {'x_gas':<12} {'ppm (mol)':<12} {'Sm³/m³':<10} {'Relative':<10}")
    print("-" * 55)

    results = {}
    for gas in gases:
        try:
            vle = SWBinaryVLE(gas, salinity_molal=salinity_molal)
            x_gas = vle.calc_gas_solubility(T_K, P_Pa)
            results[gas] = x_gas
        except Exception as e:
            results[gas] = None
            print(f"{gas:<6} Error: {e}")

    x_H2 = results.get('H2', 1e-10)
    for gas in gases:
        x_gas = results[gas]
        if x_gas is not None:
            x_ppm = x_gas * 1e6
            sm3_m3 = x_gas_to_sm3_per_m3(x_gas)
            relative = x_gas / x_H2 if x_H2 > 0 else 0
            print(f"{gas:<6} {x_gas:<12.6f} {x_ppm:<12.0f} {sm3_m3:<10.2f} {relative:<10.2f}x")

    print("\nNote: H2S and CO2 are significantly more soluble than H2, CH4, and N2.")
    print("      Sm³/m³ = standard cubic meters gas per m³ liquid at 15°C, 1 atm")


# =============================================================================
# EXAMPLE 4: Salting-Out Effect (Metric)
# =============================================================================
def example_salting_out():
    """Demonstrate salting-out effect on H2 solubility."""
    print_header("EXAMPLE 4: Salting-Out Effect on H2 Solubility")

    T_C = 80
    P_bar = 200

    print(f"\nConditions:")
    print(f"  Temperature: {T_C}°C")
    print(f"  Pressure: {P_bar} bar")
    print(f"  Gas: Pure H2")

    T_K = celsius_to_kelvin(T_C)
    P_Pa = bar_to_pascal(P_bar)

    # Fresh water baseline
    vle_fresh = SWBinaryVLE('H2', salinity_molal=0.0)
    x_fresh = vle_fresh.calc_gas_solubility(T_K, P_Pa)
    sm3_fresh = x_gas_to_sm3_per_m3(x_fresh)

    print(f"\n{'Salinity':<15} {'Molality':<10} {'x_H2':<12} {'Sm³/m³':<10} {'Reduction':<10}")
    print("-" * 60)
    print(f"{'Fresh water':<15} {0.0:<10.2f} {x_fresh:<12.6f} {sm3_fresh:<10.2f} {'-':<10}")

    salinities = [
        ("Seawater", 35000),      # ~35,000 ppm
        ("Formation", 100000),    # 100,000 ppm
        ("High salinity", 200000) # 200,000 ppm
    ]

    for name, ppm in salinities:
        molal = ppm_to_molality(ppm)
        vle = SWBinaryVLE('H2', salinity_molal=molal)
        x_brine = vle.calc_gas_solubility(T_K, P_Pa)
        sm3_brine = x_gas_to_sm3_per_m3(x_brine)
        reduction = (1 - x_brine / x_fresh) * 100 if x_fresh > 0 else 0
        print(f"{name:<15} {molal:<10.2f} {x_brine:<12.6f} {sm3_brine:<10.2f} {reduction:<10.1f}%")

    # Show Sechenov coefficient
    ks = sw_equation_8_ks(T_C, COMPONENTS['H2'].Tb)
    print(f"\nSechenov coefficient (S&W Eq. 8): ks = {ks:.4f}")
    print(f"H2 boiling point used: Tb = {COMPONENTS['H2'].Tb} K")
    print(f"\nNote: Sm³/m³ = standard cubic meters of dissolved H2 per m³ of liquid")


# =============================================================================
# EXAMPLE 5: Multi-Component Gas Mixture (Field Units)
# =============================================================================
def example_multicomponent_field():
    """Multi-component gas mixture with oilfield units."""
    print_header("EXAMPLE 5: Multi-Component Gas (Field Units)")

    print("\nGas Composition (Underground Hydrogen Storage scenario):")
    print("  H2:  85 mol%  (stored hydrogen)")
    print("  CH4: 10 mol%  (cushion gas)")
    print("  CO2:  3 mol%  (impurity)")
    print("  N2:   2 mol%  (impurity)")

    print("\nReservoir Conditions:")
    print("  Temperature: 175°F")
    print("  Pressure: 2500 psia")
    print("  Salinity: 3.5 wt% NaCl (seawater equivalent)")

    # Call multi-component API
    x_gas, water_content = calc_gas_brine_equilibrium(
        salinity_wt_pct=3.5,
        temperature_F=175,
        pressure_psia=2500,
        y_H2=0.85,
        y_CH4=0.10,
        y_CO2=0.03,
        y_N2=0.02
    )

    print("\n" + "=" * 58)
    print(" RESULTS (Field Units)")
    print("=" * 58)

    print("\nDissolved Gas in Brine:")
    print(f"  {'Gas':<6} {'mol/mol':<12} {'ppm (mol)':<12} {'scf/stb':<12}")
    print("  " + "-" * 48)
    total_dissolved = 0
    total_scf = 0
    for gas, x in sorted(x_gas.items(), key=lambda kv: -kv[1]):
        x_ppm = x * 1e6
        scf_stb = x_gas_to_scf_per_stb(x)
        total_dissolved += x
        total_scf += scf_stb
        print(f"  {gas:<6} {x:<12.6f} {x_ppm:<12,.0f} {scf_stb:<12.1f}")
    print("  " + "-" * 48)
    print(f"  {'TOTAL':<6} {total_dissolved:<12.6f} {total_dissolved*1e6:<12,.0f} {total_scf:<12.1f}")

    print("\nWater Content in Gas Phase:")
    print(f"  y_H2O:       {water_content['y_H2O']:.6f} mol/mol ({water_content['y_H2O']*100:.4f} mol%)")
    print(f"  stb/mmscf:   {water_content['stb_mmscf']:.3f}")
    print(f"  lb/mmscf:    {water_content['lb_mmscf']:.0f}")


# =============================================================================
# EXAMPLE 6: Multi-Component Gas Mixture (Metric Units)
# =============================================================================
def example_multicomponent_metric():
    """Multi-component gas mixture with metric units."""
    print_header("EXAMPLE 6: Multi-Component Gas (Metric Units)")

    print("\nGas Composition (Natural Gas with H2 blend):")
    print("  CH4: 80 mol%")
    print("  H2:  15 mol%  (hydrogen blend)")
    print("  CO2:  5 mol%")

    # Convert metric inputs to API units
    T_C = 60
    P_bar = 100
    salinity_molal = 1.0  # ~5.5 wt%

    T_F = T_C * 9/5 + 32
    P_psia = P_bar * 14.5038
    salinity_wt = molality_to_wt_pct(salinity_molal)

    print(f"\nConditions:")
    print(f"  Temperature: {T_C}°C")
    print(f"  Pressure: {P_bar} bar")
    print(f"  Salinity: {salinity_molal} molal ({salinity_wt:.1f} wt%)")

    x_gas, water_content = calc_gas_brine_equilibrium(
        salinity_wt_pct=salinity_wt,
        temperature_F=T_F,
        pressure_psia=P_psia,
        y_CH4=0.80,
        y_H2=0.15,
        y_CO2=0.05
    )

    print("\n" + "=" * 58)
    print(" RESULTS (Metric Units)")
    print("=" * 58)

    print("\nDissolved Gas in Brine:")
    print(f"  {'Gas':<6} {'mol/mol':<12} {'ppm (mol)':<12} {'Sm³/m³':<12}")
    print("  " + "-" * 48)
    total_dissolved = 0
    total_sm3 = 0
    for gas, x in sorted(x_gas.items(), key=lambda kv: -kv[1]):
        x_ppm = x * 1e6
        sm3_m3 = x_gas_to_sm3_per_m3(x)
        total_dissolved += x
        total_sm3 += sm3_m3
        print(f"  {gas:<6} {x:<12.6f} {x_ppm:<12,.0f} {sm3_m3:<12.2f}")
    print("  " + "-" * 48)
    print(f"  {'TOTAL':<6} {total_dissolved:<12.6f} {total_dissolved*1e6:<12,.0f} {total_sm3:<12.2f}")

    print("\nWater Content in Gas Phase:")
    water_mg = y_h2o_to_mg_per_sm3(water_content['y_H2O'])
    water_g = water_mg / 1000
    print(f"  y_H2O:    {water_content['y_H2O']:.6f} mol/mol ({water_content['y_H2O']*100:.4f} mol%)")
    print(f"  mg/Sm³:   {water_mg:.1f}")
    print(f"  g/Sm³:    {water_g:.3f}")


# =============================================================================
# EXAMPLE 7: Temperature Sensitivity Analysis
# =============================================================================
def example_temperature_sensitivity():
    """Show how solubility varies with temperature (U-shaped curve)."""
    print_header("EXAMPLE 7: Temperature Sensitivity (U-Shaped Solubility)")

    P_bar = 100
    salinity_molal = 0.0

    print(f"\nConditions:")
    print(f"  Pressure: {P_bar} bar")
    print(f"  Salinity: Fresh water")
    print(f"  Gas: Pure H2")

    vle = SWBinaryVLE('H2', salinity_molal=salinity_molal)
    P_Pa = bar_to_pascal(P_bar)

    print(f"\n{'T (°C)':<10} {'T (K)':<10} {'kij_AQ':<12} {'x_H2 (mol/mol)':<18}")
    print("-" * 55)

    temperatures = [0, 25, 50, 75, 100, 125, 150]
    results = []

    for T_C in temperatures:
        T_K = celsius_to_kelvin(T_C)
        kij = kij_aq_h2(T_K, salinity_molal)
        x_H2 = vle.calc_gas_solubility(T_K, P_Pa)
        results.append((T_C, x_H2))
        print(f"{T_C:<10} {T_K:<10.1f} {kij:<12.4f} {x_H2:<18.6f}")

    # Find minimum
    min_idx = min(range(len(results)), key=lambda i: results[i][1])
    min_T, min_x = results[min_idx]
    print(f"\nSolubility minimum near {min_T}°C (characteristic U-shape)")


# =============================================================================
# EXAMPLE 8: Using Parse Functions for Flexible Input
# =============================================================================
def example_flexible_units():
    """Demonstrate flexible unit parsing."""
    print_header("EXAMPLE 8: Flexible Unit Input")

    print("\nThe VLE engine supports flexible unit inputs:")
    print("  Temperature: K, C, F, degC, degF, celsius, fahrenheit")
    print("  Pressure: Pa, bar, bara, psia, psi, MPa, kPa")
    print("  Salinity: molal, mol/kg, ppm, wt%, wtpct")

    # Example: same conditions in different units
    cases = [
        ("Metric", 80, "C", 150, "bar", 35000, "ppm"),
        ("Field", 176, "F", 2175.6, "psia", 35000, "ppm"),
        ("SI", 353.15, "K", 15000000, "Pa", 0.599, "molal"),
    ]

    print(f"\n{'System':<10} {'T input':<15} {'P input':<15} {'Sal input':<18} {'x_H2':<15}")
    print("-" * 80)

    for name, T_val, T_unit, P_val, P_unit, sal_val, sal_unit in cases:
        T_K = parse_temperature(T_val, T_unit)
        P_Pa = parse_pressure(P_val, P_unit)
        sal_molal = parse_salinity(sal_val, sal_unit)

        vle = SWBinaryVLE('H2', salinity_molal=sal_molal)
        x_H2 = vle.calc_gas_solubility(T_K, P_Pa)

        T_str = f"{T_val} {T_unit}"
        P_str = f"{P_val} {P_unit}"
        sal_str = f"{sal_val} {sal_unit}"
        print(f"{name:<10} {T_str:<15} {P_str:<15} {sal_str:<18} {x_H2:<15.6f}")

    print("\nNote: All three cases represent the same physical conditions,")
    print("      confirming unit conversion consistency.")


# =============================================================================
# EXAMPLE 9: Water Content for Dehydration Design
# =============================================================================
def example_dehydration_design():
    """Water content calculations for gas dehydration design."""
    print_header("EXAMPLE 9: Gas Dehydration Design")

    print("\nScenario: Design dehydration for H2 storage withdrawal")
    print("  Reservoir gas: 95% H2, 5% CH4 (cushion gas)")
    print("  Pipeline spec: <7 lb/mmscf water content (US) or <112 mg/Sm³ (Europe)")

    salinity_wt = 5.0  # 5 wt% brine

    print(f"\nBrine salinity: {salinity_wt} wt%")

    # Field units table
    print("\n--- Field Units ---\n")
    print(f"{'T (°F)':<8} {'P (psia)':<10} {'y_H2O (%)':<11} {'lb/mmscf':<11} {'stb/mmscf':<11} {'Spec?':<8}")
    print("-" * 62)

    conditions = [
        (120, 1500),
        (120, 2000),
        (150, 1500),
        (150, 2000),
        (180, 1500),
        (180, 2000),
    ]

    for T_F, P_psia in conditions:
        x_gas, water = calc_gas_brine_equilibrium(
            salinity_wt_pct=salinity_wt,
            temperature_F=T_F,
            pressure_psia=P_psia,
            y_H2=0.95,
            y_CH4=0.05
        )

        y_pct = water['y_H2O'] * 100
        lb_mmscf = water['lb_mmscf']
        stb_mmscf = water['stb_mmscf']
        meets_spec = "YES" if lb_mmscf < 7 else "NO"

        print(f"{T_F:<8} {P_psia:<10} {y_pct:<11.4f} {lb_mmscf:<11.1f} {stb_mmscf:<11.3f} {meets_spec:<8}")

    # Metric units table
    print("\n--- Metric Units ---\n")
    print(f"{'T (°C)':<8} {'P (bar)':<10} {'y_H2O (%)':<11} {'mg/Sm³':<11} {'g/Sm³':<11} {'Spec?':<8}")
    print("-" * 62)

    for T_F, P_psia in conditions:
        T_C = (T_F - 32) * 5/9
        P_bar = P_psia / 14.5038

        x_gas, water = calc_gas_brine_equilibrium(
            salinity_wt_pct=salinity_wt,
            temperature_F=T_F,
            pressure_psia=P_psia,
            y_H2=0.95,
            y_CH4=0.05
        )

        y_pct = water['y_H2O'] * 100
        mg_sm3 = y_h2o_to_mg_per_sm3(water['y_H2O'])
        g_sm3 = mg_sm3 / 1000
        meets_spec = "YES" if mg_sm3 < 112 else "NO"

        print(f"{T_C:<8.1f} {P_bar:<10.1f} {y_pct:<11.4f} {mg_sm3:<11.1f} {g_sm3:<11.3f} {meets_spec:<8}")

    print("\nNotes:")
    print("  - Higher pressure and lower temperature reduce water content")
    print("  - US pipeline spec: typically <7 lb/mmscf (varies by operator)")
    print("  - European spec: typically <50-150 mg/Sm³ depending on application")


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print(" SOREIDE-WHITSON VLE ENGINE - USAGE EXAMPLES")
    print(" Hydrogen-Water-Brine Phase Equilibria")
    print("=" * 70)

    print("\nThis script demonstrates the VLE engine capabilities for:")
    print("  - Binary gas-water/brine systems")
    print("  - Multi-component gas mixtures")
    print("  - Various unit systems (metric and field)")
    print("  - Underground hydrogen storage applications")

    # Run all examples
    example_binary_h2_metric()
    example_binary_h2_field()
    example_multigas_comparison()
    example_salting_out()
    example_multicomponent_field()
    example_multicomponent_metric()
    example_temperature_sensitivity()
    example_flexible_units()
    example_dehydration_design()

    print_header("EXAMPLES COMPLETE")
    print("\nFor more information, see:")
    print("  - vle_engine.py: Full VLE implementation")
    print("  - manuscript/manuscript.tex: Technical details and correlations")
    print("  - GitHub: github.com/mwburgoyne/H2-Solubility-In-S-W-Framework")


if __name__ == "__main__":
    main()
