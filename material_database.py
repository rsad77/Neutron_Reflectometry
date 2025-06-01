from dataclasses import dataclass

@dataclass
class Material:
    name: str
    u_real: float  # Real part of scattering length density (10⁻⁶ Å⁻²)
    u_imag: float = 0.0  # Imaginary part (absorption, 10⁻⁶ Å⁻²)
    density: float = None  # g/cm³
    formula: str = None
    M_x: float = 0.0  # Magnetic moment x-component (T)
    M_y: float = 0.0  # Magnetic moment y-component (T)
    M_z: float = 0.0  # Magnetic moment z-component (T)

MATERIALS = {
    'Ni': Material('Nickel', 9.408, 0.0, None, 'Ni', 0.0, 0.0, 0.3),
    'Nb': Material('Niobium', 3.919, 0.0, None, 'Nb', 0.0, 0.0, 0),
    'Si': Material('Silicon', 2.074, 0.0, None, 'Si', 0.0, 0.0, 0),
    'Ti': Material('Titanium', -1.925, -0.001, None, 'Ti', 0.0, 0.0, 0.0),
    'D2O': Material('Heavy Water', 5.75, 0.0, None, 'D₂O', 0.0, 0.0, 0.0),
    'H2O': Material('Water', -0.561, 0.0, None, 'H₂O', 0.0, 0.0, 0.0),
    'Al': Material('Aluminum', 2.078, 0.0, None, 'Al', 0.0, 0.0, 0.0),
    'Pd': Material('Palladium', 4.020, -0.001, None, 'Pd', 0.0, 0.0, 0.0),
    'Co': Material('Cobalt', 2.265, -0.009, None, 'Co', 0.0, 0.0, 1.8),
    'Cu': Material('Copper', 6.554, -0.001, None, 'Cu', 0.0, 0.0, 0.0),
    'Cr': Material('Chromium', 3.027, -0.001, None, 'Cr', 0.0, 0.0, 0.0),
    'Au': Material('Gold', 4.662, -0.016, None, 'Au', 0.0, 0.0, 0.0),
    'Pt': Material('Platinum', 6.357, -0.002, None, 'Pt', 0.0, 0.0, 0.0),
    'Zr': Material('Zirconium', 3.075, 0.00, None, 'Zr', 0.0, 0.0, 0.0),
    'NiH': Material('Hydride Nickel', 0.662, 0.0, None, 'NiH', 0.0, 0.0, 0.0),
    'ZrH': Material('Hydride Zirconium', 0.223, 0.0, None, 'ZrH', 0.0, 0.0, 0.0),
    'NbH': Material('Hydride Niobium', 0.212, 0.0, None, 'NbH', 0.0, 0.0, 0.0),
    'PdH': Material('Hydride Paladium', 0.122, 0.0, None, 'PdH', 0.0, 0.0, 0.0),
    'NiD': Material('Deuteride Nickel', 1.683, 0.0, None, 'NiD', 0.0, 0.0, 0.0),
    'ZrD': Material('Deuteride Zirconium', 0.893, 0.0, None, 'ZrD', 0.0, 0.0, 0.0),
    'NbD': Material('Deuteride Niobium', 0.871, 0.0, None, 'NbD', 0.0, 0.0, 0.0),
    'PdD': Material('Deuteride Paladium', 0.699, 0.0, None, 'PdD', 0.0, 0.0, 0.0),
    'Al2O3': Material('Sapphire', 1.436, 0.0, None, 'Al₂O₃', 0.0, 0.0, 0),
    'Fe': Material('Iron', 8.024, -0.001, None, 'Fe', 0.0, 0.0, 2.1),
    'Gd': Material('Gadolinium', 6.5, -0.001, None, 'Gd', 0.0, 0.0, 7.6),
}
