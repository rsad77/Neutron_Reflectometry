from dataclasses import dataclass

@dataclass
class Material:
    name: str
    u_real: float  # Real part of scattering length density (10⁻⁶ Å⁻²)
    u_imag: float = 0.0  # Imaginary part (absorption, 10⁻⁶ Å⁻²)
    density: float = None  # g/cm³
    formula: str = None
    B_x: float = 0.0  # Magnetic moment x-component (T)
    B_y: float = 0.0  # Magnetic moment y-component (T)
    B_z: float = 0.0  # Magnetic moment z-component (T)

MATERIALS = {
    'Ni': Material('Nickel', 9.408, 0.0, 8.90, 'Ni', 0, 0, 0.61),  # Ni ферромагнетик (~0.61 T)
    'Nb': Material('Niobium', 3.919, 0.0, 8.57, 'Nb', 0, 0, 0),    # Сверхпроводник (диамагнетик)
    'Si': Material('Silicon', 2.074, 0.0, 2.33, 'Si', 0, 0, 0),    # Диамагнетик
    'Ti': Material('Titanium', -1.925, -0.001, 4.54, 'Ti', 0, 0, 0),  # Парамагнетик (слабый)
    'D2O': Material('Heavy Water', 5.75, 0.0, 1.00, 'D₂O', 0, 0, 0),  # Немагнитный
    'H2O': Material('Water', -0.561, 0.0, 1.00, 'H₂O', 0, 0, 0),     # Немагнитный
    'Al': Material('Aluminum', 2.078, 0.0, 2.70, 'Al', 0, 0, 0),     # Парамагнетик
    'Pd': Material('Palladium', 4.020, -0.001, 12.02, 'Pd', 0, 0, 0),  # Парамагнетик
    'Co': Material('Cobalt', 2.265, -0.009, 8.90, 'Co', 0, 0, 1.76),   # Co ферромагнетик (~1.76 T)
    'Cu': Material('Copper', 6.554, -0.001, 8.96, 'Cu', 0, 0, 0),      # Диамагнетик
    'Cr': Material('Chromium', 3.027, -0.001, 7.19, 'Cr', 0, 0, 0),    # Антиферромагнетик (нет чистого момента)
    'Au': Material('Gold', 4.662, -0.016, 19.30, 'Au', 0, 0, 0),       # Диамагнетик
    'Pt': Material('Platinum', 6.357, -0.002, 21.45, 'Pt', 0, 0, 0),   # Парамагнетик
    'Zr': Material('Zirconium', 3.075, 0.00, 6.51, 'Zr', 0, 0, 0),     # Парамагнетик
    'NiH': Material('Hydride Nickel', 0.662, 0.0, None, 'NiH', 0, 0, 0),  # Слабый магнетик
    'ZrH': Material('Hydride Zirconium', 0.223, 0.0, None, 'ZrH', 0, 0, 0),  # Немагнитный
    'NbH': Material('Hydride Niobium', 0.212, 0.0, None, 'NbH', 0, 0, 0),   # Немагнитный
    'PdH': Material('Hydride Paladium', 0.122, 0.0, None, 'PdH', 0, 0, 0),   # Немагнитный
    'NiD': Material('Deuteride Nickel', 1.683, 0.0, None, 'NiD', 0, 0, 0),  # Слабый магнетик
    'ZrD': Material('Deuteride Zirconium', 0.893, 0.0, None, 'ZrD', 0, 0, 0),  # Немагнитный
    'NbD': Material('Deuteride Niobium', 0.871, 0.0, None, 'NbD', 0, 0, 0),    # Немагнитный
    'PdD': Material('Deuteride Palladium', 0.699, 0.0, None, 'PdD', 0, 0, 0),  # Немагнитный
    'Al2O3': Material('Sapphire', 1.436, 0.0, 3.95, 'Al₂O₃', 0, 0, 0),         # Диамагнетик
    'U': Material('Uranium', 4.035, -0.001, 18.95, 'U', 0, 0, 0),              # Парамагнетик (антиферро при низких T)
    'Gd': Material('Gadolineum', 4.150, -2.171, 7.90, 'Gd', 0, 0, 2.2),       # Gd ферромагнетик (~2.2 T)
    'Ge': Material('Germanium', 3.613, 0.000, 5.32, 'Ge', 0, 0, 0)            # Диамагнетик
}