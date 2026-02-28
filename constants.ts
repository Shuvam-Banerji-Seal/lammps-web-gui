
// CPK/Jmol colors for all 118 elements
export const ATOM_COLORS: Record<number, string> = {
  1: "#FFFFFF",   // H  - Hydrogen
  2: "#D9FFFF",   // He - Helium
  3: "#CC80FF",   // Li - Lithium
  4: "#C2FF00",   // Be - Beryllium
  5: "#FFB5B5",   // B  - Boron
  6: "#909090",   // C  - Carbon
  7: "#3050F8",   // N  - Nitrogen
  8: "#FF0D0D",   // O  - Oxygen
  9: "#90E050",   // F  - Fluorine
  10: "#B3E3F5",  // Ne - Neon
  11: "#AB5CF2",  // Na - Sodium
  12: "#8AFF00",  // Mg - Magnesium
  13: "#BFA6A6",  // Al - Aluminium
  14: "#F0C8A0",  // Si - Silicon
  15: "#FF8000",  // P  - Phosphorus
  16: "#FFFF30",  // S  - Sulfur
  17: "#1FF01F",  // Cl - Chlorine
  18: "#80D1E3",  // Ar - Argon
  19: "#8F40D4",  // K  - Potassium
  20: "#3DFF00",  // Ca - Calcium
  21: "#E6E6E6",  // Sc - Scandium
  22: "#BFC2C7",  // Ti - Titanium
  23: "#A6A6AB",  // V  - Vanadium
  24: "#8A99C7",  // Cr - Chromium
  25: "#9C7AC7",  // Mn - Manganese
  26: "#E06633",  // Fe - Iron
  27: "#F090A0",  // Co - Cobalt
  28: "#50D050",  // Ni - Nickel
  29: "#C88033",  // Cu - Copper
  30: "#7D80B0",  // Zn - Zinc
  31: "#C28F8F",  // Ga - Gallium
  32: "#668F8F",  // Ge - Germanium
  33: "#BD80E3",  // As - Arsenic
  34: "#FFA100",  // Se - Selenium
  35: "#A62929",  // Br - Bromine
  36: "#5CB8D1",  // Kr - Krypton
  37: "#702EB0",  // Rb - Rubidium
  38: "#00FF00",  // Sr - Strontium
  39: "#94FFFF",  // Y  - Yttrium
  40: "#94E0E0",  // Zr - Zirconium
  41: "#73C2C9",  // Nb - Niobium
  42: "#54B5B5",  // Mo - Molybdenum
  43: "#3B9E9E",  // Tc - Technetium
  44: "#248F8F",  // Ru - Ruthenium
  45: "#0A7D8C",  // Rh - Rhodium
  46: "#006985",  // Pd - Palladium
  47: "#C0C0C0",  // Ag - Silver
  48: "#FFD98F",  // Cd - Cadmium
  49: "#A67573",  // In - Indium
  50: "#668080",  // Sn - Tin
  51: "#9E63B5",  // Sb - Antimony
  52: "#D47A00",  // Te - Tellurium
  53: "#940094",  // I  - Iodine
  54: "#429EB0",  // Xe - Xenon
  55: "#57178F",  // Cs - Caesium
  56: "#00C900",  // Ba - Barium
  57: "#70D4FF",  // La - Lanthanum
  58: "#FFFFC7",  // Ce - Cerium
  59: "#D9FFC7",  // Pr - Praseodymium
  60: "#C7FFC7",  // Nd - Neodymium
  61: "#A3FFC7",  // Pm - Promethium
  62: "#8FFFC7",  // Sm - Samarium
  63: "#61FFC7",  // Eu - Europium
  64: "#45FFC7",  // Gd - Gadolinium
  65: "#30FFC7",  // Tb - Terbium
  66: "#1FFFC7",  // Dy - Dysprosium
  67: "#00FF9C",  // Ho - Holmium
  68: "#00E675",  // Er - Erbium
  69: "#00D452",  // Tm - Thulium
  70: "#00BF38",  // Yb - Ytterbium
  71: "#00AB24",  // Lu - Lutetium
  72: "#4DC2FF",  // Hf - Hafnium
  73: "#4DA6FF",  // Ta - Tantalum
  74: "#2194D6",  // W  - Tungsten
  75: "#267DAB",  // Re - Rhenium
  76: "#266696",  // Os - Osmium
  77: "#175487",  // Ir - Iridium
  78: "#D0D0E0",  // Pt - Platinum
  79: "#FFD123",  // Au - Gold
  80: "#B8B8D0",  // Hg - Mercury
  81: "#A6544D",  // Tl - Thallium
  82: "#575961",  // Pb - Lead
  83: "#9E4FB5",  // Bi - Bismuth
  84: "#AB5C00",  // Po - Polonium
  85: "#754F45",  // At - Astatine
  86: "#428296",  // Rn - Radon
  87: "#420066",  // Fr - Francium
  88: "#007D00",  // Ra - Radium
  89: "#70ABFA",  // Ac - Actinium
  90: "#00BAFF",  // Th - Thorium
  91: "#00A1FF",  // Pa - Protactinium
  92: "#008FFF",  // U  - Uranium
  93: "#0080FF",  // Np - Neptunium
  94: "#006BFF",  // Pu - Plutonium
  95: "#545CF2",  // Am - Americium
  96: "#785CE3",  // Cm - Curium
  97: "#8A4FE3",  // Bk - Berkelium
  98: "#A136D4",  // Cf - Californium
  99: "#B31FD4",  // Es - Einsteinium
  100: "#B31FBA", // Fm - Fermium
  101: "#B30DA6", // Md - Mendelevium
  102: "#BD0D87", // No - Nobelium
  103: "#C70066", // Lr - Lawrencium
  104: "#CC0059", // Rf - Rutherfordium
  105: "#D1004F", // Db - Dubnium
  106: "#D90045", // Sg - Seaborgium
  107: "#E00038", // Bh - Bohrium
  108: "#E6002E", // Hs - Hassium
  109: "#EB0026", // Mt - Meitnerium
  110: "#EB0026", // Ds - Darmstadtium
  111: "#EB0026", // Rg - Roentgenium
  112: "#EB0026", // Cn - Copernicium
  113: "#EB0026", // Nh - Nihonium
  114: "#EB0026", // Fl - Flerovium
  115: "#EB0026", // Mc - Moscovium
  116: "#EB0026", // Lv - Livermorium
  117: "#EB0026", // Ts - Tennessine
  118: "#EB0026", // Og - Oganesson
};

export const DEFAULT_ATOM_COLOR = "#FF00FF"; // Magenta for unknown

// Van der Waals radii in Angstroms (standard references; estimated where unavailable)
export const ELEMENT_RADII: Record<number, number> = {
  1: 1.20,   // H  - Hydrogen
  2: 1.40,   // He - Helium
  3: 1.82,   // Li - Lithium
  4: 1.53,   // Be - Beryllium
  5: 1.92,   // B  - Boron
  6: 1.70,   // C  - Carbon
  7: 1.55,   // N  - Nitrogen
  8: 1.52,   // O  - Oxygen
  9: 1.47,   // F  - Fluorine
  10: 1.54,  // Ne - Neon
  11: 2.27,  // Na - Sodium
  12: 1.73,  // Mg - Magnesium
  13: 1.84,  // Al - Aluminium
  14: 2.10,  // Si - Silicon
  15: 1.80,  // P  - Phosphorus
  16: 1.80,  // S  - Sulfur
  17: 1.75,  // Cl - Chlorine
  18: 1.88,  // Ar - Argon
  19: 2.75,  // K  - Potassium
  20: 2.31,  // Ca - Calcium
  21: 2.11,  // Sc - Scandium (est.)
  22: 1.87,  // Ti - Titanium (est.)
  23: 1.79,  // V  - Vanadium (est.)
  24: 1.89,  // Cr - Chromium (est.)
  25: 1.97,  // Mn - Manganese (est.)
  26: 1.94,  // Fe - Iron (est.)
  27: 1.92,  // Co - Cobalt (est.)
  28: 1.63,  // Ni - Nickel
  29: 1.40,  // Cu - Copper
  30: 1.39,  // Zn - Zinc
  31: 1.87,  // Ga - Gallium
  32: 2.11,  // Ge - Germanium
  33: 1.85,  // As - Arsenic
  34: 1.90,  // Se - Selenium
  35: 1.85,  // Br - Bromine
  36: 2.02,  // Kr - Krypton
  37: 3.03,  // Rb - Rubidium
  38: 2.49,  // Sr - Strontium
  39: 2.12,  // Y  - Yttrium (est.)
  40: 2.06,  // Zr - Zirconium (est.)
  41: 1.98,  // Nb - Niobium (est.)
  42: 1.90,  // Mo - Molybdenum (est.)
  43: 1.83,  // Tc - Technetium (est.)
  44: 1.82,  // Ru - Ruthenium (est.)
  45: 1.83,  // Rh - Rhodium (est.)
  46: 1.63,  // Pd - Palladium
  47: 1.72,  // Ag - Silver
  48: 1.58,  // Cd - Cadmium
  49: 1.93,  // In - Indium
  50: 2.17,  // Sn - Tin
  51: 2.06,  // Sb - Antimony
  52: 2.06,  // Te - Tellurium
  53: 1.98,  // I  - Iodine
  54: 2.16,  // Xe - Xenon
  55: 3.43,  // Cs - Caesium
  56: 2.68,  // Ba - Barium
  57: 2.40,  // La - Lanthanum (est.)
  58: 2.35,  // Ce - Cerium (est.)
  59: 2.39,  // Pr - Praseodymium (est.)
  60: 2.29,  // Nd - Neodymium (est.)
  61: 2.36,  // Pm - Promethium (est.)
  62: 2.29,  // Sm - Samarium (est.)
  63: 2.33,  // Eu - Europium (est.)
  64: 2.37,  // Gd - Gadolinium (est.)
  65: 2.21,  // Tb - Terbium (est.)
  66: 2.29,  // Dy - Dysprosium (est.)
  67: 2.16,  // Ho - Holmium (est.)
  68: 2.35,  // Er - Erbium (est.)
  69: 2.27,  // Tm - Thulium (est.)
  70: 2.42,  // Yb - Ytterbium (est.)
  71: 2.21,  // Lu - Lutetium (est.)
  72: 2.12,  // Hf - Hafnium (est.)
  73: 2.00,  // Ta - Tantalum (est.)
  74: 1.93,  // W  - Tungsten (est.)
  75: 1.97,  // Re - Rhenium (est.)
  76: 1.85,  // Os - Osmium (est.)
  77: 1.80,  // Ir - Iridium (est.)
  78: 1.75,  // Pt - Platinum
  79: 1.66,  // Au - Gold
  80: 1.55,  // Hg - Mercury
  81: 1.96,  // Tl - Thallium
  82: 2.02,  // Pb - Lead
  83: 2.07,  // Bi - Bismuth
  84: 1.97,  // Po - Polonium (est.)
  85: 2.02,  // At - Astatine (est.)
  86: 2.20,  // Rn - Radon
  87: 3.48,  // Fr - Francium (est.)
  88: 2.83,  // Ra - Radium (est.)
  89: 2.47,  // Ac - Actinium (est.)
  90: 2.45,  // Th - Thorium (est.)
  91: 2.43,  // Pa - Protactinium (est.)
  92: 1.86,  // U  - Uranium
  93: 2.39,  // Np - Neptunium (est.)
  94: 2.43,  // Pu - Plutonium (est.)
  95: 2.44,  // Am - Americium (est.)
  96: 2.45,  // Cm - Curium (est.)
  97: 2.44,  // Bk - Berkelium (est.)
  98: 2.45,  // Cf - Californium (est.)
  99: 2.45,  // Es - Einsteinium (est.)
  100: 2.45, // Fm - Fermium (est.)
  101: 2.46, // Md - Mendelevium (est.)
  102: 2.46, // No - Nobelium (est.)
  103: 2.46, // Lr - Lawrencium (est.)
  104: 2.07, // Rf - Rutherfordium (est.)
  105: 2.04, // Db - Dubnium (est.)
  106: 2.02, // Sg - Seaborgium (est.)
  107: 2.01, // Bh - Bohrium (est.)
  108: 2.00, // Hs - Hassium (est.)
  109: 1.99, // Mt - Meitnerium (est.)
  110: 1.99, // Ds - Darmstadtium (est.)
  111: 1.98, // Rg - Roentgenium (est.)
  112: 1.98, // Cn - Copernicium (est.)
  113: 1.97, // Nh - Nihonium (est.)
  114: 1.97, // Fl - Flerovium (est.)
  115: 1.96, // Mc - Moscovium (est.)
  116: 1.96, // Lv - Livermorium (est.)
  117: 1.95, // Ts - Tennessine (est.)
  118: 1.95, // Og - Oganesson (est.)
};

// Complete periodic table: all 118 elements with IUPAC atomic masses
export const ELEMENT_DATA: { mass: number; symbol: string; name: string; number: number }[] = [
  { mass: 1.008,   symbol: 'H',  name: 'Hydrogen',      number: 1 },
  { mass: 4.0026,  symbol: 'He', name: 'Helium',        number: 2 },
  { mass: 6.941,   symbol: 'Li', name: 'Lithium',       number: 3 },
  { mass: 9.0122,  symbol: 'Be', name: 'Beryllium',     number: 4 },
  { mass: 10.81,   symbol: 'B',  name: 'Boron',         number: 5 },
  { mass: 12.011,  symbol: 'C',  name: 'Carbon',        number: 6 },
  { mass: 14.007,  symbol: 'N',  name: 'Nitrogen',      number: 7 },
  { mass: 15.999,  symbol: 'O',  name: 'Oxygen',        number: 8 },
  { mass: 18.998,  symbol: 'F',  name: 'Fluorine',      number: 9 },
  { mass: 20.180,  symbol: 'Ne', name: 'Neon',          number: 10 },
  { mass: 22.990,  symbol: 'Na', name: 'Sodium',        number: 11 },
  { mass: 24.305,  symbol: 'Mg', name: 'Magnesium',     number: 12 },
  { mass: 26.982,  symbol: 'Al', name: 'Aluminium',     number: 13 },
  { mass: 28.085,  symbol: 'Si', name: 'Silicon',       number: 14 },
  { mass: 30.974,  symbol: 'P',  name: 'Phosphorus',    number: 15 },
  { mass: 32.06,   symbol: 'S',  name: 'Sulfur',        number: 16 },
  { mass: 35.45,   symbol: 'Cl', name: 'Chlorine',      number: 17 },
  { mass: 39.948,  symbol: 'Ar', name: 'Argon',         number: 18 },
  { mass: 39.098,  symbol: 'K',  name: 'Potassium',     number: 19 },
  { mass: 40.078,  symbol: 'Ca', name: 'Calcium',       number: 20 },
  { mass: 44.956,  symbol: 'Sc', name: 'Scandium',      number: 21 },
  { mass: 47.867,  symbol: 'Ti', name: 'Titanium',      number: 22 },
  { mass: 50.942,  symbol: 'V',  name: 'Vanadium',      number: 23 },
  { mass: 51.996,  symbol: 'Cr', name: 'Chromium',      number: 24 },
  { mass: 54.938,  symbol: 'Mn', name: 'Manganese',     number: 25 },
  { mass: 55.845,  symbol: 'Fe', name: 'Iron',          number: 26 },
  { mass: 58.933,  symbol: 'Co', name: 'Cobalt',        number: 27 },
  { mass: 58.693,  symbol: 'Ni', name: 'Nickel',        number: 28 },
  { mass: 63.546,  symbol: 'Cu', name: 'Copper',        number: 29 },
  { mass: 65.38,   symbol: 'Zn', name: 'Zinc',          number: 30 },
  { mass: 69.723,  symbol: 'Ga', name: 'Gallium',       number: 31 },
  { mass: 72.630,  symbol: 'Ge', name: 'Germanium',     number: 32 },
  { mass: 74.922,  symbol: 'As', name: 'Arsenic',       number: 33 },
  { mass: 78.971,  symbol: 'Se', name: 'Selenium',      number: 34 },
  { mass: 79.904,  symbol: 'Br', name: 'Bromine',       number: 35 },
  { mass: 83.798,  symbol: 'Kr', name: 'Krypton',       number: 36 },
  { mass: 85.468,  symbol: 'Rb', name: 'Rubidium',      number: 37 },
  { mass: 87.62,   symbol: 'Sr', name: 'Strontium',     number: 38 },
  { mass: 88.906,  symbol: 'Y',  name: 'Yttrium',       number: 39 },
  { mass: 91.224,  symbol: 'Zr', name: 'Zirconium',     number: 40 },
  { mass: 92.906,  symbol: 'Nb', name: 'Niobium',       number: 41 },
  { mass: 95.95,   symbol: 'Mo', name: 'Molybdenum',    number: 42 },
  { mass: 97.0,    symbol: 'Tc', name: 'Technetium',    number: 43 },
  { mass: 101.07,  symbol: 'Ru', name: 'Ruthenium',     number: 44 },
  { mass: 102.91,  symbol: 'Rh', name: 'Rhodium',       number: 45 },
  { mass: 106.42,  symbol: 'Pd', name: 'Palladium',     number: 46 },
  { mass: 107.87,  symbol: 'Ag', name: 'Silver',        number: 47 },
  { mass: 112.41,  symbol: 'Cd', name: 'Cadmium',       number: 48 },
  { mass: 114.82,  symbol: 'In', name: 'Indium',        number: 49 },
  { mass: 118.71,  symbol: 'Sn', name: 'Tin',           number: 50 },
  { mass: 121.76,  symbol: 'Sb', name: 'Antimony',      number: 51 },
  { mass: 127.60,  symbol: 'Te', name: 'Tellurium',     number: 52 },
  { mass: 126.90,  symbol: 'I',  name: 'Iodine',        number: 53 },
  { mass: 131.29,  symbol: 'Xe', name: 'Xenon',         number: 54 },
  { mass: 132.91,  symbol: 'Cs', name: 'Caesium',       number: 55 },
  { mass: 137.33,  symbol: 'Ba', name: 'Barium',        number: 56 },
  { mass: 138.91,  symbol: 'La', name: 'Lanthanum',     number: 57 },
  { mass: 140.12,  symbol: 'Ce', name: 'Cerium',        number: 58 },
  { mass: 140.91,  symbol: 'Pr', name: 'Praseodymium',  number: 59 },
  { mass: 144.24,  symbol: 'Nd', name: 'Neodymium',     number: 60 },
  { mass: 145.0,   symbol: 'Pm', name: 'Promethium',    number: 61 },
  { mass: 150.36,  symbol: 'Sm', name: 'Samarium',      number: 62 },
  { mass: 151.96,  symbol: 'Eu', name: 'Europium',      number: 63 },
  { mass: 157.25,  symbol: 'Gd', name: 'Gadolinium',    number: 64 },
  { mass: 158.93,  symbol: 'Tb', name: 'Terbium',       number: 65 },
  { mass: 162.50,  symbol: 'Dy', name: 'Dysprosium',    number: 66 },
  { mass: 164.93,  symbol: 'Ho', name: 'Holmium',       number: 67 },
  { mass: 167.26,  symbol: 'Er', name: 'Erbium',        number: 68 },
  { mass: 168.93,  symbol: 'Tm', name: 'Thulium',       number: 69 },
  { mass: 173.05,  symbol: 'Yb', name: 'Ytterbium',     number: 70 },
  { mass: 174.97,  symbol: 'Lu', name: 'Lutetium',      number: 71 },
  { mass: 178.49,  symbol: 'Hf', name: 'Hafnium',       number: 72 },
  { mass: 180.95,  symbol: 'Ta', name: 'Tantalum',      number: 73 },
  { mass: 183.84,  symbol: 'W',  name: 'Tungsten',      number: 74 },
  { mass: 186.21,  symbol: 'Re', name: 'Rhenium',       number: 75 },
  { mass: 190.23,  symbol: 'Os', name: 'Osmium',        number: 76 },
  { mass: 192.22,  symbol: 'Ir', name: 'Iridium',       number: 77 },
  { mass: 195.08,  symbol: 'Pt', name: 'Platinum',      number: 78 },
  { mass: 196.97,  symbol: 'Au', name: 'Gold',          number: 79 },
  { mass: 200.59,  symbol: 'Hg', name: 'Mercury',       number: 80 },
  { mass: 204.38,  symbol: 'Tl', name: 'Thallium',      number: 81 },
  { mass: 207.2,   symbol: 'Pb', name: 'Lead',          number: 82 },
  { mass: 208.98,  symbol: 'Bi', name: 'Bismuth',       number: 83 },
  { mass: 209.0,   symbol: 'Po', name: 'Polonium',      number: 84 },
  { mass: 210.0,   symbol: 'At', name: 'Astatine',      number: 85 },
  { mass: 222.0,   symbol: 'Rn', name: 'Radon',         number: 86 },
  { mass: 223.0,   symbol: 'Fr', name: 'Francium',      number: 87 },
  { mass: 226.0,   symbol: 'Ra', name: 'Radium',        number: 88 },
  { mass: 227.0,   symbol: 'Ac', name: 'Actinium',      number: 89 },
  { mass: 232.04,  symbol: 'Th', name: 'Thorium',       number: 90 },
  { mass: 231.04,  symbol: 'Pa', name: 'Protactinium',  number: 91 },
  { mass: 238.03,  symbol: 'U',  name: 'Uranium',       number: 92 },
  { mass: 237.0,   symbol: 'Np', name: 'Neptunium',     number: 93 },
  { mass: 244.0,   symbol: 'Pu', name: 'Plutonium',     number: 94 },
  { mass: 243.0,   symbol: 'Am', name: 'Americium',     number: 95 },
  { mass: 247.0,   symbol: 'Cm', name: 'Curium',        number: 96 },
  { mass: 247.0,   symbol: 'Bk', name: 'Berkelium',     number: 97 },
  { mass: 251.0,   symbol: 'Cf', name: 'Californium',   number: 98 },
  { mass: 252.0,   symbol: 'Es', name: 'Einsteinium',   number: 99 },
  { mass: 257.0,   symbol: 'Fm', name: 'Fermium',       number: 100 },
  { mass: 258.0,   symbol: 'Md', name: 'Mendelevium',   number: 101 },
  { mass: 259.0,   symbol: 'No', name: 'Nobelium',      number: 102 },
  { mass: 266.0,   symbol: 'Lr', name: 'Lawrencium',    number: 103 },
  { mass: 267.0,   symbol: 'Rf', name: 'Rutherfordium', number: 104 },
  { mass: 268.0,   symbol: 'Db', name: 'Dubnium',       number: 105 },
  { mass: 269.0,   symbol: 'Sg', name: 'Seaborgium',    number: 106 },
  { mass: 270.0,   symbol: 'Bh', name: 'Bohrium',       number: 107 },
  { mass: 277.0,   symbol: 'Hs', name: 'Hassium',       number: 108 },
  { mass: 278.0,   symbol: 'Mt', name: 'Meitnerium',    number: 109 },
  { mass: 281.0,   symbol: 'Ds', name: 'Darmstadtium',  number: 110 },
  { mass: 282.0,   symbol: 'Rg', name: 'Roentgenium',   number: 111 },
  { mass: 285.0,   symbol: 'Cn', name: 'Copernicium',   number: 112 },
  { mass: 286.0,   symbol: 'Nh', name: 'Nihonium',      number: 113 },
  { mass: 289.0,   symbol: 'Fl', name: 'Flerovium',     number: 114 },
  { mass: 290.0,   symbol: 'Mc', name: 'Moscovium',     number: 115 },
  { mass: 293.0,   symbol: 'Lv', name: 'Livermorium',   number: 116 },
  { mass: 294.0,   symbol: 'Ts', name: 'Tennessine',    number: 117 },
  { mass: 294.0,   symbol: 'Og', name: 'Oganesson',     number: 118 },
];