
export const ATOM_COLORS: Record<number, string> = {
  1: "#FFFFFF",   // H  - Hydrogen (White)
  2: "#D9FFFF",   // He - Helium (Cyan-White)
  3: "#CC80FF",   // Li - Lithium (Violet)
  4: "#C2FF00",   // Be - Beryllium (Yellow-Green)
  5: "#FFB5B5",   // B  - Boron (Pinkish)
  6: "#909090",   // C  - Carbon (Grey)
  7: "#3050F8",   // N  - Nitrogen (Blue)
  8: "#FF0D0D",   // O  - Oxygen (Red)
  9: "#90E050",   // F  - Fluorine (Green)
  10: "#B3E3F5",  // Ne - Neon (Cyan)
  11: "#AB5CF2",  // Na - Sodium (Violet)
  12: "#8AFF00",  // Mg - Magnesium (Green)
  13: "#BFA6A6",  // Al - Aluminium (Grey-Pink)
  14: "#F0C8A0",  // Si - Silicon (Tan)
  15: "#FF8000",  // P  - Phosphorus (Orange)
  16: "#FFFF30",  // S  - Sulfur (Yellow)
  17: "#1FF01F",  // Cl - Chlorine (Green)
  18: "#80D1E3",  // Ar - Argon (Cyan)
  19: "#8F40D4",  // K  - Potassium (Violet)
  20: "#3DFF00",  // Ca - Calcium (Green)
  21: "#E6E6E6",  // Sc - Scandium
  22: "#BFC2C7",  // Ti - Titanium (Grey)
  23: "#A6A6AB",  // V  - Vanadium
  24: "#8A99C7",  // Cr - Chromium
  25: "#9C7AC7",  // Mn - Manganese
  26: "#E06633",  // Fe - Iron (Orange-Red)
  27: "#F090A0",  // Co - Cobalt
  28: "#50D050",  // Ni - Nickel (Green)
  29: "#C88033",  // Cu - Copper (Brown)
  30: "#7D80B0",  // Zn - Zinc (Slate)
  31: "#C28F8F",  // Ga - Gallium
  32: "#668F8F",  // Ge - Germanium
  33: "#BD80E3",  // As - Arsenic
  34: "#FFA100",  // Se - Selenium
  35: "#A62929",  // Br - Bromine (Dark Red)
  36: "#5CB8D1",  // Kr - Krypton
  37: "#702EB0",  // Rb - Rubidium
  38: "#00FF00",  // Sr - Strontium
  46: "#006985",  // Pd - Palladium
  47: "#C0C0C0",  // Ag - Silver
  48: "#FFD98F",  // Cd - Cadmium
  50: "#668080",  // Sn - Tin
  53: "#940094",  // I  - Iodine (Purple)
  54: "#429EB0",  // Xe - Xenon
  78: "#D0D0E0",  // Pt - Platinum
  79: "#FFD123",  // Au - Gold
  80: "#B8B8D0",  // Hg - Mercury
  82: "#575961",  // Pb - Lead
  92: "#008FFF",  // U  - Uranium
};

export const DEFAULT_ATOM_COLOR = "#FF00FF"; // Magenta for unknown

// Helper to guess element from mass
export const ELEMENT_DATA: { mass: number; symbol: string; name: string; number: number }[] = [
  { mass: 1.008, symbol: 'H', name: 'Hydrogen', number: 1 },
  { mass: 4.003, symbol: 'He', name: 'Helium', number: 2 },
  { mass: 6.941, symbol: 'Li', name: 'Lithium', number: 3 },
  { mass: 9.012, symbol: 'Be', name: 'Beryllium', number: 4 },
  { mass: 10.81, symbol: 'B', name: 'Boron', number: 5 },
  { mass: 12.01, symbol: 'C', name: 'Carbon', number: 6 },
  { mass: 14.01, symbol: 'N', name: 'Nitrogen', number: 7 },
  { mass: 16.00, symbol: 'O', name: 'Oxygen', number: 8 },
  { mass: 19.00, symbol: 'F', name: 'Fluorine', number: 9 },
  { mass: 20.18, symbol: 'Ne', name: 'Neon', number: 10 },
  { mass: 22.99, symbol: 'Na', name: 'Sodium', number: 11 },
  { mass: 24.31, symbol: 'Mg', name: 'Magnesium', number: 12 },
  { mass: 26.98, symbol: 'Al', name: 'Aluminium', number: 13 },
  { mass: 28.09, symbol: 'Si', name: 'Silicon', number: 14 },
  { mass: 30.97, symbol: 'P', name: 'Phosphorus', number: 15 },
  { mass: 32.07, symbol: 'S', name: 'Sulfur', number: 16 },
  { mass: 35.45, symbol: 'Cl', name: 'Chlorine', number: 17 },
  { mass: 39.95, symbol: 'Ar', name: 'Argon', number: 18 },
  { mass: 39.10, symbol: 'K', name: 'Potassium', number: 19 },
  { mass: 40.08, symbol: 'Ca', name: 'Calcium', number: 20 },
  { mass: 47.87, symbol: 'Ti', name: 'Titanium', number: 22 },
  { mass: 55.85, symbol: 'Fe', name: 'Iron', number: 26 },
  { mass: 63.55, symbol: 'Cu', name: 'Copper', number: 29 },
  { mass: 107.9, symbol: 'Ag', name: 'Silver', number: 47 },
  { mass: 196.97, symbol: 'Au', name: 'Gold', number: 79 },
];