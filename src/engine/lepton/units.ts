/*
 * Angstroms per distance unit of each units style (docs.lammps.org/units.html), as
 * the ZBL function of the Lepton styles needs them (pair_zbl.html: "LAMMPS will
 * automatically convert these values to the distance unit of the specified LAMMPS
 * units setting"). Same table as pair style zbl.
 */
export const ANGSTROM_PER_UNIT: Record<string, number> = {
  lj: 1, real: 1, metal: 1, si: 1e10, cgs: 1e8,
  electron: 0.52917721092, micro: 1e4, nano: 10,
};
