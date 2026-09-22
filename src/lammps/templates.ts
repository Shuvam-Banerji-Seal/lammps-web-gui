/**
 * Ready-made starter pipelines for the Script Builder.
 * Every template is assembled from catalog CommandDefs (so ids stay valid)
 * and is guaranteed to generate with ZERO warnings — enforced by tests.
 *
 * Sources: docs.lammps.org example scripts (in.lj, in.eam, log kspace
 * examples) — parameter values are the docs' canonical demo values.
 */

import {
  COMMAND_BY_ID,
  ScriptModel,
  ScriptStep,
  defaultParams,
} from './catalog';

export interface ScriptTemplate {
  id: string;
  label: string;
  description: string;
  /** defId + param overrides, in pipeline (emission) order. */
  steps: { defId: string; params?: Record<string, string> }[];
}

const step = (defId: string, params?: Record<string, string>, note?: string): ScriptStep => {
  const def = COMMAND_BY_ID[defId];
  if (!def) throw new Error(`template references unknown command ${defId}`);
  return {
    uid: `tpl-${defId}-${Math.random().toString(36).slice(2, 8)}`,
    defId,
    params: { ...defaultParams(def), ...(params ?? {}) },
    enabled: true,
    note,
  };
};

export const buildTemplate = (tpl: ScriptTemplate): ScriptModel => ({
  title: tpl.label,
  steps: tpl.steps.map(s => step(s.defId, s.params)),
});

export const SCRIPT_TEMPLATES: ScriptTemplate[] = [
  {
    id: 'lj-nvt',
    label: 'Lennard-Jones fluid (NVT)',
    description: 'Canonical LJ liquid: fcc lattice → NVT at T=1.0, lj units.',
    steps: [
      { defId: 'units', params: { style: 'lj' } },
      { defId: 'dimension' },
      { defId: 'boundary' },
      { defId: 'atom_style_cmd', params: { style: 'atomic' } },
      // create_atoms fills from the CURRENT lattice — without this the default
      // `lattice none 1.0` applies and the fill is undefined
      // (docs.lammps.org/create_atoms.html, Restrictions).
      { defId: 'lattice', params: { style: 'fcc', scale: '0.8442' } },
      { defId: 'region_block', params: { id: 'box', xhi: '10', yhi: '10', zhi: '10' } },
      { defId: 'create_box', params: { ntypes: '1', region: 'box' } },
      { defId: 'create_atoms', params: { type: '1', where: 'box' } },
      { defId: 'mass', params: { type: '1', value: '1.0' } },
      { defId: 'velocity_create', params: { temp: '1.0', seed: '4928459' } },
      { defId: 'pair_style_popular', params: { style: 'lj/cut', args: '2.5' } },
      { defId: 'pair_coeff', params: { types: '* *', coeffs: '1.0 1.0 2.5' } },
      { defId: 'neighbor' },
      { defId: 'timestep' },
      { defId: 'thermo_style', params: { fields: 'step temp pe press vol density' } },
      { defId: 'thermo' },
      { defId: 'dump_custom', params: { id: 'traj', file: 'traj.lammpstrj', fields: 'id type x y z vx vy vz' } },
      { defId: 'fix_nvt', params: { temp_start: '1.0', temp_end: '1.0', temp_damp: '0.5' } },
      { defId: 'run', params: { steps: '50000' } },
    ],
  },
  {
    id: 'eam-metal',
    label: 'Metal (EAM) minimize + NPT',
    description: 'Read a metal data file, EAM potential, relax, then NPT at 300 K.',
    steps: [
      { defId: 'units', params: { style: 'metal' } },
      { defId: 'boundary' },
      { defId: 'atom_style_cmd', params: { style: 'atomic' } },
      { defId: 'read_data', params: { file: 'metal.data' } },
      // `pair_style eam` reads funcfl files, which are single-element: the docs
      // require one pair_coeff per I,I pair with a single filename argument —
      // `pair_coeff * * file` is only legal for eam/alloy and eam/fs.
      { defId: 'pair_style_popular', params: { style: 'eam', args: '' } },
      { defId: 'pair_coeff', params: { types: '1 1', coeffs: 'Cu_u3.eam' } },
      { defId: 'neighbor' },
      { defId: 'fix_minimize' },
      { defId: 'fix_npt', params: { temp_start: '300', temp_end: '300', temp_damp: '0.1', p_start: '0.0', p_end: '0.0', p_damp: '1.0' } },
      { defId: 'timestep', params: { dt: '0.003' } },
      { defId: 'thermo_style', params: { fields: 'step temp pe etotal press vol' } },
      { defId: 'dump_custom', params: { file: 'metal.lammpstrj' } },
      { defId: 'run', params: { steps: '20000' } },
    ],
  },
  {
    id: 'reaxff',
    label: 'Reactive chemistry (ReaxFF)',
    description: 'Charge-tagged system with ReaxFF + charge equilibration, NVT.',
    steps: [
      { defId: 'units', params: { style: 'real' } },
      { defId: 'boundary' },
      { defId: 'atom_style_cmd', params: { style: 'charge' } },
      { defId: 'read_data', params: { file: 'reactive.data' } },
      // The control file is an argument of pair_style (NULL = built-in
      // defaults); pair_coeff takes the force field plus the element mapping.
      { defId: 'pair_style_popular', params: { style: 'reaxff', args: 'NULL' } },
      { defId: 'pair_coeff', params: { types: '* *', coeffs: 'ffield.reax C H O' } },
      // LAMMPS requires a charge-equilibration fix with pair_style reaxff.
      // SHAKE is deliberately absent: constraining bonds contradicts a
      // reactive potential (and atom_style charge carries no bond topology).
      { defId: 'fix_qeq_reaxff', params: { id: 'qeq', group: 'all' } },
      { defId: 'fix_langevin', params: { temp_start: '300', temp_end: '300', damp: '100.0' } },
      { defId: 'fix_nve' },
      { defId: 'timestep', params: { dt: '0.25' } },
      { defId: 'thermo_style', params: { fields: 'step temp press etotal density' } },
      { defId: 'run', params: { steps: '100000' } },
    ],
  },
  {
    id: 'granular-pour',
    label: 'Granular pouring',
    description: 'Spheres raining onto a flat wall under gravity (granular pair style).',
    // Modelled on lammps/lammps examples/granular/in.pour.flatwall — granular
    // runs need atom_style sphere (per-particle mass, so NO `mass` command),
    // `comm_modify vel yes`, an nve/sphere integrator, a wall/gran wall and a
    // separate insertion region for fix pour.
    steps: [
      { defId: 'units', params: { style: 'lj' } },
      { defId: 'dimension', params: { n: '3' } },
      { defId: 'atom_style_cmd', params: { style: 'sphere' } },
      { defId: 'boundary', params: { bx: 'p', by: 'p', bz: 'f' } },
      { defId: 'comm_modify', params: { args: 'vel yes' } },
      { defId: 'neighbor', params: { skin: '0.15', style: 'bin' } },
      { defId: 'neigh_modify', params: { args: 'delay 0 every 1 check yes' } },
      { defId: 'region_block', params: { id: 'boxreg', xhi: '20', yhi: '20', zhi: '30', units: 'box' } },
      { defId: 'create_box', params: { ntypes: '1', region: 'boxreg' } },
      { defId: 'pair_style_popular', params: { style: 'granular', args: '' } },
      { defId: 'pair_coeff', params: { types: '* *', coeffs: 'hertz/material 1e5 1e3 0.3 tangential mindlin NULL 1.0 0.5' } },
      // insertion volume: a cylinder high above the wall
      { defId: 'region_cylinder', params: { id: 'insreg', axis: 'z', c1: '10', c2: '10', radius: '5', lo: '15', hi: '30', units: 'box' } },
      { defId: 'fix_any', params: { id: 'integrate', group: 'all', style: 'nve/sphere', args: '' } },
      { defId: 'fix_gravity', params: { id: 'grav', magnitude: '10.0', direction: 'vector 0 0 -1' } },
      { defId: 'fix_any', params: { id: 'wall', group: 'all', style: 'wall/gran', args: 'granular hertz/material 1e5 1e3 0.3 tangential mindlin NULL 1.0 0.5 zplane 0 NULL' } },
      { defId: 'fix_pour', params: { id: 'ins', group: 'all', n: '1500', type: '1', seed: '3123', region: 'insreg', diam: 'range 0.5 1.0', vol: '', rate: '0', extra: 'dens 1.0 1.0' } },
      { defId: 'timestep', params: { dt: '0.001' } },
      { defId: 'thermo_style', params: { fields: 'step atoms ke' } },
      { defId: 'thermo', params: { n: '100' } },
      { defId: 'dump_custom', params: { id: 'traj', file: 'pour.lammpstrj', fields: 'id type radius mass x y z' } },
      { defId: 'run', params: { steps: '25000' } },
    ],
  },
  {
    id: 'shear-nemd',
    label: 'Shear flow (NEMD)',
    description: 'fix deform shear (remap v) thermostatted with fix nvt/sllod.',
    steps: [
      { defId: 'units', params: { style: 'lj' } },
      { defId: 'boundary' },
      { defId: 'atom_style_cmd', params: { style: 'atomic' } },
      { defId: 'read_data', params: { file: 'fluid.data' } },
      { defId: 'pair_style_popular', params: { style: 'lj/cut', args: '2.5' } },
      { defId: 'pair_coeff', params: { types: '* *', coeffs: '1.0 1.0 2.5' } },
      { defId: 'fix_deform', params: { param: 'xy', style: 'erate', v1: '0.1', remap: 'v', units: 'box' } },
      // Plain `fix nvt` thermostats the TOTAL velocity, which under shear
      // includes the streaming profile and gives the wrong temperature.
      // fix nvt/sllod creates its own `compute temp/deform` and pairs with
      // `fix deform ... remap v` (docs.lammps.org/fix_nvt_sllod.html).
      { defId: 'fix_any', params: { id: 'integrate', group: 'all', style: 'nvt/sllod', args: 'temp 1.0 1.0 0.5' } },
      { defId: 'compute_rdf', params: { id: 'myRDF', nbins: '200' } },
      { defId: 'fix_ave_time', params: { values: 'c_myRDF[*]', mode: 'vector', file: 'rdf.txt', ave: 'running' } },
      { defId: 'thermo_style', params: { fields: 'step temp press xy' } },
      { defId: 'timestep' },
      { defId: 'run', params: { steps: '100000' } },
    ],
  },
];
