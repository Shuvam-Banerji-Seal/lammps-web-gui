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
  /** defId + param overrides, in emission order (sections reorder anyway). */
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
      { defId: 'region_block', params: { id: 'box', xhi: '16.8', yhi: '16.8', zhi: '16.8', units: 'box' } },
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
      { defId: 'pair_style_popular', params: { style: 'eam', args: '' } },
      { defId: 'pair_coeff', params: { types: '* *', coeffs: 'Cu_u3.eam' } },
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
      { defId: 'pair_style_popular', params: { style: 'reaxff', args: '' } },
      { defId: 'pair_coeff', params: { types: '* *', coeffs: 'ffield.reax control.reax' } },
      { defId: 'fix_shake', params: { group: 'water', bonds: 'b 1', angles: 'a 1' } },
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
    description: 'Sphere particles raining into a box under gravity (si units).',
    steps: [
      { defId: 'units', params: { style: 'si' } },
      { defId: 'dimension', params: { n: '3' } },
      { defId: 'boundary', params: { bz: 'f' } },
      { defId: 'atom_style_cmd', params: { style: 'sphere' } },
      { defId: 'region_block', params: { id: 'box', xhi: '0.02', yhi: '0.02', zhi: '0.06', units: 'box' } },
      { defId: 'create_box', params: { ntypes: '1', region: 'box' } },
      { defId: 'neighbor' },
      { defId: 'fix_gravity', params: { magnitude: '9.81', direction: 'down' } },
      { defId: 'fix_wall_reflect', params: { face: 'zlo', pos: '0.0' } },
      { defId: 'fix_nve' },
      { defId: 'fix_pour', params: { n: '2000', type: '1', seed: '12345', region: 'box', diam: 'range 0.0005 0.0009', vol: '0.3 10', rate: '0.001' } },
      { defId: 'timestep', params: { dt: '1e-6' } },
      { defId: 'thermo_style', params: { fields: 'step atoms ke temp' } },
      { defId: 'dump_custom', params: { file: 'pour.lammpstrj', fields: 'id type x y z diameter' } },
      { defId: 'run', params: { steps: '100000' } },
    ],
  },
  {
    id: 'shear-nemd',
    label: 'Shear flow (NEMD)',
    description: 'fix deform shear with remap v + NVT/sllod-style thermostat.',
    steps: [
      { defId: 'units', params: { style: 'lj' } },
      { defId: 'boundary' },
      { defId: 'atom_style_cmd', params: { style: 'atomic' } },
      { defId: 'read_data', params: { file: 'fluid.data' } },
      { defId: 'pair_style_popular', params: { style: 'lj/cut', args: '2.5' } },
      { defId: 'pair_coeff', params: { types: '* *', coeffs: '1.0 1.0 2.5' } },
      { defId: 'fix_deform', params: { param: 'xy', style: 'erate', v1: '0.1', remap: 'v', units: 'box' } },
      { defId: 'fix_nvt', params: { temp_start: '1.0', temp_end: '1.0', temp_damp: '0.5' } },
      { defId: 'compute_rdf', params: { id: 'myRDF', nbins: '200' } },
      { defId: 'fix_ave_time', params: { values: 'c_myRDF[*]', mode: 'vector', file: 'rdf.txt', ave: 'running' } },
      { defId: 'thermo_style', params: { fields: 'step temp press xy' } },
      { defId: 'timestep' },
      { defId: 'run', params: { steps: '100000' } },
    ],
  },
];
