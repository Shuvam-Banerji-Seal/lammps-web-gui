/**
 * LAMMPS workbench data model.
 *
 * Every user-visible command in the Script Builder is a CommandDef:
 * declarative metadata (params, categories, docs link) plus a `build`
 * function that renders the exact LAMMPS input line(s).
 *
 * Sources: docs.lammps.org (git 4Jul2026) — see plans/08-lammps-research.md.
 * [VERIFIED] for enumerations; curated parameter sets for popular commands.
 */

export type ScriptSection =
  | 'setup'
  | 'system'
  | 'interactions'
  | 'output'
  | 'control';

export type ParamType = 'enum' | 'number' | 'string' | 'flag' | 'text';

export interface ParamOption {
  value: string;
  label?: string;
}

export interface ParamDef {
  key: string;
  label: string;
  type: ParamType;
  options?: ParamOption[];
  default?: string;
  placeholder?: string;
  /** Shown as hint text under the control. */
  help?: string;
}

export interface CommandDef {
  /** Unique id, e.g. "fix.nvt". */
  id: string;
  /** Primary keyword, e.g. "fix", "pair_style". */
  command: string;
  label: string;
  section: ScriptSection;
  category: string;
  doc?: string;
  params: ParamDef[];
  /**
   * Render the exact input line(s). Return empty array to skip emission
   * (e.g. optional blocks the user left blank).
   */
  build: (p: Record<string, string>) => string[];
}

/** One user-added step in the builder pipeline. */
export interface ScriptStep {
  uid: string;
  defId: string;
  params: Record<string, string>;
  enabled: boolean;
  /** Optional user comment rendered above the command. */
  note?: string;
}

export interface ScriptModel {
  title: string;
  steps: ScriptStep[];
}

export const SECTION_ORDER: ScriptSection[] = [
  'setup',
  'system',
  'interactions',
  'output',
  'control',
];

export const SECTION_LABELS: Record<ScriptSection, string> = {
  setup: '1 · Simulation setup',
  system: '2 · System definition',
  interactions: '3 · Interactions',
  output: '4 · Output',
  control: '5 · Run control',
};

const p = (
  key: string,
  label: string,
  type: ParamType,
  extra: Partial<ParamDef> = {}
): ParamDef => ({ key, label, type, ...extra });

const num = (key: string, label: string, def: string, help?: string): ParamDef =>
  p(key, label, 'number', { default: def, help });

const en = (key: string, label: string, values: string[], def?: string, help?: string): ParamDef =>
  p(key, label, 'enum', {
    options: values.map(v => ({ value: v })),
    default: def,
    help,
  });

const str = (key: string, label: string, def = '', placeholder?: string, help?: string): ParamDef =>
  p(key, label, 'string', { default: def, placeholder, help });

const flag = (key: string, label: string, def = 'no'): ParamDef =>
  p(key, label, 'flag', { default: def });

/** Join non-empty tokens with single spaces. */
const line = (...tokens: (string | number | undefined | null | false)[]): string =>
  tokens.filter(t => t !== '' && t !== undefined && t !== null && t !== false)
    .map(String).join(' ');

/* ------------------------------------------------------------------ */
/* 1 · SETUP                                                           */
/* ------------------------------------------------------------------ */

export const SETUP_COMMANDS: CommandDef[] = [
  {
    id: 'units',
    command: 'units',
    label: 'units — unit system',
    section: 'setup',
    category: 'Fundamentals',
    doc: 'https://docs.lammps.org/units.html',
    params: [en('style', 'Style', ['lj', 'real', 'metal', 'si', 'cgs', 'electron', 'micro', 'nano'], 'metal')],
    build: v => [line('units', v.style)],
  },
  {
    id: 'dimension',
    command: 'dimension',
    label: 'dimension — 2d/3d',
    section: 'setup',
    category: 'Fundamentals',
    doc: 'https://docs.lammps.org/dimension.html',
    params: [en('n', 'Dimensions', ['3', '2'], '3')],
    build: v => [line('dimension', v.n)],
  },
  {
    id: 'boundary',
    command: 'boundary',
    label: 'boundary — periodicity per axis',
    section: 'setup',
    category: 'Fundamentals',
    doc: 'https://docs.lammps.org/boundary.html',
    params: ['x', 'y', 'z'].flatMap(ax => [en(`b${ax}`, `${ax.toUpperCase()} boundary`, ['p', 'f', 's', 'm'], 'p')]),
    build: v => [line('boundary', v.bx, v.by, v.bz)],
  },
  {
    id: 'atom_style_cmd',
    command: 'atom_style',
    label: 'atom_style — particle attributes',
    section: 'setup',
    category: 'Fundamentals',
    doc: 'https://docs.lammps.org/atom_style.html',
    params: [
      en('style', 'Style',
        ['atomic', 'charge', 'molecular', 'full', 'angle', 'bond', 'sphere', 'ellipsoid',
         'line', 'tri', 'body', 'dipole', 'spin', 'peri', 'sph', 'smd', 'dpd', 'edpd',
         'mdpd', 'tdpd', 'oxdna', 'electron', 'dielectric', 'template', 'hybrid'],
        'full'),
      str('args', 'Extra args (hybrid substyles / tdpd Nspecies / template ID)', '', 'e.g. hybrid charge bond'),
    ],
    build: v => [line('atom_style', v.style, v.args)],
  },
  {
    id: 'processors',
    command: 'processors',
    label: 'processors — MPI grid layout',
    section: 'setup',
    category: 'Performance',
    doc: 'https://docs.lammps.org/processors.html',
    params: [
      str('grid', 'Grid', '* * *', '* * *'),
      flag('local', 'one proc per local region'),
    ],
    build: v => [v.local === 'yes' ? line('processors * * * local') : line('processors', v.grid)],
  },
  {
    id: 'neighbor',
    command: 'neighbor',
    label: 'neighbor — skin distance + style',
    section: 'setup',
    category: 'Neighbor lists',
    doc: 'https://docs.lammps.org/neighbor.html',
    params: [
      num('skin', 'Skin distance', '2.0'),
      en('style', 'Build style', ['bin', 'nsq', 'multi'], 'bin'),
    ],
    build: v => [line('neighbor', v.skin, v.style)],
  },
  {
    id: 'neigh_modify',
    command: 'neigh_modify',
    label: 'neigh_modify — tuning',
    section: 'setup',
    category: 'Neighbor lists',
    doc: 'https://docs.lammps.org/neigh_modify.html',
    params: [
      num('every', 'delay = every steps', '0'),
      num('check_dist', 'check yes interval', '1'),
      str('page', 'one/page keywords', '', 'e.g. exclude group A B'),
    ],
    build: v => [line('neigh_modify', v.every ? line('delay', v.every) : '', v.check_dist ? line('every', v.check_dist) : '', v.page)],
  },
  {
    id: 'timestep',
    command: 'timestep',
    label: 'timestep — dt',
    section: 'setup',
    category: 'Integration',
    doc: 'https://docs.lammps.org/timestep.html',
    params: [num('dt', 'Timestep', '0.001', 'metal: ps · real: fs · lj: tau')],
    build: v => [line('timestep', v.dt)],
  },
  {
    id: 'reset_timestep',
    command: 'reset_timestep',
    label: 'reset_timestep',
    section: 'setup',
    category: 'Integration',
    doc: 'https://docs.lammps.org/reset_timestep.html',
    params: [num('n', 'Start at step', '0')],
    build: v => [line('reset_timestep', v.n)],
  },
];

/* ------------------------------------------------------------------ */
/* 2 · SYSTEM                                                          */
/* ------------------------------------------------------------------ */

export const SYSTEM_COMMANDS: CommandDef[] = [
  {
    id: 'read_data',
    command: 'read_data',
    label: 'read_data — import structure file',
    section: 'system',
    category: 'Structure source',
    doc: 'https://docs.lammps.org/read_data.html',
    params: [
      str('file', 'Data file', 'structure.data'),
      str('extra', 'Keywords', '', 'e.g. fix prop/atom … / nolabelmap'),
    ],
    build: v => [line('read_data', v.file || 'structure.data', v.extra)],
  },
  {
    id: 'read_restart',
    command: 'read_restart',
    label: 'read_restart — continue from restart',
    section: 'system',
    category: 'Structure source',
    doc: 'https://docs.lammps.org/read_restart.html',
    params: [str('file', 'Restart file', 'restart.equil')],
    build: v => [line('read_restart', v.file)],
  },
  {
    id: 'region_block',
    command: 'region',
    label: 'region — block box',
    section: 'system',
    category: 'Regions',
    doc: 'https://docs.lammps.org/region.html',
    params: [
      str('id', 'Region ID', 'box'),
      en('side', 'Side', ['in', 'out'], 'in'),
      num('xlo', 'xlo', '0'), num('xhi', 'xhi', '10'),
      num('ylo', 'ylo', '0'), num('yhi', 'yhi', '10'),
      num('zlo', 'zlo', '0'), num('zhi', 'zhi', '10'),
      en('units', 'units kw', ['box', 'lattice', ''], ''),
    ],
    build: v => [line('region', v.id, 'block', v.xlo, v.xhi, v.ylo, v.yhi, v.zlo, v.zhi, v.side === 'out' ? 'side out' : '', v.units && line('units', v.units))],
  },
  {
    id: 'region_sphere',
    command: 'region',
    label: 'region — sphere',
    section: 'system',
    category: 'Regions',
    doc: 'https://docs.lammps.org/region.html',
    params: [
      str('id', 'Region ID', 'ball'),
      num('cx', 'center x', '0'), num('cy', 'center y', '0'), num('cz', 'center z', '0'),
      num('r', 'radius', '5'),
      en('units', 'units kw', ['box', 'lattice', ''], ''),
    ],
    build: v => [line('region', v.id, 'sphere', v.cx, v.cy, v.cz, v.r, v.units && line('units', v.units))],
  },
  {
    id: 'create_box',
    command: 'create_box',
    label: 'create_box — from region',
    section: 'system',
    category: 'Box creation',
    doc: 'https://docs.lammps.org/create_box.html',
    params: [
      num('ntypes', 'Number of atom types', '1'),
      str('region', 'Region ID', 'box'),
    ],
    build: v => [line('create_box', v.ntypes, v.region)],
  },
  {
    id: 'lattice',
    command: 'lattice',
    label: 'lattice — crystal lattice',
    section: 'system',
    category: 'Lattice & fill',
    doc: 'https://docs.lammps.org/lattice.html',
    params: [
      en('style', 'Style', ['sc', 'bcc', 'fcc', 'hcp', 'diamond', 'sq', 'sq2', 'hex', 'custom'], 'fcc'),
      num('scale', 'Scale (density or a)', '3.615'),
      str('basis', 'basis keywords', '', 'optional'),
    ],
    build: v => [line('lattice', v.style, v.scale, v.basis)],
  },
  {
    id: 'create_atoms',
    command: 'create_atoms',
    label: 'create_atoms — fill region',
    section: 'system',
    category: 'Lattice & fill',
    doc: 'https://docs.lammps.org/create_atoms.html',
    params: [
      num('type', 'Atom type', '1'),
      en('where', 'Where', ['box', 'region', 'single'], 'box'),
      str('region', 'Region ID (if region)', '', ''),
      str('single', 'x y z (if single)', '', ''),
    ],
    build: v => [line('create_atoms', v.type, v.where, v.where === 'region' ? v.region : '', v.where === 'single' ? v.single : '')],
  },
  {
    id: 'mass',
    command: 'mass',
    label: 'mass — per type',
    section: 'system',
    category: 'Types',
    doc: 'https://docs.lammps.org/mass.html',
    params: [
      str('type', 'Type (* or N)', '1'),
      num('value', 'Mass (g/mol)', '12.011'),
    ],
    build: v => [line('mass', v.type, v.value)],
  },
  {
    id: 'group',
    command: 'group',
    label: 'group — define atom group',
    section: 'system',
    category: 'Groups',
    doc: 'https://docs.lammps.org/group.html',
    params: [
      str('id', 'Group ID', 'mobile'),
      en('style', 'Style', ['type', 'id', 'region', 'molecule', 'all', 'subtract', 'union', 'intersect'], 'type'),
      str('args', 'Style args', '1', 'e.g. 1 2 3'),
    ],
    build: v => [line('group', v.id, v.style, v.args)],
  },
  {
    id: 'set_type',
    command: 'set',
    label: 'set — assign attributes',
    section: 'system',
    category: 'Types',
    doc: 'https://docs.lammps.org/set.html',
    params: [
      en('scope', 'Scope', ['group', 'type', 'region'], 'group'),
      str('scopeval', 'Scope value', 'all'),
      en('attr', 'Attribute', ['type', 'charge', 'mass', 'diameter'], 'charge'),
      str('value', 'Value', '0.0'),
    ],
    build: v => [line('set', v.scope, v.scopeval, v.attr, v.value)],
  },
  {
    id: 'velocity_create',
    command: 'velocity',
    label: 'velocity — create thermal velocities',
    section: 'system',
    category: 'Velocities',
    doc: 'https://docs.lammps.org/velocity.html',
    params: [
      str('group', 'Group', 'all'),
      en('style', 'Style', ['create', 'set', 'scale'], 'create'),
      num('temp', 'Temperature (T)', '300.0'),
      num('seed', 'RNG seed', '4928459'),
      en('dist_kw', 'Distribution', ['gaussian', 'uniform'], 'gaussian'),
      flag('mom', 'zero total momentum (mom yes)'),
    ],
    build: v => [line('velocity', v.group, v.style, v.temp, v.seed, line('dist', v.dist_kw), v.mom === 'yes' ? 'mom yes' : '')],
  },
  {
    id: 'replicate',
    command: 'replicate',
    label: 'replicate — tile the system',
    section: 'system',
    category: 'Box creation',
    doc: 'https://docs.lammps.org/replicate.html',
    params: [num('nx', 'nx', '2'), num('ny', 'ny', '2'), num('nz', 'nz', '2')],
    build: v => [line('replicate', v.nx, v.ny, v.nz)],
  },
];

/* ------------------------------------------------------------------ */
/* 3 · INTERACTIONS                                                    */
/* ------------------------------------------------------------------ */

const PAIR_POPULAR: { style: string; coeffHelp: string }[] = [
  { style: 'lj/cut', coeffHelp: 'type1 type2 epsilon sigma cutoff' },
  { style: 'lj/cut/coul/long', coeffHelp: 'type1 type2 epsilon sigma; coulomb cutoff via pair_coeff * * rc' },
  { style: 'lj/cut/coul/cut', coeffHelp: 'type1 type2 epsilon sigma rc' },
  { style: 'lj/smooth/linear', coeffHelp: 'type1 type2 epsilon sigma cutinner cut' },
  { style: 'lj96/cut', coeffHelp: 'type1 type2 epsilon sigma cutoff' },
  { style: 'buck', coeffHelp: 'type1 type2 A rho C cutoff' },
  { style: 'buck/coul/long', coeffHelp: 'type1 type2 A rho C cutoff' },
  { style: 'morse', coeffHelp: 'type1 type2 D alpha r0 cutoff' },
  { style: 'yukawa', coeffHelp: 'type1 type2 kappa cutoff' },
  { style: 'eam', coeffHelp: 'type1..N potential-file' },
  { style: 'eam/alloy', coeffHelp: 'type1..N potential-file' },
  { style: 'eam/fs', coeffHelp: 'type1..N potential-file' },
  { style: 'tersoff', coeffHelp: 'type1..N potential-file' },
  { style: 'sw', coeffHelp: 'type1..N potential-file' },
  { style: 'reaxff', coeffHelp: '* * control-field potential-file' },
  { style: 'table', coeffHelp: 'style N type1 type2 table-file keyword' },
  { style: 'soft', coeffHelp: 'type1 type2 epsilon lambda-start lambda-stop' },
  { style: 'coul/long', coeffHelp: '(no coeffs needed; cutoff via pair_coeff * * rc)' },
  { style: 'coul/dsf', coeffHelp: 'alpha rc' },
  { style: 'zbl', coeffHelp: 'type1 type2 inner outer' },
  { style: 'meam', coeffHelp: 'type1..N library-file element-list parameter-file' },
  { style: 'hybrid', coeffHelp: 'sub-style args… (advanced)' },
];

export const INTERACTION_COMMANDS: CommandDef[] = [
  {
    id: 'pair_style_popular',
    command: 'pair_style',
    label: 'pair_style — interaction model',
    section: 'interactions',
    category: 'Pair styles',
    doc: 'https://docs.lammps.org/pair_style.html',
    params: [
      en('style',
        'Style',
        PAIR_POPULAR.map(x => x.style),
        'lj/cut',
        'Full catalog (~280 styles) linked in docs'),
      str('args', 'pair_style args', '10.0', 'e.g. global cutoff / hybrid substyles'),
    ],
    build: v => [line('pair_style', v.style, v.args)],
  },
  {
    id: 'pair_coeff',
    command: 'pair_coeff',
    label: 'pair_coeff — parameters',
    section: 'interactions',
    category: 'Pair styles',
    doc: 'https://docs.lammps.org/pair_coeff.html',
    params: [
      str('types', 'Type selection', '1 1'),
      str('coeffs', 'Coefficients / files', '0.0103 3.4 10.0'),
    ],
    build: v => (v.coeffs.trim() ? [line('pair_coeff', v.types, v.coeffs)] : []),
  },
  {
    id: 'pair_modify',
    command: 'pair_modify',
    label: 'pair_modify — mixing/tail options',
    section: 'interactions',
    category: 'Pair styles',
    doc: 'https://docs.lammps.org/pair_modify.html',
    params: [
      en('mix', 'Mixing rule', ['', 'geometric', 'arithmetic', 'sixthpower'], ''),
      flag('tail', 'tail corrections (tail yes)'),
    ],
    build: v => {
      const kws = [
        v.mix ? line('mix', v.mix) : '',
        v.tail === 'yes' ? 'tail yes' : '',
      ].filter(Boolean);
      return kws.length > 0 ? [line('pair_modify', ...kws)] : [];
    },
  },
  {
    id: 'kspace_style',
    command: 'kspace_style',
    label: 'kspace_style — long-range electrostatics',
    section: 'interactions',
    category: 'Long range',
    doc: 'https://docs.lammps.org/kspace_style.html',
    params: [
      en('style', 'Style', ['pppm', 'pppm/disp', 'ewald', 'msm', 'none'], 'pppm'),
      num('precision', 'Relative force accuracy', '1.0e-4'),
    ],
    build: v => [line('kspace_style', v.style, v.style === 'none' ? '' : v.precision)],
  },
  {
    id: 'special_bonds',
    command: 'special_bonds',
    label: 'special_bonds — 1-2/1-3/1-4 scaling',
    section: 'interactions',
    category: 'Molecular topology',
    doc: 'https://docs.lammps.org/special_bonds.html',
    params: [
      en('preset', 'Preset', ['amber', 'charmm', 'dreiding', 'custom'], 'amber'),
      str('weights', 'Custom w1 w2 w3 (if custom)', '0.0 0.0 0.5'),
    ],
    build: v => [line('special_bonds', v.preset === 'custom' ? v.weights : v.preset)],
  },
  {
    id: 'bond_style',
    command: 'bond_style',
    label: 'bond_style',
    section: 'interactions',
    category: 'Topology styles',
    doc: 'https://docs.lammps.org/bond_style.html',
    params: [en('style', 'Style', ['harmonic', 'fene', 'class2', 'morse', 'quartic', 'hybrid'], 'harmonic')],
    build: v => [line('bond_style', v.style)],
  },
  {
    id: 'angle_style',
    command: 'angle_style',
    label: 'angle_style',
    section: 'interactions',
    category: 'Topology styles',
    doc: 'https://docs.lammps.org/angle_style.html',
    params: [en('style', 'Style', ['harmonic', 'charmm', 'cosine', 'cosine/squared', 'class2', 'hybrid'], 'harmonic')],
    build: v => [line('angle_style', v.style)],
  },
  {
    id: 'dihedral_style',
    command: 'dihedral_style',
    label: 'dihedral_style',
    section: 'interactions',
    category: 'Topology styles',
    doc: 'https://docs.lammps.org/dihedral_style.html',
    params: [en('style', 'Style', ['harmonic', 'charmm', 'multi/harmonic', 'opls', 'fourier', 'quadratic', 'hybrid'], 'harmonic')],
    build: v => [line('dihedral_style', v.style)],
  },
];

/* ------------------------------------------------------------------ */
/* 4 · OUTPUT                                                          */
/* ------------------------------------------------------------------ */

export const OUTPUT_COMMANDS: CommandDef[] = [
  {
    id: 'thermo_style',
    command: 'thermo_style',
    label: 'thermo_style — screen/log columns',
    section: 'output',
    category: 'Thermodynamics',
    doc: 'https://docs.lammps.org/thermo_style.html',
    params: [
      en('style', 'Style', ['one', 'two', 'multi', 'custom'], 'custom'),
      str('fields', 'Fields', 'step temp pe etotal press vol density'),
    ],
    build: v => [line('thermo_style', v.style, v.fields)],
  },
  {
    id: 'thermo',
    command: 'thermo',
    label: 'thermo — print every N steps',
    section: 'output',
    category: 'Thermodynamics',
    doc: 'https://docs.lammps.org/thermo.html',
    params: [num('n', 'Every N steps', '1000')],
    build: v => [line('thermo', v.n)],
  },
  {
    id: 'compute',
    command: 'compute',
    label: 'compute — derived quantities',
    section: 'output',
    category: 'Computes',
    doc: 'https://docs.lammps.org/compute.html',
    params: [
      str('id', 'Compute ID', 'myMSD'),
      str('group', 'Group', 'all'),
      str('style', 'Style + args', 'msd com yes'),
    ],
    build: v => [line('compute', v.id, v.group, v.style)],
  },
  {
    id: 'dump_custom',
    command: 'dump',
    label: 'dump — trajectory (custom)',
    section: 'output',
    category: 'Trajectories',
    doc: 'https://docs.lammps.org/dump.html',
    params: [
      str('id', 'Dump ID', 'traj'),
      str('group', 'Group', 'all'),
      en('style', 'Style', ['custom', 'xyz', 'cfg', 'local'], 'custom'),
      num('n', 'Every N steps', '1000'),
      str('file', 'File pattern', 'traj.dump.*'),
      str('fields', 'Fields', 'id type x y z vx vy vz'),
    ],
    build: v => [line('dump', v.id, v.group, v.style, v.n, v.file, v.fields)],
  },
  {
    id: 'dump_modify',
    command: 'dump_modify',
    label: 'dump_modify — sort/scale',
    section: 'output',
    category: 'Trajectories',
    doc: 'https://docs.lammps.org/dump_modify.html',
    params: [
      str('dumpid', 'Dump ID', 'traj'),
      en('sort', 'Sort by id', ['id', 'no'], 'id'),
    ],
    build: v => [line('dump_modify', v.dumpid, v.sort && line('sort', v.sort))],
  },
  {
    id: 'restart_out',
    command: 'restart',
    label: 'restart — periodic checkpoints',
    section: 'output',
    category: 'Checkpoints',
    doc: 'https://docs.lammps.org/restart.html',
    params: [
      num('n', 'Every N steps', '50000'),
      str('f1', 'File A pattern', 'prod.restart.*'),
      str('f2', 'File B pattern (optional)', ''),
    ],
    build: v => [line('restart', v.n, v.f1, v.f2)],
  },
  {
    id: 'write_data_out',
    command: 'write_data',
    label: 'write_data — final structure',
    section: 'output',
    category: 'Checkpoints',
    doc: 'https://docs.lammps.org/write_data.html',
    params: [str('file', 'File', 'final.data'), str('kw', 'Keywords', '', 'e.g. nocoeff')],
    build: v => [line('write_data', v.file || 'final.data', v.kw)],
  },
  {
    id: 'log_cmd',
    command: 'log',
    label: 'log — redirect logfile',
    section: 'output',
    category: 'Misc',
    doc: 'https://docs.lammps.org/log.html',
    params: [str('file', 'Log file', 'log.lammps')],
    build: v => [line('log', v.file)],
  },
];

/* ------------------------------------------------------------------ */
/* 5 · RUN CONTROL                                                     */
/* ------------------------------------------------------------------ */

export const CONTROL_COMMANDS: CommandDef[] = [
  {
    id: 'fix_nve',
    command: 'fix',
    label: 'fix nve — NVE integration',
    section: 'control',
    category: 'Integrators',
    doc: 'https://docs.lammps.org/fix_nve.html',
    params: [str('group', 'Group', 'all')],
    build: v => [line('fix', 'integrate', v.group, 'nve')],
  },
  {
    id: 'fix_nvt',
    command: 'fix',
    label: 'fix nvt — Nose-Hoover thermostat',
    section: 'control',
    category: 'Integrators',
    doc: 'https://docs.lammps.org/fix_nh.html',
    params: [
      str('group', 'Group', 'all'),
      num('temp_start', 'T start', '300.0'),
      num('temp_end', 'T end', '300.0'),
      num('temp_damp', 'T damp (≈100·dt)', '0.1'),
    ],
    build: v => [line('fix', 'integrate', v.group, 'nvt', 'temp', v.temp_start, v.temp_end, v.temp_damp)],
  },
  {
    id: 'fix_npt',
    command: 'fix',
    label: 'fix npt — thermostat + barostat',
    section: 'control',
    category: 'Integrators',
    doc: 'https://docs.lammps.org/fix_nh.html',
    params: [
      str('group', 'Group', 'all'),
      num('temp_start', 'T start', '300.0'),
      num('temp_end', 'T end', '300.0'),
      num('temp_damp', 'T damp', '0.1'),
      en('pstyle', 'Pressure components', ['iso', 'aniso', 'tri'], 'iso'),
      num('p_start', 'P start', '0.0'),
      num('p_end', 'P end', '0.0'),
      num('p_damp', 'P damp', '1.0'),
    ],
    build: v => [line('fix', 'integrate', v.group, 'npt', 'temp', v.temp_start, v.temp_end, v.temp_damp, v.pstyle, v.p_start, v.p_end, v.p_damp)],
  },
  {
    id: 'fix_langevin',
    command: 'fix',
    label: 'fix langevin — stochastic thermostat',
    section: 'control',
    category: 'Thermostats',
    doc: 'https://docs.lammps.org/fix_langevin.html',
    params: [
      str('group', 'Group', 'all'),
      num('temp_start', 'T start', '300.0'),
      num('temp_end', 'T end', '300.0'),
      num('damp', 'Damping', '0.1'),
      num('seed', 'Seed', '12345'),
    ],
    build: v => [line('fix', 'lang', v.group, 'langevin', v.temp_start, v.temp_end, v.damp, v.seed)],
  },
  {
    id: 'fix_berendsen',
    command: 'fix',
    label: 'fix berendsen — weak-coupling T',
    section: 'control',
    category: 'Thermostats',
    doc: 'https://docs.lammps.org/fix_berendsen.html',
    params: [
      str('group', 'Group', 'all'),
      num('temp_start', 'T start', '300.0'),
      num('temp_end', 'T end', '300.0'),
      num('tdamp', 'T damp', '0.1'),
    ],
    build: v => [line('fix', 'berend', v.group, 'berendsen', v.temp_start, v.temp_end, v.tdamp)],
  },
  {
    id: 'fix_minimize',
    command: 'minimize',
    label: 'minimize — energy minimization',
    section: 'control',
    category: 'Minimization',
    doc: 'https://docs.lammps.org/minimize.html',
    params: [
      num('etol', 'Energy tolerance', '0.0'),
      num('ftol', 'Force tolerance', '1.0e-6'),
      num('maxiter', 'Max iterations', '10000'),
      num('maxeval', 'Max force evals', '100000'),
    ],
    build: v => [line('minimize', v.etol, v.ftol, v.maxiter, v.maxeval)],
  },
  {
    id: 'min_style_cmd',
    command: 'min_style',
    label: 'min_style — minimizer',
    section: 'control',
    category: 'Minimization',
    doc: 'https://docs.lammps.org/min_style.html',
    params: [en('style', 'Style', ['cg', 'hftn', 'sd', 'fire', 'quickmin'], 'cg')],
    build: v => [line('min_style', v.style)],
  },
  {
    id: 'fix_shake',
    command: 'fix',
    label: 'fix shake — constrain bonds/angles',
    section: 'control',
    category: 'Constraints',
    doc: 'https://docs.lammps.org/fix_shake.html',
    params: [
      str('group', 'Group', 'water'),
      num('tol', 'Tolerance', '1.0e-4'),
      str('bonds', 'bonds keyword', 'b 1'),
      str('angles', 'angles keyword', 'a 1'),
    ],
    build: v => [line('fix', 'shake', v.group, 'shake', v.tol, v.bonds, v.angles)],
  },
  {
    id: 'fix_rigid',
    command: 'fix',
    label: 'fix rigid — rigid bodies',
    section: 'control',
    category: 'Constraints',
    doc: 'https://docs.lammps.org/fix_rigid.html',
    params: [
      str('group', 'Group', 'molecules'),
      en('style', 'Style', ['small', 'nve', 'nvt', 'npt'], 'nvt'),
      str('args', 'Style args', 'temp 300 300 100'),
    ],
    build: v => [line('fix', 'rigid', v.group, 'rigid', v.style, v.args)],
  },
  {
    id: 'fix_wall',
    command: 'fix',
    label: 'fix wall — reflective walls',
    section: 'control',
    category: 'Walls',
    doc: 'https://docs.lammps.org/fix_wall.html',
    params: [
      str('group', 'Group', 'all'),
      en('style', 'Potential', ['lj93', 'lj126', 'colloid', 'harmonic', 'reflect'], 'lj126'),
      en('axis', 'Axis', ['xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi'], 'zlo'),
      num('pos', 'Position', '0.0'),
      num('epsilon', 'epsilon', '1.0'),
      num('sigma', 'sigma', '1.0'),
      num('cutoff', 'cutoff', '3.0'),
    ],
    build: v => [line('fix', 'walls', v.group, 'wall', v.style, v.axis, v.pos, v.epsilon, v.sigma, v.cutoff)],
  },
  {
    id: 'fix_addforce',
    command: 'fix',
    label: 'fix addforce — constant force',
    section: 'control',
    category: 'Forcing',
    doc: 'https://docs.lammps.org/fix_addforce.html',
    params: [
      str('group', 'Group', 'pull'),
      num('fx', 'Fx', '0.0'), num('fy', 'Fy', '0.0'), num('fz', 'Fz', '1.0'),
    ],
    build: v => [line('fix', 'pull', v.group, 'addforce', v.fx, v.fy, v.fz)],
  },
  {
    id: 'fix_momentum',
    command: 'fix',
    label: 'fix momentum — drift removal',
    section: 'control',
    category: 'Constraints',
    doc: 'https://docs.lammps.org/fix_momentum.html',
    params: [
      str('group', 'Group', 'all'),
      num('n', 'Every N steps', '100'),
      flag('linear', 'zero linear momentum'),
      flag('angular', 'zero angular momentum'),
    ],
    build: v => [line('fix', 'drift', v.group, 'momentum', v.n, v.linear === 'yes' ? 'linear 1 1 1' : '', v.angular === 'yes' ? 'angular 1 1 1' : '')],
  },
  {
    id: 'fix_recenter',
    command: 'fix',
    label: 'fix recenter — hold COM position',
    section: 'control',
    category: 'Constraints',
    doc: 'https://docs.lammps.org/fix_recenter.html',
    params: [
      str('group', 'Group', 'all'),
      str('coords', 'INIT INIT INIT mode…', 'INIT INIT INIT'),
    ],
    build: v => [line('fix', 'rcm', v.group, 'recenter', v.coords)],
  },
  {
    id: 'run',
    command: 'run',
    label: 'run — molecular dynamics',
    section: 'control',
    category: 'Execution',
    doc: 'https://docs.lammps.org/run.html',
    params: [
      num('steps', 'Number of steps', '100000'),
      str('post_kw', 'Post-run keywords', '', 'e.g. up 100 every 1000 "print …"'),
    ],
    build: v => [line('run', v.steps, v.post_kw)],
  },
  {
    id: 'variable_eq',
    command: 'variable',
    label: 'variable — equal-style expression',
    section: 'control',
    category: 'Scripting',
    doc: 'https://docs.lammps.org/variable.html',
    params: [
      str('name', 'Name', 'myTemp'),
      str('expr', 'Expression', 'equal temp'),
    ],
    build: v => [line('variable', v.name, v.expr)],
  },
  {
    id: 'print_cmd',
    command: 'print',
    label: 'print — emit text to log',
    section: 'control',
    category: 'Scripting',
    doc: 'https://docs.lammps.org/print.html',
    params: [str('text', 'Text', '"Hello from Molecule3D builder"')],
    build: v => [line('print', v.text)],
  },
];

export const ALL_COMMANDS: CommandDef[] = [
  ...SETUP_COMMANDS,
  ...SYSTEM_COMMANDS,
  ...INTERACTION_COMMANDS,
  ...OUTPUT_COMMANDS,
  ...CONTROL_COMMANDS,
];

export const COMMAND_BY_ID: Record<string, CommandDef> = Object.fromEntries(
  ALL_COMMANDS.map(d => [d.id, d])
);

/** Default param map for a fresh step of the given command. */
export const defaultParams = (def: CommandDef): Record<string, string> =>
  Object.fromEntries(def.params.map(pd => [pd.key, pd.default ?? '']));
