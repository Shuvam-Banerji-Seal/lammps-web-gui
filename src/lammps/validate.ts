/**
 * LAMMPS input-script validator.
 *
 * Works on the FINAL text (generated or hand-written), so what is checked is
 * exactly what the user will feed to `lmp -in`. Every rule below quotes the
 * requirement it enforces from docs.lammps.org; `doc` links back to the page.
 *
 * Verified against docs.lammps.org on 2026-09-22:
 *  - Commands_structure.html — "The last 2 parts can be repeated as many
 *    times as desired. I.e. run a simulation, change some settings, run some
 *    more, etc."
 *  - units.html / dimension.html / boundary.html / atom_style.html —
 *    "cannot be used after the simulation box is defined by a read_data or
 *    create_box command" (boundary/atom_style also name read_restart).
 *  - pair_coeff.html / mass.html — "must come after the simulation box is
 *    defined by a read_data, read_restart, or create_box command".
 *  - mass.html — "All masses must be defined before a simulation is run.
 *    They must also all be defined before a velocity or fix shake command
 *    is used."
 *  - create_atoms.html — "An atom_style must be previously defined to use
 *    this command"; a lattice is required except for style `single` with
 *    box units and style `random`.
 */

import { scriptStatementsDetailed, tokenizeLine } from './scriptParser';

export type DiagnosticLevel = 'error' | 'warning' | 'info';

export interface Diagnostic {
  level: DiagnosticLevel;
  /** 1-based source line of the offending statement (0 = whole script). */
  line: number;
  /** The LAMMPS command keyword the rule fired on. */
  command: string;
  message: string;
  /** Stable rule id, e.g. "order/units-after-box". */
  rule: string;
  /** docs.lammps.org page backing the rule. */
  doc?: string;
}

const DOC = (page: string) => `https://docs.lammps.org/${page}`;

/** Commands that define the simulation box (and therefore close the header). */
const BOX_DEFINING = new Set(['read_data', 'read_restart', 'create_box']);

/**
 * Header-only commands → the doc sentence that forbids them after the box.
 * Every entry is quoted from that command's Restrictions section.
 */
const HEADER_ONLY: Record<string, { doc: string; after: string }> = {
  units: { doc: 'units.html', after: 'read_data or create_box' },
  dimension: { doc: 'dimension.html', after: 'read_data or create_box' },
  boundary: { doc: 'boundary.html', after: 'read_data, create_box or read_restart' },
  atom_style: { doc: 'atom_style.html', after: 'read_data or create_box' },
  newton: { doc: 'newton.html', after: 'read_data, create_box or read_restart' },
};

/** Commands that require the box to exist first. */
const NEEDS_BOX: Record<string, string> = {
  pair_coeff: 'pair_coeff.html',
  bond_coeff: 'bond_coeff.html',
  angle_coeff: 'angle_coeff.html',
  dihedral_coeff: 'dihedral_coeff.html',
  improper_coeff: 'improper_coeff.html',
  mass: 'mass.html',
  velocity: 'velocity.html',
  create_atoms: 'create_atoms.html',
};

/** `unfix`/`undump`/`uncompute` → the command that defines the id. */
const UNDEFINERS: Record<string, { defines: string; doc: string }> = {
  unfix: { defines: 'fix', doc: 'unfix.html' },
  undump: { defines: 'dump', doc: 'undump.html' },
  uncompute: { defines: 'compute', doc: 'uncompute.html' },
};

/** Time-integration fix styles — `run` without one leaves atoms frozen. */
const INTEGRATORS = [
  'nve', 'nvt', 'npt', 'nph', 'nvt/sllod', 'npt/sphere', 'nve/limit',
  'nve/sphere', 'nve/noforce', 'nve/asphere', 'nvt/sphere', 'nvt/asphere',
  'nph/sphere', 'npt/asphere', 'nve/body', 'nvt/body', 'rigid/nve',
  'rigid/nvt', 'rigid/npt', 'rigid/nph', 'rigid', 'rigid/small',
  'rigid/nve/small', 'rigid/nvt/small', 'nve/line', 'nve/tri',
  'nve/spin', 'nve/dot', 'nve/dotc/langevin', 'nph/asphere',
];

const isIntegrator = (style: string): boolean =>
  INTEGRATORS.includes(style) ||
  style.startsWith('rigid') ||
  /^(nve|nvt|npt|nph)(\/|$)/.test(style);

/**
 * Atom styles that carry per-atom mass (finite-size or per-particle density),
 * so the per-TYPE `mass` command is neither used nor required.
 * docs.lammps.org/atom_style.html — "sphere … per-particle diameter and mass".
 */
const PER_ATOM_MASS_STYLES = [
  'sphere', 'granular', 'ellipsoid', 'line', 'tri', 'body', 'peri',
  'bpm/sphere', 'oxdna', 'edpd', 'mdpd', 'smd', 'rheo',
];

/** Handles `atom_style sphere` and `atom_style hybrid sphere charge` alike. */
const derivesMass = (atomStyleLine: string | null): boolean =>
  !!atomStyleLine &&
  atomStyleLine
    .split(/\s+/)
    .some(tok => PER_ATOM_MASS_STYLES.includes(tok));

/**
 * Commands LAMMPS has REMOVED. Quoted from
 * docs.lammps.org/Commands_removed.html (2026-09-22).
 */
const REMOVED_COMMANDS: Record<string, string> = {
  box: 'removed in 22Dec2022 — LAMMPS ignores it and prints a warning; ' +
    'triclinic tilt limits no longer need declaring',
  reset_ids: 'removed in 22Dec2022 — folded into `reset_atoms`',
  reset_atom_ids: 'removed in 22Dec2022 — folded into `reset_atoms`',
  reset_mol_ids: 'removed in 22Dec2022 — folded into `reset_atoms`',
};

/** Fix styles that create atoms at run time, so a run with 0 atoms is fine. */
const ATOM_CREATING_FIXES = ['pour', 'deposit', 'gcmc', 'widom', 'append/atoms'];

/** kspace solvers need a matching long-range pair style. */
const LONG_RANGE_PAIR = /(coul\/long|coul\/msm|coul\/wolf\/cs|long\/long|coul\/esp|tip4p\/long|dsf)/;

interface State {
  boxDefined: boolean;
  qeqFix: boolean;
  atomStyle: string | null;
  atomsCreated: boolean;
  latticeDefined: boolean;
  pairStyle: string | null;
  pairCoeffSeen: boolean;
  kspaceLine: number | null;
  massSeen: boolean;
  velocitySeen: number | null;
  runSeen: boolean;
  dumpBeforeRun: number | null;
  fixIds: Map<string, string>;
  dumpIds: Map<string, string>;
  computeIds: Map<string, string>;
  regionIds: Set<string>;
  groupIds: Set<string>;
  activeIntegrators: Set<string>;
  readData: boolean;
}

/**
 * Validate a LAMMPS input script.
 *
 * Returns diagnostics ordered by source line. `error` = LAMMPS will refuse to
 * run (or silently do the wrong thing); `warning` = runs but very likely not
 * what was intended; `info` = style nudges.
 */
export const validateScript = (text: string): Diagnostic[] => {
  const out: Diagnostic[] = [];
  const add = (
    level: DiagnosticLevel,
    line: number,
    command: string,
    rule: string,
    message: string,
    doc?: string,
  ) => out.push({ level, line, command, rule, message, doc: doc ? DOC(doc) : undefined });

  const st: State = {
    boxDefined: false,
    qeqFix: false,
    atomStyle: null,
    atomsCreated: false,
    latticeDefined: false,
    pairStyle: null,
    pairCoeffSeen: false,
    kspaceLine: null,
    massSeen: false,
    velocitySeen: null,
    runSeen: false,
    dumpBeforeRun: null,
    fixIds: new Map(),
    dumpIds: new Map(),
    computeIds: new Map(),
    regionIds: new Set(),
    groupIds: new Set(),
    activeIntegrators: new Set(),
    readData: false,
  };

  const statements = scriptStatementsDetailed(text);
  let sawRunOrMinimize = false;

  for (const { text: stmt, line } of statements) {
    const tok = tokenizeLine(stmt);
    if (tok.length === 0) continue;
    const cmd = tok[0];

    /* ---- commands LAMMPS has removed -------------------------------- */
    if (REMOVED_COMMANDS[cmd]) {
      add('warning', line, cmd, 'deprecated/removed-command',
        `\`${cmd}\` was ${REMOVED_COMMANDS[cmd]}.`,
        'Commands_removed.html');
    }

    /* ---- header-only commands after the box ------------------------- */
    const header = HEADER_ONLY[cmd];
    if (header && st.boxDefined) {
      add(
        'error', line, cmd, `order/${cmd}-after-box`,
        `\`${cmd}\` cannot be used after the simulation box is defined ` +
          `(${header.after}). Move it above the box-defining command` +
          (cmd === 'boundary' ? ', or use `change_box` instead.' : '.'),
        header.doc,
      );
    }

    /* ---- commands that need the box --------------------------------- */
    if (NEEDS_BOX[cmd] && !st.boxDefined) {
      add(
        'error', line, cmd, `order/${cmd}-before-box`,
        `\`${cmd}\` must come after the simulation box is defined by ` +
          '`read_data`, `read_restart` or `create_box`.',
        NEEDS_BOX[cmd],
      );
    }

    switch (cmd) {
      case 'units':
      case 'dimension':
      case 'newton':
        break;

      case 'atom_style':
        // keep the whole spec so `hybrid sphere charge` is understood
        st.atomStyle = tok.slice(1).join(' ') || null;
        break;

      case 'lattice':
        st.latticeDefined = true;
        break;

      case 'region':
        if (tok[1]) st.regionIds.add(tok[1]);
        break;

      case 'group':
        if (tok[1]) st.groupIds.add(tok[1]);
        if (tok[2] === 'region' && tok[3] && !st.regionIds.has(tok[3])) {
          add('error', line, cmd, 'ref/unknown-region',
            `\`group ... region ${tok[3]}\` refers to a region that has not been defined.`,
            'group.html');
        }
        break;

      case 'create_box':
        if (!st.atomStyle) {
          add('warning', line, cmd, 'order/create_box-without-atom_style',
            '`create_box` uses the default `atom_style atomic`. Declare `atom_style` ' +
              'explicitly above it if you need charges, bonds or molecule IDs.',
            'atom_style.html');
        }
        if (tok[2] && !st.regionIds.has(tok[2])) {
          add('error', line, cmd, 'ref/unknown-region',
            `\`create_box\` refers to region "${tok[2]}", which has not been defined.`,
            'create_box.html');
        }
        st.boxDefined = true;
        break;

      case 'read_data':
      case 'read_restart':
        st.boxDefined = true;
        st.atomsCreated = true;
        st.readData = true;
        break;

      case 'read_dump':
        st.atomsCreated = true;
        break;

      case 'create_atoms': {
        if (!st.atomStyle && !st.readData) {
          add('error', line, cmd, 'order/create_atoms-without-atom_style',
            'An `atom_style` must be previously defined to use `create_atoms`.',
            'create_atoms.html');
        }
        const style = tok[2];
        const needsLattice = style !== 'random' && !(style === 'single' && stmt.includes('units box'));
        if (needsLattice && !st.latticeDefined) {
          add('warning', line, cmd, 'order/create_atoms-without-lattice',
            `\`create_atoms ... ${style ?? 'box'}\` fills from a lattice, but no ` +
              '`lattice` command was issued — LAMMPS falls back to the default ' +
              '`lattice none 1.0`, which is almost never intended.',
            'lattice.html');
        }
        st.atomsCreated = true;
        break;
      }

      case 'mass':
        st.massSeen = true;
        if (st.velocitySeen !== null) {
          add('warning', line, cmd, 'order/mass-after-velocity',
            'All masses must be defined before a `velocity` or `fix shake` command ' +
              `is used — a \`velocity\` appears earlier on line ${st.velocitySeen}.`,
            'mass.html');
        }
        break;

      case 'velocity':
        if (st.velocitySeen === null) st.velocitySeen = line;
        if (!st.atomsCreated) {
          add('error', line, cmd, 'order/velocity-without-atoms',
            '`velocity` needs atoms: add `read_data`, `read_restart` or `create_atoms` first.',
            'velocity.html');
        }
        break;

      case 'pair_style':
        st.pairStyle = tok.slice(1).join(' ');
        break;

      case 'pair_coeff':
        st.pairCoeffSeen = true;
        if (!st.pairStyle && !st.readData) {
          add('error', line, cmd, 'order/pair_coeff-without-pair_style',
            '`pair_coeff` sets coefficients for the current pair style, but no ' +
              '`pair_style` has been declared.',
            'pair_coeff.html');
        }
        break;

      case 'kspace_style':
        if (tok[1] && tok[1] !== 'none') st.kspaceLine = line;
        break;

      case 'fix': {
        const [, id, group, style] = tok;
        if (id) {
          const prev = st.fixIds.get(id);
          if (prev !== undefined && prev !== style) {
            add('error', line, cmd, 'id/fix-reuse',
              `Fix ID "${id}" is already in use with style \`${prev}\`. LAMMPS only ` +
                'allows re-issuing a fix ID with the SAME style; otherwise `unfix` it first.',
              'fix.html');
          }
          st.fixIds.set(id, style ?? '');
        }
        if (group && group !== 'all' && !st.groupIds.has(group)) {
          add('error', line, cmd, 'ref/unknown-group',
            `\`fix ${id ?? ''} ${group} ...\` uses group "${group}", which has not been defined.`,
            'group.html');
        }
        if (style && isIntegrator(style) && id) st.activeIntegrators.add(id);
        if (style && ATOM_CREATING_FIXES.some(f => style === f || style.startsWith(f))) {
          st.atomsCreated = true;
        }
        if (style && /^(qeq|acks2|qtpie)\//.test(style)) st.qeqFix = true;
        break;
      }

      case 'unfix':
      case 'undump':
      case 'uncompute': {
        const info = UNDEFINERS[cmd];
        const id = tok[1];
        const table =
          cmd === 'unfix' ? st.fixIds : cmd === 'undump' ? st.dumpIds : st.computeIds;
        if (id && !table.has(id)) {
          add('error', line, cmd, `ref/unknown-${info.defines}`,
            `\`${cmd} ${id}\` refers to a ${info.defines} ID that was never defined ` +
              '(or was already removed).',
            info.doc);
        } else if (id) {
          table.delete(id);
          if (cmd === 'unfix') st.activeIntegrators.delete(id);
        }
        break;
      }

      case 'dump': {
        const id = tok[1];
        if (id) {
          if (st.dumpIds.has(id)) {
            add('error', line, cmd, 'id/dump-reuse',
              `Dump ID "${id}" is already in use — pick a new ID or \`undump ${id}\` first.`,
              'dump.html');
          }
          st.dumpIds.set(id, tok[3] ?? '');
        }
        const group = tok[2];
        if (group && group !== 'all' && !st.groupIds.has(group)) {
          add('error', line, cmd, 'ref/unknown-group',
            `\`dump\` uses group "${group}", which has not been defined.`,
            'group.html');
        }
        if (!st.runSeen && st.dumpBeforeRun === null) st.dumpBeforeRun = line;
        break;
      }

      case 'compute': {
        const id = tok[1];
        if (id) {
          if (st.computeIds.has(id)) {
            add('error', line, cmd, 'id/compute-reuse',
              `Compute ID "${id}" is already in use — pick a new ID or ` +
                `\`uncompute ${id}\` first.`,
              'compute.html');
          }
          st.computeIds.set(id, tok[3] ?? '');
        }
        break;
      }

      case 'run':
      case 'minimize': {
        sawRunOrMinimize = true;
        if (!st.atomsCreated) {
          add('error', line, cmd, 'run/no-atoms',
            `\`${cmd}\` has no atoms to act on — add \`read_data\`, \`read_restart\`, ` +
              '`read_dump` or `create_atoms` first.',
            `${cmd}.html`);
        }
        if (cmd === 'run' && st.activeIntegrators.size === 0) {
          add('warning', line, cmd, 'run/no-integrator',
            'No time-integration fix is active (`fix nve`, `fix nvt`, `fix npt`, ' +
              '`fix rigid`, …) — LAMMPS will run but the atoms will not move.',
            'run.html');
        }
        if (!st.pairStyle && !st.readData) {
          add('warning', line, cmd, 'run/no-pair-style',
            'No `pair_style` has been declared — atoms will not interact.',
            'pair_style.html');
        }
        st.runSeen = true;
        break;
      }

      default:
        break;
    }
  }

  /* ---- whole-script rules ------------------------------------------- */
  if (statements.length > 0 && !sawRunOrMinimize) {
    add('warning', 0, 'run', 'script/no-run',
      'The script never calls `run` or `minimize`, so nothing is simulated.',
      'run.html');
  }
  if (st.pairStyle && !st.pairCoeffSeen && !st.readData) {
    add('warning', 0, 'pair_coeff', 'script/pair-style-without-coeff',
      `\`pair_style ${st.pairStyle}\` is declared but no \`pair_coeff\` follows, and ` +
        'no data file supplies the coefficients.',
      'pair_coeff.html');
  }
  if (
    st.pairStyle &&
    /^reaxff/.test(st.pairStyle) &&
    !st.qeqFix &&
    !/checkqeq\s+no/.test(st.pairStyle)
  ) {
    add('error', 0, 'pair_style', 'reaxff/missing-qeq',
      'LAMMPS requires `fix qeq/reaxff` (or qeq/shielded, acks2/reaxff, ' +
        'qtpie/reaxff) alongside `pair_style reaxff` so charges are equilibrated ' +
        'every step — unless `checkqeq no` is passed to pair_style.',
      'pair_reaxff.html');
  }
  if (st.kspaceLine !== null && st.pairStyle && !LONG_RANGE_PAIR.test(st.pairStyle)) {
    add('warning', st.kspaceLine, 'kspace_style', 'kspace/pair-mismatch',
      `A long-range solver needs a matching long-range pair style (e.g. ` +
        `\`lj/cut/coul/long\`); the current style is \`${st.pairStyle}\`.`,
      'kspace_style.html');
  }
  if (st.boxDefined && !st.massSeen && !st.readData && !derivesMass(st.atomStyle)) {
    add('warning', 0, 'mass', 'script/no-mass',
      'No `mass` command and no data file — every atom type needs a mass before a run.',
      'mass.html');
  }

  return out.sort((a, b) => a.line - b.line);
};

/** Convenience counters for badge UI. */
export const diagnosticCounts = (diags: Diagnostic[]) => ({
  errors: diags.filter(d => d.level === 'error').length,
  warnings: diags.filter(d => d.level === 'warning').length,
  infos: diags.filter(d => d.level === 'info').length,
});
