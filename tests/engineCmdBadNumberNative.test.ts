import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * An input command or a coefficient line whose first numeric word is replaced by the word "bogus", refused with native
 * LAMMPS's message: the value is read by utils::numeric, inumeric or bounds, or a check of the command comes first (the
 * TIP4P pair styles read their type words at pair_coeff, so native first finds the pair coefficients unset). Each row
 * takes an oracle case up to one of its lines, replaces the word (from word 1 of a command, word 2 of a coefficient
 * line) and runs 0 steps. Measured with native LAMMPS (/home/roy/.local/bin/lmp, b332f28).
 */

const ORACLE = join(__dirname, 'oracle');
const POTENTIALS = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const EXAMPLES = join(__dirname, '..', 'third_party', 'lammps', 'examples');
const ORACLE_FILES = new Set(readdirSync(ORACLE).filter((f) => !f.endsWith('.in')));
const directive = (text: string, key: string): string[] => {
  const m = new RegExp(`^# ${key}:(.*)$`, 'm').exec(text);
  return m ? m[1].trim().split(/\s+/).filter((w) => w !== '') : [];
};
const readInput = (path: string): string => readFileSync(path, path.endsWith('.gz') ? 'latin1' : 'utf8');

const NUMBER = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;

/** The line with its first numeric word from word `first` on replaced by "bogus". */
const badNumber = (line: string, first: number): string => {
  const w = line.split('#')[0].trim().split(/\s+/);
  const i = w.findIndex((x, j) => j >= first && NUMBER.test(x));
  if (i >= 0) w[i] = 'bogus';
  return w.join(' ');
};

const caseFiles = (text: string, body: string): Record<string, string> => {
  const files: Record<string, string> = {};
  for (const f of directive(text, 'oracle-inputs')) files[f] = readInput(join(ORACLE, f));
  for (const f of directive(text, 'oracle-potentials')) files[f] = readFileSync(join(POTENTIALS, f), 'utf8');
  for (const f of directive(text, 'oracle-examples')) files[f.split('/').pop()!] = readInput(join(EXAMPLES, f));
  // files of the oracle directory that the lines name without a directive
  for (const w of body.split(/\s+/)) if (ORACLE_FILES.has(w) && !(w in files)) files[w] = readInput(join(ORACLE, w));
  return files;
};

/** The case's input up to line k, whose first numeric word from word `first` on becomes "bogus", then run 0. */
const withBadNumber = (name: string, k: number, first: number): [string, Record<string, string>] => {
  const text = readFileSync(join(ORACLE, `${name}.in`), 'utf8');
  const lines = text.split('\n');
  const body = lines.slice(0, k).filter((l) => !l.startsWith('# oracle')).join('\n') + '\n' + badNumber(lines[k], first) + '\nrun 0\n';
  return [body, caseFiles(text, body)];
};

const session = (files: Record<string, string>): Session => {
  const s = new Session({ emit: () => {}, writeFile: () => {} } as never);
  for (const [name, text] of Object.entries(files)) s.addFile(name, text);
  return s;
};

const outcome = async (input: string, files: Record<string, string>): Promise<string> => {
  // the engine's message starts with the line of the input ("line N: "), which native prints elsewhere
  try { await session(files).execute(input); } catch (e) { return 'ERROR: ' + (e as Error).message.split('\n')[0].replace(/^line \d+: /, ''); }
  return 'OK';
};

/** [oracle case, index of the line (0-based), first word that may be replaced, native result] */
const ROWS: [string, number, number, string][] = [
  ["w225cm_angle_write_charmm", 13, 1, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w50bc_static", 5, 1, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w378balance_out_tri2d", 13, 1, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w144bpm_rot_chain", 31, 1, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w144bpm_spring_store", 31, 1, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w137cf_delbonds_stale", 18, 1, "ERROR: Invalid range string: bogus"],
  ["w106gs_tribox_surf", 31, 1, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w225cm_dihedral_write", 14, 1, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w19trigen_general", 13, 1, "ERROR: Could not find dump_modify ID: bogus"],
  ["w209ms_water_flex", 17, 1, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w209ms_cg_peratom", 33, 1, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w243de_efield_ljcoulmsm", 10, 1, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w54accel_melt", 2, 1, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w12tip4p_cut", 8, 1, "ERROR: All pair coeffs are not set. Status:"],
  ["w117cf_tip4p_pppm_water", 7, 1, "ERROR: All pair coeffs are not set. Status:"],
  ["w69fep_tip4plj", 124, 1, "ERROR: All pair coeffs are not set. Status:"],
  ["w242tp_ljcut_pppm_tip4p", 8, 1, "ERROR: All pair coeffs are not set. Status:"],
  ["w69fep_morse", 15, 1, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w142mx_multi_lucy_lookup", 25, 1, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w55reax_rdx_tab", 8, 1, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w69fep_tip4p", 122, 1, "ERROR: All pair coeffs are not set. Status:"],
  ["w37respa_bond", 17, 1, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w96rheo_balloon", 61, 1, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w255cs_mesocnt", 11, 2, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w255cs_mesocnt", 9, 2, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w147ab_class2xe", 25, 2, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w217xb_dihedral_spherical", 19, 2, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w80superell_ellipsoid_gran", 25, 2, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["hybrid_molecular", 10, 2, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
];

describe('a bad number in a command or coefficient line, as native LAMMPS', () => {
  it.each(ROWS)('%s line %i (from word %i): %s', async (name, k, first, expected) => {
    const [input, files] = withBadNumber(name, k, first);
    expect(await outcome(input, files)).toBe(expected);
  });
});

/*
 * The TIP4P pair styles keep the otype htype btype atype words of pair_style and read them at each pair_coeff, as native
 * LAMMPS does (src/MOLECULE/pair_lj_cut_tip4p_cut.cpp coeff at b332f28, utils::expand_type_int with verify): a type
 * label is its number, a number outside the types is refused. Measured with native LAMMPS: the case below with types
 * 5, a bond type 3 or an angle type 0 gives Atom type 5 is out of bounds (1-2), Bond type 3 is out of bounds (1-1) and
 * Angle type 0 is out of bounds (1-1); the label words OW HW OW-HW HW-OW-HW give the thermo of the numeric ones
 * (step 10: pe 4628.83804004455, press 135644.516854304); an unknown label gives Atom type string HX not found in
 * labelmap, and OW without a labelmap Expected integer parameter instead of 'OW' in input script or data file.
 */
describe('the TIP4P type words of pair_style, read at pair_coeff', () => {
  const text = readFileSync(join(ORACLE, 'w12tip4p_cut.in'), 'utf8');
  const files = caseFiles(text, text);
  const LABELS = 'labelmap atom 1 OW 2 HW\nlabelmap bond 1 OW-HW\nlabelmap angle 1 HW-OW-HW\n';
  const input = (style: string, labels = ''): string => text.split('\n').filter((l) => !l.startsWith('#'))
    .map((l) => (l.startsWith('pair_style') ? style : l.startsWith('read_data') ? `${l}\n${labels}` : l.startsWith('run') ? 'run 10' : l)).join('\n');

  it.each([
    ['pair_style lj/cut/tip4p/cut 5 2 1 1 0.1546 4.5', '', 'ERROR: Atom type 5 is out of bounds (1-2)'],
    ['pair_style lj/cut/tip4p/cut 1 2 3 1 0.1546 4.5', '', 'ERROR: Bond type 3 is out of bounds (1-1)'],
    ['pair_style lj/cut/tip4p/cut 1 2 1 0 0.1546 4.5', '', 'ERROR: Angle type 0 is out of bounds (1-1)'],
    ['pair_style lj/cut/tip4p/cut OW HX OW-HW HW-OW-HW 0.1546 4.5', LABELS, 'ERROR: Atom type string HX not found in labelmap'],
    ['pair_style lj/cut/tip4p/cut OW HW 1 1 0.1546 4.5', '', "ERROR: Expected integer parameter instead of 'OW' in input script or data file"],
  ])('%s', async (style, labels, expected) => {
    expect(await outcome(input(style, labels), files)).toBe(expected);
  });

  it('type labels give the thermo of the numeric types', async () => {
    const thermo = async (style: string, labels = ''): Promise<Record<string, number>[]> => {
      const rows: Record<string, number>[] = [];
      const s = new Session({ emit: (e: { kind: string; row?: Record<string, number> }) => { if (e.kind === 'thermo' && e.row) rows.push(e.row); }, writeFile: () => {} } as never);
      for (const [name, t] of Object.entries(files)) s.addFile(name, t);
      await s.execute(input(style, labels));
      return rows;
    };
    const num = await thermo('pair_style lj/cut/tip4p/cut 1 2 1 1 0.1546 4.5');
    const lab = await thermo('pair_style lj/cut/tip4p/cut OW HW OW-HW HW-OW-HW 0.1546 4.5', LABELS);
    expect(lab.length).toBe(2);
    expect(lab).toEqual(num);
    expect(Math.abs(lab[1].pe / 4628.83804004455 - 1)).toBeLessThan(1e-12);
    expect(Math.abs(lab[1].press / 135644.516854304 - 1)).toBeLessThan(1e-10);
  });
});
