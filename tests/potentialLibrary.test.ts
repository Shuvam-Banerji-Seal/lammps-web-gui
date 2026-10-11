import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import { styleNames } from '../src/engine/styles';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { buildIndex } from '../scripts/potential-index.mjs';
import {
  engineSupport, missingPotentials, scriptWords, searchPotentials, type PotentialEntry, type PotentialIndex,
} from '../src/components/workbench/notebook/potentialLibrary';

const DIR = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const INDEX: PotentialIndex = { format: 1, source: { upstream: '', commit: 'b332f28', license: 'GPL-2.0', note: '' }, files: buildIndex(DIR).files };

describe('potential library: which library files a script names', () => {
  it('finds the potential files of pair_coeff that the inputs lack', () => {
    const script = 'pair_style eam\npair_coeff 1 1 Cu_u3.eam   # the Cu file\n';
    expect(missingPotentials(INDEX, script, new Set()).map((f) => f.name)).toEqual(['Cu_u3.eam']);
  });

  it('a file the user provides always wins (it is not in the missing list)', () => {
    const script = 'pair_style eam\npair_coeff 1 1 Cu_u3.eam\n';
    expect(missingPotentials(INDEX, script, new Set(['Cu_u3.eam']))).toEqual([]);
  });

  it('ignores names in comments and names the library does not have', () => {
    expect(missingPotentials(INDEX, '# pair_coeff * * Cu_u3.eam Cu\npair_coeff * * nothere.eam Cu\n', new Set())).toEqual([]);
  });

  it('finds both files of a snap or meam pair', () => {
    const script = 'pair_style snap\npair_coeff * * Cu_Zuo_JPCA2020.snapcoeff Cu_Zuo_JPCA2020.snapparam Cu\n';
    expect(missingPotentials(INDEX, script, new Set()).map((f) => f.name).sort())
      .toEqual(['Cu_Zuo_JPCA2020.snapcoeff', 'Cu_Zuo_JPCA2020.snapparam']);
  });

  it('does not offer README files', () => {
    expect(missingPotentials(INDEX, 'include README\n', new Set())).toEqual([]);
  });

  it('splits words on whitespace and strips quotes', () => {
    expect(scriptWords('pair_coeff * * "Cu_u3.eam"  Cu\t#x')).toEqual(['pair_coeff', '*', '*', 'Cu_u3.eam', 'Cu']);
  });
});

describe('potential library: browser lists', () => {
  const files = INDEX.files as PotentialEntry[];
  it('searches by name, element and style', () => {
    expect(searchPotentials(files, 'Cu_u3').map((f) => f.name)).toContain('Cu_u3.eam');
    expect(searchPotentials(files, 'Ni').map((f) => f.name)).toContain('Ni_u3.eam');
    expect(searchPotentials(files, 'eam/alloy').map((f) => f.name)).toContain('AlCu.eam.alloy');
    expect(searchPotentials(files, '').length).toBe(files.length);
  });

  it('says which of a file\'s styles the engine has', () => {
    const entry = files.find((f) => f.name === 'Cu_u3.eam') as PotentialEntry;
    expect(engineSupport(entry, new Set(['eam']))).toEqual({ supported: ['eam'], unsupported: [] });
    expect(engineSupport(entry, new Set())).toEqual({ supported: [], unsupported: ['eam'] });
  });
});

/*
 * Engine compatibility: each representative potential file is read from third_party/lammps/potentials and
 * run for 0 steps on a small crystal; the energy must be finite. Only the file contents are passed to the
 * session (the notebook sends the same text through addFile).
 */
interface Case {
  label: string;
  files: string[];
  lattice: string;
  types: number;
  pair: string;
}
const CASES: Case[] = [
  { label: 'eam (Cu_u3.eam, funcfl)', files: ['Cu_u3.eam'], lattice: 'fcc 3.615', types: 1,
    pair: 'pair_style eam\npair_coeff 1 1 Cu_u3.eam' },
  { label: 'eam/alloy (AlCu.eam.alloy, setfl)', files: ['AlCu.eam.alloy'], lattice: 'fcc 4.05', types: 1,
    pair: 'pair_style eam/alloy\npair_coeff * * AlCu.eam.alloy Cu' },
  { label: 'eam/fs (CuZr_mm.eam.fs)', files: ['CuZr_mm.eam.fs'], lattice: 'fcc 3.615', types: 1,
    pair: 'pair_style eam/fs\npair_coeff * * CuZr_mm.eam.fs Cu' },
  { label: 'tersoff (Si.tersoff)', files: ['Si.tersoff'], lattice: 'diamond 5.431', types: 1,
    pair: 'pair_style tersoff\npair_coeff * * Si.tersoff Si' },
  { label: 'sw (Si.sw)', files: ['Si.sw'], lattice: 'diamond 5.431', types: 1,
    pair: 'pair_style sw\npair_coeff * * Si.sw Si' },
  { label: 'meam (library.meam + Cu.meam)', files: ['library.meam', 'Cu.meam'], lattice: 'fcc 3.615', types: 1,
    pair: 'pair_style meam\npair_coeff * * library.meam Cu Cu.meam Cu' },
  { label: 'snap (Cu_Zuo_JPCA2020.snapcoeff + snapparam)', files: ['Cu_Zuo_JPCA2020.snapcoeff', 'Cu_Zuo_JPCA2020.snapparam'], lattice: 'fcc 3.615', types: 1,
    pair: 'pair_style snap\npair_coeff * * Cu_Zuo_JPCA2020.snapcoeff Cu_Zuo_JPCA2020.snapparam Cu' },
];

const runCase = async (c: Case): Promise<ThermoRow[]> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const f of c.files) session.addFile(f, readFileSync(join(DIR, f), 'utf8'));
  await session.execute(`units metal
atom_style atomic
boundary p p p
lattice ${c.lattice}
region box block 0 2 0 2 0 2
create_box ${c.types} box
create_atoms 1 box
mass * 28.0
${c.pair}
thermo_style custom step pe
thermo_modify format float %.15g
run 0
`);
  return events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
};

describe('potential library: the engine runs the library files (run 0, finite energy)', () => {
  const pairStyles = new Set(styleNames().pair_style);
  for (const c of CASES) {
    it(c.label, async () => {
      const rows = await runCase(c);
      expect(rows.length).toBeGreaterThan(0);
      expect(Number.isFinite(rows[rows.length - 1].pe)).toBe(true);
    });
  }
  it('the representative pair styles are registered in the engine', () => {
    for (const s of ['eam', 'eam/alloy', 'eam/fs', 'tersoff', 'sw', 'meam', 'snap']) expect(pairStyles.has(s)).toBe(true);
  });
});
