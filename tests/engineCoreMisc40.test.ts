import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Core command behaviours (w40core): dynamic group keywords within, include
 * molecule and exclude (group.html), change_box with lost atoms (change_box.html
 * and thermo_modify lost), the warning of an auto-deleted file variable
 * (variable.html), and the neighbor rebuild on restart steps (restart.html).
 * Whole-run parity with native LAMMPS is in tests/oracle/w40core_*.in.
 */

const ORACLE = join(__dirname, 'oracle');
const PEPTIDE = readFileSync(join(ORACLE, 'w40core_peptide.data'), 'utf8');

interface Out {
  error: string | null;
  logs: string[];
  files: Map<string, string>;
  session: Session;
}

const runText = async (text: string, inputs: Record<string, string> = {}): Promise<Out> => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (ev) => events.push(ev),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  for (const [name, body] of Object.entries(inputs)) session.addFile(name, body);
  let error: string | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e): e is Extract<EngineEvent, { kind: 'error' }> => e.kind === 'error');
  const logs = events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);
  return { error: err?.message ?? error, logs, files, session };
};

/** Member IDs of a dynamic group after run 0, read from a dump written at the setup of the run. */
const members = async (dynamic: string): Promise<number[]> => {
  const out = await runText(`units lj
atom_style full
read_data w40core_peptide.data
pair_style lj/cut 2.5
pair_coeff * * 0.0 1.0
group solute type 2
group solv type 1
region left block 0 5 0 10 0 10 units box
${dynamic}
dump 1 d custom 1 probe.dump id
run 0
`, { 'w40core_peptide.data': PEPTIDE });
  expect(out.error).toBeNull();
  const text = out.files.get('probe.dump') ?? '';
  const ids: number[] = [];
  let atoms = false;
  for (const line of text.split('\n')) {
    if (line.startsWith('ITEM: ATOMS')) { atoms = true; continue; }
    if (atoms && line.trim()) ids.push(Number(line.trim().split(/\s+/)[0]));
  }
  return ids.sort((a, b) => a - b);
};

describe('group dynamic within, include molecule and exclude: membership (group.html)', () => {
  // Measured with native LAMMPS (black box) on the same 60-atom data file, after run 0.
  const cases: [string, number[]][] = [
    ['group d dynamic solute within 2.5', [1, 2, 3, 4, 5, 6, 11, 16, 17, 21, 26, 27, 34, 41, 57]],
    ['group d dynamic solute within 2.5 include molecule', [1, 2, 3, 4, 5, 6, 11, 12, 15, 16, 17, 18, 21, 22, 25, 26, 27, 28, 33, 34, 41, 42, 57, 58]],
    ['group d dynamic all region left within 2.5', [1, 2, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 23, 30, 31, 32, 33, 37, 38, 39, 40, 44, 46, 48, 49, 50, 51, 56, 58]],
    ['group d dynamic solute region left within 2.5', [1, 2, 5, 11, 17, 26, 41]],
    ['group d dynamic solute within 2.5 exclude solv', [1, 2, 3, 4]],
    ['group d dynamic solute within 2.5 exclude solute', [5, 6, 11, 16, 17, 21, 26, 27, 34, 41, 57]],
    ['group d dynamic solute include molecule within 2.5 exclude solv', [1, 2, 3, 4]],
    ['group d dynamic solute within 2.5 every 3', [1, 2, 3, 4, 5, 6, 11, 16, 17, 21, 26, 27, 34, 41, 57]],
  ];
  for (const [dynamic, want] of cases) {
    it(dynamic, async () => {
      expect(await members(dynamic)).toEqual(want);
    });
  }
});

describe('group dynamic keywords: parsing and refusals (native texts)', () => {
  const cases: [string, string | RegExp][] = [
    ['group d dynamic solute within', 'Illegal group dynamic within command: missing argument(s)'],
    ['group d dynamic solute within -1', 'Illegal within cutoff value -1 for dynamic group d'],
    ['group d dynamic solute within 0.0', 'Illegal within cutoff value 0 for dynamic group d'],
    ['group d dynamic solute within abc', "Expected floating point parameter instead of 'abc'"],
    ['group d dynamic solute include atom', 'Unknown include setting atom in dynamic group command'],
    ['group d dynamic solute include', 'Illegal group dynamic include command: missing argument(s)'],
    ['group d dynamic solute exclude', 'Illegal group dynamic exclude command: missing argument(s)'],
    ['group d dynamic solute exclude nosuch\nrun 0', 'Group ID nosuch requested by group dynamic exclude does not exist'],
    ['group d dynamic solute within 2.9\nrun 0', 'Custom neighbor list cutoff 3.1999999999999997 is too large for communication cutoff 2.8'],
    ['group d dynamic solute include molecule molecule', 'Unknown keyword molecule in dynamic group command'],
  ];
  for (const [text, want] of cases) {
    it(text.split('\n').slice(-2).join(' | '), async () => {
      const out = await runText(`units lj\natom_style full\nread_data w40core_peptide.data\npair_style lj/cut 2.5\npair_coeff * * 0.0 1.0\ngroup solute type 2\n${text}`, { 'w40core_peptide.data': PEPTIDE });
      expect(out.error).not.toBeNull();
      if (typeof want === 'string') expect(out.error).toContain(want);
      else expect(out.error).toMatch(want);
    });
  }

  it('include molecule needs molecule IDs (atom_style atomic)', async () => {
    const out = await runText('units lj\natom_style atomic\nlattice fcc 0.8442\nregion b block 0 3 0 3 0 3\ncreate_box 1 b\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\ngroup d dynamic all include molecule\nrun 0');
    expect(out.error).toContain('Dynamic Group include molecule setting requires atom attribute molecule');
  });

  it('accepts within, include molecule and exclude on a valid command', async () => {
    const out = await runText('units lj\natom_style full\nread_data w40core_peptide.data\npair_style lj/cut 2.5\npair_coeff * * 0.0 1.0\ngroup solute type 2\ngroup d dynamic solute within 2.5 include molecule exclude solute every 2\nrun 0', { 'w40core_peptide.data': PEPTIDE });
    expect(out.error).toBeNull();
    expect(out.logs).toContain('dynamic group d defined');
    expect(out.session.sys.groups.dynamic.size).toBe(1);
  });
});

describe('change_box with atoms left outside a fixed boundary (thermo_modify lost)', () => {
  const CB = `units lj
atom_style atomic
boundary f p p
read_data w40core_cb.data
pair_style lj/cut 2.5
pair_coeff * * 0.1 1.0
`;
  const CB_DATA = readFileSync(join(ORACLE, 'w40core_cb.data'), 'utf8');

  it('default lost error stops the command with the native text', async () => {
    // Measured with native LAMMPS (black box): Lost atoms via change_box: original 20 current 9
    const out = await runText(`${CB}change_box all x final 0 5\n`, { 'w40core_cb.data': CB_DATA });
    expect(out.error).toContain('Lost atoms via change_box: original 20 current 9');
  });

  it('thermo_modify lost warn prints the warning and deletes the atoms', async () => {
    const out = await runText(`${CB}thermo_modify lost warn\nchange_box all x final 0 5\nrun 0\n`, { 'w40core_cb.data': CB_DATA });
    expect(out.error).toBeNull();
    expect(out.logs).toContain('WARNING: Lost atoms via change_box: original 20 current 9');
    expect(out.session.sys.state.n).toBe(9);
  });

  it('thermo_modify lost ignore deletes the atoms silently', async () => {
    const out = await runText(`${CB}thermo_modify lost ignore\nchange_box all x final 0 5\nrun 0\n`, { 'w40core_cb.data': CB_DATA });
    expect(out.error).toBeNull();
    expect(out.logs.some((l) => l.includes('Lost atoms via change_box'))).toBe(false);
    expect(out.session.sys.state.n).toBe(9);
  });

  it('remap keeps every atom inside the new box', async () => {
    const out = await runText(`${CB}change_box all x final 0 5 remap\nrun 0\n`, { 'w40core_cb.data': CB_DATA });
    expect(out.error).toBeNull();
    expect(out.session.sys.state.n).toBe(20);
  });
});

describe('file variables: auto-deleted warning and the classic loop (variable.html)', () => {
  it('prints Auto-deleting variable once, when next exhausts the file, and the jump is skipped', async () => {
    const out = await runText(`units lj
variable a file vals.txt
label loop
print "value $a"
next a
jump SELF loop
print "after"
`, { 'vals.txt': '1.5\n2.5\n3.5\n' });
    expect(out.error).toBeNull();
    const warns = out.logs.filter((l) => l.includes('Auto-deleting variable a'));
    expect(warns).toEqual(['WARNING: Auto-deleting variable a']);
    expect(out.logs.filter((l) => l.startsWith('value ')).slice(0, 3)).toEqual(['value 1.5', 'value 2.5', 'value 3.5']);
    expect(out.logs).toContain('after');
  });

  it('index variables are exhausted silently', async () => {
    const out = await runText('units lj\nvariable i index 1 2\nlabel l\nprint "i=$i"\nnext i\njump SELF l\nprint done\n');
    expect(out.error).toBeNull();
    expect(out.logs.some((l) => l.includes('Auto-deleting'))).toBe(false);
  });
});

describe('restart N forces a neighbor list rebuild on the steps it writes (restart.html)', () => {
  const melt = (extra: string) => `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 5 0 5 0 5
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 3.0 87287
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
neighbor 0.3 bin
neigh_modify every 10 delay 0 check yes
fix 1 all nve
${extra}
thermo 25
run 50
`;
  const builds = async (extra: string): Promise<number> => {
    const out = await runText(melt(extra));
    expect(out.error).toBeNull();
    return (out.session as unknown as { sys: { nb: { nbuild: number } } }).sys.nb.nbuild;
  };

  it('a restart step adds a rebuild (25 is not a multiple of every 10)', async () => {
    const without = await builds('');
    const withRestart = await builds('restart 25 melt.*.restart');
    expect(withRestart).toBe(without + 1);
  });
});
