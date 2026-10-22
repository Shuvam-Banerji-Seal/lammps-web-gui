import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * fix qeq/point, qeq/shielded, qeq/dynamic, qeq/fire and qeq/slater: their argument checks, init checks and warnings as
 * native LAMMPS has them (src/QEQ/fix_qeq.cpp and fix_qeq_{point,shielded,dynamic,fire,slater}.cpp at b332f28).
 * Measured with native LAMMPS (/home/roy/.local/bin/lmp, b332f28): each row is the input below with one fix line and
 * run 0; null when it runs, else the start of native's error message.
 */

const PARAMS = '# UNITS: metal\n1     0.00000         7.25028        0.010000        0.772871        0.000000\n2    11.26882        15.37920        0.010000        0.243072        0.000000\n';

const input = (fix: string, opts: { style?: string; pre?: string; charges?: string; extra?: string; runs?: number } = {}): string => {
  const style = opts.style ?? 'charge';
  return `units metal
atom_style ${style}
boundary f f f
${opts.pre ?? ''}region box block 0 30 0 30 0 30
create_box 2 box
create_atoms 1 single 5 15 15
create_atoms 2 single 7 15 15
create_atoms 1 single 9 15 15
create_atoms 2 single 11 15 15
mass * 1.0
${style === 'charge' ? opts.charges ?? 'set group all charge 0.0\n' : ''}pair_style ${style === 'charge' ? 'lj/cut/coul/cut 10.0 8.0' : 'lj/cut 8.0'}
pair_coeff * * 0.0 1.0
${opts.extra ?? ''}${fix}
thermo_style custom step pe f_q
${'run 0\n'.repeat(opts.runs ?? 1)}`;
};

const run = async (text: string, files: Record<string, string> = {}): Promise<{ err: string | null; log: string[] }> => {
  const log: string[] = [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const s = new Session({ emit: (e: any) => { if (e.kind === 'log') log.push(e.text); }, writeFile: () => {} });
  s.addFile('qp.params', PARAMS);
  for (const [k, v] of Object.entries(files)) s.addFile(k, v);
  let err: string | null = null;
  try { await s.execute(text); } catch (e) { err = (e as Error).message; }
  return { err, log };
};

const ROWS: [string, string | null][] = [];
for (const st of ['shielded', 'point']) {
  ROWS.push(
    [`fix q all qeq/${st} 1 8 1.0e-14 200 qp.params`, null],
    [`fix q all qeq/${st} 1 8 1.0e-14 200`, `Illegal fix qeq/${st} command: missing argument(s)`],
    [`fix q all qeq/${st} 0 8 1.0e-14 200 qp.params`, 'Illegal fix qeq command'],
    [`fix q all qeq/${st} 1 0 1.0e-14 200 qp.params`, 'Illegal fix qeq command'],
    [`fix q all qeq/${st} 1 8 0 200 qp.params`, 'Illegal fix qeq command'],
    [`fix q all qeq/${st} 1 8 1.0e-14 0 qp.params`, 'Illegal fix qeq command'],
    [`fix q all qeq/${st} 1.5 8 1.0e-14 200 qp.params`, "Expected integer parameter instead of '1.5' in input script or data file"],
    [`fix q all qeq/${st} 1 abc 1.0e-14 200 qp.params`, "Expected floating point parameter instead of 'abc' in input script or data file"],
    [`fix q all qeq/${st} 1 8 1.0e-14 200 qp.params warn no`, null],
    [`fix q all qeq/${st} 1 8 1.0e-14 200 qp.params warn maybe`, "Expected boolean parameter instead of 'maybe' in input script or data file"],
    [`fix q all qeq/${st} 1 8 1.0e-14 200 qp.params bogus 1`, `Illegal fix qeq/${st} command`],
    [`fix q all qeq/${st} 1 8 1.0e-14 200 qp.params warn`, `Illegal fix qeq/${st} command`],
    [`fix q all qeq/${st} 1 8 1.0e-14 200 qp.params warn no extra`, `Illegal fix qeq/${st} command`],
    [`fix q all qeq/${st} 1 8 1.0e-14 200 qp_one.params`, 'Parameters for atom type 2 missing in qeq parameter file'],
    [`fix q all qeq/${st} 1 8 1.0e-14 200 qp_g0.params`, null],
    [`fix q all qeq/${st} 1 4 1.0e-14 200 qp.params`, null],
  );
}
ROWS.push(['fix q all qeq/shielded 1 8 1.0e-14 200 reaxff', 'No reaxff pair style for fix qeq/shielded']);
for (const st of ['dynamic', 'fire']) {
  ROWS.push(
    [`fix q all qeq/${st} 1 8 1.0e-3 200 qp.params`, null],
    [`fix q all qeq/${st} 1 8 1.0e-3 200 qp.params qdamp 0.2 qstep 0.01 warn no`, null],
    [`fix q all qeq/${st} 1 8 1.0e-3 200 qp.params qdamp`, `Illegal fix qeq/${st} qdamp command: missing argument(s)`],
    [`fix q all qeq/${st} 1 8 1.0e-3 200 qp.params bogus 1`, `Unknown fix qeq/${st} keyword: bogus`],
    [`fix q all qeq/${st} 1 8 1.0e-3 200 qp.params warn maybe`, "Expected boolean parameter instead of 'maybe' in input script or data file"],
    [`fix q all qeq/${st} 1 8 1.0e-3 200 qp.params qstep 0`, null],
    [`fix q all qeq/${st} 1 8 1.0e-3 200 qp.params qdamp abc`, "Expected floating point parameter instead of 'abc' in input script or data file"],
  );
}
ROWS.push(
  ['fix q all qeq/slater 1 8 1.0e-6 200 qp.params', null],
  ['fix q all qeq/slater 1 8 1.0e-6 200 qp.params alpha 0.3', null],
  ['fix q all qeq/slater 1 8 1.0e-6 200 qp.params alpha', 'Illegal fix qeq/slater alpha command: missing argument(s)'],
  ['fix q all qeq/slater 1 8 1.0e-6 200 qp.params alpha 0.3 dsf', null],
  ['fix q all qeq/slater 1 8 1.0e-6 200 qp.params alpha 0.3 dsf 1.0 warn no', null],
  ['fix q all qeq/slater 1 8 1.0e-6 200 qp.params bogus 1', 'Unknown fix qeq/slater keyword: bogus'],
  ['fix q all qeq/slater 1 8 1.0e-6 200 qp.params alpha 0', null],
  ['fix q all qeq/slater 1 8 1.0e-6 200 qp.params alpha -1', null],
  ['fix q all qeq/slater 1 8 1.0e-6 200 coul/streitz', 'No pair style coul/streitz for fix qeq/slater'],
  ['fix q all qeq/slater 1 8 1.0e-6 200 qp.params warn maybe', "Expected boolean parameter instead of 'maybe' in input script or data file"],
);

const FILES = {
  'qp_one.params': '1 0.0 7.25 0.01 0.77 0.0\n',
  'qp_g0.params': '1 0.0 7.25 0.01 0.77 0.0\n2 11.26 15.37 0.0 0.24 0.0\n',
};

describe('fix qeq as native LAMMPS', () => {
  it.each(ROWS)('%s', async (fix, expected) => {
    const { err } = await run(input(fix), FILES);
    if (expected === null) expect(err).toBeNull();
    else expect(err).toContain(expected);
  });

  it.each(['shielded', 'point', 'dynamic', 'fire', 'slater'])('qeq/%s: the init and setup stops', async (st) => {
    const fix = (g: string) => `fix q ${g} qeq/${st} 1 8 1.0e-6 200 qp.params`;
    expect((await run(input(fix('all'), { style: 'atomic' }))).err).toContain(`Fix qeq/${st} requires atom attribute q`);
    expect((await run(input(fix('none'), { extra: 'group none empty\n' }))).err).toContain(`Fix qeq/${st} group has no atoms`);
    expect((await run(input(fix('all'), { pre: 'newton off\n' }))).err).toContain("QEQ with 'newton pair off' not supported");
  });

  it('qeq/shielded: a gamma of 0 stops only from the second run on (native checks over the nlocal of the last solve)', async () => {
    const fix = 'fix q all qeq/shielded 1 8 1.0e-14 200 qp_g0.params';
    expect((await run(input(fix), FILES)).err).toBeNull();
    expect((await run(input(fix, { runs: 2 }), FILES)).err).toContain('Invalid QEq parameters for atom type 2 provided');
  });

  it('the warnings of init', async () => {
    const warnings = async (fix: string, opts: Parameters<typeof input>[1] = {}): Promise<string[]> =>
      (await run(input(fix, opts), FILES)).log.filter((l) => l.startsWith('WARNING'));
    expect(await warnings('fix q all qeq/shielded 1 4 1.0e-14 200 qp.params')).toContain('WARNING: Fix qeq has very low Taper radius cutoff');
    expect(await warnings('fix q all qeq/point 1 8 1.0e-14 200 qp.params', { extra: 'fix ef all efield 0.1 0 0\n' }))
      .toContain('WARNING: Fix efield is ignored during charge equilibration');
    expect(await warnings('fix q all qeq/shielded 1 8 1.0e-14 200 qp.params', { charges: 'set group all charge 0.0\nset atom 1 charge 0.5\n' }))
      .toContain('WARNING: Fix qeq/shielded group is not charge neutral, net charge = 0.5\nFor more information see https://docs.lammps.org/err0029');
    expect(await warnings('fix q all qeq/dynamic 1 8 1.0e-12 200 qp.params'))
      .toContain('WARNING: Fix qeq/dynamic tolerance 1e-12 may be too small for damped dynamics');
    expect(await warnings('fix q all qeq/fire 1 8 1.0e-12 200 qp.params'))
      .toContain('WARNING: Fix qeq/fire tolerance 1e-12 may be too small for damped fires');
    // the tolerance warning belongs to the damped styles only
    expect((await warnings('fix q all qeq/point 1 8 1.0e-14 200 qp.params')).some((w) => w.includes('tolerance'))).toBe(false);
  });
});
