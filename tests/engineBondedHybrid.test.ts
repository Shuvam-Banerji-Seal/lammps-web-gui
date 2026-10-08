import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { BondedHybrid } from '../src/engine/force/bonded_hybrid';
import { BOND_STYLES } from '../src/engine/styles';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * bond_style hybrid, angle_style hybrid, dihedral_style hybrid, improper_style hybrid
 * (force/bonded_hybrid.ts) and compute bond | angle | dihedral | improper
 * (compute/bonded_energy.ts). Native parity for the two-sub-style cases is in
 * tests/oracle/w21bhyb_*.in; the messages checked here were measured with native
 * LAMMPS (black box). Doc page: docs.lammps.org/bond_hybrid.html.
 */

/** Four atoms in a chain with two types of every kind (the energies are checked against hand values). */
const TINY = `tiny four atoms

4 atoms
3 bonds
2 angles
2 dihedrals
2 impropers
2 atom types
2 bond types
2 angle types
2 dihedral types
2 improper types

0 10.0 xlo xhi
0 10.0 ylo yhi
0 10.0 zlo zhi

Masses

1 1.0
2 1.0

Atoms # molecular

1 1 1 1.0 1.0 1.0
2 1 1 2.1 1.0 1.0
3 1 2 3.2 1.0 1.0
4 1 2 3.2 2.0 1.0

Bonds

1 1 1 2
2 2 2 3
3 1 3 4

Angles

1 1 1 2 3
2 2 2 3 4

Dihedrals

1 1 1 2 3 4
2 2 1 2 3 4

Impropers

1 1 2 1 3 4
2 2 2 1 3 4
`;

/*
 * Atoms 1, 2 and 3 of TINY are collinear, so the dihedral 1-2-3-4 and the improper 2-1-3-4 are undefined
 * there: the engine gives 0 for both, native gives other values (see the remaining issues). TINY_DIH moves
 * atoms 3 and 4 off the line, and the dihedral and improper checks use it.
 */
const TINY_DIH = TINY.replace('3 1 2 3.2 1.0 1.0\n4 1 2 3.2 2.0 1.0', '3 1 2 2.6 1.9 1.0\n4 1 2 3.6 2.0 1.4');
const HEAD = 'units lj\natom_style molecular\nread_data tiny.data\npair_style zero 2.5\npair_coeff * *\n';
const THERMO = 'thermo_style custom step ebond eangle edihed eimp\nthermo_modify norm no\n';

const runScript = async (script: string, files: Record<string, string> = { 'tiny.data': TINY }) => {
  const events: EngineEvent[] = [];
  const written = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => written.set(n, (ap ? written.get(n) ?? '' : '') + t),
  });
  for (const [name, text] of Object.entries(files)) session.addFile(name, text);
  try {
    await session.execute(script);
  } catch {
    // the failure is reported as an error event below
  }
  const err = events.find((e) => e.kind === 'error');
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as ThermoRow);
  return { error: err && 'message' in err ? String(err.message) : undefined, rows, written, session };
};

describe('bond_style hybrid: settings', () => {
  it('refuses hybrid and none as a sub-style, and a repeated sub-style', async () => {
    expect((await runScript(`${HEAD}bond_style hybrid hybrid\n`)).error).toMatch(/Bond style hybrid cannot have hybrid as an argument/);
    expect((await runScript(`${HEAD}angle_style hybrid none\n`)).error).toMatch(/Angle style hybrid cannot have none as an argument/);
    expect((await runScript(`${HEAD}bond_style hybrid harmonic harmonic\n`)).error).toMatch(/Bond style hybrid cannot use same bond style twice/);
  });

  it('refuses a settings word that is not a style name, and a missing list', async () => {
    expect((await runScript(`${HEAD}bond_style hybrid harmonic 1.0\n`)).error).toMatch(/Illegal bond_style hybrid argument: 1\.0/);
    expect((await runScript(`${HEAD}bond_style hybrid\n`)).error).toMatch(/missing argument/);
  });

  it('names an unsupported first sub-style in the error', async () => {
    expect((await runScript(`${HEAD}bond_style hybrid foo\n`)).error).toMatch(/'foo' is not supported by the browser engine/);
  });

  it('re-issuing bond_style hybrid clears the coefficients (measured: All bond coeffs are not set)', async () => {
    const r = await runScript(`${HEAD}bond_style hybrid harmonic zero\nbond_coeff 1 harmonic 100 1.0\nbond_coeff 2 zero\nrun 0\nbond_style hybrid harmonic zero\nrun 0\n`);
    expect(r.error).toMatch(/All bond coeffs are not set/);
  });
});

describe('bond_style hybrid: coefficients', () => {
  it('refuses a style that is not in the list', async () => {
    const r = await runScript(`${HEAD}bond_style hybrid harmonic zero\nbond_coeff 1 harmonic 100 1.0\nbond_coeff 2 foo 1\n`);
    expect(r.error).toMatch(/Expected hybrid sub-style instead of foo in bond_coeff command/);
  });

  it('refuses skip for bond (measured: Expected hybrid sub-style instead of skip)', async () => {
    const r = await runScript(`${HEAD}bond_style hybrid harmonic\nbond_coeff 1 harmonic 100 1.0\nbond_coeff 2 skip\n`);
    expect(r.error).toMatch(/Expected hybrid sub-style instead of skip in bond_coeff command/);
  });

  it('angle skip leaves the type unset, so the run stops (measured)', async () => {
    const r = await runScript(`${HEAD}angle_style hybrid harmonic\nangle_coeff 1 harmonic 40 109.5\nangle_coeff 2 skip\nrun 0\n`);
    expect(r.error).toMatch(/All angle coeffs are not set \(type 2\)/);
  });

  it('a later line overrides the sub-style of a type (measured energies below)', async () => {
    // angle type 1 harmonic (K=40, 109.5 deg) on the 180 deg angle; type 2 cosine (K=50) on the 90 deg angle
    const r = await runScript(`${HEAD}angle_style hybrid harmonic cosine\nangle_coeff 1 cosine 10.0\nangle_coeff 1 harmonic 40 109.5\nangle_coeff 2 cosine 50\ncompute ca all angle\nthermo_style custom step eangle c_ca[1] c_ca[2]\nthermo_modify norm no\nrun 0\n`);
    expect(r.error).toBeUndefined();
    const row = r.rows[0] as Record<string, number>;
    // measured with native LAMMPS (black box): E_angle 110.560989227796 with these coefficients
    expect(row.eangle).toBeCloseTo(110.560989227796, 9);
    expect(row['c_ca[1]']).toBeCloseTo(60.560989227796, 9);
    expect(row['c_ca[2]']).toBeCloseTo(50, 9);
  });
});

describe('bond_style hybrid: run-time checks', () => {
  it('a listed sub-style that no type uses stops the run (measured)', async () => {
    const r = await runScript(`${HEAD}bond_style hybrid harmonic zero\nbond_coeff 1 harmonic 100 1.0\nbond_coeff 2 none\nrun 0\n`);
    expect(r.error).toMatch(/Bond hybrid sub-style zero is not used/);
  });

  it('a none bond type stops every run with the equilibrium message (measured)', async () => {
    const r = await runScript(`${HEAD}bond_style hybrid harmonic\nbond_coeff 1 harmonic 100 1.0\nbond_coeff 2 none\nrun 0\n`);
    expect(r.error).toMatch(/Invoked bond equil distance on bond style none/);
  });

  it('a none angle type gives no energy (measured: E_angle 60.5609892277955 with one harmonic type)', async () => {
    const r = await runScript(`${HEAD}angle_style hybrid harmonic\nangle_coeff 1 harmonic 40 109.5\nangle_coeff 2 none\n${THERMO}run 0\n`);
    expect(r.error).toBeUndefined();
    expect((r.rows[0] as Record<string, number>).eangle).toBeCloseTo(60.5609892277955, 9);
  });

  it('an unset type stops the run', async () => {
    const r = await runScript(`${HEAD}bond_style hybrid harmonic zero\nbond_coeff 1 harmonic 100 1.0\nrun 0\n`);
    expect(r.error).toMatch(/All bond coeffs are not set/);
  });
});

describe('compute bond | angle | dihedral | improper', () => {
  const HYB = `${HEAD}bond_style hybrid harmonic zero
bond_coeff 1 harmonic 100 1.0
bond_coeff 2 zero
angle_style hybrid harmonic cosine
angle_coeff 1 harmonic 40 109.5
angle_coeff 2 cosine 50
dihedral_style hybrid harmonic zero
dihedral_coeff 1 harmonic 1.5 -1 3
dihedral_coeff 2 zero
improper_style hybrid cvff zero
improper_coeff 1 cvff 12 -1 2
improper_coeff 2 zero
`;

  it('the vectors hold the per-sub-style energies and add up to the totals (measured values)', async () => {
    const r = await runScript(`${HYB}compute cb all bond
compute ca all angle
compute cd all dihedral
compute ci all improper
thermo_style custom step ebond eangle edihed eimp c_cb[1] c_cb[2] c_ca[1] c_ca[2] c_cd[1] c_cd[2] c_ci[1] c_ci[2]
thermo_modify norm no
run 0
`);
    expect(r.error).toBeUndefined();
    const row = r.rows[0] as Record<string, number>;
    expect(row['c_cb[1]'] + row['c_cb[2]']).toBeCloseTo(row.ebond, 12);
    expect(row['c_ca[1]'] + row['c_ca[2]']).toBeCloseTo(row.eangle, 12);
    expect(row['c_cd[1]'] + row['c_cd[2]']).toBeCloseTo(row.edihed, 12);
    expect(row['c_ci[1]'] + row['c_ci[2]']).toBeCloseTo(row.eimp, 12);
    // measured with native LAMMPS (black box) for this input: bond 1 with K=100, r0=1.0 gives 1
    expect(row['c_cb[1]']).toBeCloseTo(1, 12);
    expect(row['c_cb[2]']).toBe(0);
    expect(row['c_ca[1]']).toBeCloseTo(60.560989227796, 9);
    expect(row['c_ca[2]']).toBeCloseTo(50, 12);
  });

  it('the dihedral vector holds the harmonic sub-style energy (measured on a non-degenerate chain)', async () => {
    const r = await runScript(`${HYB}compute cd all dihedral\nthermo_style custom step edihed eimp c_cd[1] c_cd[2]\nthermo_modify norm no\nrun 0\n`, { 'tiny.data': TINY_DIH });
    expect(r.error).toBeUndefined();
    const row = r.rows[0] as Record<string, number>;
    // measured with native LAMMPS (black box) on this geometry: edihed 1.823366 and c_cd[1] 1.823366, c_cd[2] 0
    expect(row['c_cd[1]']).toBeCloseTo(1.823366, 6);
    expect(row['c_cd[2]']).toBe(0);
    expect(row.edihed).toBeCloseTo(row['c_cd[1]'], 12);
    // measured with native LAMMPS (black box) on this geometry: eimp 11.907251 from the cvff improper 1
    expect(row.eimp).toBeCloseTo(11.907251, 6);
  });

  it('the group is ignored (a group that exists gives the same values)', async () => {
    const r = await runScript(`${HYB}group g id 1 2\ncompute cb g bond\ncompute cb2 all bond\nthermo_style custom step c_cb[1] c_cb2[1]\nthermo_modify norm no\nrun 0\n`);
    expect(r.error).toBeUndefined();
    const row = r.rows[0] as Record<string, number>;
    expect(row['c_cb[1]']).toBe(row['c_cb2[1]']);
  });

  it('refuses a plain bond style (measured: Bond style for compute bond command is not hybrid)', async () => {
    const r = await runScript(`${HEAD}bond_style harmonic\nbond_coeff * 100 1.0\ncompute cb all bond\n`);
    expect(r.error).toMatch(/Bond style for compute bond command is not hybrid/);
  });

  it('refuses a missing angle style (measured: Angle style for compute angle command is not hybrid)', async () => {
    const r = await runScript(`${HEAD}compute ca all angle\n`);
    expect(r.error).toMatch(/Angle style for compute angle command is not hybrid/);
  });

  it('refuses an extra argument (measured: Illegal compute bond command)', async () => {
    const r = await runScript(`${HYB}compute cb all bond extra\n`);
    expect(r.error).toMatch(/Illegal compute bond command/);
  });

  it('refuses a compute whose style was replaced by a plain one (measured at the next evaluation)', async () => {
    const r = await runScript(`${HYB}compute cb all bond\nbond_style harmonic\nbond_coeff * 100 1.0\nthermo_style custom step c_cb[1]\nrun 0\n`);
    expect(r.error).toMatch(/Bond style for compute bond command is not hybrid/);
  });
});

describe('write_data and read_data with hybrid styles', () => {
  it('write_data writes no coefficient section for a hybrid style (measured)', async () => {
    const r = await runScript(`${HEAD}bond_style hybrid harmonic zero\nbond_coeff 1 harmonic 100 1.0\nbond_coeff 2 zero\nangle_style hybrid harmonic\nangle_coeff 1 harmonic 40 109.5\nangle_coeff 2 none\nwrite_data out.data\n`);
    expect(r.error).toBeUndefined();
    const text = r.written.get('out.data') ?? '';
    expect(text).toContain('Atoms # molecular');
    expect(text).not.toMatch(/Bond Coeffs/);
    expect(text).not.toMatch(/Angle Coeffs/);
  });

  it('a data file coefficient line naming a style that is not listed is refused', async () => {
    const data = TINY.replace('Masses', 'Bond Coeffs # hybrid\n\n1 harmonic 100 1.0\n2 fene 30 1.5 1 1\n\nMasses');
    const r = await runScript(`units lj\natom_style molecular\nbond_style hybrid harmonic zero\nread_data tiny.data\npair_style zero 2.5\npair_coeff * *\nrun 0\n`, { 'tiny.data': data });
    expect(r.error).toMatch(/Expected hybrid sub-style instead of fene in bond_coeff command/);
  });
});

describe('restart round trip', () => {
  it('keeps the sub-style list and drops the coefficients (measured)', async () => {
    const first = await runScript(`${HEAD}bond_style hybrid harmonic zero\nbond_coeff 1 harmonic 100 1.0\nbond_coeff 2 zero\nrun 0\nwrite_restart rt.restart\n`);
    expect(first.error).toBeUndefined();
    const text = first.written.get('rt.restart');
    expect(text).toBeDefined();
    const second = await runScript(`read_restart rt.restart\npair_style zero 2.5\npair_coeff * *\nrun 0\n`, { 'rt.restart': text! });
    expect(second.error).toMatch(/All bond coeffs are not set/);
    const third = await runScript(`read_restart rt.restart\npair_style zero 2.5\npair_coeff * *\nbond_coeff 1 harmonic 100 1.0\nbond_coeff 2 zero\n${THERMO}run 0\n`, { 'rt.restart': text! });
    expect(third.error).toBeUndefined();
    expect((third.rows[0] as Record<string, number>).ebond).toBeCloseTo(1, 12);
  });

  it('a restored hybrid still refuses a style that was not listed', async () => {
    const first = await runScript(`${HEAD}bond_style hybrid harmonic zero\nbond_coeff 1 harmonic 100 1.0\nbond_coeff 2 zero\nrun 0\nwrite_restart rt.restart\n`);
    const r = await runScript(`read_restart rt.restart\nbond_coeff 1 fene 30 1.5 1 1\n`, { 'rt.restart': first.written.get('rt.restart')! });
    expect(r.error).toMatch(/Expected hybrid sub-style instead of fene in bond_coeff command/);
  });
});

describe('BondedHybrid: equilibrium lengths route to the sub-style', () => {
  it('bond equilibrium comes from the harmonic sub-style of the type; none stops with the measured message', () => {
    const st = new BondedHybrid('bond', BOND_STYLES);
    st.settings(['harmonic', 'zero']);
    st.allocate(2);
    st.coeff(['1', 'harmonic', '100', '1.05']);
    st.coeff(['2', 'zero']);
    expect(st.equilibrium(1)).toBeCloseTo(1.05, 12);
    st.coeff(['2', 'none']);
    expect(() => st.equilibrium(2)).toThrow(/Invoked bond equil distance on bond style none/);
  });
});
