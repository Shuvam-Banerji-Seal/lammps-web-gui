import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/session';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { UNIT_SYSTEMS } from '../src/engine/units';

/*
 * Wave-18 simple computes: compute nbond/atom, compute count/type and compute
 * erotate/sphere/atom (docs.lammps.org/compute_nbond_atom.html,
 * compute_count_type.html, compute_erotate_sphere_atom.html; page text in
 * plans/lammps-docs). Native parity is in tests/oracle/w18cmisc.in and
 * tests/oracle/w18cerot.in; here the rules the doc pages state are checked
 * directly, including the group semantics and the StyleError paths.
 *
 * Measured with native LAMMPS (black box) rules that the doc pages leave open
 * are quoted in the test names, e.g. that a bond which is off (type < 1) is
 * not counted by nbond/atom but is counted by count/type in its own type slot.
 */

const CHAINS = readFileSync(join(__dirname, 'oracle', 'mol_chains.data'), 'utf8');

/** A branched-chain molecular system: 90 atoms, 72 bonds, 72 angles, 36 dihedrals, 18 impropers. */
const MOLECULAR = `units lj
atom_style molecular
bond_style harmonic
angle_style harmonic
dihedral_style harmonic
improper_style harmonic
pair_style lj/cut 2.5
read_data mol_chains.data
pair_coeff * * 0.5 0.9
bond_coeff 1 120.0 1.05
bond_coeff 2 90.0 1.1
angle_coeff 1 40.0 109.5
angle_coeff 2 25.0 120.0
dihedral_coeff 1 1.5 -1 3
improper_coeff 1 12.0 20.0
special_bonds lj 0.0 0.5 0.8
velocity all set 0.0 0.0 0.0
`;

/** atom_style sphere with explicit per-atom mass and radius: 3 atoms, mass 3, radius 1. */
const SPHERE = (units: string) => `${units}
atom_style sphere
lattice sc 2.0
region box block 0 3 0 1 0 1
create_box 1 box
create_atoms 1 box
set group all diameter 2.0
set group all mass 3.0
variable wx atom 0.5*sin(0.3*id)
variable wy atom 0.5*cos(0.7*id)
variable wz atom 0.2*sin(1.1*id)
set group all omega v_wx v_wy v_wz
velocity all set 0.0 0.0 0.0
`;

/** The omega set above for the atom with the given ID. */
const omegaOf = (id: number): [number, number, number] => [
  0.5 * Math.sin(0.3 * id), 0.5 * Math.cos(0.7 * id), 0.2 * Math.sin(1.1 * id),
];

/** Runs an input; returns the session and the thermo rows, or the error message. */
async function run(script: string, files: Record<string, string> = { 'mol_chains.data': CHAINS }) {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const [name, text] of Object.entries(files)) session.addFile(name, text);
  try {
    await session.execute(script);
  } catch (err) {
    return { error: (err as Error).message, session, rows: [] as ThermoRow[] };
  }
  const errEvent = events.find((e) => e.kind === 'error');
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { error: errEvent && 'message' in errEvent ? String(errEvent.message) : undefined, session, rows };
}

/** Per-atom values of `compute c all <style>` after `run 0`. */
async function peratom(script: string, style: string): Promise<Float64Array> {
  const r = await run(`${script}compute c all ${style}\nrun 0\n`);
  expect(r.error).toBeUndefined();
  return r.session.sys.compute('c').peratomValues();
}

/** Degrees of every atom of a group (bonds that are off are not counted), by atom ID. */
function degreesIn(session: Session, groupBit: number): Map<number, number> {
  const s = session.sys.state;
  const map = new Map<number, number>();
  for (let e = 0; e < s.topo.bonds.n; e++) {
    if (s.topo.bonds.type[e] < 1) continue;
    for (let w = 0; w < 2; w++) {
      const i = session.sys.indexOfId(s.topo.bonds.atoms[2 * e + w]);
      if (i < 0 || !(s.mask[i] & groupBit)) continue;
      const id = s.id[i];
      map.set(id, (map.get(id) ?? 0) + 1);
    }
  }
  return map;
}

describe('compute nbond/atom', () => {
  it('counts every bond of every atom of the group and is zero outside it', async () => {
    const r = await run(`${MOLECULAR}group frag id 1:3\ncompute nb frag nbond/atom\nrun 0\n`);
    expect(r.error).toBeUndefined();
    const c = r.session.sys.compute('nb');
    expect(c.style).toBe('nbond/atom');
    expect(c.peratomFlag).toBe(true);
    expect(c.sizePeratomCols).toBe(0);
    const vals = c.peratomValues();
    const want = degreesIn(r.session, c.groupBit);
    for (let i = 0; i < r.session.sys.state.n; i++) {
      const id = r.session.sys.state.id[i];
      // atoms outside the group are 0 even where they have bonds (atoms 4 and 5)
      expect(vals[i]).toBe(id <= 3 ? want.get(id) : 0);
    }
    // measured with native LAMMPS (black box): group id 1:3 gives 1, 3, 2 — the two
    // bonds of atom 2 that reach atoms outside the group count as well
    expect([vals[0], vals[1], vals[2]]).toEqual([1, 3, 2]);
  });

  it('sums to twice the number of bonds for group all', async () => {
    const r = await run(`${MOLECULAR}compute nb all nbond/atom
compute s all reduce sum c_nb
thermo_style custom step c_s
thermo_modify format float %.15g norm no
run 0
`);
    expect(r.error).toBeUndefined();
    const n = r.session.sys.state.topo.bonds.n;
    expect(n).toBe(72);
    let sum = 0;
    for (const v of r.session.sys.compute('nb').peratomValues()) sum += v;
    expect(sum).toBe(2 * n);
    expect(r.rows[0].c_s).toBe(2 * n);
  });

  it('bond/type counts only the listed type', async () => {
    const r = await run(`${MOLECULAR}group one id 2
compute nb1 all nbond/atom bond/type 1
compute nb2 all nbond/atom bond/type 2
compute v1 one nbond/atom bond/type 1
compute v2 one nbond/atom bond/type 2
compute s1 all reduce sum c_nb1
compute s2 all reduce sum c_nb2
compute r1 all reduce sum c_v1
compute r2 all reduce sum c_v2
thermo_style custom step c_s1 c_s2 c_r1 c_r2
thermo_modify format float %.15g norm no
run 0
`);
    expect(r.error).toBeUndefined();
    const bonds = r.session.sys.state.topo.bonds;
    const tally = (type: number) => {
      let n = 0;
      for (let e = 0; e < bonds.n; e++) if (bonds.type[e] === type) n += 2;
      return n;
    };
    const row = r.rows[0];
    expect(row.c_s1).toBe(tally(1));
    expect(row.c_s2).toBe(tally(2));
    // atom 2 of the data file has the type-1 bonds (1,2) and (2,3) and the type-2 bond (2,5)
    expect(row.c_r1).toBe(2);
    expect(row.c_r2).toBe(1);
    // a type no bond has is accepted by native LAMMPS and gives a zero vector
    for (const btype of ['3', '-1']) {
      const empty = await run(`${MOLECULAR}compute nb all nbond/atom bond/type ${btype}\nrun 0\n`);
      expect(empty.error).toBeUndefined();
      for (const v of empty.session.sys.compute('nb').peratomValues()) expect(v).toBe(0);
    }
  });

  it('does not count bonds that are off (type < 1)', async () => {
    const script = (setup: string) => `${MOLECULAR}${setup}compute nb all nbond/atom
compute nb2 all nbond/atom bond/type 2
compute s all reduce sum c_nb
compute s2 all reduce sum c_nb2
thermo_style custom step c_s c_s2
thermo_modify format float %.15g norm no
run 0
`;
    const base = await run(script(''));
    expect(base.error).toBeUndefined();
    // `set ... bond` is the engine-only way to reach an off bond here (native LAMMPS
    // rejects a non-positive type in set; delete_bonds turns types negative there)
    const off = await run(script('group frag id 3:4\nset group frag bond -2\n'));
    expect(off.error).toBeUndefined();
    // measured with native LAMMPS (black box): after delete_bonds frag bond 2 on the
    // atoms of the single type-2 bond (3,4) the tally falls by two, once for each of
    // its atoms (native only touches bonds of the named type, `set` rewrites all)
    expect(base.rows[0].c_s).toBe(144);
    expect(base.rows[0].c_s2).toBe(72);
    expect(off.rows[0].c_s).toBe(142);
    expect(off.rows[0].c_s2).toBe(70);
    // atom 4 had only that bond, so it now counts none
    expect(off.session.sys.compute('nb').peratomValues()[3]).toBe(0);
  });

  it('rejects an unknown keyword and the bond/type keyword without a value', async () => {
    const a = await run(`${MOLECULAR}compute nb all nbond/atom badkw 1\n`);
    expect(a.error).toMatch(/Unknown compute nbond\/atom command badkw/);
    const b = await run(`${MOLECULAR}compute nb all nbond/atom bond/type\n`);
    expect(b.error).toMatch(/Illegal compute nbond\/atom bond\/type command: missing argument/);
  });
});

describe('compute count/type', () => {
  /** The vector of `compute ct <group> count/type <mode>` after `run 0`. */
  const counts = async (mode: string, group = 'all', setup = '') => {
    const r = await run(`${MOLECULAR}${setup}compute ct ${group} count/type ${mode}\nrun 0\n`);
    expect(r.error).toBeUndefined();
    return Array.from(r.session.sys.compute('ct').vectorValues());
  };

  it('atom: one count per atom type, only atoms of the group', async () => {
    expect(await counts('atom')).toEqual([36, 36, 18]);
    expect(await counts('atom', 'half', 'group half molecule 1:9\n')).toEqual([18, 18, 9]);
    // a group cut through the chains counts only its own atoms
    expect(await counts('atom', 'frag', 'group frag id 1:3\n')).toEqual([1, 2, 0]);
  });

  it('bond: only bonds with both atoms in the group', async () => {
    expect(await counts('bond')).toEqual([36, 36]);
    expect(await counts('bond', 'half', 'group half molecule 1:9\n')).toEqual([18, 18]);
    // the type-2 bonds (3,4) and (2,5) leave the group, the two type-1 bonds do not
    expect(await counts('bond', 'frag', 'group frag id 1:3\n')).toEqual([2, 0]);
  });

  it('angle needs all three atoms, dihedral and improper all four', async () => {
    expect(await counts('angle')).toEqual([54, 18]);
    expect(await counts('angle', 'half', 'group half molecule 1:9\n')).toEqual([27, 9]);
    expect(await counts('angle', 'frag', 'group frag id 1:3\n')).toEqual([1, 0]);
    expect(await counts('dihedral')).toEqual([36]);
    expect(await counts('dihedral', 'half', 'group half molecule 1:9\n')).toEqual([18]);
    expect(await counts('dihedral', 'frag', 'group frag id 1:3\n')).toEqual([0]);
    expect(await counts('improper')).toEqual([18]);
    expect(await counts('improper', 'half', 'group half molecule 1:9\n')).toEqual([9]);
    expect(await counts('improper', 'frag', 'group frag id 1:3\n')).toEqual([0]);
  });

  it('vector length is the number of types and the values are intensive', async () => {
    const r = await run(`${MOLECULAR}compute ct all count/type atom
compute cb all count/type bond
compute ca all count/type angle
compute cd all count/type dihedral
compute ci all count/type improper
thermo_style custom step c_ct[1] c_ct[2] c_ct[3] c_cb[1] c_cb[2] c_ca[1] c_ca[2] c_cd[1] c_ci[1]
thermo_modify format float %.15g
run 0
`);
    expect(r.error).toBeUndefined();
    const sys = r.session.sys;
    expect(sys.compute('ct').sizeVector).toBe(3);
    expect(sys.compute('cb').sizeVector).toBe(2);
    expect(sys.compute('ca').sizeVector).toBe(2);
    expect(sys.compute('cd').sizeVector).toBe(1);
    expect(sys.compute('ci').sizeVector).toBe(1);
    for (const id of ['ct', 'cb', 'ca', 'cd', 'ci']) expect(sys.compute(id).extvector).toBe(0);
    // intensive: the counts are the full counts, not divided by the atom count
    // (lj units default to thermo_modify norm yes)
    expect(r.rows[0]['c_ct[1]']).toBe(36);
    expect(r.rows[0]['c_cb[1]']).toBe(36);
    expect(r.rows[0]['c_ca[1]']).toBe(54);
    expect(r.rows[0]['c_cd[1]']).toBe(36);
    expect(r.rows[0]['c_ci[1]']).toBe(18);
  });

  it('counts a turned-off bond in the slot of its own type, and type 0 in the scalar', async () => {
    // docs: "This command includes the turned-off bonds (angles, etc) in the count for
    // each type"; measured with native LAMMPS (black box) on delete_bonds: after
    // turning off two type-1 bonds of one molecule the type-1 count is still 36 of 36.
    const off = await run(`${MOLECULAR}group frag id 1:3\nset group frag bond -1\ncompute cb all count/type bond\nrun 0\n`);
    expect(off.error).toBeUndefined();
    expect(Array.from(off.session.sys.compute('cb').vectorValues())).toEqual([36, 36]);

    // docs: "If the mode is bond this compute also calculates a global scalar which is
    // the number of broken bonds with type = 0" and "the group setting is ignored for
    // broken bonds; all broken bonds in the system are counted"
    const broken = await run(`${MOLECULAR}group frag id 1:3\nset group frag bond 0
group idone id 1
compute cb all count/type bond
compute cbone idone count/type bond
run 0
`);
    expect(broken.error).toBeUndefined();
    const cb = broken.session.sys.compute('cb');
    expect(cb.scalarFlag).toBe(true);
    expect(cb.scalarValue()).toBe(2); // the bonds (1,2) and (2,3) are broken
    expect(Array.from(cb.vectorValues())).toEqual([34, 36]); // not in the vector
    expect(broken.session.sys.compute('cbone').scalarValue()).toBe(2); // group ignored
    // only mode bond has the scalar
    const atoms = await run(`${MOLECULAR}compute ca all count/type atom\nrun 0\n`);
    expect(atoms.session.sys.compute('ca').scalarFlag).toBe(false);
  });

  it('rejects a missing, extra or unknown mode, and a mode with no types defined', async () => {
    expect((await run(`${MOLECULAR}compute ct all count/type\n`)).error).toMatch('Incorrect number of args for compute count/type command');
    expect((await run(`${MOLECULAR}compute ct all count/type atom extra\n`)).error).toMatch('Incorrect number of args for compute count/type command');
    expect((await run(`${MOLECULAR}compute ct all count/type bogus\n`)).error).toMatch('Invalid compute count/type keyword bogus');
    const atomic = `units lj
atom_style atomic
lattice sc 1.0
region box block 0 2 0 1 0 1
create_box 2 box
create_atoms 1 box
mass * 1.0
`;
    const none = await run(`${atomic}compute cb all count/type bond\nrun 0\n`);
    expect(none.error).toMatch('Compute cb (count/type) bond command with no bonds defined');
    // mode atom works on a system without any topology
    const ok = await run(`${atomic}compute ct all count/type atom\nrun 0\n`);
    expect(ok.error).toBeUndefined();
    expect(Array.from(ok.session.sys.compute('ct').vectorValues())).toEqual([2, 0]);
  });
});

describe('compute erotate/sphere/atom', () => {
  it('is 1/2 I omega^2 with I = 2/5 m r^2 and sums to erotate/sphere', async () => {
    const r = await run(`${SPHERE('units lj')}compute era all erotate/sphere/atom
compute er all erotate/sphere
compute s all reduce sum c_era
thermo_style custom step c_s c_er
thermo_modify format float %.15g norm no
run 0
`);
    expect(r.error).toBeUndefined();
    const c = r.session.sys.compute('era');
    expect(c.style).toBe('erotate/sphere/atom');
    expect(c.peratomFlag).toBe(true);
    expect(c.sizePeratomCols).toBe(0);
    const vals = c.peratomValues();
    let sum = 0;
    for (let i = 0; i < 3; i++) {
      const w = omegaOf(i + 1);
      const want = 0.5 * 0.4 * 3.0 * (w[0] ** 2 + w[1] ** 2 + w[2] ** 2); // m = 3, r = 1, mvv2e = 1
      expect(vals[i]).toBeCloseTo(want, 12);
      sum += want;
    }
    expect(r.rows[0].c_s).toBeCloseTo(sum, 8);
    expect(r.rows[0].c_er).toBeCloseTo(sum, 8);
  });

  it('is in energy units: the same system in real units is scaled by mvv2e', async () => {
    const lj = await peratom(`${SPHERE('units lj')}`, 'erotate/sphere/atom');
    const real = await peratom(`${SPHERE('units real')}`, 'erotate/sphere/atom');
    const k = UNIT_SYSTEMS.real.mvv2e;
    for (let i = 0; i < 3; i++) expect(real[i] / lj[i]).toBeCloseTo(k, 12);
  });

  it('is 0 for a point particle (radius 0) and for atoms outside the group', async () => {
    const r = await run(`${SPHERE('units lj')}group pt id 1\nset group pt diameter 0.0\ngroup two id 2:3
compute era all erotate/sphere/atom
compute era2 two erotate/sphere/atom
run 0
`);
    expect(r.error).toBeUndefined();
    const all = r.session.sys.compute('era').peratomValues();
    const sub = r.session.sys.compute('era2').peratomValues();
    expect(all[0]).toBe(0); // radius 0: a point particle does not rotate
    expect(all[1]).toBeGreaterThan(0);
    expect(sub[0]).toBe(0); // outside the group
    expect(sub[1]).toBeCloseTo(all[1], 12);
    expect(sub[2]).toBeCloseTo(all[2], 12);
  });

  it('rejects arguments and needs atom_style sphere', async () => {
    expect((await run(`${SPHERE('units lj')}compute era all erotate/sphere/atom extra\n`)).error)
      .toMatch('compute erotate/sphere/atom takes no arguments');
    const atomic = `units lj
atom_style atomic
lattice sc 1.0
region box block 0 2 0 1 0 1
create_box 1 box
create_atoms 1 box
mass * 1.0
`;
    expect((await run(`${atomic}compute era all erotate/sphere/atom\n`)).error)
      .toMatch('Compute erotate/sphere/atom requires atom attribute omega');
  });
});