import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Spatial stress profiles (src/engine/compute/stressprof.ts): parsing, the refusals of the unimplemented
 * contributions, and the sum rules. Native parity itself is in the w47sp_* oracle cases.
 */

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  await session.execute(text);
  return session;
};

/** Two atoms 1.5 apart along x crossing the bin edge at x = 5 (the measured case in the header of stressprof.ts). */
const TWO = `
units lj
atom_style atomic
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 4.2 5.0 5.0
create_atoms 1 single 5.7 5.0 5.0
mass 1 1.0
set atom 1 vx 0.3
set atom 2 vx -0.2
set atom 2 vy 0.1
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
`;

const PERIODIC_LJ = `
units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
variable vxa atom 0.7*sin(2.3*y+0.4*z)+0.3*cos(1.1*x)
variable vya atom 0.6*cos(1.7*z-0.9*x)
variable vza atom 0.5*sin(1.3*x+0.8*y)
velocity all set v_vxa v_vya v_vza units box
variable dxa atom 0.05*sin(2.3*y+0.4*z)
variable dya atom 0.04*cos(1.7*z)
variable dza atom 0.03*sin(1.3*z+0.5)
displace_atoms all move v_dxa v_dya v_dza units box
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
`;

describe('compute stress/cartesian: bins, columns and the measured rules', () => {
  it('two atoms straddling a bin edge reproduce the native per-bin values', async () => {
    const s = await runScript(`${TWO}
compute p all stress/cartesian x 1.0 NULL 0 ke pair
run 0`);
    const c = s.sys.compute('p');
    expect(c.sizeArrayCols).toBe(8);
    expect(c.sizeArrayRows).toBe(10);
    const a = c.arrayValues();
    const row = (k: number) => Array.from(a.slice(8 * k, 8 * k + 8));
    // bin centres 0.5 ... 9.5; rows 5 and 6 (1-based) hold the atoms: density 0.01, kinetic 0.0009 / 0.0004 and 0.0001
    expect(row(4)[0]).toBeCloseTo(4.5, 12);
    expect(row(4)[1]).toBeCloseTo(0.01, 12);
    expect(row(4)[2]).toBeCloseTo(0.0009, 12);
    expect(row(5)[2]).toBeCloseTo(0.0004, 12);
    expect(row(5)[3]).toBeCloseTo(0.0001, 12);
    // native (black box): virial -0.00926423 and -0.0081062 in the two bins (the pair line split 0.8/1.5 and 0.7/1.5)
    expect(row(4)[5]).toBeCloseTo(-0.00926423, 7);
    expect(row(5)[5]).toBeCloseTo(-0.0081062, 7);
  });

  it('two dimensions give 9 columns and the first dimension varies fastest', async () => {
    const s = await runScript(`${TWO}
compute p all stress/cartesian x 1.0 y 2.0 ke pair
run 0`);
    const c = s.sys.compute('p');
    expect(c.sizeArrayCols).toBe(9);
    expect(c.sizeArrayRows).toBe(10 * 5);
    const a = c.arrayValues();
    // row 1 is (x 0.5, y 1.0): the first dimension is the fastest index
    expect(a[0]).toBeCloseTo(0.5, 12);
    expect(a[9]).toBeCloseTo(1.5, 12);
    expect(a[10 * 9]).toBeCloseTo(0.5, 12);
  });

  it('the bin count is floor(L / b) and the width is then L / nbins (native: box -5..5, b = 3.0 gives 3 bins)', async () => {
    const s = await runScript(`units lj
atom_style atomic
region box block -5 5 -5 5 -5 5
create_box 1 box
create_atoms 1 single -4.2 0.0 0.0
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
compute p all stress/cartesian x 3.0 NULL 0
run 0`);
    const c = s.sys.compute('p');
    expect(c.sizeArrayRows).toBe(3);
    const a = c.arrayValues();
    // centres -3.333, 0, 3.333 (width 10 / 3)
    expect(a[0]).toBeCloseTo(-10 / 3, 12);
    expect(a[8]).toBeCloseTo(0, 12);
    expect(a[16]).toBeCloseTo(10 / 3, 12);
  });

  it('the ke keyword gives kinetic and density only; the pair keyword gives the virial only', async () => {
    const s = await runScript(`${TWO}
compute kk all stress/cartesian x 1.0 NULL 0 ke
compute pp all stress/cartesian x 1.0 NULL 0 pair
run 0`);
    const kk = s.sys.compute('kk').arrayValues();
    const pp = s.sys.compute('pp').arrayValues();
    // columns: 0 coord, 1 density, 2-4 kinetic, 5-7 virial
    expect(kk[8 * 4 + 5]).toBe(0);
    expect(kk[8 * 4 + 1]).toBeCloseTo(0.01, 12);
    expect(pp[8 * 4 + 1]).toBe(0);
    expect(pp[8 * 4 + 2]).toBe(0);
    expect(pp[8 * 4 + 5]).toBeCloseTo(-0.00926423, 7);
  });

  it('sums over bins of the profile times the bin volume give the total virial of a periodic lj box', async () => {
    const s = await runScript(`${PERIODIC_LJ}
compute p all stress/cartesian x 0.7 NULL 0 pair
compute sa all stress/atom NULL pair
run 0`);
    const c = s.sys.compute('p');
    const a = c.arrayValues();
    const L = s.sys.state.box.hi[0] - s.sys.state.box.lo[0];
    const n = c.sizeArrayRows;
    const w = L / n;
    const vbin = w * (s.sys.state.box.hi[1] - s.sys.state.box.lo[1]) * (s.sys.state.box.hi[2] - s.sys.state.box.lo[2]);
    let total = 0;
    for (let r = 0; r < n; r++) total += a[8 * r + 5] * vbin;
    // stress/atom with the pair keyword gives -W per atom, so the sum over atoms is minus the total virial
    const sa = s.sys.compute('sa').peratomValues();
    let w0 = 0;
    for (let i = 0; i < s.sys.state.n; i++) w0 += sa[6 * i];
    expect(total).toBeCloseTo(-w0, 8);
  });
});

describe('compute stress/mop and stress/mop/profile', () => {
  it('a profile row equals the stress/mop vector at the same plane', async () => {
    // the box starts at 0 (region 0 3), so the profile planes are 0, 1, 2, 3, 4, 5 (the box is 5.04 long)
    const s = await runScript(`${PERIODIC_LJ}
compute mv all stress/mop x 2.0 kin conf
compute mp all stress/mop/profile x 0.0 1.0 kin conf
run 0`);
    const v = Array.from(s.sys.compute('mv').vectorValues());
    const mp = s.sys.compute('mp');
    const cols = mp.sizeArrayCols;
    const p = mp.arrayValues();
    expect(cols).toBe(7);
    let row = -1;
    for (let r = 0; r < mp.sizeArrayRows; r++) if (Math.abs(p[cols * r] - 2.0) < 1e-12) row = r;
    expect(row).toBeGreaterThanOrEqual(0);
    // mv = [kin xyz, conf xyz] and the profile row holds [kin xyz, conf xyz] after the position column
    const fromRow = Array.from(p.slice(cols * row + 1, cols * row + 7));
    for (let k = 0; k < 6; k++) expect(fromRow[k]).toBeCloseTo(v[k], 12);
  });

  it('the planes are origin + k delta inside the box (origin 0, delta 2.5, box -5..5: five planes)', async () => {
    const s = await runScript(`units lj
atom_style atomic
region box block -5 5 0 10 0 10
create_box 1 box
create_atoms 1 single -0.2 5.0 5.0
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
compute mp all stress/mop/profile x 0.0 2.5 kin
run 0`);
    const c = s.sys.compute('mp');
    expect(c.sizeArrayRows).toBe(5);
    expect(Array.from(c.arrayValues().filter((_, i) => i % c.sizeArrayCols === 0))).toEqual([-5, -2.5, 0, 2.5, 5]);
  });

  it('the kinetic part of an atom crossing the plane is m v_b s / (A dt) (measured: 2.0 for v = 1, A = 100, dt = 0.005)', async () => {
    const s = await runScript(`units lj
atom_style atomic
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 4.998 5.0 5.0
create_atoms 1 single 8.0 5.0 5.0
mass 1 1.0
set atom 1 vx 1.0
pair_style lj/cut 1.0
pair_coeff * * 1.0 1.0
fix nv all nve
compute m1 all stress/mop x 5.0 kin
run 1`);
    expect(s.sys.compute('m1').vectorValues()[0]).toBeCloseTo(2.0, 9);
  });
});

describe('units of the method-of-planes kinetic part', () => {
  it('units real scales the kinetic part by nktv2p * mvv2e (measured natively: 6.5553e+06 for m = 2, v = 2, A = 100, dt = 1)', async () => {
    const s = await runScript(`units real
atom_style atomic
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 4.998 5.0 5.0
create_atoms 1 single 8.0 5.0 5.0
mass 1 2.0
set atom 1 vx 2.0
set atom 1 vy 0.3
pair_style lj/cut 1.0
pair_coeff * * 0.1 1.0
fix nv all nve
compute m1 all stress/mop x 5.0 kin
run 1`);
    const v = s.sys.compute('m1').vectorValues();
    expect(v[0]).toBeCloseTo(6.5553e6, -2);
    expect(v[1]).toBeCloseTo(983295, -1);
  });
});

describe('the setup row', () => {
  it('stress/mop evaluated at the setup of a run is an EngineError (native setup row not reproduced)', async () => {
    await expect(runScript(`${PERIODIC_LJ}
compute m1 all stress/mop x 2.5 kin conf
thermo_style custom step c_m1[1]
run 0`)).rejects.toThrow(/setup row/);
    await expect(runScript(`${PERIODIC_LJ}
compute mp all stress/mop/profile x 0.0 1.0 kin
thermo_style custom step c_mp[1][2]
run 1`)).rejects.toThrow(/setup row/);
  });
});

describe('refusals', () => {
  it('stress/cylinder and stress/spherical are EngineErrors naming the style', async () => {
    await expect(runScript(`${TWO}\ncompute c all stress/cylinder 0 10 15 0.25\nrun 0`)).rejects.toThrow(/stress\/cylinder/);
    await expect(runScript(`${TWO}\ncompute c all stress/spherical 0 0 0 0.25 10\nrun 0`)).rejects.toThrow(/stress\/spherical/);
  });

  it('the bond keyword is refused by stress/cartesian and stress/mop', async () => {
    await expect(runScript(`${TWO}\ncompute c all stress/cartesian x 1.0 NULL 0 bond\nrun 0`)).rejects.toThrow(/bond/);
    await expect(runScript(`${TWO}\ncompute c all stress/mop x 5.0 bond\nrun 0`)).rejects.toThrow(/bond/);
    await expect(runScript(`${TWO}\ncompute c all stress/mop/profile x 0.0 1.0 angle\nrun 0`)).rejects.toThrow(/angle/);
  });

  it('unknown keywords and dimensions are rejected', async () => {
    await expect(runScript(`${TWO}\ncompute c all stress/cartesian q 1.0 NULL 0\nrun 0`)).rejects.toThrow(/dimension/);
    await expect(runScript(`${TWO}\ncompute c all stress/cartesian x 1.0 NULL 0 virial\nrun 0`)).rejects.toThrow(/keyword/);
    await expect(runScript(`${TWO}\ncompute c all stress/mop x 5.0 frob\nrun 0`)).rejects.toThrow(/keyword/);
  });

  it('a bin width larger than the box is rejected', async () => {
    await expect(runScript(`${TWO}\ncompute c all stress/cartesian x 20.0 NULL 0\nrun 0`)).rejects.toThrow(/larger than the box/);
  });

  it('the default keyword set of stress/cartesian is refused on a system with bonds', async () => {
    await expect(runScript(`units lj
atom_style bond
region box block 0 10 0 10 0 10
create_box 1 box bond/types 1 extra/bond/per/atom 1
create_atoms 1 single 4.2 5.0 5.0
create_atoms 1 single 5.7 5.0 5.0
mass 1 1.0
create_bonds single/bond 1 1 2
bond_style harmonic
bond_coeff 1 100.0 1.5
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
compute c all stress/cartesian x 1.0 NULL 0
run 0`)).rejects.toThrow(/default keywords include bond/);
  });
});
