import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * Unit tests for the class 2 Coulomb styles and the SPICA family (lj/spica, lj/spica/coul/long, the lj/sdk
 * originals) and angle_style spica / sdk. The reference values are measured with native LAMMPS (black box) on
 * the same one-dimer or three-atom inputs; the documented forms are checked against them analytically.
 *
 * docs.lammps.org/pair_class2.html: "The *lj/class2* styles compute a 6/9 Lennard-Jones potential given by"
 * docs.lammps.org/pair_spica.html: "The *lj/spica* styles compute a 9/6, 12/4, 12/5, or 12/6 Lennard-Jones potential,"
 * docs.lammps.org/angle_spica.html: "with the *repulsive* part of the non-bonded *lj/spica* pair style between the atoms 1 and 3."
 */

/** Runs a script; returns the thermo rows of the last run and the error message, if any. */
const runScript = async (text: string, files: Record<string, string> = {}) => {
  const events: { kind: string; row?: Record<string, number>; message?: string }[] = [];
  const session = new Session({ emit: (e: never) => events.push(e), writeFile: () => {} } as never, undefined, 0, 1);
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  let error = '';
  try {
    await session.execute(text);
  } catch (e) {
    error = String(e instanceof Error ? e.message : e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && err.message) error = err.message;
  const rows = events.filter((e) => e.kind === 'thermo').map((e) => e.row as Record<string, number>);
  return { rows, error };
};

/** A two-atom dimer in a 40 A box: type 1 at x = 10, type 2 at x = 10 + r. */
const dimer = (pair: string, r: number, types = [1, 2], charges = [0, 0], extra = '') => `units real
atom_style charge
boundary p p p
region box block 0 40 0 40 0 40
create_box ${Math.max(...types)} box
create_atoms ${types[0]} single 10 10 10
create_atoms ${types[1]} single ${10 + r} 10 10
mass * 1.0
${charges[0] || charges[1] ? `set type ${types[0]} charge ${charges[0]}\nset type ${types[1]} charge ${charges[1]}\n` : ''}${pair}
${extra}thermo_style custom step evdwl ecoul
thermo_modify format float %.15g
run 0
`;

const evdwlOf = async (pair: string, r: number, types?: number[], charges?: number[], extra?: string) => {
  const out = await runScript(dimer(pair, r, types, charges, extra));
  expect(out.error).toBe('');
  return out.rows[0];
};

describe('pair_style lj/class2/coul/cut and lj/class2/coul/long', () => {
  it('the 6/9 form with sixthpower mixing, measured with native LAMMPS (black box)', async () => {
    // lj/class2/coul/cut 9.0 with 1-1 (eps 1, sigma 1) and 2-2 (eps 0.5, sigma 1.2); the 1-2 pair is mixed
    const pair = 'pair_style lj/class2/coul/cut 9.0\npair_coeff 1 1 1.0 1.0\npair_coeff 2 2 0.5 1.2\npair_modify shift no';
    // measured: r = 1.1 gives evdwl -0.6060478520396 for the mixed pair (types 1 and 2)
    const row = await evdwlOf(pair, 1.1);
    expect(row.evdwl).toBeCloseTo(-0.6060478520396, 11);
    expect(row.ecoul).toBe(0);
  });

  it('the plain 6/9 energy is 2 (sigma/r)^9 - 3 (sigma/r)^6 for one set pair', async () => {
    const pair = 'pair_style lj/class2/coul/cut 9.0\npair_coeff 1 1 1.0 1.0\npair_modify shift no';
    const r = 1.5;
    const row = await evdwlOf(pair, r, [1, 1]);
    expect(row.evdwl).toBeCloseTo(2 * (1 / r) ** 9 - 3 * (1 / r) ** 6, 12);
  });

  it('the Coulomb term is the bare C q_i q_j / r inside the Coulomb cutoff (measured: -83.0159275 at r = 1, C = 332.06371)', async () => {
    const pair = 'pair_style lj/class2/coul/cut 9.0 8.0\npair_coeff 1 1 1.0 1.0\npair_coeff 2 2 0.5 1.2\npair_modify shift no';
    const row = await evdwlOf(pair, 1.0, [1, 2], [0.5, -0.5]);
    expect(row.ecoul).toBeCloseTo((332.06371 * 0.5 * -0.5) / 1.0, 6);
    // beyond the Coulomb cutoff (8.0) there is no Coulomb energy
    const far = await evdwlOf(pair, 8.5, [1, 2], [0.5, -0.5]);
    expect(far.ecoul).toBe(0);
  });

  it('lj/class2/coul/long refuses two cutoffs in pair_coeff (measured: Incorrect args for pair coefficients)', async () => {
    const out = await runScript(dimer('pair_style lj/class2/coul/long 9.0\npair_coeff 1 1 1.0 1.0 9.0 8.0\nkspace_style ewald 1e-6\nkspace_modify gewald 0.3', 1.5));
    expect(out.error).toContain('Incorrect args for pair coefficients');
  });
});

describe('pair_style lj/spica family', () => {
  it('lj9_6 energy at r = 1.2 (measured with native LAMMPS: -0.920066410949035, sigma 1.1, eps 1)', async () => {
    const pair = 'pair_style lj/spica 2.8\npair_coeff 1 1 lj9_6 1.0 1.1 2.8\npair_modify shift no';
    const row = await evdwlOf(pair, 1.2, [1, 1]);
    expect(row.evdwl).toBeCloseTo(-0.920066410949035, 11);
  });

  it('each cg type has its minimum at -epsilon at r = sigma (n/m)^(1/(n-m)), with the documented prefactors', async () => {
    // measured with native LAMMPS (black box): lj12_6 at r = 1.2 is -0.965186265806962
    const pair = 'pair_style lj/spica 2.8\npair_coeff 1 1 lj12_6 1.0 1.1 2.8\npair_modify shift no';
    expect((await evdwlOf(pair, 1.2, [1, 1])).evdwl).toBeCloseTo(-0.965186265806962, 11);
    for (const cg of ['lj9_6', 'lj12_4', 'lj12_5', 'lj12_6']) {
      const n = cg === 'lj9_6' ? 9 : 12, m = Number(cg.split('_')[1]);
      const rmin = 1.1 * (n / m) ** (1 / (n - m));
      const e = (await evdwlOf(`pair_style lj/spica 2.8\npair_coeff 1 1 ${cg} 1.0 1.1 2.8\npair_modify shift no`, rmin, [1, 1])).evdwl;
      expect(e, cg).toBeCloseTo(-1, 9);
    }
  });

  it('lj/sdk gives the same energies as lj/spica (measured: lj9_6 shift yes at r = 1.0 is 3.98142043532433)', async () => {
    const spica = 'pair_style lj/spica 2.8\npair_coeff 1 1 lj9_6 1.0 1.1 2.8\npair_modify shift yes';
    const sdk = 'pair_style lj/sdk 2.8\npair_coeff 1 1 lj9_6 1.0 1.1 2.8\npair_modify shift yes';
    expect((await evdwlOf(spica, 1.0, [1, 1])).evdwl).toBeCloseTo(3.98142043532433, 11);
    expect((await evdwlOf(sdk, 1.0, [1, 1])).evdwl).toBeCloseTo(3.98142043532433, 11);
  });

  it('refuses mixing of unset pairs (measured: No mixing support for lj/spica. Coefficients for all pairs need to be set explicitly.)', async () => {
    const out = await runScript(dimer('pair_style lj/spica 2.8\npair_coeff 1 1 lj9_6 1.0 1.1 2.8\npair_coeff 2 2 lj9_6 1.0 1.1 2.8', 1.5));
    expect(out.error).toContain('No mixing support for lj/spica. Coefficients for all pairs need to be set explicitly.');
  });

  it('rejects an unknown cg_type (measured: Cannot parse LJ type flag.)', async () => {
    const out = await runScript(dimer('pair_style lj/spica 2.8\npair_coeff 1 1 lj8_6 1.0 1.1 2.8', 1.5));
    expect(out.error).toContain('Cannot parse LJ type flag.');
  });

  it('refuses pair_modify tail yes (measured: Tail flag not supported by lj/spica pair style)', async () => {
    // one atom type: with an unset type 2 native stops first with All pair coeffs are not set. Status:
    const out = await runScript(dimer('pair_style lj/spica 2.8\npair_coeff 1 1 lj9_6 1.0 1.1 2.8\npair_modify tail yes', 1.5, [1, 1]));
    expect(out.error).toContain('Tail flag not supported by lj/spica pair style');
  });

  it('lj/spica/coul/long takes one cutoff per pair (measured: Incorrect args for pair coefficients)', async () => {
    const out = await runScript(dimer('pair_style lj/spica/coul/long 9.0\npair_coeff 1 1 lj9_6 1.0 1.1 9.0 8.0\nkspace_style ewald 1e-6\nkspace_modify gewald 0.3', 1.5));
    expect(out.error).toContain('Incorrect args for pair coefficients');
  });

  it('the msm variants (engine-gpl/msm): kspace msm, refused with ewald and with pressure/scalar yes (measured)', async () => {
    // Measured with native LAMMPS: q = +-0.5 at r = 2 (one atom type), pair_coeff 1 1 lj9_6 1.0 1.1 9.0, kspace_style msm 1e-4 with
    // pressure/scalar no cutoff/adjust no order 6: evdwl -0.155758099776856, ecoul -22.2977238633228, elong
    // -19.2194800416347. With kspace_style ewald: KSpace style ewald is incompatible with Pair style lj/spica/coul/msm.
    // With the default pressure/scalar yes: Must use 'kspace_modify pressure/scalar no' with Pair style.
    const pair = (name: string) => `pair_style ${name} 9.0\npair_coeff * * lj9_6 1.0 1.1 9.0`;
    const msm = 'kspace_style msm 1e-4\nkspace_modify pressure/scalar no cutoff/adjust no order 6\n';
    for (const name of ['lj/spica/coul/msm', 'lj/sdk/coul/msm']) {
      const row = await evdwlOf(pair(name), 2.0, [1, 2], [0.5, -0.5], msm);
      expect(row.evdwl).toBeCloseTo(-0.155758099776856, 12);
      expect(row.ecoul).toBeCloseTo(-22.2977238633228, 10);
    }
    const ew = await runScript(dimer(pair('lj/spica/coul/msm'), 2.0, [1, 2], [0.5, -0.5], 'kspace_style ewald 1e-6\n'));
    expect(ew.error).toContain('KSpace style ewald is incompatible with Pair style lj/spica/coul/msm');
    const sc = await runScript(dimer(pair('lj/spica/coul/msm'), 2.0, [1, 2], [0.5, -0.5], 'kspace_style msm 1e-4\n'));
    expect(sc.error).toContain("Must use 'kspace_modify pressure/scalar no' with Pair style");
  });

  it('lj/spica/coul/long: the Coulomb part with ewald (measured: ecoul 16.4431231190563 at r = 2 with table 0, q = 0.5 each)', async () => {
    const pair = 'pair_style lj/spica/coul/long 9.0\npair_coeff 1 1 lj9_6 1.0 1.1 9.0\npair_modify table 0';
    const out = await runScript(dimer(pair, 2.0, [1, 1], [0.5, 0.5], 'kspace_style ewald 1e-6\nkspace_modify gewald 0.3\n'));
    expect(out.error).toBe('');
    expect(out.rows[0].ecoul).toBeCloseTo(16.4431231190563, 9);
    expect(out.rows[0].evdwl).toBeCloseTo(-0.155758099776856, 11);
  });
});

describe('angle_style spica and sdk: the 1-3 repulsion', () => {
  // three atoms on a straight line, 1-3 distance r13 = 2 d; angle type 1 has K = 0 (pure repulsion)
  const triple = (d: number) => `triple ${d}

3 atoms
2 bonds
1 angles
1 atom types
1 bond types
1 angle types

0 40 xlo xhi
0 40 ylo yhi
0 40 zlo zhi

Masses

1 1.0

Atoms # angle

1 1 1 10.0 10.0 10.0
2 1 1 ${10 + d} 10.0 10.0
3 1 1 ${10 + 2 * d} 10.0 10.0

Bonds

1 1 1 2
2 1 2 3

Angles

1 1 1 2 3
`;
  const angleRun = async (style: string, r13: number) => {
    const files = { 'triple.data': triple(r13 / 2) };
    const text = `units real
atom_style angle
read_data triple.data
pair_style lj/spica 2.8
pair_coeff 1 1 lj12_6 1.0 1.1 2.8
bond_style harmonic
bond_coeff 1 0.0 1.0
angle_style ${style}
angle_coeff 1 0.0 180.0
thermo_style custom step eangle evdwl
thermo_modify format float %.15g
run 0
`;
    const out = await runScript(text, files);
    expect(out.error).toBe('');
    return out.rows[0].eangle;
  };

  it('E = LJ(r13) + epsilon below the LJ minimum (measured: 6.46746950688401 at r13 = 1.0)', async () => {
    expect(await angleRun('spica', 1.0)).toBeCloseTo(6.46746950688401, 9);
    expect(await angleRun('sdk', 1.0)).toBeCloseTo(6.46746950688401, 9);
  });

  it('is zero at and beyond the LJ minimum, and continuous at it (measured: 0.0348137341930385 at r13 = 1.2)', async () => {
    expect(await angleRun('spica', 1.2)).toBeCloseTo(0.0348137341930385, 9);
    expect(await angleRun('spica', 1.2355)).toBe(0);
    expect(await angleRun('spica', 2.0)).toBe(0);
  });
});
