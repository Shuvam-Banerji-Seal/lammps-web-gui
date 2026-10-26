import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { UEFBox, rotationMatrix, zeroRot, virialRot, type M3 } from '../src/engine-gpl/uef/uef_utils';

/*
 * Unit checks of the UEF port (src/engine-gpl/uef): the reduced-cell box keeps the image translations exact under
 * every lattice reduction (checked against the change of basis, column edges as the engine's box), the rotation
 * is orthogonal, and the keywords that are not ported are refused by name. The native parity runs are the oracle
 * cases tests/oracle/w129uef_*.in.
 */

describe('UEF reduced cell (uef_utils)', () => {
  it('keeps the image translations exact under every lattice reduction', () => {
    const box = new UEFBox();
    const Q = zeroRot();
    const Bcur = zeroRot();
    const Bnew = zeroRot();
    const R: M3 = zeroRot();
    const ri = zeroRot();
    // deterministic integer image vectors of 20 atoms
    let seed = 7;
    const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    const imgs = Array.from({ length: 20 }, () => [0, 1, 2].map(() => Math.floor(rnd() * 5) - 2));
    let reductions = 0;
    let worst = 0;
    for (let step = 0; step < 1500; step++) {
      box.stepDeform(-0.0025, -0.0025);
      box.getRot(Q);
      box.getBox(Bcur, 1);
      const old = Q.map((row) => row.slice());
      if (!box.reduce()) continue;
      reductions++;
      box.getRot(R);
      box.getBox(Bnew, 1);
      box.getInverseCob(ri);
      // rotation from the old upper-triangular frame to the new one: R = Q_new Q_old^T
      const T: M3 = zeroRot();
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) for (let c = 0; c < 3; c++) T[a][b] += R[a][c] * old[b][c];
      for (const n of imgs) {
        const v = [0, 1, 2].map((d) => n[0] * Bcur[d][0] + n[1] * Bcur[d][1] + n[2] * Bcur[d][2]);
        const w = [0, 1, 2].map((a) => T[a][0] * v[0] + T[a][1] * v[1] + T[a][2] * v[2]);
        const nn = [0, 1, 2].map((k) => ri[k][0] * n[0] + ri[k][1] * n[1] + ri[k][2] * n[2]);
        const u = [0, 1, 2].map((d) => nn[0] * Bnew[d][0] + nn[1] * Bnew[d][1] + nn[2] * Bnew[d][2]);
        for (let d = 0; d < 3; d++) worst = Math.max(worst, Math.abs(u[d] - w[d]));
      }
    }
    expect(reductions).toBeGreaterThan(10);
    expect(worst).toBeLessThan(1e-12);
  });

  it('rotation_matrix gives an orthogonal Q with Q m = r upper triangular', () => {
    const m: M3 = [[1.2, 0.3, -0.4], [0.1, 0.9, 0.5], [-0.2, 0.7, 1.1]];
    const q = zeroRot();
    const r = zeroRot();
    rotationMatrix(q, r, m);
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
      const qtq = q[0][a] * q[0][b] + q[1][a] * q[1][b] + q[2][a] * q[2][b];
      expect(Math.abs(qtq - (a === b ? 1 : 0))).toBeLessThan(1e-12);
    }
    expect(Math.abs(r[1][0])).toBeLessThan(1e-12);
    expect(Math.abs(r[2][0])).toBeLessThan(1e-12);
    expect(Math.abs(r[2][1])).toBeLessThan(1e-12);
    for (let a = 0; a < 3; a++) expect(r[a][a]).toBeGreaterThan(0);
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
      const qm = q[a][0] * m[0][b] + q[a][1] * m[1][b] + q[a][2] * m[2][b];
      expect(Math.abs(qm - r[a][b])).toBeLessThan(1e-12);
    }
  });

  it('virialRot preserves the trace of the tensor', () => {
    const q = zeroRot();
    rotationMatrix(q, zeroRot(), [[1, 0.2, 0.1], [0, 1, 0.3], [0, 0, 1]]);
    const x = Float64Array.from([1.5, 2.0, 0.7, 0.2, -0.3, 0.4]);
    const tr = x[0] + x[1] + x[2];
    virialRot(x, q);
    expect(x[0] + x[1] + x[2]).toBeCloseTo(tr, 12);
  });
});

const BASE = [
  'units lj', 'atom_style atomic', 'lattice fcc 0.8442', 'region box block 0 2 0 2 0 2', 'create_box 1 box',
  'create_atoms 1 box', 'mass 1 1.0', 'pair_style lj/cut 2.5', 'pair_coeff 1 1 1.0 1.0 2.5', 'change_box all triclinic',
];
const run = (lines: string[]) => {
  const session = new Session({ emit: () => {}, writeFile: () => {} }, undefined, 0, 1);
  return session.execute([...BASE, ...lines].join('\n') + '\n');
};

describe('fix nvt/uef and npt/uef keywords', () => {
  it('refuses the keywords the UEF integrator does not use, by name', async () => {
    await expect(run(['fix 1 all nvt/uef temp 1 1 0.5 erate -0.5 -0.5 fixedpoint 0 0 0', 'run 1'])).rejects.toThrow(/fixedpoint/);
    await expect(run(['fix 1 all nvt/uef temp 1 1 0.5 erate -0.5 -0.5 dilate all', 'run 1'])).rejects.toThrow(/dilate/);
  });

  it('needs erate, a triclinic box, and the matching pressure keywords', async () => {
    await expect(run(['fix 1 all nvt/uef temp 1 1 0.5', 'run 1'])).rejects.toThrow(/erate/);
    await expect(run(['fix 1 all nvt/uef temp 1 1 0.5 erate -0.5 -0.5 iso 1 1 1', 'run 1'])).rejects.toThrow(/Pressure control/);
    await expect(run(['fix 1 all npt/uef temp 1 1 0.5 erate 0.5 0.5', 'run 1'])).rejects.toThrow(/Pressure control must be used/);
    await expect(run(['fix 1 all npt/uef temp 1 1 0.5 aniso 1 1 1 erate 0.5 0.5 ext z', 'run 1'])).rejects.toThrow(/ext keyword/);
  });

  it('dump cfg/uef writes the box in the flow frame (native files compared by the oracle case dump_cfg_uef)', async () => {
    const files = new Map<string, string>();
    const session = new Session({ emit: () => {}, writeFile: (n: string, t: string) => { files.set(n, t); } }, undefined, 0, 1);
    await session.execute([...BASE, 'fix 1 all nvt/uef temp 1 1 0.5 erate -0.5 -0.5', 'dump 1 all cfg/uef 25 dump.*.cfg mass type xs ys zs', 'run 1'].join('\n') + '\n');
    const text = files.get('dump.0.cfg') ?? '';
    expect(text).toMatch(/^Number of particles = \d+\nA = 1 Angstrom \(basic length-scale\)\nH0\(1,1\) = /);
    expect(text.split('\n').filter((l) => /^H0\(\d,\d\) = /.test(l))).toHaveLength(9);
  });
});
