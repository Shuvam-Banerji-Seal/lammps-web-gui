/**
 * Unit checks of the AMOEBA permanent multipole pieces (src/engine-gpl/amoeba/multipole.ts, params.ts). The native
 * parity of the whole term is the oracle case w119amoeba_mpole; these tests pin the pieces with closed forms.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseAmoebaPrm, logicalLines } from '../src/engine-gpl/amoeba/params';
import { kmpole, mapAxes, multipoleReal, rotmat, rotsite, torque2force, type MpoleSites, type MpoleTopo } from '../src/engine-gpl/amoeba/multipole';

const ROOT = join(__dirname, 'oracle');
const prmText = readFileSync(join(ROOT, 'w119amoeba_mpole.prm'), 'utf8');
const BOHR = 0.52917721067;

/** The water dimer of w119amoeba_dimer.data: atoms 1 (O), 2, 3 (H), 4 (O), 5, 6 (H); bonds 1-2, 1-3, 4-5, 4-6. */
const bondsOf: Record<number, number[]> = { 1: [2, 3], 2: [1], 3: [1], 4: [5, 6], 5: [4], 6: [4] };
const amtypeOf: Record<number, number> = { 1: 1, 2: 2, 3: 2, 4: 1, 5: 2, 6: 2 };
const x = Float64Array.from([
  -0.024616, -0.001154, -0.003748, -0.244211, -0.000666, 0.933978, 0.932234, -0.000406, -0.008705,
  -0.892721, 0.000120, 2.773674, -1.462996, 0.755120, 2.933870, -1.461809, -0.755549, 2.934934,
]);
const topo: MpoleTopo = {
  nb12: (id) => [...bondsOf[id]].sort((a, b) => a - b),
  nb13: (id) => {
    const direct = new Set(bondsOf[id]);
    const out = new Set<number>();
    for (const p of bondsOf[id]) for (const q of bondsOf[p]) if (q !== id && !direct.has(q)) out.add(q);
    return [...out].sort((a, b) => a - b);
  },
  bonded: (a, b) => bondsOf[a].includes(b),
};

const buildSites = (): MpoleSites => {
  const prm = parseAmoebaPrm(prmText);
  const ids = Int32Array.from([1, 2, 3, 4, 5, 6]);
  const amt = Int32Array.from([1, 2, 2, 1, 2, 2].map((t) => t));
  const sites = kmpole((t) => prm.multipoles.get(t) ?? [], ids, amt, (id) => amtypeOf[id], topo);
  mapAxes(sites, (_i, id) => ids.indexOf(id));
  for (let i = 0; i < 6; i++) {
    rotmat(sites, x, i);
    rotsite(sites, i);
  }
  return sites;
};

describe('AMOEBA multipole parsing (amoeba_file.cpp file_multipole)', () => {
  it('joins continuation lines and reads the axis kinds of the water types', () => {
    const lines = logicalLines('multipole 1 -2 -2 0.5\n  0.1 0.2 0.3\n  0.4\n  0.5 0.6\n  0.7 0.8 0.9\n#multipole 2 1 2 0.2\n  0.1\n');
    expect(lines).toHaveLength(1);
    const prm = parseAmoebaPrm(prmText + '\n');
    const o = prm.multipoles.get(1)![0];
    expect(o.axis).toBe('BISECTOR');
    expect([o.z, o.x, o.y]).toEqual([2, 2, 0]);
    const h = prm.multipoles.get(2)![0];
    expect(h.axis).toBe('ZTHENX');
    expect([h.z, h.x, h.y]).toEqual([1, 2, 0]);
  });

  it('converts dipoles with Bohr and quadrupoles with Bohr^2/3, symmetrized row-major', () => {
    const prm = parseAmoebaPrm(prmText);
    const o = prm.multipoles.get(1)![0].pole;
    expect(o[0]).toBeCloseTo(-0.51966, 12);
    expect(o[1]).toBeCloseTo(0.02 * BOHR, 12);
    expect(o[3]).toBeCloseTo(0.05 * BOHR, 12);
    expect(o[4]).toBeCloseTo(0.20 * BOHR * BOHR / 3, 12);
    expect(o[5]).toBeCloseTo(-0.05 * BOHR * BOHR / 3, 12);
    expect(o[7]).toBe(o[5]);
    expect(o[8]).toBeCloseTo(0.10 * BOHR * BOHR / 3, 12);
    expect(o[12]).toBeCloseTo(-0.10 * BOHR * BOHR / 3, 12);
  });
});

describe('AMOEBA multipole frames (amoeba_utils.cpp kmpole, rotmat, rotsite)', () => {
  it('assigns the water frames from the bonded partners', () => {
    const sites = buildSites();
    expect(sites.axis[0]).toBe('BISECTOR');
    expect([sites.zid[0], sites.xid[0], sites.yid[0]]).toEqual([2, 3, 0]);
    expect(sites.axis[1]).toBe('ZTHENX');
    expect([sites.zid[1], sites.xid[1]]).toEqual([1, 3]);
    expect(sites.axis[4]).toBe('ZTHENX');
    expect([sites.zid[4], sites.xid[4]]).toEqual([4, 6]);
  });

  it('builds orthonormal rotation matrices and keeps the charge and the traceless quadrupole', () => {
    const sites = buildSites();
    for (let i = 0; i < 6; i++) {
      const R = sites.rotate.subarray(9 * i, 9 * i + 9);
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        let d = 0;
        for (let k = 0; k < 3; k++) d += R[3 * a + k] * R[3 * b + k];
        expect(Math.abs(d - (a === b ? 1 : 0))).toBeLessThan(1e-12);
      }
      const rp = sites.rpole.subarray(13 * i, 13 * i + 13);
      expect(rp[0]).toBe(sites.pole[13 * i]);
      // a rotation keeps the trace of the quadrupole
      const pl = sites.pole.subarray(13 * i, 13 * i + 13);
      expect(rp[4] + rp[8] + rp[12]).toBeCloseTo(pl[4] + pl[8] + pl[12], 14);
    }
  });
});

describe('AMOEBA multipole energy and torques (amoeba_multipole.cpp multipole_real, torque2force)', () => {
  it('reproduces Coulomb\'s law for two charges and removes the pair when the special factor is 0', () => {
    const n = 2;
    const sites: MpoleSites = {
      n, axis: ['NOFRAME', 'NOFRAME'], zid: new Float64Array(n), xid: new Float64Array(n), yid: new Float64Array(n),
      zl: Int32Array.from([0, 1]), xl: Int32Array.from([0, 1]), yl: Int32Array.from([0, 1]),
      pole: new Float64Array(26), rpole: new Float64Array(26), rotate: new Float64Array(18),
    };
    sites.rpole[0] = 0.5;
    sites.rpole[13] = -0.25;
    sites.rotate.set([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const xy = Float64Array.from([0, 0, 0, 0, 0, 2]);
    const f = new Float64Array(6);
    const pi = Int32Array.from([0]), pj = Int32Array.from([1]);
    const own = Int32Array.from([0, 1]);
    const full = multipoleReal(sites, xy, f, pi, pj, Int8Array.from([0]), 1, [1, 1, 1, 1, 1], 1.0, 10, own, 0.0);
    expect(full.e).toBeCloseTo((332.063713 * 0.5 * -0.25) / 2, 10);
    // the force on atom 1 (at z = 2) is attractive for opposite charges: along -z
    expect(f[5]).toBeLessThan(0);
    const f2 = new Float64Array(6);
    const none = multipoleReal(sites, xy, f2, pi, pj, Int8Array.from([1]), 1, [1, 0, 0, 0, 0], 1.0, 10, own, 0.0);
    expect(none.e).toBe(0);
    expect(f2.every((v) => v === 0)).toBe(true);
  });

  it('conserves the total force of a torque conversion on a ZTHENX site', () => {
    const sites = buildSites();
    const f = new Float64Array(18);
    const trq = Float64Array.from([0.3, -0.2, 0.5]);
    torque2force(sites, 1, trq, x, f);
    for (let k = 0; k < 3; k++) {
      let s = 0;
      for (const a of [0, 1, 2]) s += f[3 * a + k];
      expect(Math.abs(s)).toBeLessThan(1e-12);
    }
  });

  it('refuses the torque of a frame type that is not ported yet, by name', () => {
    const sites = buildSites();
    sites.axis[0] = 'THREEFOLD';
    expect(() => torque2force(sites, 0, Float64Array.from([1, 0, 0]), x, new Float64Array(18))).toThrow(/THREEFOLD/);
  });
});
