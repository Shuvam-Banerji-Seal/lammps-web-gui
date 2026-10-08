import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { clebsch, wignerU, ComputeSnaAtom } from '../src/engine/compute/sna';
import { parseSnapCoeff, parseSnapParam } from '../src/engine/force/pair/snap';
import type { EngineEvent } from '../src/engine/types';

/** Runs input text in a fresh session. */
const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  await session.execute(text);
  return session;
};

/** Three atoms in a 20 A box; the central atom is at the origin of the probe. */
const probe = (pts: number[][]) => `
units lj
atom_style atomic
boundary p p p
region box block -10 10 -10 10 -10 10
create_box 2 box
${pts.map((p, i) => `create_atoms ${i === 0 ? 1 : 2} single ${p.join(' ')}`).join('\n')}
mass * 1.0
pair_style zero 4.5
pair_coeff * *
`;

/** Rotation about an axis (unit vector) by an angle, applied to a point. */
const rotate = (p: number[], axis: number[], ang: number): number[] => {
  const [ux, uy, uz] = axis;
  const c = Math.cos(ang), s = Math.sin(ang), t = 1 - c;
  const [x, y, z] = p;
  return [
    (t * ux * ux + c) * x + (t * ux * uy - s * uz) * y + (t * ux * uz + s * uy) * z,
    (t * ux * uy + s * uz) * x + (t * uy * uy + c) * y + (t * uy * uz - s * ux) * z,
    (t * ux * uz - s * uy) * x + (t * uy * uz + s * ux) * y + (t * uz * uz + c) * z,
  ];
};

const NEIGH = [[0.7, 0.3, -0.4], [-0.2, 0.9, 0.5], [1.1, -0.8, 0.6], [-0.9, -0.5, -0.7]];
const SNA_ARGS = 'sna/atom 1.1 0.85 4 1.6 2.0 1.0 1.0 rmin0 0.1';

describe('compute sna/atom: building blocks', () => {
  it('Clebsch-Gordan values match textbook tables', () => {
    // doubled arguments: <1/2 1/2, 1/2 -1/2 | 0 0> = 1/sqrt2
    expect(clebsch(1, 1, 1, -1, 0, 0)).toBeCloseTo(Math.SQRT1_2, 12);
    // <1 1, 1 -1 | 0 0> = 1/sqrt3
    expect(clebsch(2, 2, 2, -2, 0, 0)).toBeCloseTo(1 / Math.sqrt(3), 12);
    // <1 0, 1 0 | 2 0> = sqrt(2/3)
    expect(clebsch(2, 0, 2, 0, 4, 0)).toBeCloseTo(Math.sqrt(2 / 3), 12);
    // selection rules give zero
    expect(clebsch(2, 2, 2, 0, 2, 0)).toBe(0);
  });

  it('Wigner U is unitary and the identity at (a, b) = (1, 0)', () => {
    const ar = 0.6, ai = 0.3, br = 0.2, bi = -0.5;
    const nrm = Math.sqrt(ar * ar + ai * ai + br * br + bi * bi);
    for (let J = 0; J <= 4; J++) {
      const n = J + 1;
      const U = wignerU(J, ar / nrm, ai / nrm, br / nrm, bi / nrm);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          let sr = 0, si = 0;
          for (let k = 0; k < n; k++) {
            sr += U.re[i * n + k] * U.re[j * n + k] + U.im[i * n + k] * U.im[j * n + k];
            si += U.im[i * n + k] * U.re[j * n + k] - U.re[i * n + k] * U.im[j * n + k];
          }
          expect(sr).toBeCloseTo(i === j ? 1 : 0, 12);
          expect(Math.abs(si)).toBeLessThan(1e-12);
        }
      }
      const I = wignerU(J, 1, 0, 0, 0);
      for (let q = 0; q < n * n; q++) {
        expect(I.re[q]).toBeCloseTo(q % (n + 1) === 0 ? 1 : 0, 14);
        expect(I.im[q]).toBeCloseTo(0, 14);
      }
    }
  });
});

describe('compute sna/atom: invariance and output layout', () => {
  it('bispectrum is invariant under a rigid rotation of the neighbours', async () => {
    const axis = [0.36, 0.48, 0.8];
    const rot = NEIGH.map((p) => rotate(p, axis, 0.7));
    const shift = (pts: number[][]) => pts.map((p) => p.map((v) => v + 5));
    const a = await runScript(`${probe(shift([[0, 0, 0], ...NEIGH]))}\ncompute b all ${SNA_ARGS} bzeroflag 0\nrun 0\n`);
    const b = await runScript(`${probe(shift([[0, 0, 0], ...rot]))}\ncompute b all ${SNA_ARGS} bzeroflag 0\nrun 0\n`);
    const va = a.sys.compute('b').peratomValues();
    const vb = b.sys.compute('b').peratomValues();
    const K = a.sys.compute('b').sizePeratomCols;
    expect(K).toBe(14); // twojmax 4: m = 3, K = m(m+1)(2m+1)/6
    // atom 0 is the central atom of both runs
    for (let c = 0; c < K; c++) expect(vb[c]).toBeCloseTo(va[c], 9);
    expect(Math.abs(va[2])).toBeGreaterThan(1e-6);
  });

  it('an isolated atom has B0 subtracted by default and identity values without bzeroflag', async () => {
    const s = await runScript(`${probe([[0, 0, 0]])}\ncompute b all ${SNA_ARGS}\ncompute r all ${SNA_ARGS} bzeroflag 0\nrun 0\n`);
    const vb = s.sys.compute('b').peratomValues();
    const vr = s.sys.compute('r').peratomValues();
    // B0 of the identity: B_000 = 1, (1/2,0,1/2) = 2, (1,0,1) = 3, (3/2,0,3/2)... first column is 1
    expect(vr[0]).toBeCloseTo(1, 12);
    expect(vb[0]).toBeCloseTo(0, 12);
  });

  it('quadraticflag appends K(K+1)/2 products after the K linear terms', async () => {
    const s = await runScript(`${probe(shiftAll([[0, 0, 0], ...NEIGH]))}\ncompute q all sna/atom 1.1 0.85 2 1.6 2.0 1.0 1.0 rmin0 0.1 quadraticflag 1\nrun 0\n`);
    const c = s.sys.compute('q');
    expect(c.sizePeratomCols).toBe(5 + 15);
  });
});

const shiftAll = (pts: number[][]) => pts.map((p) => p.map((v) => v + 5));

describe('compute sna/atom: argument errors', () => {
  const fails = async (cmd: string, re: RegExp) => {
    let err: unknown;
    try {
      await runScript(`${probe([[0, 0, 0]])}\ncompute b all ${cmd}\nrun 0\n`);
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(Error);
    expect((err as Error).message).toMatch(re);
  };

  it('rejects too few arguments', async () => {
    await fails('sna/atom 1.1 0.85 4 1.6', /expects rcutfac rfac0 twojmax/);
  });

  it('rejects a non-integer twojmax', async () => {
    await fails('sna/atom 1.1 0.85 4.5 1.6 2.0 1.0 1.0', /twojmax/);
  });

  it('rejects unsupported keywords by name', async () => {
    await fails('sna/atom 1.1 0.85 4 1.6 2.0 1.0 1.0 chem 2 0 1', /keyword 'chem' is not implemented/);
    await fails('sna/atom 1.1 0.85 4 1.6 2.0 1.0 1.0 nnn 12', /keyword 'nnn' is not implemented/);
  });

  it('rejects unknown keywords and non-binary flags', async () => {
    await fails('sna/atom 1.1 0.85 4 1.6 2.0 1.0 1.0 frobnicate 1', /unknown keyword 'frobnicate'/);
    await fails('sna/atom 1.1 0.85 4 1.6 2.0 1.0 1.0 switchflag 2', /switchflag must be 0 or 1/);
  });

  it('is registered under the sna/atom style name', () => {
    expect(ComputeSnaAtom).toBeDefined();
  });
});

/* ---- pair_style snap: synthetic 2-element files, energy and forces ---- */

const K3 = 8; // twojmax 3: K = 8 bispectrum components

/** Deterministic pseudo-random coefficients (small, so the energies stay O(1)). */
const coefLines = (n: number, seed: number): string[] => {
  const out: string[] = [];
  let v = seed;
  for (let k = 0; k < n; k++) {
    v = (v * 16807) % 2147483647;
    out.push(String((v / 2147483647 - 0.5) * (k === 0 ? 0.5 : 0.2)));
  }
  return out;
};

const snapCoeffText = (quadratic: boolean): string => {
  const n = 1 + K3 + (quadratic ? (K3 * (K3 + 1)) / 2 : 0);
  return [
    `# synthetic two-element SNAP coefficients (test only)`,
    `2 ${n}`,
    `A 1.4 1.0`, ...coefLines(n, 12345),
    `B 1.6 0.8`, ...coefLines(n, 67891),
  ].join('\n') + '\n';
};

const snapParamText = (quadratic: boolean): string => [
  '# synthetic SNAP parameters (test only)',
  'rcutfac 1.1',
  'twojmax 3',
  'rfac0 0.9',
  'rmin0 0.1',
  ...(quadratic ? ['quadraticflag 1'] : []),
].join('\n') + '\n';

/** Energy (sum of atom energies) and forces of a 4-atom configuration. */
const snapRun = async (quadratic: boolean, pos: number[][]) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  session.addFile('t.snapcoeff', snapCoeffText(quadratic));
  session.addFile('t.snapparam', snapParamText(quadratic));
  const types = [1, 2, 1, 2];
  const create = pos.map((p, i) => `create_atoms ${types[i]} single ${p.join(' ')}`).join('\n');
  await session.execute(`
units lj
atom_style atomic
boundary p p p
region box block -10 10 -10 10 -10 10
create_box 2 box
${create}
mass * 1.0
pair_style snap
pair_coeff * * t.snapcoeff t.snapparam A B
run 0
`);
  const sys = session.sys;
  const E = sys.forces().evdwl;
  return { E, f: Array.from(sys.state.f.slice(0, 3 * pos.length)) };
};

const SNAP_POS = [[0.0, 0.0, 0.0], [0.9, 0.4, -0.5], [-0.3, 1.0, 0.6], [1.1, -0.7, 0.5]];

describe('pair_style snap', () => {
  for (const quadratic of [false, true]) {
    it(`forces equal -dE/dx by finite differences (quadraticflag ${quadratic ? 1 : 0})`, async () => {
      const ref = await snapRun(quadratic, SNAP_POS);
      expect(Number.isFinite(ref.E)).toBe(true);
      // guard against a trivially passing test: the forces must be sizeable
      expect(Math.max(...ref.f.map(Math.abs))).toBeGreaterThan(1e-3);
      const h = 1e-5;
      // check atoms 0 and 2 in all three directions
      for (const atom of [0, 2]) {
        for (let c = 0; c < 3; c++) {
          const plus = SNAP_POS.map((p) => p.slice());
          const minus = SNAP_POS.map((p) => p.slice());
          plus[atom][c] += h;
          minus[atom][c] -= h;
          const Ep = (await snapRun(quadratic, plus)).E;
          const Em = (await snapRun(quadratic, minus)).E;
          const fd = -(Ep - Em) / (2 * h);
          const f = ref.f[3 * atom + c];
          expect(Math.abs(f - fd), `atom ${atom} component ${c}: engine ${f} vs fd ${fd}`).toBeLessThan(1e-6 * Math.max(1, Math.abs(fd)));
        }
      }
    }, 60_000);
  }

  it('forces stay consistent with the energy for a pair closer than rmin0', async () => {
    // rmin0 = 0.1: the second atom is at 0.08 from the first (theta0 < 0 branch, f_c = 1 there)
    const close = [[0.0, 0.0, 0.0], [0.08, 0.02, -0.01], [-0.3, 1.0, 0.6], [1.1, -0.7, 0.5]];
    const ref = await snapRun(false, close);
    const h = 1e-6;
    for (let c = 0; c < 3; c++) {
      const plus = close.map((p) => p.slice());
      const minus = close.map((p) => p.slice());
      plus[1][c] += h;
      minus[1][c] -= h;
      const fd = -((await snapRun(false, plus)).E - (await snapRun(false, minus)).E) / (2 * h);
      expect(Math.abs(ref.f[3 + c] - fd)).toBeLessThan(1e-5 * Math.max(1, Math.abs(fd)));
    }
  }, 60_000);

  it('the quadratic energy differs from the linear one (the quadratic block is used)', async () => {
    const lin = await snapRun(false, SNAP_POS);
    const quad = await snapRun(true, SNAP_POS);
    expect(Math.abs(lin.E - quad.E)).toBeGreaterThan(1e-6);
  });

  it('total force on an isolated pair is zero (Newton third law)', async () => {
    const r = await snapRun(false, SNAP_POS);
    for (let c = 0; c < 3; c++) {
      let sum = 0;
      for (let a = 0; a < SNAP_POS.length; a++) sum += r.f[3 * a + c];
      expect(Math.abs(sum)).toBeLessThan(1e-9);
    }
  });

  it('parses the documented coefficient and parameter layouts', () => {
    const lin = parseSnapCoeff(snapCoeffText(false), 'x', K3, false);
    expect(lin.elems).toEqual(['A', 'B']);
    expect(lin.ncoeff).toBe(1 + K3);
    expect(lin.radius[1]).toBeCloseTo(1.6, 12);
    const quad = parseSnapCoeff(snapCoeffText(true), 'x', K3, true);
    expect(quad.ncoeff).toBe(1 + K3 + (K3 * (K3 + 1)) / 2);
    expect(() => parseSnapCoeff(snapCoeffText(false), 'x', K3, true)).toThrow(/does not match twojmax/);
    const prm = parseSnapParam(snapParamText(true), 'x');
    expect(prm.quadraticflag).toBe(true);
    expect(prm.bzeroflag).toBe(true);
    expect(() => parseSnapParam('rcutfac 1.0\nchemflag 1\ntwojmax 2\n', 'x')).toThrow(/chemflag 1 is not implemented/);
    expect(() => parseSnapParam('rcutfac 1.0\n', 'x')).toThrow(/twojmax are required/);
  });
});
