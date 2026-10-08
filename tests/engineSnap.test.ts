import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { clebsch, wignerU, ComputeSnaAtom } from '../src/engine/compute/sna';
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
