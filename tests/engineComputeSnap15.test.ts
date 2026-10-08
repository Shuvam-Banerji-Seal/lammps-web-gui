import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * compute snap, sna/grid and sna/grid/local (wave 15). Parity with native LAMMPS is in
 * tests/oracle/w15snap_*.in (run by tests/engineOracle.test.ts); this file checks the layouts,
 * internal identities and argument errors of the engine.
 *
 * Docs: docs.lammps.org/compute_sna_atom.html (compute snap, sna/grid, sna/grid/local).
 */

/** Four atoms (two types) in a bcc cell with a smooth displacement: the same crystal as the oracle cases. */
const HEADER = `units metal
atom_modify map hash
lattice bcc 2.0
region box block 0 2 0 1 0 1
create_box 2 box
create_atoms 1 box
group odd id 1:4:2
set group odd type 2
mass * 180.88
variable dx atom 0.05*sin(1.3*x+0.7*y)
variable dy atom 0.05*cos(0.9*y-1.1*z)
variable dz atom 0.05*sin(1.7*z+0.5*x)
displace_atoms all move v_dx v_dy v_dz units box
`;
const PAIR = 'pair_style zbl 4 4.8\npair_coeff * * 73 73\n';
const SNAP = '1.0 0.99363 2 2.3 2.0 1.0 0.96';

const runScript = async (text: string): Promise<Session> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  await session.execute(text);
  return session;
};

/** Runs HEADER + a compute line (with run 0) and returns the session. */
const withCompute = (line: string, pair = PAIR) => runScript(`${HEADER}${pair}${line}\nrun 0\n`);

describe('compute snap: array layout', () => {
  it('has 1 + 3N + 6 rows and ntypes K + 1 columns', async () => {
    const s = await withCompute(`compute snap all snap ${SNAP} rmin0 0.1`);
    const c = s.sys.compute('snap');
    c.arrayValues();
    expect(c.sizeArrayRows).toBe(1 + 3 * 4 + 6);
    expect(c.sizeArrayCols).toBe(2 * 5 + 1);
  });

  it('bikflag gives N per-atom rows, then 3N snad rows and 6 snav rows', async () => {
    const s = await withCompute(`compute snap all snap ${SNAP} rmin0 0 bikflag 1`);
    const c = s.sys.compute('snap');
    c.arrayValues();
    expect(c.sizeArrayRows).toBe(4 + 3 * 4 + 6);
    expect(c.sizeArrayCols).toBe(11);
  });

  it('dgradflag gives N + 3N^2 + 1 rows and K + 3 columns', async () => {
    const s = await withCompute(`compute snap all snap ${SNAP} rmin0 0 bikflag 1 dgradflag 1`);
    const c = s.sys.compute('snap');
    c.arrayValues();
    expect(c.sizeArrayRows).toBe(4 + 3 * 16 + 1);
    expect(c.sizeArrayCols).toBe(5 + 3);
  });

  it('quadraticflag adds K(K+1)/2 columns per type block', async () => {
    const s = await withCompute(`compute snap all snap ${SNAP} rmin0 0 quadraticflag 1`);
    const c = s.sys.compute('snap');
    c.arrayValues();
    expect(c.sizeArrayCols).toBe(2 * (5 + 15) + 1);
  });

  it('the first row sums the per-atom sna/atom values of each type', async () => {
    const s = await withCompute(`compute snap all snap ${SNAP} rmin0 0.1\ncompute b all sna/atom ${SNAP} rmin0 0.1`);
    const row0 = s.sys.compute('snap').arrayValues();
    const cols = s.sys.compute('snap').sizeArrayCols;
    const b = s.sys.compute('b').peratomValues();
    const type = s.sys.state.type;
    for (let t = 1; t <= 2; t++) {
      for (let k = 0; k < 5; k++) {
        let sum = 0;
        for (let i = 0; i < 4; i++) if (type[i] === t) sum += b[i * 5 + k];
        expect(row0[(t - 1) * 5 + k]).toBeCloseTo(sum, 6);
      }
    }
    expect(row0[cols - 1]).toBeCloseTo(s.sys.forces().evdwl + s.sys.forces().ecoul, 9);
  });

  it('the force rows hold the forces of the atoms (rows ordered by atom ID)', async () => {
    const s = await withCompute(`compute snap all snap ${SNAP} rmin0 0.1`);
    const c = s.sys.compute('snap');
    const a = c.arrayValues();
    const cols = c.sizeArrayCols;
    const f = s.sys.state.f;
    for (let i = 0; i < 4; i++) {
      for (let k = 0; k < 3; k++) expect(a[(1 + 3 * i + k) * cols + cols - 1]).toBeCloseTo(f[3 * i + k], 9);
    }
  });
});

describe('compute snap: derivative identities', () => {
  it('dgradflag rows sum over i to minus the snad rows, and sum over j to zero (translation invariance)', async () => {
    const args = `${SNAP} rmin0 0`;
    const s = await withCompute(`compute dg all snap ${args} bikflag 1 dgradflag 1\ncompute sn all snap ${args}`);
    const dg = s.sys.compute('dg').arrayValues();
    const sn = s.sys.compute('sn');
    const snv = sn.arrayValues();
    const N = 4, K = 5;
    const type = s.sys.state.type;
    const cell = (i: number, j: number, a: number, k: number) => dg[(N + (j * N + i) * 3 + a) * 8 + 3 + k];
    for (let j = 0; j < N; j++) {
      for (let a = 0; a < 3; a++) {
        for (let t = 1; t <= 2; t++) {
          for (let k = 0; k < K; k++) {
            let sum = 0;
            for (let i = 0; i < N; i++) if (type[i] === t) sum += cell(i, j, a, k);
            // snad row of atom j in its ID order (atoms are created in ID order here); the
            // dgrad rows hold -dB/dr (native sign), so their sum over i is the snad row itself
            const row = 1 + 3 * j + a;
            expect(sum).toBeCloseTo(snv[row * sn.sizeArrayCols + (t - 1) * K + k], 6);
          }
        }
      }
    }
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < K; k++) {
        let sum = 0;
        for (let j = 0; j < N; j++) sum += cell(i, j, 0, k);
        expect(Math.abs(sum)).toBeLessThan(1e-9);
      }
    }
  });

  it('wselfallflag has no effect without chem (documented)', async () => {
    const s = await withCompute(`compute a all snap ${SNAP} rmin0 0.1 wselfallflag 1\ncompute b all snap ${SNAP} rmin0 0.1 wselfallflag 0`);
    const x = s.sys.compute('a').arrayValues();
    const y = s.sys.compute('b').arrayValues();
    expect(Array.from(x)).toEqual(Array.from(y));
  });
});

describe('compute sna/grid and sna/grid/local', () => {
  /** The same crystal without displacement: grid point 0 is the origin atom, point 7 the basis atom 2. */
  const PERFECT = HEADER.replace(/variable dx[\s\S]*displace_atoms[^\n]*\n/, '');

  it('has nx ny nz rows and 3 + K columns; the local array has the global indexes first', async () => {
    const s = await runScript(`${PERFECT}${PAIR}compute g all sna/grid grid 2 2 2 ${SNAP} rmin0 0\ncompute l all sna/grid/local grid 2 2 2 ${SNAP} rmin0 0\nrun 0\n`);
    const g = s.sys.compute('g');
    const garr = g.arrayValues();
    expect(g.sizeArrayRows).toBe(8);
    expect(g.sizeArrayCols).toBe(3 + 5);
    expect(garr.length).toBe(8 * 8);
    const l = s.sys.compute('l');
    const larr = l.localValues();
    expect(l.sizeLocalCols).toBe(6 + 5);
    expect(l.localRows).toBe(8);
    // point 1 is (ix=1, iy=0, iz=0): same bispectrum as the global row 1
    expect(larr[1 * 11 + 0]).toBe(1);
    for (let c = 0; c < 5; c++) expect(larr[1 * 11 + 6 + c]).toBeCloseTo(garr[1 * 8 + 3 + c], 12);
  });

  it('a grid point on an atom reproduces that atom (the self term is the identity)', async () => {
    // one type: the grid point takes the type of its neighbours (doc), so a single type matches sna/atom
    const ONE = PERFECT.replace('create_box 2 box', 'create_box 1 box').replace('set group odd type 2\n', '').replace('group odd id 1:4:2\n', '').replace('mass * 180.88', 'mass 1 180.88');
    const ARGS = '1.0 0.99363 2 2.3 1.0';
    const s = await runScript(`${ONE}${PAIR}compute b all sna/atom ${ARGS} rmin0 0\ncompute g all sna/grid grid 2 2 2 ${ARGS} rmin0 0\nrun 0\n`);
    const x = s.sys.state.x;
    const b = s.sys.compute('b').peratomValues();
    const garr = s.sys.compute('g').arrayValues();
    // grid point 0 is the origin; grid point 7 is the body centre (the basis atom 2 at 0.5a)
    const origin = [0, 1, 2, 3].find((i) => Math.abs(x[3 * i]) + Math.abs(x[3 * i + 1]) + Math.abs(x[3 * i + 2]) < 1e-9);
    expect(origin).toBeDefined();
    for (let c = 0; c < 5; c++) expect(garr[0 * 8 + 3 + c]).toBeCloseTo(b[origin! * 5 + c], 9);
  });

  it('switchinnerflag needs sinner and dinner, and the grid accepts it', async () => {
    const s = await runScript(`${PERFECT}${PAIR}compute g all sna/grid grid 2 2 2 ${SNAP} rmin0 0 switchinnerflag 1 sinner 1.35 1.6 dinner 0.25 0.3\nrun 0\n`);
    expect(s.sys.compute('g').arrayValues().length).toBe(8 * 8);
  });
});

describe('compute snap, sna/grid: argument errors', () => {
  const fails = async (line: string, re: RegExp, pair = PAIR) => {
    let err: unknown;
    try {
      await withCompute(line, pair);
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(Error);
    expect((err as Error).message).toMatch(re);
  };

  it('rejects chem, nnn, wmode and delta (not implemented, documented for sna/atom only)', async () => {
    await fails(`compute s all snap ${SNAP} chem 2 0 1`, /keyword 'chem' .* not implemented/);
    await fails(`compute s all snap ${SNAP} nnn 4`, /keyword 'nnn' is only implemented for compute sna\/atom/);
    await fails(`compute s all snap ${SNAP} wmode 1 delta 0.2`, /keyword 'wmode' is only implemented for compute sna\/atom/);
    await fails(`compute g all sna/grid grid 2 2 2 ${SNAP} nnn 4`, /keyword 'nnn' is only implemented for compute sna\/atom/);
  });

  it('rejects the snap-only keywords on sna/grid, and dgradflag without bikflag (doc)', async () => {
    await fails(`compute g all sna/grid grid 2 2 2 ${SNAP} bikflag 1`, /keyword 'bikflag' is only implemented for compute snap/);
    await fails(`compute g all sna/grid grid 2 2 2 ${SNAP} dgradflag 1`, /keyword 'dgradflag' is only implemented for compute snap/);
    await fails(`compute s all snap ${SNAP} dgradflag 1`, /dgradflag 1 requires bikflag 1/);
  });

  it('rejects dgradflag with quadraticflag (native LAMMPS refuses the combination)', async () => {
    await fails(`compute s all snap ${SNAP} bikflag 1 dgradflag 1 quadraticflag 1`, /dgradflag 1 with quadraticflag 1 is not implemented/);
  });

  it('pairs sinner/dinner with switchinnerflag', async () => {
    await fails(`compute s all snap ${SNAP} sinner 1.35 1.6 dinner 0.25 0.3`, /only used with switchinnerflag 1/);
    await fails(`compute s all snap ${SNAP} switchinnerflag 1`, /needs the keywords sinner and dinner/);
  });

  it('rejects unknown keywords, non-binary flags and a missing grid keyword', async () => {
    await fails(`compute s all snap ${SNAP} frobnicate 1`, /unknown keyword 'frobnicate'/);
    await fails(`compute s all snap ${SNAP} switchflag 2`, /switchflag must be 0 or 1/);
    await fails(`compute g all sna/grid 2 2 2 ${SNAP}`, /expects 'grid nx ny nz/);
  });

  it('rejects a snap cutoff longer than the pair cutoff (native refuses it too)', async () => {
    let err: unknown;
    try {
      const s = await withCompute(`compute s all snap ${SNAP}`, 'pair_style zero 1.0\npair_coeff * *\n');
      s.sys.compute('s').arrayValues();
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(Error);
    expect((err as Error).message).toMatch(/is longer than pairwise cutoff/);
  });
});
