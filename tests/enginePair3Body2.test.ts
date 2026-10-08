import { describe, expect, it } from 'vitest';
import { PairATM, atmTriple } from '../src/engine/force/pair/atm';
import { PairNB3BHarmonic, PairNB3BScreened } from '../src/engine/force/pair/nb3b';
import { StyleError, newAccum, type PairCompute, type StyleContext } from '../src/engine/force/types';

/*
 * Three-body pair styles atm (Axilrod-Teller-Muto) and nb3b/harmonic.
 * Reference formulas: docs.lammps.org/pair_atm.html and docs.lammps.org/pair_nb3b.html
 * (plans/lammps-docs/pair_atm.rst, pair_nb3b.rst). Forces are checked against
 * central finite differences of the energy the styles report.
 */

/** Deterministic pseudo-random numbers (LCG), so failures reproduce. */
const rng = (seed: number) => {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
};

const ctxWith = (files: Record<string, string> = {}): StyleContext => ({
  s: null,
  readFile: (name: string) => {
    if (!(name in files)) throw new StyleError(`file ${name} not found`);
    return files[name];
  },
  log: () => {},
});

/** A PairCompute over n owned atoms with a full neighbour list of all pairs (no ghosts, no bonds). */
const computeInput = (x: Float64Array, type: Int32Array) => {
  const n = type.length;
  const neighbors = new Int32Array(n * (n - 1));
  const firstneigh = new Int32Array(n), numneigh = new Int32Array(n);
  let p = 0;
  for (let i = 0; i < n; i++) {
    firstneigh[i] = p;
    for (let j = 0; j < n; j++) if (j !== i) neighbors[p++] = j;
    numneigh[i] = n - 1;
  }
  const owner = new Int32Array(n);
  for (let i = 0; i < n; i++) owner[i] = i;
  const pc = {
    x, f: new Float64Array(3 * n), type, q: new Float64Array(n), nlocal: n, nall: n,
    half: null, full: { inum: n, ilist: null, firstneigh, numneigh, neighbors },
    nb: { owner, gimage: new Int32Array(3 * n) },
    specialLJ: new Float64Array(4), specialCoul: new Float64Array(4), qqrd2e: 1,
    acc: newAccum(), eatom: null, vatom: null,
    s: null, geom: null,
  } as unknown as PairCompute;
  return pc;
};

/** Energy and forces of a style at positions x (forces = -grad E from the style itself). */
const evalStyle = (style: PairATM | PairNB3BHarmonic, x: Float64Array, type: Int32Array) => {
  const pc = computeInput(x, type);
  style.compute(pc);
  return { e: pc.acc.evdwl, f: pc.f };
};

/** Central finite-difference force -dE/dx of a style, component by component. */
const fdForces = (style: PairATM | PairNB3BHarmonic, x: Float64Array, type: Int32Array, h = 1e-6) => {
  const g = new Float64Array(x.length);
  for (let k = 0; k < x.length; k++) {
    const xp = Float64Array.from(x), xm = Float64Array.from(x);
    xp[k] += h; xm[k] -= h;
    g[k] = -(evalStyle(style, xp, type).e - evalStyle(style, xm, type).e) / (2 * h);
  }
  return g;
};

const randomPositions = (n: number, span: number, seed: number) => {
  const r = rng(seed);
  const x = new Float64Array(3 * n);
  for (let k = 0; k < x.length; k++) x[k] = (r() - 0.5) * span;
  return x;
};

describe('atm triple energy and gradient', () => {
  it('matches the doc formula with the triangle angles (equilateral side 1 gives 1.375)', () => {
    const p = [0, 0, 0, 1, 0, 0, 0.5, Math.sqrt(3) / 2, 0];
    expect(atmTriple(p, 1, null)).toBeCloseTo(1.375, 12);
    expect(atmTriple(p, 2.5, null)).toBeCloseTo(2.5 * 1.375, 12);
  });

  it('matches 1 + 3 cos g1 cos g2 cos g3 with the interior angles for a generic triangle', () => {
    const p = [0, 0, 0, 1.3, 0.2, -0.1, 0.4, 0.9, 0.6];
    const pts = [[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], p[8]]];
    const dist = (u: number[], v: number[]) => Math.hypot(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
    const r12 = dist(pts[0], pts[1]), r23 = dist(pts[1], pts[2]), r31 = dist(pts[2], pts[0]);
    const angle = (a: number[], b: number[], c: number[]) => {
      const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
      const d = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
      return Math.acos(d / (Math.hypot(...u) * Math.hypot(...v)));
    };
    const g1 = angle(pts[0], pts[1], pts[2]), g2 = angle(pts[1], pts[2], pts[0]), g3 = angle(pts[2], pts[0], pts[1]);
    const ref = (1 + 3 * Math.cos(g1) * Math.cos(g2) * Math.cos(g3)) / (r12 * r23 * r31) ** 3;
    expect(atmTriple(p, 1, null)).toBeCloseTo(ref, 12);
  });

  it('gradient equals central finite differences for random triplets', () => {
    const r = rng(7);
    for (let t = 0; t < 40; t++) {
      const p = new Float64Array(9);
      for (let k = 0; k < 9; k++) p[k] = (r() - 0.5) * 4;
      const nu = 0.2 + r();
      const g = new Float64Array(9);
      atmTriple(p, nu, g);
      for (let k = 0; k < 9; k++) {
        const pp = Float64Array.from(p), pm = Float64Array.from(p);
        const h = 1e-6;
        pp[k] += h; pm[k] -= h;
        const fd = (atmTriple(pp, nu, null) - atmTriple(pm, nu, null)) / (2 * h);
        expect(Math.abs(g[k] - fd)).toBeLessThanOrEqual(1e-6 * (1 + Math.abs(fd)));
      }
      // the three gradients sum to zero (translation invariance)
      for (let c = 0; c < 3; c++) expect(Math.abs(g[c] + g[3 + c] + g[6 + c])).toBeLessThan(1e-12);
    }
  });
});

describe('pair_style atm', () => {
  const setup = (cut: string, ctrip: string, coeff: string[]) => {
    const pair = new PairATM();
    const ctx = ctxWith();
    pair.allocate(2);
    pair.settings([cut, ctrip], ctx);
    for (const c of coeff) pair.coeff(c.split(' '), ctx);
    pair.init(ctx);
    return pair;
  };

  it('forces are minus the gradient of the total energy (two types, periodic-free cluster)', () => {
    const pair = setup('4.5', '100', ['* * * 0.7']);
    const n = 6;
    const type = Int32Array.from([1, 2, 1, 2, 2, 1]);
    const x = randomPositions(n, 3.2, 11);
    const { f } = evalStyle(pair, x, type);
    const fd = fdForces(pair, x, type);
    for (let k = 0; k < 3 * n; k++) expect(Math.abs(f[k] - fd[k])).toBeLessThanOrEqual(1e-5 * (1 + Math.abs(fd[k])));
  });

  it('the triplet cutoffs exclude triangles as documented', () => {
    // equilateral side 1 (energy 1.375 nu): excluded by pair cutoff 0.99, by cutoff_triple 0.99 (product 1 vs 0.99^3)
    const x = Float64Array.from([0, 0, 0, 1, 0, 0, 0.5, Math.sqrt(3) / 2, 0]);
    const type = Int32Array.from([1, 1, 1]);
    const e = (cut: string, ctrip: string) => evalStyle(setup(cut, ctrip, ['* * * 1.0']), x, type).e;
    expect(e('4.5', '100')).toBeCloseTo(1.375, 10);
    expect(e('0.99', '100')).toBe(0);
    expect(e('4.5', '0.99')).toBe(0);
    expect(e('4.5', '1.01')).toBeCloseTo(1.375, 10);
  });

  it('an unset (I,J) pair is an error, as the doc requires a pair_coeff for all I,J', () => {
    const pair = new PairATM();
    const ctx = ctxWith();
    pair.allocate(2);
    pair.settings(['4.5', '2.5'], ctx);
    pair.coeff(['1', '1', '1', '0.1'], ctx);
    expect(() => pair.init(ctx)).toThrow(StyleError);
    expect(() => pair.init(ctx)).toThrow(/all pair coeffs are not set/);
  });

  it('argument errors name the problem', () => {
    const pair = new PairATM();
    const ctx = ctxWith();
    pair.allocate(2);
    expect(() => pair.settings(['4.5'], ctx)).toThrow(StyleError);
    expect(() => pair.settings(['-1', '2'], ctx)).toThrow(/cutoff must be > 0/);
    expect(() => pair.settings(['4.5', '0'], ctx)).toThrow(/cutoff_triple must be > 0/);
    pair.settings(['4.5', '2.5'], ctx);
    expect(() => pair.coeff(['1', '1', '1'], ctx)).toThrow(/usage: pair_coeff I J K nu/);
    expect(() => pair.coeff(['1', '1', '3', '0.1'], ctx)).toThrow(StyleError);
    expect(() => pair.coeff(['1', '1', '1', 'abc'], ctx)).toThrow(StyleError);
  });
});

describe('pair_style nb3b/harmonic', () => {
  const ELEMS = ['Mg', 'O', 'H'];
  /** All ordered element triples; cutoff 3.0 for the legs; K and theta0 random per entry. */
  const potentialFile = (seed: number) => {
    const r = rng(seed);
    const lines: string[] = [];
    for (const a of ELEMS) for (const b of ELEMS) for (const c of ELEMS) {
      if (b === c) lines.push(`${a} ${b} ${c} 0 0 3.0`);
      else lines.push(`${a} ${b} ${c} ${(0.5 + r()).toFixed(6)} ${(80 + 60 * r()).toFixed(4)} 0`);
    }
    return lines.join('\n') + '\n';
  };

  const setup = (file: string, elems: string[]) => {
    const pair = new PairNB3BHarmonic();
    const ctx = ctxWith({ 'pot.nb3b': file });
    pair.allocate(elems.length);
    pair.coeff(['*', '*', 'pot.nb3b', ...elems], ctx);
    pair.init(ctx);
    return pair;
  };

  /** Brute-force energy: each owned center, each unordered neighbour pair within the leg cutoffs. */
  const bruteEnergy = (file: string, elems: string[], x: Float64Array, type: Int32Array) => {
    const params = new Map<string, { K: number; th: number; cut: number }>();
    for (const line of file.trim().split('\n')) {
      const t = line.trim().split(/\s+/);
      params.set(`${t[0]} ${t[1]} ${t[2]}`, { K: Number(t[3]), th: Number(t[4]) * Math.PI / 180, cut: Number(t[5]) });
    }
    const n = type.length;
    let e = 0;
    const d = (i: number, j: number) => [x[3 * j] - x[3 * i], x[3 * j + 1] - x[3 * i + 1], x[3 * j + 2] - x[3 * i + 2]];
    for (let c = 0; c < n; c++) {
      const ec = elems[type[c] - 1];
      const nbs: number[] = [];
      for (let j = 0; j < n; j++) {
        if (j === c) continue;
        const ej = elems[type[j] - 1];
        const rr = Math.hypot(...d(c, j));
        if (rr <= params.get(`${ec} ${ej} ${ej}`)!.cut) nbs.push(j);
      }
      for (let a = 0; a < nbs.length; a++) for (let b = a + 1; b < nbs.length; b++) {
        const j = nbs[a], k = nbs[b];
        // the entry for the angle: the neighbour of lower type index comes second in the argument order
        const lo = type[j] < type[k] ? j : k, hi = lo === j ? k : j;
        const p = params.get(`${ec} ${elems[type[lo] - 1]} ${elems[type[hi] - 1]}`)!;
        const u = d(c, j), v = d(c, k);
        const cos = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (Math.hypot(...u) * Math.hypot(...v));
        const th = Math.acos(Math.min(1, Math.max(-1, cos)));
        e += p.K * (th - p.th) ** 2;
      }
    }
    return e;
  };

  it('energy equals the brute-force sum over centers and neighbour pairs (K (theta - theta0)^2)', () => {
    const file = potentialFile(3);
    const type = Int32Array.from([1, 2, 3, 2, 1]);
    const x = randomPositions(5, 4.0, 5);
    const pair = setup(file, ELEMS);
    expect(evalStyle(pair, x, type).e).toBeCloseTo(bruteEnergy(file, ELEMS, x, type), 10);
  });

  it('forces are minus the gradient of the energy', () => {
    const file = potentialFile(9);
    const type = Int32Array.from([1, 2, 3, 2, 1, 3]);
    const x = randomPositions(6, 4.0, 21);
    const pair = setup(file, ELEMS);
    const { f } = evalStyle(pair, x, type);
    const fd = fdForces(pair, x, type);
    for (let k = 0; k < 3 * 6; k++) expect(Math.abs(f[k] - fd[k])).toBeLessThanOrEqual(1e-5 * (1 + Math.abs(fd[k])));
  });

  it('the NULL mapping leaves the type out of the three-body term', () => {
    const file = potentialFile(4);
    const type = Int32Array.from([1, 2, 3, 2, 1]);
    const x = randomPositions(5, 3.5, 2);
    const e = evalStyle(setup(file, ['Mg', 'O', 'NULL']), x, type).e;
    // atoms of type 3 are neither centers nor neighbours: only the atoms 0, 1, 3, 4 count
    const keep = [0, 1, 3, 4];
    const xs = new Float64Array(3 * keep.length);
    keep.forEach((a, q) => xs.set(x.slice(3 * a, 3 * a + 3), 3 * q));
    const ts = Int32Array.from(keep.map((a) => type[a]));
    expect(e).toBeCloseTo(bruteEnergy(file, ELEMS, xs, ts), 10);
  });

  it('argument and file errors name the problem', () => {
    const ctx = ctxWith({ 'pot.nb3b': potentialFile(1), 'short.nb3b': 'Mg O H 1.0 110.0\n' });
    const pair = new PairNB3BHarmonic();
    expect(() => pair.settings(['2.0'], ctx)).toThrow(/takes no arguments/);
    pair.allocate(3);
    expect(() => pair.coeff(['1', '1', 'pot.nb3b', 'Mg', 'O', 'H'], ctx)).toThrow(/first 2 arguments/);
    expect(() => pair.coeff(['*', '*', 'pot.nb3b', 'Mg', 'O'], ctx)).toThrow(/one element name per atom type/);
    expect(() => pair.coeff(['*', '*', 'pot.nb3b', 'Mg', 'O', 'Cl'], ctx)).toThrow(/element 'Cl' is not in nb3b potential file/);
    expect(() => pair.coeff(['*', '*', 'short.nb3b', 'Mg', 'O', 'H'], ctx)).toThrow(/expected 'element1 element2 element3 K theta0 cutoff'/);
    const partial = 'Mg O H 1 110 0\nMg O O 0 0 3\n';
    const ctx2 = ctxWith({ 'partial.nb3b': partial });
    const p2 = new PairNB3BHarmonic();
    p2.allocate(3);
    expect(() => p2.coeff(['*', '*', 'partial.nb3b', 'Mg', 'O', 'H'], ctx2)).toThrow(/missing an entry for: Mg Mg Mg/);
  });

  it('nb3b/screened is refused with a StyleError (its file layout is not in the docs)', () => {
    expect(() => new PairNB3BScreened().settings([], ctxWith())).toThrow(/pair_style nb3b\/screened is not supported/);
  });
});
