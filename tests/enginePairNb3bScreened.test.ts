import { describe, expect, it } from 'vitest';
import { PairNB3BScreened } from '../src/engine/force/pair/nb3b';
import { StyleError, newAccum, type PairCompute, type StyleContext } from '../src/engine/force/types';

/*
 * pair_style nb3b/screened (docs.lammps.org/pair_nb3b.html, plans/lammps-docs/pair_nb3b.rst):
 * E = K (theta - theta0)^2 exp(-r_ij/rho_ij - r_ik/rho_ik) over the angles of neighbour pairs.
 * The screening lengths and the leg cutoffs come from the (i j j) entries, K and theta0 from
 * (i j k); the file layout is examples/nb3b/README.nb3b.screened (7 columns). Native parity of
 * the style is in tests/oracle/w38nb3b_*.in.
 */

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

const ELEMS = ['P', 'O', 'Si'];

/**
 * All ordered triples of ELEMS. K and theta0 are the same for (i j k) and (i k j) (the order-independent
 * case); the legs (i j j) have their own rho and cutoff; the (i j k) rho and cutoff are junk that must be ignored.
 */
const potentialFile = (seed: number): string => {
  const r = rng(seed);
  const kth = new Map<string, [number, number]>();
  const lines: string[] = [];
  for (const c of ELEMS) for (const a of ELEMS) for (const b of ELEMS) {
    if (a === b) {
      const K = 0.5 + 3 * r(), th = 70 + 60 * r();
      lines.push(`${c} ${a} ${b} ${K.toFixed(6)} ${th.toFixed(4)} ${(0.8 + 0.6 * r()).toFixed(4)} ${(2.6 + 1.0 * r()).toFixed(4)}`);
    } else {
      const key = [c, [a, b].sort().join(' ')].join(' ');
      if (!kth.has(key)) kth.set(key, [0.5 + 3 * r(), 70 + 60 * r()]);
      const [K, th] = kth.get(key)!;
      lines.push(`${c} ${a} ${b} ${K.toFixed(6)} ${th.toFixed(4)} ${(0.3 + r()).toFixed(4)} 9.9`);
    }
  }
  return lines.join('\n') + '\n';
};

interface Ent { K: number; th: number; rho: number; cut: number }

const parse = (file: string) => {
  const m = new Map<string, Ent>();
  for (const line of file.trim().split('\n')) {
    const t = line.trim().split(/\s+/);
    m.set(`${t[0]} ${t[1]} ${t[2]}`, { K: Number(t[3]), th: Number(t[4]) * Math.PI / 180, rho: Number(t[5]), cut: Number(t[6]) });
  }
  return m;
};

/** Brute-force energy of the documented formula; NULL types (not in `elems`) take no part. */
const bruteEnergy = (file: string, names: string[], x: Float64Array, type: Int32Array) => {
  const elems = names.map((e) => (e === 'NULL' ? null : e));
  const P = parse(file);
  const n = type.length;
  const d = (i: number, j: number) => [x[3 * j] - x[3 * i], x[3 * j + 1] - x[3 * i + 1], x[3 * j + 2] - x[3 * i + 2]];
  const norm = (u: number[]) => Math.hypot(u[0], u[1], u[2]);
  let e = 0;
  for (let c = 0; c < n; c++) {
    const ec = elems[type[c] - 1];
    if (ec === null) continue;
    const nbs: { j: number; e: string; u: number[]; r: number }[] = [];
    for (let j = 0; j < n; j++) {
      const ej = elems[type[j] - 1];
      if (j === c || ej === null) continue;
      const u = d(c, j), r = norm(u);
      const leg = P.get(`${ec} ${ej} ${ej}`)!;
      if (r <= leg.cut) nbs.push({ j, e: ej, u, r });
    }
    for (let a = 0; a < nbs.length; a++) for (let b = a + 1; b < nbs.length; b++) {
      const A = nbs[a], B = nbs[b];
      const p = P.get(`${ec} ${A.e} ${B.e}`)!;
      const cos = (A.u[0] * B.u[0] + A.u[1] * B.u[1] + A.u[2] * B.u[2]) / (A.r * B.r);
      const th = Math.acos(Math.min(1, Math.max(-1, cos)));
      const rhoA = P.get(`${ec} ${A.e} ${A.e}`)!.rho, rhoB = P.get(`${ec} ${B.e} ${B.e}`)!.rho;
      e += p.K * (th - p.th) ** 2 * Math.exp(-A.r / rhoA - B.r / rhoB);
    }
  }
  return e;
};

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
  return {
    x, f: new Float64Array(3 * n), type, q: new Float64Array(n), nlocal: n, nall: n,
    half: null, full: { inum: n, ilist: null, firstneigh, numneigh, neighbors },
    nb: { owner, gimage: new Int32Array(3 * n) },
    specialLJ: new Float64Array(4), specialCoul: new Float64Array(4), qqrd2e: 1,
    acc: newAccum(), eatom: null, vatom: null,
    s: null, geom: null,
  } as unknown as PairCompute;
};

const evalStyle = (style: PairNB3BScreened, x: Float64Array, type: Int32Array) => {
  const pc = computeInput(x, type);
  style.compute(pc);
  return { e: pc.acc.evdwl, f: pc.f };
};

const fdForces = (style: PairNB3BScreened, x: Float64Array, type: Int32Array, h = 1e-6) => {
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

const setup = (file: string, mapping: string[]) => {
  const pair = new PairNB3BScreened();
  const ctx = ctxWith({ 'pot.nb3b.screened': file });
  pair.allocate(mapping.length);
  pair.coeff(['*', '*', 'pot.nb3b.screened', ...mapping], ctx);
  pair.init(ctx);
  return pair;
};

describe('pair_style nb3b/screened', () => {
  it('runs: no settings arguments, and the style name is nb3b/screened', () => {
    const pair = new PairNB3BScreened();
    expect(pair.name).toBe('nb3b/screened');
    expect(() => pair.settings([], ctxWith())).not.toThrow();
    expect(() => pair.settings(['2.0'], ctxWith())).toThrow(/takes no arguments/);
  });

  it('energy equals the brute-force screened sum, with a NULL type taking no part', () => {
    const file = potentialFile(7);
    const mapping = ['P', 'NULL', 'O'];
    const type = Int32Array.from([1, 2, 3, 3, 1, 2, 3]);
    const x = randomPositions(7, 4.5, 11);
    const pair = setup(file, mapping);
    const e = evalStyle(pair, x, type).e;
    expect(e).toBeCloseTo(bruteEnergy(file, mapping, x, type), 10);
  });

  it('forces are minus the gradient of the energy (screening terms included)', () => {
    const file = potentialFile(19);
    const mapping = ['P', 'O', 'Si'];
    const type = Int32Array.from([1, 2, 3, 2, 1, 3]);
    const x = randomPositions(6, 4.0, 23);
    const pair = setup(file, mapping);
    const { f } = evalStyle(pair, x, type);
    const fd = fdForces(pair, x, type);
    for (let k = 0; k < 3 * 6; k++) expect(Math.abs(f[k] - fd[k])).toBeLessThanOrEqual(1e-5 * (1 + Math.abs(fd[k])));
  });

  it('the cutoff of an (i j j) leg is kept while r <= cut, and the (i j k) cutoff is not used', () => {
    // center O (type 2), neighbours P at 1.5 and P at 1.5 with the angle 100 deg; (O P P) cut 1.5 keeps both
    const file = [
      'O P P 10 100 1.0 1.5', 'O P O 10 100 1.0 0.5', 'O O P 10 100 1.0 0.5', 'O O O 0 0 1.0 0.5',
      'P P P 0 0 1.0 0', 'P P O 0 0 1.0 0', 'P O P 0 0 1.0 0', 'P O O 0 0 1.0 0',
      'O P Si 0 0 1 0', 'O Si P 0 0 1 0', 'O Si Si 0 0 1 0', 'O O Si 0 0 1 0', 'O Si O 0 0 1 0',
      'P P Si 0 0 1 0', 'P Si P 0 0 1 0', 'P Si Si 0 0 1 0', 'P Si O 0 0 1 0', 'P O Si 0 0 1 0',
      'Si P P 0 0 1 0', 'Si P Si 0 0 1 0', 'Si P O 0 0 1 0', 'Si Si P 0 0 1 0', 'Si Si Si 0 0 1 0',
      'Si Si O 0 0 1 0', 'Si O P 0 0 1 0', 'Si O Si 0 0 1 0', 'Si O O 0 0 1 0',
    ].join('\n') + '\n';
    const mapping = ['P', 'O', 'Si'];
    const r = 1.5, th = (100 * Math.PI) / 180;
    const x = Float64Array.from([0, 0, 0, r, 0, 0, r * Math.cos(th), r * Math.sin(th), 0]);
    const type = Int32Array.from([2, 1, 1]);
    const pair = setup(file, mapping);
    const e = evalStyle(pair, x, type).e;
    // the angle is theta0 = 100 deg here, so the energy is zero
    expect(e).toBeCloseTo(0, 12);
    const th2 = (110 * Math.PI) / 180;
    const x2 = Float64Array.from([0, 0, 0, r, 0, 0, r * Math.cos(th2), r * Math.sin(th2), 0]);
    const ref = 10 * (th2 - (100 * Math.PI) / 180) ** 2 * Math.exp(-2 * r);
    expect(evalStyle(pair, x2, type).e).toBeCloseTo(ref, 12);
    // the same geometry with r just above the leg cutoff drops the term
    const rOut = 1.5000001;
    const x3 = Float64Array.from([0, 0, 0, rOut, 0, 0, rOut * Math.cos(th2), rOut * Math.sin(th2), 0]);
    expect(evalStyle(pair, x3, type).e).toBe(0);
  });

  it('argument and file errors name the problem', () => {
    const good = potentialFile(3);
    const ctx = ctxWith({ 'good.nb3b.screened': good, 'six.nb3b.screened': good.replace(/^(\S+ \S+ \S+ \S+ \S+) \S+ (\S+)$/gm, '$1 $2') });
    const pair = new PairNB3BScreened();
    pair.allocate(3);
    expect(() => pair.coeff(['1', '1', 'good.nb3b.screened', 'P', 'O', 'Si'], ctx)).toThrow(/first 2 arguments/);
    expect(() => pair.coeff(['*', '*', 'good.nb3b.screened', 'P', 'O'], ctx)).toThrow(/one element name per atom type/);
    expect(() => pair.coeff(['*', '*', 'six.nb3b.screened', 'P', 'O', 'Si'], ctx)).toThrow(/cutoff of the nb3b file entry|got 6/);
    expect(() => pair.coeff(['*', '*', 'short.nb3b.screened', 'P', 'O', 'Si'], ctxWith({ 'short.nb3b.screened': 'P P P 1 100 0.8\n' })))
      .toThrow(/expected 'element1 element2 element3 K theta0 rho cutoff' \(7 values\), got 6/);
    const bad = good.split('\n').map((l) => (l.startsWith('O P P') ? 'O P P 1 100 0 3.0' : l)).join('\n');
    const ctx2 = ctxWith({ 'rho0.nb3b.screened': bad, 'neg.nb3b.screened': good.split('\n').map((l) => (l.startsWith('O P P') ? 'O P P 1 100 1.0 -3.0' : l)).join('\n') });
    const p2 = new PairNB3BScreened();
    p2.allocate(3);
    expect(() => p2.coeff(['*', '*', 'rho0.nb3b.screened', 'P', 'O', 'Si'], ctx2)).toThrow(/rho must be > 0/);
    const p3 = new PairNB3BScreened();
    p3.allocate(3);
    expect(() => p3.coeff(['*', '*', 'neg.nb3b.screened', 'P', 'O', 'Si'], ctx2)).toThrow(/cutoff must be >= 0/);
    // missing combination
    const partial = 'P P P 0 0 1 0\nO O O 0 0 1 3\nO P P 1 100 1 3\n';
    const p4 = new PairNB3BScreened();
    p4.allocate(2);
    expect(() => p4.coeff(['*', '*', 'partial.nb3b.screened', 'P', 'O'], ctxWith({ 'partial.nb3b.screened': partial }))).toThrow(/missing an entry for/);
  });

  it('asymmetric K or theta0 between (i j k) and (i k j) is refused, since native does not follow the parameters there', () => {
    const file = potentialFile(5).split('\n').map((l) => (l.startsWith('O P O ') ? 'O P O 9.0 100 1.0 3.0' : l)).join('\n');
    const pair = new PairNB3BScreened();
    pair.allocate(2);
    expect(() => pair.coeff(['*', '*', 'asym.nb3b.screened', 'P', 'O'], ctxWith({ 'asym.nb3b.screened': file })))
      .toThrow(/different K or theta0/);
  });
});
