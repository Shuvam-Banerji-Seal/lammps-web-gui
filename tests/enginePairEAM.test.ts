import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { PairEAM, PairEAMAlloy, PairEAMFS } from '../src/engine/force/pair/eam';
import { StyleError, type Pair, type StyleContext } from '../src/engine/force/types';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Unit checks for the EAM pair styles, written against the documented
 * formulas (docs.lammps.org/pair_eam.html): the pair term of style eam is
 * r*phi = 27.2*0.529*Z_i(r)*Z_j(r), evaluated exactly at table grid points
 * by synthetic analytic potentials; style eam/alloy and eam/fs read setfl /
 * Finnis-Sinclair files whose r*phi arrays are checked the same way; a
 * two-atom run checks the density sum, the embedding energies and the
 * asymmetric rho_{alpha beta} indexing of eam/fs. Native LAMMPS agreement
 * is covered by the tests/oracle/w2eam_*.in cases.
 */

const ctx: StyleContext = { s: null, readFile: () => '', log: () => {} };

const NRHO = 200, DRHO = 0.05, NR = 300, DR = 0.01;
const CUTOFF = 2.5;

const num = (v: number): string => v.toExponential(16);

/** Tabulated values f(k*dx), five per line. */
const block = (f: (x: number) => number, n: number, dx: number): string => {
  const lines: string[] = [];
  for (let k = 0; k < n; k += 5) {
    const row: string[] = [];
    for (let t = 0; t < 5 && k + t < n; t++) row.push(num(f((k + t) * dx)));
    lines.push(row.join(' '));
  }
  return lines.join('\n');
};

// analytic element tables; every function is evaluated exactly at grid points in the expectations
const F_A = (rho: number) => -0.2 * rho + 0.01 * rho * rho;
const F_B = (rho: number) => -0.15 * rho + 0.005 * rho * rho;
const Z_A = (r: number) => 2.0 - 0.05 * r;
const Z_B = (r: number) => 1.5 - 0.03 * r;
const rho_A = (r: number) => 1.0 + 0.5 * r;
const rho_B = (r: number) => 1.1 + 0.5 * r;
const g_AA = (r: number) => 3 - 0.2 * r + 0.01 * r * r;
const g_AB = (r: number) => 2 + 0.1 * r;
const g_BB = (r: number) => 2.5 - 0.1 * r;
const RHO_FS: ((r: number) => number)[][] = [
  // fsRho[beta][alpha]: density of element beta at a site of element alpha
  [(r) => 1.0 + 0.5 * r, (r) => 0.9 + 0.5 * r],   // section A
  [(r) => 1.2 + 0.5 * r, (r) => 0.8 + 0.5 * r],   // section B
];

const funcfl = (name: string, mass: string, cutoff: number, F: (rho: number) => number, Z: (r: number) => number, rho: (r: number) => number): string =>
  [
    `${name} test element`,
    `29 ${mass} 3.615 FCC`,
    `${NRHO} ${num(DRHO)} ${NR} ${num(DR)} ${cutoff}`,
    block(F, NRHO, DRHO),
    block(Z, NR, DR),
    block(rho, NR, DR),
  ].join('\n');

const fileA = funcfl('A.eam', '63.55', CUTOFF, F_A, Z_A, rho_A);
const fileB = funcfl('B.eam', '58.69', 2.0, F_B, Z_B, rho_B);

const setfl = (fs: boolean): string => {
  const out = ['setfl test file A B', 'second comment line', '-'];
  out.push('2 A B');
  out.push(`${NRHO} ${num(DRHO)} ${NR} ${num(DR)} ${CUTOFF}`);
  out.push('10 63.55 3.615 FCC');
  out.push(block(F_A, NRHO, DRHO));
  if (!fs) out.push(block(rho_A, NR, DR));
  else {
    out.push(block(RHO_FS[0][0], NR, DR));
    out.push(block(RHO_FS[0][1], NR, DR));
  }
  out.push('11 58.69 3.52 FCC');
  out.push(block(F_B, NRHO, DRHO));
  if (!fs) out.push(block(rho_B, NR, DR));
  else {
    out.push(block(RHO_FS[1][0], NR, DR));
    out.push(block(RHO_FS[1][1], NR, DR));
  }
  out.push(block(g_AA, NR, DR));
  out.push(block(g_AB, NR, DR));
  out.push(block(g_BB, NR, DR));
  return out.join('\n');
};

const make = (s: Pair, coeffs: string[], files: Record<string, string> = {}): Pair => {
  s.settings([], ctx);
  s.allocate(2);
  const full: StyleContext = {
    s: null,
    log: () => {},
    readFile: (n) => {
      if (!(n in files)) throw new StyleError(`cannot open file ${n}`);
      return files[n];
    },
  };
  for (const c of coeffs) s.coeff(c.split(/\s+/), full);
  s.init(full);
  return s;
};

const close = (a: number, b: number, rel: number, abs: number) =>
  Math.abs(a - b) <= abs + rel * Math.max(Math.abs(a), Math.abs(b));

const runSession = async (script: string, files: Record<string, string>): Promise<ThermoRow[]> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  await session.execute(script);
  return events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
};

const dimerScript = (style: string, coeffs: string): string => `
units metal
atom_style atomic
region box block 0 10 0 10 0 10
create_box 2 box
create_atoms 1 single 2 2 2
create_atoms 2 single 3.2 2 2
mass * 1.0
pair_style ${style}
${coeffs}
thermo_style custom step pe
run 0
`;

describe('eam (funcfl): documented r*phi formula at table grid points', () => {
  it('pair energy 27.2*0.529*Zi(r)*Zj(r)/r and force = -dphi/dr for I=I and mixed I!=J', () => {
    const s = make(new PairEAM(), ['1 1 A.eam', '2 2 B.eam'], { 'A.eam': fileA, 'B.eam': fileB });
    for (const k of [30, 95, 160, 249]) {
      const r = k * DR;
      const self = s.single!(0, 0, 1, 1, r * r, 1, 1, 0, 0);
      expect(close(self.eng, (27.2 * 0.529 * Z_A(r) * Z_A(r)) / r, 1e-12, 1e-12)).toBe(true);
    }
    for (const k of [30, 95, 160, 220]) {
      const r = k * DR;
      const mixed = s.single!(0, 0, 1, 2, r * r, 1, 1, 0, 0);
      expect(close(mixed.eng, (27.2 * 0.529 * Z_A(r) * Z_B(r)) / r, 1e-12, 1e-12)).toBe(true);
    }
    const h = 1e-6;
    for (const k of [70, 130, 210]) {
      const r = k * DR;
      const em = s.single!(0, 0, 1, 1, (r - h) * (r - h), 1, 1, 0, 0).eng;
      const ep = s.single!(0, 0, 1, 1, (r + h) * (r + h), 1, 1, 0, 0).eng;
      const m = s.single!(0, 0, 1, 1, r * r, 1, 1, 0, 0);
      expect(close(m.fforce * r, -(ep - em) / (2 * h), 1e-5, 1e-7)).toBe(true);
    }
  });

  it('cutoffs come from the files; the mixed pair cutoff is the average', () => {
    const s = make(new PairEAM(), ['1 1 A.eam', '2 2 B.eam'], { 'A.eam': fileA, 'B.eam': fileB });
    expect(s.cut[1 * 3 + 1]).toBe(CUTOFF);
    expect(s.cut[2 * 3 + 2]).toBe(2.0);
    expect(close(s.cut[1 * 3 + 2], 0.5 * (CUTOFF + 2.0), 0, 1e-12)).toBe(true);
  });

  it('two-atom energy: embedding at rho of the neighbour element plus the mixed pair term', async () => {
    const d = 1.2;
    const rho1 = rho_B(d), rho2 = rho_A(d);
    const want = F_A(rho1) + F_B(rho2) + (27.2 * 0.529 * Z_A(d) * Z_B(d)) / d;
    const rows = await runSession(dimerScript('eam', 'pair_coeff 1 1 A.eam\npair_coeff 2 2 B.eam'), { 'A.eam': fileA, 'B.eam': fileB });
    expect(rows.length).toBe(1);
    expect(close(rows[0].pe, want, 1e-9, 1e-9)).toBe(true);
  });

  it('errors: pair_style arguments, short pair_coeff, unreadable file, unset pair', () => {
    const s = new PairEAM();
    expect(() => s.settings(['3.0'], ctx)).toThrow(StyleError);
    s.allocate(2);
    expect(() => s.coeff(['1', '1'], ctx)).toThrow(StyleError);
    expect(() => make(new PairEAM(), ['1 1 nope.eam'], {})).toThrow(StyleError);
    const bare = new PairEAM();
    bare.settings([], ctx);
    bare.allocate(2);
    expect(() => bare.init(ctx)).toThrow(StyleError);
  });
});

describe('eam/alloy (setfl): r*phi tables and element mapping', () => {
  it('pair energies from the r*phi arrays at grid points (i,i and mixed)', () => {
    const s = make(new PairEAMAlloy(), ['* * AB.eam.alloy A B'], { 'AB.eam.alloy': setfl(false) });
    for (const k of [40, 120, 200]) {
      const r = k * DR;
      expect(close(s.single!(0, 0, 1, 1, r * r, 1, 1, 0, 0).eng, g_AA(r) / r, 1e-12, 1e-12)).toBe(true);
      expect(close(s.single!(0, 0, 1, 2, r * r, 1, 1, 0, 0).eng, g_AB(r) / r, 1e-12, 1e-12)).toBe(true);
      expect(close(s.single!(0, 0, 2, 2, r * r, 1, 1, 0, 0).eng, g_BB(r) / r, 1e-12, 1e-12)).toBe(true);
    }
    expect(s.cut[1 * 3 + 2]).toBe(CUTOFF);
  });

  it('two-atom energy: F_A(rho_B(d)) + F_B(rho_A(d)) + phi_AB(d)', async () => {
    const d = 1.2;
    const want = F_A(rho_B(d)) + F_B(rho_A(d)) + g_AB(d) / d;
    const rows = await runSession(dimerScript('eam/alloy', 'pair_coeff * * AB.eam.alloy A B'), { 'AB.eam.alloy': setfl(false) });
    expect(rows.length).toBe(1);
    expect(close(rows[0].pe, want, 1e-9, 1e-9)).toBe(true);
  });

  it('NULL mapping leaves the type without interactions', () => {
    const s = make(new PairEAMAlloy(), ['* * AB.eam.alloy A NULL'], { 'AB.eam.alloy': setfl(false) });
    expect(s.initOne(2, 2)).toBe(0);
    expect(s.initOne(1, 2)).toBe(0);
    expect(s.initOne(1, 1)).toBe(CUTOFF);
  });

  it('errors: non-wildcard I J, wrong element count, unknown element', () => {
    const s = new PairEAMAlloy();
    s.allocate(2);
    expect(() => s.coeff(['1', '1', 'AB.eam.alloy', 'A', 'B'], ctx)).toThrow(StyleError);
    expect(() => make(new PairEAMAlloy(), ['* * AB.eam.alloy A'], { 'AB.eam.alloy': setfl(false) })).toThrow(StyleError);
    expect(() => make(new PairEAMAlloy(), ['* * AB.eam.alloy A X'], { 'AB.eam.alloy': setfl(false) })).toThrow(StyleError);
  });
});

describe('eam/fs (Finnis-Sinclair): asymmetric densities rho_{alpha beta}', () => {
  it('two-atom energy uses rho at site A from element B and at site B from element A', async () => {
    const d = 1.2;
    const rhoSiteA = RHO_FS[1][0](d);   // section B, array A: element B at site A
    const rhoSiteB = RHO_FS[0][1](d);   // section A, array B: element A at site B
    const want = F_A(rhoSiteA) + F_B(rhoSiteB) + g_AB(d) / d;
    const rows = await runSession(dimerScript('eam/fs', 'pair_coeff * * AB.eam.fs A B'), { 'AB.eam.fs': setfl(true) });
    expect(rows.length).toBe(1);
    expect(close(rows[0].pe, want, 1e-9, 1e-9)).toBe(true);
  });

  it('pair term matches the r*phi arrays; NULL mapping returns a zero cutoff', () => {
    const s = make(new PairEAMFS(), ['* * AB.eam.fs A B'], { 'AB.eam.fs': setfl(true) });
    const r = 120 * DR;
    expect(close(s.single!(0, 0, 1, 2, r * r, 1, 1, 0, 0).eng, g_AB(r) / r, 1e-12, 1e-12)).toBe(true);
    const nulled = make(new PairEAMFS(), ['* * AB.eam.fs NULL B'], { 'AB.eam.fs': setfl(true) });
    expect(nulled.initOne(1, 1)).toBe(0);
    expect(nulled.initOne(2, 2)).toBe(CUTOFF);
  });

  it('errors: wrong element count and unknown element', () => {
    expect(() => make(new PairEAMFS(), ['* * AB.eam.fs A'], { 'AB.eam.fs': setfl(true) })).toThrow(StyleError);
    expect(() => make(new PairEAMFS(), ['* * AB.eam.fs A Q'], { 'AB.eam.fs': setfl(true) })).toThrow(StyleError);
  });
});
