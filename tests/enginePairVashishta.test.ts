import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { PairVashishta } from '../src/engine/force/pair/vashishta';
import { StyleError, type Pair, type StyleContext } from '../src/engine/force/types';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Unit checks for pair_style vashishta, written against the documented
 * formulas (docs.lammps.org/pair_vashishta.html):
 *
 *   U_ij^(2)(r) = H/r^eta + Zi Zj exp(-r/lambda1)/r
 *                 - D exp(-r/lambda4)/r^4 - W/r^6,   r < rc
 *   U_ijk^(3)   = B [cos t - cos t0]^2 / (1 + C [cos t - cos t0]^2)
 *                 exp(gamma_ij/(r_ij - r0_ij)) exp(gamma_ik/(r_ik - r0_ik))
 *
 * with "the twobody terms ... shifted and tilted by a linear function so that
 * the energy and force are both zero at r_c".  The Coulomb prefactor for the
 * electron-charge units is the style's own constant QQ (see the header of
 * src/engine/force/pair/vashishta.ts).  The expectations recompute these
 * sums independently in the test.  Native LAMMPS agreement (energy,
 * pressure and the final per-atom state) is covered by tests/oracle/
 * w5vash_si.in and w5vash_six.in.
 */

const QQ = 14.399645; // metal-unit qqr2e, as in the style (see src/engine/units.ts)

// synthetic two-element parameter set (magnitudes in the oracle's spirit)
const P = {
  H: 23.67, eta: 7.0, Zi: 0.4804, Zj: 0.4804, lam1: 5.0,
  D: 1.1, lam4: 3.0, W: 0.4, rc: 7.35,
  B: 9.003, gam: 1.0, r0: 2.9, C: 5.09, costheta0: -0.333333333333,
};

const entry = (e1: string, e2: string, e3: string, over: Partial<typeof P> = {}): string => {
  const p = { ...P, ...over };
  return `${e1} ${e2} ${e3} ${p.H} ${p.eta} ${p.Zi} ${p.Zj} ${p.lam1} ${p.D} ${p.lam4} ${p.W} ${p.rc} ${p.B} ${p.gam} ${p.r0} ${p.C} ${p.costheta0}`;
};

const fileContent = [
  '# tiny synthetic Vashishta file',
  entry('E', 'E', 'E'),
  entry('E', 'F', 'F', { H: 471.74, eta: 9.0, Zi: 0.4804, Zj: -0.4804, rc: 7.35, r0: 2.9 }),
  entry('F', 'E', 'E', { H: 471.74, eta: 9.0, Zi: -0.4804, Zj: 0.4804, rc: 7.35, r0: 2.9 }),
  entry('E', 'E', 'F', { H: 0, eta: 0, Zi: 0, Zj: 0, lam1: 0, D: 0, lam4: 0, W: 0, rc: 0 }),
  entry('E', 'F', 'E', { H: 0, eta: 0, Zi: 0, Zj: 0, lam1: 0, D: 0, lam4: 0, W: 0, rc: 0 }),
  entry('F', 'E', 'F', { H: 0, eta: 0, Zi: 0, Zj: 0, lam1: 0, D: 0, lam4: 0, W: 0, rc: 0, B: 7.5 }),
  entry('F', 'F', 'E', { H: 0, eta: 0, Zi: 0, Zj: 0, lam1: 0, D: 0, lam4: 0, W: 0, rc: 0, B: 7.5 }),
  entry('F', 'F', 'F', { H: 440.0, Zi: -0.4804, Zj: -0.4804, B: 0 }),
].join('\n');

const u2 = (r: number, p: Partial<typeof P> = {}): number => {
  const q = { ...P, ...p };
  return (
    q.H / Math.pow(r, q.eta) + QQ * q.Zi * q.Zj * Math.exp(-r / q.lam1) / r -
    q.D / Math.pow(r, 4) * Math.exp(-r / q.lam4) - q.W / Math.pow(r, 6)
  );
};
const du2 = (r: number, p: Partial<typeof P> = {}): number => {
  const q = { ...P, ...p };
  return (
    -q.eta * q.H / Math.pow(r, q.eta + 1) -
    QQ * q.Zi * q.Zj * Math.exp(-r / q.lam1) * (1 / (r * r) + 1 / (q.lam1 * r)) +
    q.D * Math.exp(-r / q.lam4) * (4 / Math.pow(r, 5) + 1 / (q.lam4 * Math.pow(r, 4))) +
    6 * q.W / Math.pow(r, 7)
  );
};
/** Shifted and tilted two-body: energy and force are zero at rc. */
const u2s = (r: number, p: Partial<typeof P> = {}): number => {
  const q = { ...P, ...p };
  return u2(r, q) - u2(q.rc, q) - (r - q.rc) * du2(q.rc, q);
};
const u3 = (r1: number, r2: number, cosT: number, b = P.B, c = P.C): number => {
  const dl = cosT - P.costheta0;
  const es1 = Math.exp(P.gam / (r1 - P.r0));
  const es2 = Math.exp(P.gam / (r2 - P.r0));
  return b * ((dl * dl) / (1 + c * dl * dl)) * es1 * es2;
};

const ctx: StyleContext = { s: null, readFile: () => '', log: () => {} };

const make = (coeffs: string[], files: Record<string, string> = {}, ntypes = 1): Pair => {
  const s = new PairVashishta();
  s.settings([], ctx);
  s.allocate(ntypes);
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

const close = (a: number, b: number, rel = 1e-12, abs = 1e-12): boolean =>
  Math.abs(a - b) <= abs + rel * Math.max(Math.abs(a), Math.abs(b));

const runSession = async (script: string, files: Record<string, string>): Promise<ThermoRow[]> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  await session.execute(script);
  return events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
};

const script = (atoms: string, coeff = '* * tiny.vashishta E'): string => `
units metal
atom_style atomic
region box block 0 30 0 30 0 30
create_box 1 box
${atoms}
mass * 1.0
pair_style vashishta
pair_coeff ${coeff}
thermo_style custom step pe
run 0
`;

describe('vashishta: documented shifted/tilted two-body term on a dimer', () => {
  it('pe equals U2(r) shifted and tilted at rc, and is zero at rc', async () => {
    const r = 2.3516;
    const rows = await runSession(
      script('create_atoms 1 single 2 2 2\ncreate_atoms 1 single 4.3516 2 2'),
      { 'tiny.vashishta': fileContent },
    );
    expect(rows.length).toBe(1);
    expect(close(rows[0].pe, u2s(r))).toBe(true);
    // energy and force are zero at rc: a pair sitting exactly at rc contributes nothing
    const atRc = await runSession(
      script('create_atoms 1 single 2 2 2\ncreate_atoms 1 single 9.35 2 2'),
      { 'tiny.vashishta': fileContent },
    );
    expect(Math.abs(atRc[0].pe)).toBeLessThan(1e-12);
  });

  it('pe is zero beyond rc', async () => {
    const rows = await runSession(
      script('create_atoms 1 single 2 2 2\ncreate_atoms 1 single 9.4 2 2'),
      { 'tiny.vashishta': fileContent },
    );
    expect(Math.abs(rows[0].pe)).toBe(0);
  });
});

describe('vashishta: documented three-body term on a three-atom cluster', () => {
  it('pe equals the independent sum of shifted U2 pairs plus U3 for the center', async () => {
    // center at origin-ish, two neighbors at different radii and a non-tetrahedral angle
    const pos: [number, number, number][] = [[4, 4, 4], [6.2, 4, 4], [4, 6.1, 4.3]];
    const atoms = pos.map((p) => `create_atoms 1 single ${p[0]} ${p[1]} ${p[2]}`).join('\n');
    const rows = await runSession(script(atoms), { 'tiny.vashishta': fileContent });
    const d = (i: number, j: number): number =>
      Math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1], pos[i][2] - pos[j][2]);
    const r1 = d(0, 1);
    const r2 = d(0, 2);
    const d12 = [pos[1][0] - pos[0][0], pos[1][1] - pos[0][1], pos[1][2] - pos[0][2]];
    const d13 = [pos[2][0] - pos[0][0], pos[2][1] - pos[0][1], pos[2][2] - pos[0][2]];
    const cosT =
      (d12[0] * d13[0] + d12[1] * d13[1] + d12[2] * d13[2]) / (r1 * r2);
    const r12 = d(1, 2);
    const want = u2s(r1) + u2s(r2) + u2s(r12) + u3(r1, r2, cosT);
    expect(u3(r1, r2, cosT)).not.toBe(0);
    expect(close(rows[0].pe, want)).toBe(true);
  });

  it('a three-body entry with B = 0 contributes no angle energy', async () => {
    // same geometry, but force the (E,E,E) entry to B = 0
    const noB = fileContent.replace(entry('E', 'E', 'E'), entry('E', 'E', 'E', { B: 0 }));
    const pos: [number, number, number][] = [[4, 4, 4], [6.2, 4, 4], [4, 6.1, 4.3]];
    const atoms = pos.map((p) => `create_atoms 1 single ${p[0]} ${p[1]} ${p[2]}`).join('\n');
    const rows = await runSession(script(atoms), { 'tiny.vashishta': noB });
    const d = (i: number, j: number): number =>
      Math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1], pos[i][2] - pos[j][2]);
    expect(close(rows[0].pe, u2s(d(0, 1)) + u2s(d(0, 2)) + u2s(d(1, 2)))).toBe(true);
  });
});

describe('vashishta: coefficients, mapping and cutoffs', () => {
  it('cutoff covers the two-body rc and both legs r0; NULL mapping turns the type off', () => {
    const s = make(['* * tiny.vashishta E'], { 'tiny.vashishta': fileContent });
    expect(s.cut[1 * 2 + 1]).toBe(P.rc);
    const two = make(['* * tiny.vashishta E F'], { 'tiny.vashishta': fileContent }, 2);
    // Si-X style pair: two-body rc 7.35 and both legs r0 2.9 -> 7.35
    expect(two.cut[1 * 3 + 2]).toBe(7.35);
    const nulled = make(['* * tiny.vashishta NULL'], { 'tiny.vashishta': fileContent });
    expect(nulled.initOne(1, 1)).toBe(0);
  });

  it('errors: non-* * coefficients, missing box, wrong element count, unknown element, wrong file', () => {
    const s = new PairVashishta();
    s.settings([], ctx);
    s.allocate(1);
    expect(() => s.coeff(['1', '1', 'tiny.vashishta', 'E'], ctx)).toThrow(StyleError);
    expect(() => make(['* * tiny.vashishta'], { 'tiny.vashishta': fileContent })).toThrow(StyleError);
    expect(() => make(['* * tiny.vashishta Q'], { 'tiny.vashishta': fileContent })).toThrow(StyleError);
    expect(() => make(['* * other.vashishta E'], { 'tiny.vashishta': fileContent })).toThrow(StyleError);
  });

  it('errors: bad pair_style arguments and malformed file lines', () => {
    const s = new PairVashishta();
    expect(() => s.settings(['1.0'], ctx)).toThrow(StyleError);
    const short = fileContent.split('\n').map((l, k) => (k === 1 ? l.split(' ').slice(1).join(' ') : l)).join('\n');
    expect(() => make(['* * tiny.vashishta E'], { 'tiny.vashishta': `${short}\n` })).toThrow(StyleError);
    const bad = fileContent.replace('23.67', 'x23.67');
    expect(() => make(['* * tiny.vashishta E'], { 'tiny.vashishta': `${bad}\n` })).toThrow();
    expect(() => make(['* * tiny.vashishta E'], { 'tiny.vashishta': '# no entries\n' })).toThrow(StyleError);
  });

  it('errors: missing three-body entry and pair_modify shift', () => {
    const no3b = fileContent.split('\n').filter((l) => !l.startsWith('F E E')).join('\n');
    expect(() => make(['* * tiny.vashishta E F'], { 'tiny.vashishta': `${no3b}\n` }, 2)).toThrow(StyleError);
    const s = new PairVashishta();
    s.shift = true;
    expect(() => s.initStyle(ctx)).toThrow(StyleError);
  });
});

describe('vashishta: force = -dE/dx by central differences (two elements, three atoms)', () => {
  // center type 1 (E); neighbors type 2 (F) and type 1 (E): the three-body term uses
  // the EFE / EEF / EFF / EEE entries and the two-body term both (E,F) orderings
  const pos: [number, number, number][] = [[4, 4, 4], [6.2, 4, 4], [4, 6.1, 4.3]];
  const types = [1, 2, 1];

  const probe = async (p: [number, number, number][]) => {
    const rows: ThermoRow[] = [];
    const files = new Map<string, string>();
    const session = new Session({
      emit: (e: EngineEvent) => { if (e.kind === 'thermo') rows.push(e.row); },
      writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
    });
    const atoms = p.map((q, k) => `create_atoms ${types[k]} single ${q[0]} ${q[1]} ${q[2]}`).join('\n');
    await session.addFile('tiny.vashishta', fileContent);
    await session.execute(`
units metal
atom_style atomic
region box block 0 30 0 30 0 30
create_box 2 box
${atoms}
mass * 1.0
pair_style vashishta
pair_coeff * * tiny.vashishta E F
thermo_style custom step pe
thermo_modify format float %.17g
run 0
write_dump all custom fd.dump id fx fy fz modify format float %.17g sort id
`);
    const lines = (files.get('fd.dump') ?? '').trim().split('\n');
    const k0 = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
    const cols = lines[k0].split(/\s+/).slice(2);
    const forces = lines.slice(k0 + 1).map((l) => l.trim().split(/\s+/).map(Number)).map((v) => ({
      fx: v[cols.indexOf('fx')], fy: v[cols.indexOf('fy')], fz: v[cols.indexOf('fz')],
    }));
    return { pe: rows[0].pe, forces };
  };

  it('forces match -dE/dx for every atom and component', async () => {
    const h = 1e-5;
    const base = await probe(pos);
    expect(base.forces.length).toBe(3);
    let maxAbs = 0;
    for (let a = 0; a < 3; a++) {
      const comps = ['fx', 'fy', 'fz'] as const;
      for (let c = 0; c < 3; c++) {
        const plus = pos.map((q) => [...q]) as [number, number, number][];
        const minus = pos.map((q) => [...q]) as [number, number, number][];
        plus[a][c] += h;
        minus[a][c] -= h;
        const ep = (await probe(plus)).pe;
        const em = (await probe(minus)).pe;
        const fd = -(ep - em) / (2 * h);
        const f = base.forces[a][comps[c]];
        maxAbs = Math.max(maxAbs, Math.abs(f));
        expect(Math.abs(f - fd)).toBeLessThan(1e-6 * Math.max(1, Math.abs(fd)));
      }
    }
    // the configuration is not trivially force-free
    expect(maxAbs).toBeGreaterThan(1e-3);
  });
});
