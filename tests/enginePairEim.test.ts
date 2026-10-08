import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import { PairEIM, parseEimFile } from '../src/engine/force/pair/eim';
import { erfcExact } from '../src/engine/force/erfc';
import { StyleError } from '../src/engine/force/types';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Unit checks for pair_style eim (docs.lammps.org/pair_eim.html, source
 * plans/lammps-docs/pair_eim.rst): the energy E = 1/2 sum phi + sum E_i with
 * q_i = sum eta_ji, sigma_i = sum q_j psi_ij, E_i = 1/2 q_i sigma_i; forces are
 * checked against central finite differences of that energy, a two-atom run
 * against the closed form phi - q^2 psi, and the pair_coeff rules against the
 * errors the doc describes. Native LAMMPS agreement is covered by the
 * tests/oracle/w14eim_*.in cases.
 */

const CASES = join(__dirname, 'oracle');
const TWO = readFileSync(join(CASES, 'w14eim_two.eim'), 'utf8');
const THREE = readFileSync(join(CASES, 'w14eim_three.eim'), 'utf8');

const runEvents = async (script: string, files: Record<string, string>): Promise<EngineEvent[]> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  await session.execute(script);
  return events;
};

const thermoRows = (events: EngineEvent[]): ThermoRow[] =>
  events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);

/** Runs a script and returns the per-atom forces (id order) from a final write_dump. */
const forcesOf = async (script: string, files: Record<string, string>): Promise<number[][]> => {
  const written = new Map<string, string>();
  const session = new Session({ emit: () => {}, writeFile: (n, t) => written.set(n, t) });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  await session.execute(`${script}\nwrite_dump all custom eim_forces.dump id fx fy fz modify format float %.17g sort id\n`);
  const lines = (written.get('eim_forces.dump') ?? '').trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  return lines.slice(k + 1).map((l) => l.trim().split(/\s+/).slice(1).map(Number));
};

const peOf = async (script: string, files: Record<string, string>): Promise<number> => {
  const rows = thermoRows(await runEvents(`${script}\nthermo_style custom step pe\nthermo_modify format float %.17g\nrun 0\n`, files));
  return rows[rows.length - 1].pe as number;
};

const close = (a: number, b: number, rel: number, abs: number) =>
  Math.abs(a - b) <= abs + rel * Math.max(Math.abs(a), Math.abs(b));

/** Five atoms of the three-element file (types 1..3) in a periodic 12 A box, all pairs inside the cutoffs. */
const CLUSTER = [
  { t: 1, x: [1.0, 1.0, 1.0] },
  { t: 2, x: [3.7, 1.4, 1.2] },
  { t: 3, x: [1.5, 3.9, 0.7] },
  { t: 1, x: [4.2, 4.1, 2.3] },
  { t: 2, x: [0.3, 3.1, 3.6] },
];

const clusterScript = (positions: number[][], pre: string[] = []): string => [
  'units metal',
  'atom_style atomic',
  'boundary p p p',
  'region box block 0 12 0 12 0 12',
  'create_box 3 box',
  ...CLUSTER.map((a, i) => `create_atoms ${a.t} single ${positions[i][0]} ${positions[i][1]} ${positions[i][2]} units box`),
  ...pre,
  'pair_style eim',
  'pair_coeff * * Qa Qb Qc w14eim_three.eim Qa Qb Qc',
].join('\n');

describe('pair_style eim: file parsing', () => {
  it('reads global, element and pair lines, continuation lines and comments', () => {
    const text = [
      '# comment',
      'global: 2.0 -1.5 1.7 ### trailing comment',
      'element: Qa 11 22.99 0.90 1.30 1.30 -1.10 0.0',
      'element: Qb 17 35.45 3.10 1.40 1.40 -1.30 0.0',
      'pair: Qb Qa 6.0 6.0 -0.45 3.8 6.8 3.2 4.5 0.8 2.5 &',
      '   5.5 0.5 0.9 2.5 1.0',
    ].join('\n');
    const f = parseEimFile(text, 'inline.eim');
    expect(f.G2).toBe(-1.5);
    expect(f.G3).toBe(1.7);
    expect(f.elements.get('Qb')?.chi).toBe(3.1);
    // the pair line is found in both element orders
    expect(f.pairs.get('Qa Qb')?.rcEta).toBe(4.5);
    expect(f.pairs.get('Qb Qa')?.rcPsi).toBe(5.5);
  });

  it('rejects malformed files: missing global line, bad p, unknown keyword, wrong count', () => {
    expect(() => parseEimFile('element: Qa 11 22.99 0.90 1.30 1.30 -1.10 0.0', 'x.eim')).toThrow(StyleError);
    const base = 'global: 2.0 -1.5 1.7\nelement: Qa 11 22.99 0.90 1.30 1.30 -1.10 0.0\n';
    expect(() => parseEimFile(`${base}pair: Qa Qa 5.5 5.5 -0.2 3.5 7.0 3.0 0.0 0.0 2.0 5.0 0.3 0.6 2.0 3.0\n`, 'x.eim')).toThrow(StyleError);
    expect(() => parseEimFile(`${base}pair: Qa Qa 5.5 5.5 -0.2 3.5\n`, 'x.eim')).toThrow(StyleError);
    expect(() => parseEimFile(`${base}sphere: Qa\n`, 'x.eim')).toThrow(StyleError);
  });
});

describe('pair_style eim: energy and forces', () => {
  it('two-atom energy is phi(r) - q^2 psi(r) from the documented functions', async () => {
    // Qa (chi 0.9) and Qb (chi 3.1) with the Qa Qb line of w14eim_two.eim; the cutoff function uses
    // the file's global values G2 = -1.5 and G3 = 1.7 (fc = 1 at r_p and 0 at r_c)
    const G2 = -1.5, G3 = 1.7;
    const fc = (r: number, rp: number, rc: number) => {
      const arg = G2 + ((G3 - G2) * (r - rp)) / (rc - rp);
      return (erfcExact(arg) - erfcExact(G3)) / (erfcExact(G2) - erfcExact(G3));
    };
    const Eb = -0.45, re = 3.8, al = 6.8, be = 3.2;
    const chiA = 0.9, chiB = 3.1;
    for (const r of [2.6, 3.2, 3.9, 4.3]) {
      const phi = ((Eb * be) / (be - al)) * Math.exp((-al * (r - re)) / re) - ((Eb * al) / (be - al)) * Math.exp((-be * (r - re)) / re);
      // each term is zero beyond its own cutoff: r_c,phi = 5.41, r_c,eta = 3.5, r_c,psi = 4.6
      const pair = r < 5.41 ? phi * fc(r, re, 5.41) : 0;
      const q = r < 3.5 ? 0.8 * (chiB - chiA) * fc(r, 2.5, 3.5) : 0;
      const psi = r < 4.6 ? 0.5 * Math.exp(-0.9 * r) * fc(r, 2.5, 4.6) : 0;
      const want = pair - q * q * psi;
      const script = `units metal\natom_style atomic\nregion box block -20 20 -20 20 -20 20\ncreate_box 2 box\n` +
        `create_atoms 1 single 0 0 0 units box\ncreate_atoms 2 single ${r} 0 0 units box\n` +
        'pair_style eim\npair_coeff * * Qa Qb w14eim_two.eim Qa Qb\n';
      const pe = await peOf(script, { 'w14eim_two.eim': TWO });
      expect(close(pe, want, 1e-10, 1e-12), `r=${r}`).toBe(true);
    }
  });

  it('the two r_c,phi values gate the two phi terms: first for the beta term, second for the alpha term (measured with native LAMMPS)', async () => {
    // "redundant for historical reasons" is what the doc says of the second value; the native energies of
    // dimers with (first, second) = (5.6, 4.0) and (4.0, 5.6) follow this split form for p = 1 and p = 2
    const G2 = -1.5, G3 = 1.7;
    const fc = (r: number, rp: number, rc: number) => {
      const arg = G2 + ((G3 - G2) * (r - rp)) / (rc - rp);
      return (erfcExact(arg) - erfcExact(G3)) / (erfcExact(G2) - erfcExact(G3));
    };
    const Eb = -0.3, re = 3.4, al = 7.2, be = 3.1;
    const pre = (Eb * be) / (be - al), pre2 = (Eb * al) / (be - al);
    for (const [first, second] of [[5.6, 4.0], [4.0, 5.6]]) {
      for (const p of [1, 2]) {
        const file = `global: 2.0 -1.5 1.7\nelement: Qa 11 22.99 0.90 1.30 1.30 -1.10 0.0\npair: Qa Qa ${first} ${second} ${Eb} ${re} ${al} ${be} 0.0 0.0 2.0 4.6 0.30 0.60 2.0 ${p}\n`;
        for (const r of [3.0, 4.5]) {
          const ta = p === 1 ? pre * Math.exp((-al * (r - re)) / re) : pre * Math.pow(re / r, al);
          const tb = p === 1 ? pre2 * Math.exp((-be * (r - re)) / re) : pre2 * Math.pow(re / r, be);
          const want = (r < second ? ta * fc(r, re, second) : 0) - (r < first ? tb * fc(r, re, first) : 0);
          const script = `units metal\natom_style atomic\nregion box block -20 20 -20 20 -20 20\ncreate_box 1 box\n` +
            `create_atoms 1 single 0 0 0 units box\ncreate_atoms 1 single ${r} 0 0 units box\n` +
            'pair_style eim\npair_coeff * * Qa single.eim Qa\n';
          const pe = await peOf(script, { 'single.eim': file });
          expect(close(pe, want, 1e-10, 1e-12), `first ${first} second ${second} p ${p} r ${r}`).toBe(true);
        }
      }
    }
  });

  it('forces equal central finite differences of the energy (three elements, periodic)', async () => {
    const base = CLUSTER.map((a) => [...a.x]);
    const f = await forcesOf(clusterScript(base), { 'w14eim_three.eim': THREE });
    const h = 1e-5;
    for (let i = 0; i < CLUSTER.length; i++) {
      for (let c = 0; c < 3; c++) {
        const plus = base.map((p) => [...p]), minus = base.map((p) => [...p]);
        plus[i][c] += h;
        minus[i][c] -= h;
        const ep = await peOf(clusterScript(plus), { 'w14eim_three.eim': THREE });
        const em = await peOf(clusterScript(minus), { 'w14eim_three.eim': THREE });
        const fd = -(ep - em) / (2 * h);
        expect(close(f[i][c], fd, 1e-6, 1e-7), `atom ${i + 1} component ${c}: force ${f[i][c]} vs fd ${fd}`).toBe(true);
      }
    }
  });

  it('forces sum to zero (Newton third law over owned and periodic images)', async () => {
    const f = await forcesOf(clusterScript(CLUSTER.map((a) => [...a.x])), { 'w14eim_three.eim': THREE });
    for (let c = 0; c < 3; c++) {
      let s = 0;
      for (const row of f) s += row[c];
      expect(Math.abs(s)).toBeLessThan(1e-10);
    }
  });

  it('the energy does not depend on the unit style (the formulas are unitless; measured with native LAMMPS)', async () => {
    const metal = await peOf(clusterScript(CLUSTER.map((a) => [...a.x])), { 'w14eim_three.eim': THREE });
    const real = await peOf(clusterScript(CLUSTER.map((a) => [...a.x])).replace('units metal', 'units real'), { 'w14eim_three.eim': THREE });
    expect(real).toBe(metal);
  });

  it('the global cation/anion value G1 does not change the energy (measured with native LAMMPS)', async () => {
    const a = await peOf(clusterScript(CLUSTER.map((a) => [...a.x])), { 'w14eim_three.eim': THREE });
    const b = await peOf(clusterScript(CLUSTER.map((a) => [...a.x])), { 'w14eim_three.eim': THREE.replace('global: 2.0', 'global: 0.5') });
    expect(b).toBe(a);
  });

  it('file masses replace an earlier mass command (measured with native LAMMPS)', async () => {
    const ke = async (pre: string[]) => {
      const script = `${clusterScript(CLUSTER.map((a) => [...a.x]), pre)}\nvelocity all set 1.0 0.0 0.0 units box\nthermo_style custom step ke\nthermo_modify format float %.17g\nrun 0\n`;
      const rows = thermoRows(await runEvents(script, { 'w14eim_three.eim': THREE }));
      return rows[rows.length - 1].ke as number;
    };
    const withCmd = await ke(['mass * 5.0']);
    const without = await ke([]);
    expect(withCmd).toBe(without);
  });
});

describe('pair_style eim: pair_coeff and settings rules', () => {
  const dimer = (pairCoeff: string, types = 2): string => [
    'units metal', 'atom_style atomic', 'region box block -20 20 -20 20 -20 20',
    `create_box ${types} box`, 'create_atoms 1 single 0 0 0 units box', 'create_atoms 2 single 3.0 0 0 units box',
    'pair_style eim', pairCoeff, 'run 0',
  ].join('\n');

  it('a valid mapping (type 2 repeats element Qb of the file) runs', async () => {
    const rows = thermoRows(await runEvents(dimer('pair_coeff * * Qa Qb w14eim_two.eim Qa Qb'), { 'w14eim_two.eim': TWO }));
    expect(rows.length).toBe(1);
    expect(Number.isFinite(rows[0].etotal as number)).toBe(true);
  });

  it('refuses pair_style arguments', () => {
    const s = new PairEIM();
    expect(() => s.settings(['1.0'], { s: null, readFile: () => '', log: () => {} })).toThrow(StyleError);
  });

  it('refuses a mapped element that the file does not define', async () => {
    await expect(runEvents(dimer('pair_coeff * * Qa Zz w14eim_two.eim Qa Zz'), { 'w14eim_two.eim': TWO })).rejects.toThrow(/Zz/);
  });

  it('refuses a wrong number of element names before the file', async () => {
    await expect(runEvents(dimer('pair_coeff * * Qa Qb Qc w14eim_two.eim Qa Qb'), { 'w14eim_two.eim': TWO })).rejects.toThrow(/expected 2 element names/);
  });

  it('refuses a missing potential file', async () => {
    await expect(runEvents(dimer('pair_coeff * * Qa Qb nofile.eim Qa Qb'), {})).rejects.toThrow();
  });

  it('refuses a pair that the file does not define', async () => {
    const noPair = TWO.split('\n').filter((l) => !l.startsWith('pair: Qa Qb')).join('\n');
    await expect(runEvents(dimer('pair_coeff * * Qa Qb w14eim_two.eim Qa Qb'), { 'w14eim_two.eim': noPair })).rejects.toThrow(/Qa, Qb/);
  });

  it('refuses a pair_coeff that does not start with * *', async () => {
    await expect(runEvents(dimer('pair_coeff 1 1 Qa Qb w14eim_two.eim Qa Qb'), { 'w14eim_two.eim': TWO })).rejects.toThrow(/must start with \* \*/);
  });

  it('refuses a NULL mapping for a plain eim pair (the doc uses NULL for hybrid)', async () => {
    await expect(runEvents(dimer('mass 2 1.0\npair_coeff * * Qa w14eim_two.eim Qa NULL'), { 'w14eim_two.eim': TWO })).rejects.toThrow(/pair coeffs are not set/);
  });

  it('refuses pair_modify shift and tail', async () => {
    await expect(runEvents(`${dimer('pair_coeff * * Qa Qb w14eim_two.eim Qa Qb')}\npair_modify shift yes\nrun 0\n`, { 'w14eim_two.eim': TWO })).rejects.toThrow(/shift/);
  });
});
