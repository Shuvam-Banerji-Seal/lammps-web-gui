import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { invertDense, safeDerfcr, safeErfc, erfAccurate } from '../src/engine-gpl/electrode/electrode_math';
import { erfcExact } from '../src/engine/force/erfc';
import { eachImagePair } from '../src/engine-gpl/electrode/electrode_pairs';
import type { SimState } from '../src/engine/types';
import type { Geometry } from '../src/engine/domain';

/*
 * ELECTRODE port (src/engine-gpl/electrode): math helpers, the refusals of keywords that are not ported, the
 * write_* file layout, charge neutrality with symm on, and the periodic image enumeration. The native parity of the
 * charges and energies is in tests/oracle/w68elec_* (engineOracle.test.ts).
 */

const ORACLE = join(__dirname, 'oracle');
const madelungData = readFileSync(join(ORACLE, 'w68elec_au_elyt.data'), 'utf8');

/** Runs an input; returns the events and the first error message (null if none). */
const run = async (text: string, files: Record<string, string> = {}, written = new Map<string, string>()) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: (n, t, ap) => written.set(n, (ap ? written.get(n) ?? '' : '') + t) }, undefined, 0, 1);
  session.addFile('w68elec_au_elyt.data', madelungData);
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  let thrown: string | null = null;
  try {
    await session.execute(`${text}\n`);
  } catch (e) {
    thrown = String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  const message = err && 'message' in err ? err.message : thrown;
  return { events, message };
};

const MADELUNG = `boundary        p p f
kspace_style    ewald/electrode 1.0e-8
kspace_modify   slab 8.0
units           real
atom_style      full
pair_style      lj/cut/coul/long 12
read_data       w68elec_au_elyt.data
group           bot type 1
group           top type 2
variable        q atom q
compute         qbot bot reduce sum v_q
compute         qtop top reduce sum v_q
`;

describe('electrode math helpers', () => {
  it('safe_erfc follows erfc to the Abramowitz-Stegun accuracy and vanishes past ERFCMAX', () => {
    for (const x of [0.1, 0.5, 1, 2, 3]) expect(Math.abs(safeErfc(x) - erfcExact(x))).toBeLessThan(2e-7);
    expect(safeErfc(6)).toBe(0);
  });

  it('safe_derfcr is -erfc(x) - 2 x exp(-x^2)/sqrt(pi), the factor of the smeared Coulomb force', () => {
    // electrode_math.h: "return -erfc - 2.0 * expm2 * x / MathConst::MY_PIS"; with fpair = qq/r * derfcr / r^2 this is the
    // force of E = -qq erfc(eta r)/r (checked by hand: fpair = -qq (erfc/r^3 + 2 eta exp(-x^2) /(sqrt(pi) r^2)))
    const out = { erfc: 0 };
    for (const x of [0.3, 0.7, 2]) {
      const d = safeDerfcr(x, out);
      expect(out.erfc).toBeCloseTo(safeErfc(x), 14);
      expect(d).toBeCloseTo(-safeErfc(x) - 2 * Math.exp(-x * x) * x / Math.sqrt(Math.PI), 14);
    }
  });

  it('erfAccurate matches known erf values', () => {
    expect(erfAccurate(0.5)).toBeCloseTo(0.5204998778130465, 13);
    expect(erfAccurate(1)).toBeCloseTo(0.8427007929497149, 13);
    expect(erfAccurate(-1)).toBeCloseTo(-0.8427007929497149, 13);
    expect(erfAccurate(3)).toBeCloseTo(0.9999779095030014, 13);
  });

  it('invertDense inverts a 3x3 matrix and reports singular matrices', () => {
    const a = Float64Array.from([4, 1, 0, 1, 3, 1, 0, 1, 2]);
    const inv = Float64Array.from(a);
    expect(invertDense(inv, 3)).toBe(true);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      let p = 0;
      for (let k = 0; k < 3; k++) p += a[i * 3 + k] * inv[k * 3 + j];
      expect(p).toBeCloseTo(i === j ? 1 : 0, 12);
    }
    expect(invertDense(Float64Array.from([1, 2, 2, 4]), 2)).toBe(false);
  });
});

describe('electrode pair enumeration', () => {
  it('visits every periodic image within the cut-off, as a brute-force sum does', () => {
    // two atoms in a 1 x 1 x 20 box (periodic in x and y only), cut-off 12: many x-y images
    const geom = { lx: 1, ly: 1, lz: 20, periodic: [true, true, false] } as unknown as Geometry;
    const x = Float64Array.from([0, 0, -2, 0.5, 0.25, 2]);
    const s = { n: 2, x, type: Int32Array.from([1, 1]) } as unknown as SimState;
    const cut2 = 144;
    const cutsq = Float64Array.from([0, 0, 0, 0, cut2, cut2, 0, cut2, cut2]);
    let count = 0;
    eachImagePair(s, geom, cutsq, 3, () => { count++; });
    // brute force over a wide range of shifts
    let brute = 0;
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
      for (let sx = -40; sx <= 40; sx++) for (let sy = -40; sy <= 40; sy++) for (let sz = -1; sz <= 1; sz++) {
        if (i === j && sx === 0 && sy === 0 && sz === 0) continue;
        if (sz !== 0) continue;   // z is not periodic
        const dx = x[3 * i] - x[3 * j] - sx, dy = x[3 * i + 1] - x[3 * j + 1] - sy, dz = x[3 * i + 2] - x[3 * j + 2];
        if (dx * dx + dy * dy + dz * dz < cut2 && cutsq[1 * 3 + 1] > 0) brute++;
      }
    }
    expect(count).toBe(brute);
  });
});

describe('fix electrode/conp and ewald/electrode refusals', () => {
  const cases: [string, string, RegExp][] = [
    ['fix conp bot electrode/conp 0 2 couple top 1 symm on ffield yes', '', /ffield/],
    ['fix conp bot electrode/conp 0 2 couple top 1 symm on qtotal 1', '', /qtotal/],
    ['fix conp bot electrode/conp 0 2 couple top 1 symm on read_inv inv.csv', '', /read_inv/],
    ['fix conp bot electrode/conp 0 2 couple top 1 symm on algo cg 1e-6 write_inv inv.csv', '', /does not use inverted matrix/],
    ['fix conp bot electrode/conp 0 2 couple top 1 symm on algo cg 1e-6 write_vec vec.csv', '', /does not use matrix/],
    ['fix conp bot electrode/conp 0 2 couple top 1 symm on temp 300 1 1', '', /temp keyword/],
    ['fix conp bot electrode/conp 0 2 couple top 1 symm on algo newton 1e-6', '', /Unknown algo/],
    ['fix conp bot electrode/conp 0 2 couple nope 1 symm on', '', /Group nope does not exist/],
    ['fix conp bot electrode/conp 0 2 couple top 1 symm on mystery 1', '', /Unknown keyword mystery/],
  ];
  for (const [line, , re] of cases) {
    it(`refuses or rejects: ${line.replace(/^fix conp bot electrode\/conp 0 2 couple top 1 symm on ?/, '') || line}`, async () => {
      const r = await run(`${MADELUNG}${line}\nrun 0`);
      expect(r.message).toMatch(re);
    });
  }

  it('etypes yes needs electrode and electrolyte types that do not overlap', async () => {
    // type 3 would be both an electrode atom (id 3 in bot) and an electrolyte atom (id 4)
    const overlap = await run(`${MADELUNG.replace('group           bot type 1', 'group           bot id 1 3')}fix conp bot electrode/conp 0 2 couple top 1 symm on etypes yes\nrun 0`);
    expect(overlap.message).toMatch(/Types overlap, cannot use etypes keyword/);
    const ok = await run(`${MADELUNG}fix conp bot electrode/conp 0 2 couple top 1 symm on etypes yes\nrun 0`);
    expect(ok.message).toBeNull();
  });

  it('refuses kspace_modify keywords the port does not cover', async () => {
    const wire = await run(MADELUNG.replace('kspace_modify   slab 8.0', 'kspace_modify   wire 1') + 'fix conp bot electrode/conp 0 2 couple top 1 symm on\nrun 0');
    expect(wire.message).toMatch(/wire/);
    const nozforce = await run(MADELUNG.replace('slab 8.0', 'slab nozforce') + 'fix conp bot electrode/conp 0 2 couple top 1 symm on\nrun 0');
    expect(nozforce.message).toMatch(/nozforce/);
    const auto = await run(MADELUNG.replace('slab 8.0', 'slab auto') + 'fix conp bot electrode/conp 0 2 couple top 1 symm on\nrun 0');
    expect(auto.message).toMatch(/auto/);
  });

  it('needs the electrode k-space style', async () => {
    const r = await run(MADELUNG.replace('ewald/electrode 1.0e-8', 'ewald 1.0e-8') + 'fix conp bot electrode/conp 0 2 couple top 1 symm on\nrun 0');
    expect(r.message).toMatch(/ElectrodeKSpace/);
  });
});

describe('fix electrode/conp results', () => {
  it('keeps the total electrode charge at zero with symm on', async () => {
    const r = await run(`${MADELUNG}fix conp bot electrode/conp 0 2 couple top 1 symm on\nthermo_style custom step pe c_qbot c_qtop\nrun 0`);
    expect(r.message).toBeNull();
    const row = r.events.find((e) => e.kind === 'thermo') as { row: ThermoRow } | undefined;
    expect(row).toBeDefined();
    expect(row!.row.c_qbot + row!.row.c_qtop).toBeLessThan(1e-12);
    expect(Math.abs(row!.row.c_qbot)).toBeGreaterThan(0.1);
  });

  it('writes the capacitance matrix with the tag header and C-style columns', async () => {
    const written = new Map<string, string>();
    const r = await run(`${MADELUNG}fix conp bot electrode/conp 0 2 couple top 1 symm on write_inv inv.csv\nrun 0`, {}, written);
    expect(r.message).toBeNull();
    const lines = written.get('inv.csv')!.trim().split('\n');
    expect(lines.length).toBe(3);
    expect(lines[0].trim().split(/\s+/)).toEqual(['1', '2']);
    expect(lines[1]).toMatch(/^\s*-?[\d.]+e[+-]\d{2}\s+-?[\d.]+e[+-]\d{2}$/);
    expect(lines[1].length).toBe(40);
    // the symmetrised capacitance has zero row sums
    const row = lines[1].trim().split(/\s+/).map(Number);
    expect(row[0] + row[1]).toBeCloseTo(0, 14);
  });

  it('reads a potential from an equal-style variable (v_ potential)', async () => {
    const r = await run(`${MADELUNG}variable v equal 2.0\nfix conp bot electrode/conp v_v 2 couple top v_v symm on\nthermo_style custom step c_qbot\nrun 0`);
    expect(r.message).toBeNull();
    const row = r.events.find((e) => e.kind === 'thermo') as { row: ThermoRow } | undefined;
    expect(row).toBeDefined();
    // the variable value 2.0 is the potential of the bottom group (psi); its charge is then nonzero
    expect(Math.abs(row!.row.c_qbot)).toBeGreaterThan(0.1);
  });
});
