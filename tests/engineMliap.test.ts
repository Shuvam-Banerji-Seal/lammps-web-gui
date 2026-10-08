import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/session';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * pair_style mliap and compute mliap (wave 16): cross-checks and error paths. The native parity
 * cases are in tests/oracle/w16mliap_*.in (engineOracle.test.ts). Here:
 *  - model linear and quadratic with descriptor sna equal pair_style snap on the same numbers
 *    (energies and forces to 1e-12), since the doc says the sna descriptor is the snap descriptor;
 *  - model nn forces agree with a finite difference of the energy;
 *  - the styles and options the engine does not implement are StyleErrors that name them.
 */

const ORACLE = join(__dirname, 'oracle');
const read = (f: string) => readFileSync(join(ORACLE, f), 'utf8');

interface Result { rows: ThermoRow[]; forces: Map<number, number[]> }

const parseDump = (text: string): Map<number, number[]> => {
  const lines = text.trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const out = new Map<number, number[]>();
  for (const l of lines.slice(k + 1)) {
    const w = l.trim().split(/\s+/).map(Number);
    out.set(w[0], [w[1], w[2], w[3]]);
  }
  return out;
};

/** Runs a script; files maps names to the text of the inputs it reads. Dumps are returned by name. */
const run = async (script: string, inputs: Record<string, string>, dumps: string[] = []): Promise<Result & { dumpText: Map<string, string> }> => {
  const events: EngineEvent[] = [];
  const written = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => written.set(n, (ap ? written.get(n) ?? '' : '') + t),
  });
  for (const [name, text] of Object.entries(inputs)) session.addFile(name, text);
  await session.execute(script);
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  const dumpText = new Map<string, string>();
  for (const d of dumps) dumpText.set(d, written.get(d) ?? '');
  const forces = dumps.length ? parseDump(dumpText.get(dumps[0]) ?? '') : new Map<number, number[]>();
  return { rows, forces, dumpText };
};

/** Two-element perturbed bcc crystal of the oracle cases (16 atoms, types 1 and 2 alternating). */
const crystal = `units lj
atom_style atomic
lattice bcc 0.9
region box block 0 2 0 2 0 2
create_box 2 box
create_atoms 1 box
group odd id 1:128:2
set group odd type 2
mass * 1.0
variable dx atom 0.05*sin(1.3*x+0.7*y)
variable dy atom 0.05*cos(0.9*y-1.1*z)
variable dz atom 0.05*sin(1.7*z+0.5*x)
displace_atoms all move v_dx v_dy v_dz units box
`;

const FORCE_DUMP = 'write_dump all custom forces.dump id fx fy fz modify format float %.17g sort id';

const compareWithSnap = async (mliapScript: string, snapScript: string, inputs: Record<string, string>) => {
  const dumps = ['forces.dump'];
  const a = await run(mliapScript, inputs, dumps);
  const b = await run(snapScript, inputs, dumps);
  expect(a.rows[0].pe).toBeCloseTo(b.rows[0].pe, 12);
  expect(Math.abs(a.rows[0].pe - b.rows[0].pe)).toBeLessThan(1e-12 * Math.max(1, Math.abs(b.rows[0].pe)));
  expect(a.forces.size).toBe(16);
  let worst = 0;
  for (const [id, f] of b.forces) {
    const g = a.forces.get(id)!;
    for (let c = 0; c < 3; c++) worst = Math.max(worst, Math.abs(g[c] - f[c]));
  }
  expect(worst).toBeLessThan(1e-12);
  // the comparison is not trivially zero: some force is clearly nonzero
  let fmax = 0;
  for (const f of b.forces.values()) for (const c of f) fmax = Math.max(fmax, Math.abs(c));
  expect(fmax).toBeGreaterThan(1e-2);
  return worst;
};

describe('pair_style mliap against pair_style snap on the same numbers', () => {
  it('model linear with descriptor sna equals pair snap (energy, pressure, forces)', async () => {
    const inputs = {
      'w16mliap_linear.mliap.model': read('w16mliap_linear.mliap.model'),
      'w16mliap_linear.mliap.descriptor': read('w16mliap_linear.mliap.descriptor'),
      'w8snap_linear.snapcoeff': read('w8snap_linear.snapcoeff'),
      'w8snap_linear.snapparam': read('w8snap_linear.snapparam'),
    };
    const m = `${crystal}pair_style mliap model linear w16mliap_linear.mliap.model descriptor sna w16mliap_linear.mliap.descriptor
pair_coeff * * A B
thermo_style custom step pe
thermo_modify format float %.17g
run 0
${FORCE_DUMP}
`;
    const s = `${crystal}pair_style snap
pair_coeff * * w8snap_linear.snapcoeff w8snap_linear.snapparam A B
thermo_style custom step pe
thermo_modify format float %.17g
run 0
${FORCE_DUMP}
`;
    await compareWithSnap(m, s, inputs);
  });

  it('model quadratic with descriptor sna equals pair snap with quadraticflag 1', async () => {
    const inputs = {
      'w16mliap_quadratic.mliap.model': read('w16mliap_quadratic.mliap.model'),
      'w16mliap_quadratic.mliap.descriptor': read('w16mliap_quadratic.mliap.descriptor'),
      'w8snap_quadratic.snapcoeff': read('w8snap_quadratic.snapcoeff'),
      'w8snap_quadratic.snapparam': read('w8snap_quadratic.snapparam'),
    };
    const m = `${crystal}pair_style mliap model quadratic w16mliap_quadratic.mliap.model descriptor sna w16mliap_quadratic.mliap.descriptor
pair_coeff * * A B
thermo_style custom step pe
thermo_modify format float %.17g
run 0
${FORCE_DUMP}
`;
    const s = `${crystal}pair_style snap
pair_coeff * * w8snap_quadratic.snapcoeff w8snap_quadratic.snapparam A B
thermo_style custom step pe
thermo_modify format float %.17g
run 0
${FORCE_DUMP}
`;
    await compareWithSnap(m, s, inputs);
  });
});

describe('pair_style mliap model nn', () => {
  it('forces equal the negative finite difference of the energy', async () => {
    const inputs = {
      'w16mliap_nn.mliap.model': read('w16mliap_nn.mliap.model'),
      'w16mliap_nn.mliap.descriptor': read('w16mliap_nn.mliap.descriptor'),
    };
    const h = 1e-5;
    // lj units normalise thermo pe per atom by default; norm no gives the total energy the forces derive from
    const script = `${crystal}pair_style mliap model nn w16mliap_nn.mliap.model descriptor sna w16mliap_nn.mliap.descriptor
pair_coeff * * A B
thermo_style custom step pe
thermo_modify format float %.17g norm no
run 0
${FORCE_DUMP}
group one id 3
displace_atoms one move ${h} 0 0 units box
run 0
displace_atoms one move ${-2 * h} 0 0 units box
run 0
displace_atoms one move ${h} 0 0 units box
`;
    const r = await run(script, inputs, ['forces.dump']);
    const ep = r.rows[1].pe, em = r.rows[2].pe;
    const fd = -(ep - em) / (2 * h);
    const fx = r.forces.get(3)![0];
    expect(Math.abs(fx - fd)).toBeLessThan(1e-7);
    expect(Math.abs(fx)).toBeGreaterThan(1e-5); // the case is not trivially zero
  });
});

describe('unsupported mliap forms are StyleErrors that name the option', () => {
  const base = `${crystal}`;
  const desc = read('w16mliap_linear.mliap.descriptor');
  const model = read('w16mliap_linear.mliap.model');
  const inputs = { 'w16mliap_linear.mliap.model': model, 'w16mliap_linear.mliap.descriptor': desc, 'bad.desc': desc };
  const cases: [string, string, RegExp][] = [
    ['model mliappy', 'pair_style mliap model mliappy x.pt descriptor sna w16mliap_linear.mliap.descriptor\npair_coeff * * A B', /mliappy/],
    ['descriptor so3', 'pair_style mliap model linear w16mliap_linear.mliap.model descriptor so3 w16mliap_linear.mliap.descriptor\npair_coeff * * A B', /so3/],
    ['descriptor ace', 'pair_style mliap model linear w16mliap_linear.mliap.model descriptor ace w16mliap_linear.mliap.descriptor\npair_coeff * * A B', /ace/],
    ['unified', 'pair_style mliap unified x.pkl 0\npair_coeff * * A B', /unified/],
    ['NULL mapping', 'pair_style mliap model linear w16mliap_linear.mliap.model descriptor sna w16mliap_linear.mliap.descriptor\npair_coeff * * A NULL', /NULL/],
    ['descriptor quadraticflag', 'pair_style mliap model linear w16mliap_linear.mliap.model descriptor sna bad.desc\npair_coeff * * A B', /quadraticflag/],
  ];
  for (const [label, body, re] of cases) {
    it(label, async () => {
      const extra = label === 'descriptor quadraticflag' ? `${desc}quadraticflag 1\n` : '';
      const i2 = { ...inputs, 'bad.desc': extra || desc };
      await expect(run(`${base}${body}\nrun 0\n`, i2)).rejects.toThrow(re);
    });
  }
  it('compute mliap model nn is refused (the compute doc lists linear, quadratic and mliappy)', async () => {
    await expect(run(`${base}compute g all mliap model nn descriptor sna w16mliap_linear.mliap.descriptor\nrun 0\n`, inputs)).rejects.toThrow(/nn/);
  });
  it('compute mliap descriptor ace is refused', async () => {
    await expect(run(`${base}compute g all mliap model linear descriptor ace w16mliap_linear.mliap.descriptor\nrun 0\n`, inputs)).rejects.toThrow(/ace/);
  });
});
