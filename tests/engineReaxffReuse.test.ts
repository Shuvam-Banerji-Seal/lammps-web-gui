import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { System } from '../src/engine/system';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * pair reaxff reuses the bond list, the three-body list and the workspace of its previous evaluation (core.ts
 * allocBondList, workspaceFor, bonded.ts newThreeBodyList) instead of allocating them every step. The results must be
 * those of new lists, bit for bit: the same run with the previous evaluation forgotten before every evaluation gives
 * the same thermo rows, positions and forces. The run compresses the box (more far pairs, so the bond list grows past
 * the reused capacity) and deletes atoms between runs (a workspace of another size).
 */

const POTENTIALS = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const CASES = join(__dirname, 'oracle');

const INPUT = `units real
atom_style charge
boundary p p p
read_data w55reax_rdx.data
pair_style reaxff w55reax_rdx.control
pair_coeff * * ffield.reax.rdx H C O N
fix 2 all qeq/reax 1 0.0 10.0 1e-6 w55reax_param.qeq
neighbor 2.0 bin
neigh_modify every 1 delay 0 check no
timestep 0.25
velocity all create 600.0 4928459 loop geom
fix 1 all nve
fix 3 all deform 1 x final 0 21 y final 0 21 z final 0 21 remap x
thermo_style custom step pe ebond evdwl ecoul ke etotal press
thermo 2
run 20
unfix 3
group gone id 1:12
delete_atoms group gone
run 10
`;

const run = async (forget: boolean) => {
  const rows: ThermoRow[] = [];
  const s = new Session({ emit: (e: EngineEvent) => { if (e.kind === 'thermo') rows.push(e.row); }, writeFile: () => {} }, undefined, 0, 1);
  s.addFile('ffield.reax.rdx', readFileSync(join(POTENTIALS, 'ffield.reax.rdx'), 'utf8'));
  for (const f of ['w55reax_rdx.data', 'w55reax_rdx.control', 'w55reax_param.qeq']) s.addFile(f, readFileSync(join(CASES, f), 'utf8'));
  const lines = INPUT.split('\n');
  // everything up to the first run, then (when forgetting) the pair style's previous evaluation dropped before each
  const first = lines.findIndex((l) => l.startsWith('run '));
  await s.execute(lines.slice(0, first).join('\n') + '\n');
  const sys = (s as unknown as { sys: System }).sys;
  const pair = sys.ff.pair as unknown as { compute: (...a: unknown[]) => unknown; lastBonds: unknown; constructor: { name: string } };
  let calls = 0;
  if (forget) {
    const compute = pair.compute.bind(pair);
    pair.compute = (...a: unknown[]) => { pair.lastBonds = null; calls++; return compute(...a); };
  }
  await s.execute(lines.slice(first).join('\n'));
  return { rows, sys, calls };
};

describe('pair reaxff: lists reused between evaluations', () => {
  it('give the results of new lists, bit for bit, while the box shrinks and after atoms are deleted', async () => {
    const reused = await run(false);
    const fresh = await run(true);
    expect(fresh.calls).toBeGreaterThan(30);
    expect(reused.rows.length).toBe(17);
    expect(reused.rows).toEqual(fresh.rows);
    const a = reused.sys.state, b = fresh.sys.state;
    expect(a.n).toBe(93);
    expect(b.n).toBe(a.n);
    expect(Array.from(a.x.subarray(0, 3 * a.n))).toEqual(Array.from(b.x.subarray(0, 3 * b.n)));
    expect(Array.from(a.f.subarray(0, 3 * a.n))).toEqual(Array.from(b.f.subarray(0, 3 * b.n)));
    // the box did shrink: the bond list grew past its first capacity on the way
    expect(reused.rows[10].press).not.toBe(reused.rows[0].press);
  });
});
