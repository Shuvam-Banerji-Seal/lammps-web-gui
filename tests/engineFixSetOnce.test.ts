import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { System } from '../src/engine/system';
import type { EngineEvent } from '../src/engine/types';

/*
 * The set command and fix set, as native LAMMPS does (src/set.cpp Set::command and Set::invoke_actions, src/fix_set.cpp
 * FixSet::end_of_step at b332f28): set logs "Setting atom values ..." and one "N settings made for KEYWORD" line per
 * action; fix set runs the actions with no log, and a compute that its variables read is computed once in the step
 * (variable.cpp: "if (!(compute->invoked_flag & Compute::INVOKED_PERATOM))"). examples/PACKAGES/shock/nemd reads a
 * 200-column compute property/atom from 20 actions every step.
 */
const session = () => {
  const logs: string[] = [];
  const s = new Session({
    emit: (ev: EngineEvent) => { if (ev.kind === 'log') logs.push(ev.text); },
    writeFile: () => {},
  });
  return { s, logs, sys: () => (s as unknown as { sys: System }).sys };
};

const BOX = `units lj
atom_style charge
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 2 box
create_atoms 1 box
mass * 1.0
fix p all property/atom d_a d_b d_c
`;

describe('set and fix set', () => {
  it('set logs one line per action, with native\'s counts', async () => {
    const { s, logs } = session();
    await s.execute(`${BOX}set group all d_a 1.0 d_b 2.0 charge 0.5
set type 1 type/fraction 2 0.5 12345 charge 0.1
`);
    const text = logs.join('\n');
    // Measured with native LAMMPS: the same two commands log these lines
    expect(text).toContain('Setting atom values ...\n  108 settings made for d_a\n  108 settings made for d_b\n  108 settings made for charge');
    expect(text).toContain('Setting atom values ...\n  46 settings made for type/fraction\n  108 settings made for charge');
  });

  it('fix set logs nothing and computes a compute its actions read once per step', async () => {
    const { s, logs, sys } = session();
    await s.execute(`${BOX}velocity all create 1.0 87287 loop geom
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0 2.5
compute pa all property/atom vx vy
variable va atom c_pa[1]*2
variable vb atom c_pa[2]*3
variable vc atom c_pa[1]+c_pa[2]
fix 1 all nve
fix s all set 1 0 group all d_a v_va d_b v_vb d_c v_vc
thermo 5
run 0
`);
    const c = sys().computes.find((x) => x.id === 'pa')! as unknown as { computePeratom(): void };
    const inner = c.computePeratom.bind(c);
    let calls = 0;
    c.computePeratom = () => { calls++; inner(); };
    const before = logs.length;
    await s.execute('run 10\n');
    expect(calls).toBe(10);
    expect(logs.slice(before).some((l) => /Setting atom values|settings made/.test(l))).toBe(false);
    // the actions set this step's values: d_c = vx + vy of the atom at the end of step 10
    const st = sys().state;
    const dc = st.custom.get('c')!.data;
    for (let i = 0; i < st.n; i++) expect(dc[i]).toBe(st.v[3 * i] + st.v[3 * i + 1]);
  });
});
