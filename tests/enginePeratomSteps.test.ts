import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import type { System } from '../src/engine/system';
import type { ComputeFlags } from '../src/engine/force/forcefield';
import { runVerlet } from '../src/engine/run/verlet';

/*
 * Per-atom energy and virial are tallied on the steps a compute may read them only: a run whose fixes are quiet
 * (verlet.ts QUIET_FIXES, no variable argument) tallies them on its thermo and dump steps and its last step, as native
 * Integrate::ev_set does for the steps its computes registered (src/integrate.cpp at b332f28). A compute read on a
 * step that left them out is native's error (src/compute_pe_atom.cpp compute_peratom).
 */

const SETUP = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.5 87287 loop geom
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0 2.5
neighbor 0.3 bin
fix 1 all nve
compute pea all pe/atom
compute sa all stress/atom NULL
compute spe all reduce sum c_pea c_sa[1] c_sa[4]
thermo_style custom step pe press c_spe[1] c_spe[2] c_spe[3]
thermo_modify format float %.17g
thermo 10
dump d all custom 5 pa.dump id c_pea c_sa[1] c_sa[2] c_sa[3] c_sa[4] c_sa[5] c_sa[6]
dump_modify d sort id format float %.17g
`;

const session = () => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const s = new Session({ emit: (e) => events.push(e), writeFile: (n, t, append) => files.set(n, append ? (files.get(n) ?? '') + t : t) });
  const sys = (s as unknown as { sys: System }).sys;
  // the force evaluations of the run steps, with or without per-atom terms
  const tallied: number[] = [], lean: number[] = [];
  const compute = sys.computeForcesInRun.bind(sys);
  sys.computeForcesInRun = (flags: ComputeFlags) => {
    if (flags.step) (flags.eatom ? tallied : lean).push(sys.state.step);
    return compute(flags);
  };
  return { s, sys, events, files, tallied, lean };
};

const rows = (events: EngineEvent[]) => events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => JSON.stringify(e.row));

describe('per-atom energy and virial only on the steps a compute reads them', () => {
  it('tallies them on the thermo, dump and last steps of a run with quiet fixes, with the same output', async () => {
    const quiet = session();
    await quiet.s.execute(`${SETUP}run 23`);
    expect(quiet.tallied).toEqual([5, 10, 15, 20, 23]);
    expect(quiet.lean.length).toBe(18);
    // fix ave/time may invoke computes on its own steps: every step tallies them, as before
    const always = session();
    await always.s.execute(`${SETUP}fix at all ave/time 1 1 1 c_spe[1]\nrun 23`);
    expect(always.tallied.length).toBe(23);
    expect(rows(quiet.events)).toEqual(rows(always.events));
    expect(quiet.files.get('pa.dump')).toBe(always.files.get('pa.dump'));
    expect(quiet.files.get('pa.dump')!.split('ITEM: TIMESTEP').length - 1).toBe(5);
  });

  it('keeps every step for a fix with a variable argument or a dynamic group', async () => {
    for (const extra of [
      'variable fx equal 0.0\nfix f all addforce v_fx 0.0 0.0',
      'group g dynamic all every 2',
    ]) {
      const t = session();
      await t.s.execute(`${SETUP}${extra}\nrun 12`);
      expect(t.lean, extra).toEqual([]);
    }
  });

  it('follows thermo with a variable interval (thermo v_name)', async () => {
    const t = session();
    await t.s.execute(`${SETUP}variable t equal stride(10,100,10)\nthermo v_t\nrun 12`);
    expect(t.tallied).toEqual([5, 10, 12]);
  });

  it('runs without per-atom computes as before (no flags to leave out)', async () => {
    const t = session();
    await t.s.execute(`${SETUP.replace(/^(compute|dump|thermo_style).*$/gm, '')}thermo_style custom step pe\nrun 12`);
    expect(t.tallied).toEqual([]);
  });

  it('a compute read on a step that left them out is native\'s error', async () => {
    const t = session();
    await t.s.execute(`${SETUP}variable e equal c_spe[1]\nvariable v equal c_spe[2]\nrun 0`);
    // a run whose hooks say no step is read, and a compute read on its third step anyway
    const hooks = (name: string) => {
      const third = t.sys.state.step + 3;
      return {
        cancelled: () => false, afterSetup: () => {}, yieldMs: -1, peratomDue: () => false,
        afterStep: (step: number) => { if (step === third) t.sys.equalVariable(name); },
      };
    };
    t.sys.run.inRun = true;
    await expect(runVerlet(t.sys, 5, hooks('e'))).rejects.toThrow(/^Per-atom energy was not tallied on needed timestep\nFor more information see https:\/\/docs.lammps.org\/err0022/);
    await expect(runVerlet(t.sys, 5, hooks('v'))).rejects.toThrow(/^Per-atom (energy|virial) was not tallied on needed timestep/);
    // between runs the values are computed when asked for, as before
    t.sys.run.inRun = false;
    expect(Number.isFinite(t.sys.equalVariable('e'))).toBe(true);
  });
});
