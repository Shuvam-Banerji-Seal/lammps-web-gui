/**
 * Engine frames carry the bonds for the notebook's 3D view (atom ID pairs, EngineEvent 'frame' bonds): before, the
 * view had no topology and drew a bonded system (the n-butane notebook) as loose beads. The bonds go with the first
 * frame of a run and then only when they change; the trajectory buffer and the view keep the last ones received.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { TrajectoryBuffer } from '../src/components/workbench/notebook/trajectory';
import { frameToMoleculeData, typicalBondLength, type FrameEvent } from '../src/engine/view';

const notebook = JSON.parse(readFileSync(join(__dirname, '..', 'public', 'notebooks', 'butane-trappe.lmpnb'), 'utf8')) as {
  inputFiles: Record<string, string>;
};

const BUTANE = `
units           real
atom_style      molecular
boundary        p p p
lattice         sc 5.4
region          box block 0 2 0 2 0 2
create_box      2 box bond/types 1 angle/types 1 dihedral/types 1 extra/bond/per/atom 2 extra/angle/per/atom 2 extra/dihedral/per/atom 2 extra/special/per/atom 6
mass            1 15.035
mass            2 14.027
molecule        butane butane_trappe.mol
create_atoms    0 box mol butane 75421
pair_style      lj/cut 8.0
pair_coeff      1 1 0.194746 3.75
pair_coeff      2 2 0.091411 3.95
bond_style      harmonic
bond_coeff      1 452.9 1.54
angle_style     harmonic
angle_coeff     1 62.10 114.0
dihedral_style  opls
dihedral_coeff  1 1.41104 -0.27101 3.14505 0.0
special_bonds   lj 0.0 0.0 0.0
fix             1 all nve
timestep        0.5
run             20
run             10`;

const run = async (script: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) }, undefined, 10);
  session.addFile('butane_trappe.mol', notebook.inputFiles['butane_trappe.mol']);
  await session.execute(script);
  return { session, frames: events.filter((e): e is FrameEvent => e.kind === 'frame') };
};

describe('engine frames carry the bonds', () => {
  it('sends the bond pairs with the first frame of each run, as atom IDs of the topology, and not again', async () => {
    const { session, frames } = await run(BUTANE);
    const s = session.system!;
    const nb = s.topo.bonds.n;
    expect(nb).toBe(8 * 3); // 8 molecules of 3 bonds
    const expected = Array.from(s.topo.bonds.atoms.subarray(0, 2 * nb));
    // run 20 with a frame every 10 steps: steps 0, 10, 20; run 10: steps 20, 30
    const firsts = frames.filter((f) => f.bonds !== undefined);
    expect(firsts.length).toBe(2);
    expect(frames[0].bonds).toBeDefined();
    for (const f of firsts) expect(Array.from(f.bonds!)).toEqual(expected);
    expect(frames.filter((f) => f.bonds === undefined).length).toBe(frames.length - 2);
    // every pair joins two atoms of the frame
    const ids = new Set(Array.from(frames[0].id));
    for (const id of frames[0].bonds!) expect(ids.has(id)).toBe(true);
  });

  it('sends an empty list for an atomic system', async () => {
    const { frames } = await run(`
units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
fix 1 all nve
run 10`);
    expect(frames[0].bonds?.length).toBe(0);
    expect(frameToMoleculeData(frames[0]).bonds).toEqual([]);
  });
});

describe('the notebook view keeps the last bonds', () => {
  it('trajectory records share the bonds of the frame that sent them, and the view draws them', async () => {
    const { frames } = await run(BUTANE);
    const traj = new TrajectoryBuffer();
    for (const f of frames) traj.append(f);
    const nb = frames[0].bonds!.length / 2;
    for (let i = 0; i < traj.length; i++) {
      const ev = traj.toFrameEvent(i)!;
      expect(ev.bonds?.length).toBe(2 * nb);
      const data = frameToMoleculeData(ev);
      expect(data.bonds.length).toBe(nb);
      expect(data.bonds[0]).toEqual({ id: 1, type: 1, atom1Id: ev.bonds![0], atom2Id: ev.bonds![1] });
    }
    // later records share one array (no copy per frame)
    expect(traj.get(1)!.bonds).toBe(traj.get(2)!.bonds);
    traj.clear();
    expect(traj.length).toBe(0);
  });

  it('typicalBondLength is the median length of the bonds that do not cross the box', () => {
    const frame: FrameEvent = {
      kind: 'frame', step: 0,
      // atoms 1-2 1.5 apart, 3-4 1.6 apart, 5-6 1.4 apart, 7-8 across the x boundary of a 10-wide box
      x: Float64Array.from([1, 1, 1, 2.5, 1, 1, 3, 3, 3, 3, 4.6, 3, 5, 5, 5, 5, 5, 6.4, 0.2, 7, 7, 9.7, 7, 7]),
      image: new Int32Array(24), type: new Int32Array(8).fill(1), id: Int32Array.from([1, 2, 3, 4, 5, 6, 7, 8]),
      box: { lo: [0, 0, 0], hi: [10, 10, 10] } as unknown as FrameEvent['box'],
    };
    const bonds = Int32Array.from([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(typicalBondLength(frame, bonds)).toBeCloseTo(1.5, 12);
    expect(typicalBondLength(frame, Int32Array.from([7, 8]))).toBeNull();
  });
});
