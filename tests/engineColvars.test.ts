import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix colvars (src/engine-gpl/colvars/): what the port refuses, each with an error that names it, and the shell form of
 * rm that examples/PACKAGES/colvars uses. The values are checked against native LAMMPS by tests/oracle/w762cv_*.in.
 */

const newSession = () => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e) });
  return { session, events };
};

const BOX = `units lj
atom_modify map array
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
fix 1 all nve
`;

const config = (body: string) => `colvarsTrajFrequency 1\ncolvar {\n  name d\n  distance {\n    group1 { atomNumbers 1 2 }\n    group2 { atomNumbers 3 }\n  }\n}\n${body}`;

const run = async (files: Record<string, string>, script: string): Promise<{ events: EngineEvent[]; error: string | null; session: Session }> => {
  const { session, events } = newSession();
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  try {
    await session.execute(script);
    return { events, error: null, session };
  } catch (e) {
    return { events, error: (e as Error).message, session };
  }
};

describe('fix colvars: the supported subset and its refusals', () => {
  it('runs a distance with a harmonic restraint and reports f_ID and f_ID[1][1]', async () => {
    const { events, error } = await run({ 'a.colvars': config('harmonic {\n  colvars d\n  centers 1.0\n  forceConstant 2.0\n}\n') },
      `${BOX}fix 2 all colvars a.colvars\nthermo_style custom step f_2 f_2[1][1]\nrun 0\n`);
    expect(error).toBeNull();
    const row = events.find((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')!.row;
    const d = row['f_2[1][1]'];
    expect(d).toBeGreaterThan(0.5);
    // the scalar is the bias energy 0.5 k (d - d0)^2, extensive (thermo normalizes by the 32 atoms in lj units)
    expect(row.f_2).toBeCloseTo((0.5 * 2.0 * (d - 1.0) ** 2) / 32, 14);
  });

  it('refuses a component, a bias and a keyword outside the subset, naming them', async () => {
    const cn = await run({ 'a.colvars': 'colvar {\n  name t\n  inertia {\n    atoms { atomNumbers 1 2 3 }\n  }\n}\n' },
      `${BOX}fix 2 all colvars a.colvars\nrun 0\n`);
    expect(cn.error).toMatch(/Colvars component "inertia" is not supported by the browser engine \(supported: distance, distanceVec, distanceZ, distanceXY, angle, dihedral, cartesian, gyration, coordNum, selfCoordNum, groupCoord, rmsd, orientation, orientationAngle, orientationProj, tilt, spinAngle, eulerPhi, eulerPsi, eulerTheta\)/);
    // rmsd (colvar::rmsd::init): reference positions are required and must match the group; an explicit fit needs the
    // fit gradients, which are not ported
    const rm = (body: string) => run({ 'a.colvars': `colvar {\n  name r\n  rmsd {\n${body}  }\n}\n` }, `${BOX}fix 2 all colvars a.colvars\nrun 0\n`);
    expect((await rm('    atoms { atomNumbers 1 2 3 }\n')).error).toMatch(/no reference positions for RMSD; use either refPositions of refPositionsFile/);
    expect((await rm('    atoms { atomNumbers 1 2 3 }\n    refPositions (0,0,0) (1,0,0)\n')).error)
      .toMatch(/the number of reference positions provided \(2\) does not match the number of atoms of group "atoms" \(3\)/);
    expect((await rm('    atoms { atomNumbers 1 2 3\n      rotateToReference on }\n    refPositions (0,0,0) (1,0,0) (0,1,0)\n')).error)
      .toMatch(/atom group keyword "rotateToReference" .* is not supported by the browser engine/);
    expect((await rm('    atoms { atomNumbers 1 2 3 }\n    refPositions (0,0,0) (1,0,0) (0,1,0)\n')).error).toBeNull();
    // orientation components (colvar::orientation::init): reference coordinates are required and must match the group
    const or = (body: string) => run({ 'a.colvars': `colvar {\n  name o\n  orientationAngle {\n${body}  }\n}\n` }, `${BOX}fix 2 all colvars a.colvars\nrun 0\n`);
    expect((await or('    atoms { atomNumbers 1 2 3 }\n')).error).toMatch(/must define a set of reference coordinates/);
    expect((await or('    atoms { atomNumbers 1 2 3 }\n    refPositions (0,0,0) (1,0,0)\n')).error).toMatch(/reference positions do not match the number of requested atoms/);
    expect((await or('    atoms { atomNumbers 1 2 3 }\n    refPositions (0,0,0) (1,0,0) (0,1,0)\n')).error).toBeNull();
    const abf = await run({ 'a.colvars': config('abf {\n  colvars d\n}\n') }, `${BOX}fix 2 all colvars a.colvars\nrun 0\n`);
    expect(abf.error).toMatch(/Colvars bias "abf" is not supported by the browser engine \(supported: harmonic, harmonicWalls, metadynamics\)/);
    // colvarbias_meta::init: "Error: hillWeight must be provided, and a positive number."
    const meta = await run({ 'a.colvars': config('metadynamics {\n  colvars d\n}\n') }, `${BOX}fix 2 all colvars a.colvars\nrun 0\n`);
    expect(meta.error).toMatch(/hillWeight must be provided, and a positive number/);
    const kw = await run({ 'a.colvars': config('harmonic {\n  colvars d\n  centers 1.0\n  springiness 3\n}\n') }, `${BOX}fix 2 all colvars a.colvars\nrun 0\n`);
    expect(kw.error).toMatch(/keyword "springiness" is not supported, or not recognized in this context/);
  });

  it('checks the grid parameters and the metadynamics keywords as the library does', async () => {
    const cv = (opts: string, bias = '') => `colvar {\n  name d\n${opts}  distance {\n    group1 { atomNumbers 1 2 }\n    group2 { atomNumbers 3 }\n  }\n}\n${bias}`;
    const go = (text: string, extra = '') => run({ 'a.colvars': text }, `${BOX}${extra}fix 2 all colvars a.colvars\nrun 0\n`);
    // colvar::check_grid_parameters
    expect((await go(cv('  lowerBoundary 2.0\n  upperBoundary 1.0\n'))).error).toMatch(/the upper boundary, 1, is not higher than the lower boundary, 2/);
    // distance has a hard lower boundary at 0 (cvc::init_as_distance); hardUpperBoundary makes both hard
    expect((await go(cv('  upperBoundary 3.0\n  hardUpperBoundary on\n  expandBoundaries on\n'))).error)
      .toMatch(/trying to expand boundaries, but both hardLowerBoundary and hardUpperBoundary are enabled/);
    // colvarbias_meta::init and init_well_tempered_params
    expect((await go(cv('', 'metadynamics {\n  colvars d\n  hillWeight 0.1\n  hillWidth 2\n  gaussianSigmas 0.3\n}\n'))).error).toMatch(/hillWidth and gaussianSigmas are mutually exclusive/);
    expect((await go(cv('', 'metadynamics {\n  colvars d\n  hillWeight 0.1\n}\n'))).error).toMatch(/either hillWidth or gaussianSigmas must be set to positive values/);
    expect((await go(cv('', 'metadynamics {\n  colvars d\n  hillWeight 0.1\n  hillWidth 2\n  wellTempered on\n}\n'))).error).toMatch(/biasTemperature must be set to a positive value/);
    // FixColvars::set_thermostat_temperature: the tstat fix must give t_target
    const ts = await run({ 'a.colvars': cv('') }, `${BOX}fix 2 all colvars a.colvars tstat 1\n`);
    expect(ts.error).toMatch(/Fix ID 1 is not a thermostat fix/);
    // a metadynamics state is read back (grids, then the hills); a state from another grid size is an error
    const meta = cv('', 'metadynamics {\n  colvars d\n  hillWeight 0.1\n  hillWidth 2\n}\n');
    const first = await run({ 'a.colvars': meta }, `${BOX}fix 2 all colvars a.colvars output m\nrun 10\n`);
    expect(first.error).toBeNull();
    const state = first.session.sys.readFile('m.colvars.state');
    expect(state).toMatch(/metadynamics \{\n  configuration \{\n    step 10\n    name metadynamics1\n  \}\n\nhills_energy\ngrid_parameters/);
    const again = await run({ 'a.colvars': meta, 'm.colvars.state': state }, `${BOX}fix 2 all colvars a.colvars input m\nrun 0\n`);
    expect(again.error).toBeNull();
    const cut = state.replace(/hills_energy_gradients[\s\S]*$/, '}\n');
    const bad = await run({ 'a.colvars': meta, 'm.colvars.state': cut }, `${BOX}fix 2 all colvars a.colvars input m\nrun 0\n`);
    expect(bad.error).toMatch(/couldn't read grid data for metadynamics bias "metadynamics1"/);
  });

  it('refuses a script command other than config and configfile and a second instance; a row past the colvars reads 0', async () => {
    const cmd = await run({}, `${BOX}fix 2 all colvars none\nfix_modify 2 reset\n`);
    expect(cmd.error).toMatch(/Colvars script command "reset" is not supported by the browser engine/);
    const two = await run({}, `${BOX}fix 2 all colvars none\nfix 3 all colvars none\n`);
    expect(two.error).toMatch(/Only one fix colvars instance may be active at a time/);
    // As native LAMMPS does (src/thermo.cpp Thermo::compute_fix at b332f28): fix colvars has a variable row count,
    // and thermo prints 0 for a row past it
    const oob = await run({ 'a.colvars': config('') }, `${BOX}fix 2 all colvars a.colvars\nthermo_style custom step f_2[2][1]\nrun 0\n`);
    expect(oob.error).toBeNull();
    const row = oob.events.find((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')!.row;
    expect(row['f_2[2][1]']).toBe(0);
  });

  it('refuses a second keyword after a closing brace on the same line (each keyword starts a line)', async () => {
    const r = await run({ 'a.colvars': 'colvar {\n  name t\n  angle {\n    group1 { atomNumbers 1 } group2 { atomNumbers 2 }\n    group3 { atomNumbers 3 }\n  }\n}\n' },
      `${BOX}fix 2 all colvars a.colvars\nrun 0\n`);
    expect(r.error).toMatch(/follows the closing brace of "group1" on the same line/);
  });

  it('refuses a minimization before the first run (native reads no atoms then)', async () => {
    const r = await run({ 'a.colvars': config('') }, `${BOX}fix 2 all colvars a.colvars\nminimize 1e-6 1e-6 5 10\n`);
    expect(r.error).toMatch(/a minimization before the first run is not supported/);
  });
});

describe('shell with one quoted word: the sh form of rm', () => {
  it('removes the session files a pattern matches and keeps the others', async () => {
    const r = await run({ 'out.colvars.traj': 'x', 'out2.colvars.state': 'y', 'keep.txt': 'z' }, 'shell "rm -f out*.colvars.*"\n');
    expect(r.error).toBeNull();
    expect(r.session.sys.files.has('out.colvars.traj')).toBe(false);
    expect(r.session.sys.files.has('out2.colvars.state')).toBe(false);
    expect(r.session.sys.files.has('keep.txt')).toBe(true);
  });

  it('warns, as native does for a non-zero status, when rm without -f misses', async () => {
    const r = await run({}, 'shell "rm none*.dat"\n');
    expect(r.error).toBeNull();
    const warned = r.events.some((e) => e.kind === 'log' && e.text.includes("Shell command rm none*.dat returned with non-zero status"));
    expect(warned).toBe(true);
  });
});
