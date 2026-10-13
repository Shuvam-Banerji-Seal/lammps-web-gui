import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

// SPH package styles (src/engine-gpl/sph): pair sph/rhosum, sph/idealgas, sph/taitwater, sph/taitwater/morris,
// sph/heatconduction; fix sph, sph/stationary; compute sph/rho/atom, sph/e/atom. The two-atom velocities below were
// measured with native LAMMPS (black box): two atoms of types 1 and 2 at distance 0.5 approaching at 0.1, one step of
// 0.01 with pair sph/idealgas 0.75 4.0 (the lower-index atom owns the pair, so its viscosity entry applies).
// The oracle cases tests/oracle/w83sph_*.in compare whole runs with native LAMMPS.

const newSession = () => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  return { session, events, files };
};

/** The error a script stops with ('' when it runs through). */
const errorOf = async (session: Session, events: EngineEvent[], script: string): Promise<string> => {
  try {
    await session.execute(script);
  } catch (e) {
    return e instanceof Error ? e.message : String(e);
  }
  const e = events.find((ev) => ev.kind === 'error');
  return e && 'message' in e ? String(e.message) : '';
};

const TWO_ATOMS = (order: '12' | '21') => `units lj
atom_style sph
dimension 2
boundary f f p
region box block -5 5 -5 5 -0.5 0.5
create_box 2 box
${order === '12'
  ? 'create_atoms 1 single 0.0 0.0 0.0 units box\ncreate_atoms 2 single 0.5 0.0 0.0 units box'
  : 'create_atoms 2 single 0.5 0.0 0.0 units box\ncreate_atoms 1 single 0.0 0.0 0.0 units box'}
group g1 id ${order === '12' ? 1 : 2}
group g2 id ${order === '12' ? 2 : 1}
mass 1 1.0
mass 2 0.25
set atom 1 sph/e 1.0
set atom 2 sph/e 1.0
set atom 1 sph/rho 1.0
set atom 2 sph/rho 0.5
velocity g1 set 0.1 0.0 0.0 units box
velocity g2 set -0.1 0.0 0.0 units box
pair_style sph/idealgas
pair_coeff * * 0.75 4.0
fix 1 all sph
fix 2 all enforce2d
timestep 0.01
run 1
`;

describe('SPH pair, fix and compute styles', () => {
  it('two approaching atoms of types 1 and 2 match native LAMMPS (the lower-index atom owns the pair)', async () => {
    const { session } = newSession();
    await session.execute(TWO_ATOMS('12'));
    const v = session.sys.state.v;
    expect(v[0]).toBe(0.099677587259512646);
    expect(v[3]).toBe(-0.098710349038050552);
  });

  it('with the type-2 atom created first, native LAMMPS gives the type-2 atom (index 1) the ownership: no viscosity entry', async () => {
    const { session } = newSession();
    await session.execute(TWO_ATOMS('21'));
    const st = session.sys.state;
    // atom with id 1 is the type-2 atom at x 0.5: native vx -0.099315560588992394 (measured black box)
    const i1 = st.id[0] === 1 ? 0 : 1;
    expect(st.v[3 * i1]).toBe(-0.099315560588992394);
  });

  it('pair sph/* with a newton off setting, an unknown sph style, and fix sph on another atom style are errors', async () => {
    const a = newSession();
    expect(await errorOf(a.session, a.events, `${TWO_ATOMS('12')}pair_style sph/lj\n`)).toMatch(/sph\/lj/);
    const b = newSession();
    const head = 'units lj\natom_style atomic\ndimension 2\nregion box block -5 5 -5 5 -0.5 0.5\ncreate_box 1 box\nlattice sq 1.0\ncreate_atoms 1 box\nmass 1 1\n';
    expect(await errorOf(b.session, b.events, `${head}fix 1 all sph\n`)).toMatch(/Fix sph requires atom style sph/);
    const c = newSession();
    expect(await errorOf(c.session, c.events, `${TWO_ATOMS('12')}newton off\nrun 0\n`)).toMatch(/newton pair off/);
  });

  it('compute sph/rho/atom and sph/e/atom report the atoms of the group and 0 elsewhere', async () => {
    const { session } = newSession();
    await session.execute(`${TWO_ATOMS('12')}group g1 id 1\ncompute r all sph/rho/atom\ncompute e all sph/e/atom\nrun 0\n`);
    const st = session.sys.state;
    const rho = session.sys.compute('r').peratomValues();
    const e = session.sys.compute('e').peratomValues();
    for (let i = 0; i < st.n; i++) {
      expect(rho[i]).toBe(st.mask[i] & 1 ? st.custom.get('#rho')!.data[i] : 0);
      expect(e[i]).toBe(st.mask[i] & 1 ? st.custom.get('#esph')!.data[i] : 0);
    }
  });
});
