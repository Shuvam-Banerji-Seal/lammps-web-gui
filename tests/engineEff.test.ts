import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { ierfoverx1, kinElec, cutoff, dcutoff, elecNucNuc, smallRForce } from '../src/engine-gpl/eff/math';

/*
 * The EFF port (src/engine-gpl/eff, AGENTS.md rule 7): the math helpers of pair_eff_inline.h, the atom style electron
 * (src/engine/output/data.ts, atoms.ts), the refusals of styles and keywords that are not ported, and the
 * eFF temperature and kinetic energy by hand. Native parity of the whole port is in tests/oracle/w63eff_*.in.
 */

const newSession = () => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  return { session, events, files };
};

const thermoRows = (events: EngineEvent[]): ThermoRow[] =>
  events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);

const H2_DATA = `electron test

2 atoms
2 atom types

-1.0 1.0 xlo xhi
-1.0 1.0 ylo yhi
-1.0 1.0 zlo zhi

Masses

1 1.000794
2 1.000000

Atoms

1 1 1.0 0 0.0 0.0 0.0 0.0
2 2 0.0 1 1.0 0.0 0.0 0.0

Velocities

1 0.1 0.2 0.3 0.0
2 0.0 0.0 0.0 0.5
`;

const HEAD = `units electron
atom_style electron
`;

describe('pair_eff_inline.h math', () => {
  it('erf(x)/x at 0 is 2/sqrt(pi), and its two branches join at x = 2', () => {
    // the polynomial of pair_eff_inline.h is accurate to about 2e-12 at 0 (not exact)
    expect(ierfoverx1(0).f).toBeCloseTo(2 / Math.sqrt(Math.PI), 10);
    const lo = ierfoverx1(2 - 1e-9), hi = ierfoverx1(2 + 1e-9);
    expect(Math.abs(lo.f - hi.f)).toBeLessThan(1e-8);
    expect(Math.abs(lo.df - hi.df)).toBeLessThan(1e-7);
  });

  it('the derivative of erf(x)/x matches a central difference', () => {
    for (const x of [0.3, 1.1, 2.7]) {
      const h = 1e-6;
      const num = (ierfoverx1(x + h).f - ierfoverx1(x - h).f) / (2 * h);
      expect(ierfoverx1(x).df).toBeCloseTo(num, 6);
    }
  });

  it('the cutoff spline goes from 1 at 0 to 0 at 1 with zero slope at both ends', () => {
    expect(cutoff(0)).toBeCloseTo(1, 14);
    expect(cutoff(1)).toBeCloseTo(0, 14);
    expect(dcutoff(0)).toBeCloseTo(0, 14);
    expect(dcutoff(1)).toBeCloseTo(0, 14);
  });

  it('the electron kinetic energy and nucleus-nucleus Coulomb terms', () => {
    expect(kinElec(2).e).toBeCloseTo(1.5 / 4, 15);
    expect(kinElec(2).f).toBeCloseTo(3 / 8, 15);
    expect(elecNucNuc(2, 4)).toEqual({ e: 0.5, f: 0.125 });
  });

  it('smallRForce gives the Cartesian force along the separation', () => {
    expect(smallRForce(3, 4, 0, 5, 10)).toEqual([6, 8, 0]);
  });
});

describe('atom_style electron', () => {
  it('reads the Atoms and Velocities columns and writes them back', async () => {
    const { session, files } = newSession();
    session.addFile('h2.data', H2_DATA);
    await session.execute(`${HEAD}read_data h2.data\nwrite_data out.data\n`);
    const out = files.get('out.data') ?? '';
    expect(out).toMatch(/Atoms # electron/);
    expect(out).toMatch(/^1 1 1 0 0 0 0 0 0 0 0$/m);
    expect(out).toMatch(/^2 2 0 1 1 0 0 0 0 0 0$/m);
    expect(out).toMatch(/^2 0 0 0 0\.5$/m);
  });

  it('round-trips the electron properties through write_restart and read_restart', async () => {
    // the electron properties are stored apart from the pair style, so this round trip needs no pair_style
    const { session, files } = newSession();
    session.addFile('h2.data', H2_DATA);
    await session.execute(`${HEAD}read_data h2.data\nset atom 2 radius/electron 0.75\nwrite_data before.data\nwrite_restart h2.restart\n`);
    await session.execute('clear\nread_restart h2.restart\nwrite_data after.data\n');
    const cols = (t: string) => t.slice(t.indexOf('Atoms'), t.indexOf('Velocities'));
    expect(cols(files.get('after.data') ?? '')).toBe(cols(files.get('before.data') ?? ''));
  });

  it('pair eff/cut round-trips through write_restart and read_restart without pair_style or pair_coeff', async () => {
    // pair_eff.rst: "These pair styles write their information to binary restart files"; the run after the restart equals the uninterrupted run
    const setup = `${HEAD}read_data h2.data\nset atom 2 radius/electron 0.75\npair_style eff/cut 1.5 limit/eradius pressure/evirials\npair_coeff * *\ncomm_modify vel yes\nfix 1 all nve/eff\n`;
    const direct = newSession();
    direct.session.addFile('h2.data', H2_DATA);
    await direct.session.execute(`${setup}run 4\n`);
    const want = { x: Array.from(direct.session.sys.state.x), e: Array.from(direct.session.sys.state.custom.get('#eradius')!.data) };
    const { session, files } = newSession();
    session.addFile('h2.data', H2_DATA);
    await session.execute(`${setup}run 2\nwrite_restart eff.restart\n`);
    expect(files.get('eff.restart')).toContain('"name":"eff/cut"');
    await session.execute('clear\nread_restart eff.restart\nfix 1 all nve/eff\nrun 2\n');
    expect(session.sys.ff.pair?.name).toBe('eff/cut');
    const got = Array.from(session.sys.state.x);
    let worst = 0;
    for (let i = 0; i < got.length; i++) worst = Math.max(worst, Math.abs(got[i] - want.x[i]) / (1 + Math.abs(want.x[i])));
    expect(worst).toBeLessThan(1e-12);
    expect(Array.from(session.sys.state.custom.get('#eradius')!.data)).toEqual(want.e);
  });

  it('the set keywords radius/electron and spin/electron change the electron properties and check their values', async () => {
    const { session, files } = newSession();
    session.addFile('h2.data', H2_DATA);
    await session.execute(`${HEAD}read_data h2.data\nset atom 2 radius/electron 0.75\nset atom 2 spin/electron -1\nwrite_data o.data\n`);
    expect(files.get('o.data')).toMatch(/^2 2 0 -1 0\.75 0 0 0 0 0 0$/m);
    const bad = (set: string) => {
      const n = newSession();
      n.session.addFile('h2.data', H2_DATA);
      return n.session.execute(`${HEAD}read_data h2.data\n${set}\n`);
    };
    await expect(bad('set atom 2 radius/electron -1')).rejects.toThrow(/Invalid electron radius/);
    await expect(bad('set atom 2 spin/electron 4')).rejects.toThrow(/Invalid electron spin/);
  });

  it('refuses set radius/electron on a style without electrons', async () => {
    const { session } = newSession();
    await expect(session.execute('units lj\natom_style atomic\nregion b block 0 2 0 2 0 2\ncreate_box 1 b\ncreate_atoms 1 single 0 0 0\nset atom 1 radius/electron 1\n'))
      .rejects.toThrow(/Cannot set attribute radius\/electron/);
  });
});

describe('pair eff/cut, fix nve/eff, compute temp/eff and ke/eff', () => {
  it('compute ke/eff adds the electron radial kinetic energy to the nuclei kinetic energy', async () => {
    const { session, events } = newSession();
    session.addFile('h2.data', H2_DATA);
    await session.execute(`${HEAD}read_data h2.data\ncompute k all ke/eff\nvariable kk equal c_k\nthermo_style custom step v_kk\nrun 0\n`);
    // Measured with native LAMMPS (black box) on this data: v_kk = 0.174710504041769 (the ke/eff sum of the nuclei and electron radial terms)
    const expected = 0.174710504041769;
    expect(thermoRows(events).at(-1)!.v_kk).toBeCloseTo(expected, 12);
  });

  it('pair_style eff/cut reports its energy vector through compute pair', async () => {
    const { session, events } = newSession();
    session.addFile('h2.data', H2_DATA);
    await session.execute(`${HEAD}read_data h2.data\npair_style eff/cut 1.0\npair_coeff * *\ncompute e all pair eff/cut\nvariable ke equal c_e[1]\nthermo_style custom step v_ke pe\nrun 0\n`);
    const row = thermoRows(events).at(-1)!;
    expect(Number.isFinite(row.v_ke)).toBe(true);
    expect(row.pe).toBeLessThan(0);
  });

  it('refuses what is not ported, by name', async () => {
    const run = (text: string) => newSession().session.execute(text);
    const withData = () => { const n = newSession(); n.session.addFile('h2.data', H2_DATA); return n.session; };
    // ecp with an element that has no default ECP parameters (pair_eff_cut.cpp settings: "No default parameters for this atom ECP")
    await expect(withData().execute(`${HEAD}read_data h2.data\npair_style eff/cut 1.0 ecp 1 Xx\n`)).rejects.toThrow(/No default parameters for this atom ECP/);
    // an ECP line for a type that has no ecp element is refused (native would index element 0 silently)
    await expect(run('units electron\natom_style electron\nregion b block -1 1 -1 1 -1 1\ncreate_box 2 b\npair_style eff/cut 1.0\npair_coeff 1 s 0.3 2.2 0.8\n'))
      .rejects.toThrow(/has no element/);
    // the ECP lines take 5 (s) or 7 (p) words
    await expect(run('units electron\natom_style electron\nregion b block -1 1 -1 1 -1 1\ncreate_box 2 b\npair_style eff/cut 1.0 ecp 1 Si\npair_coeff 1 s 0.3 2.2\n'))
      .rejects.toThrow(/takes A B C/);
    await expect(run('units metal\natom_style atomic\nregion b block 0 2 0 2 0 2\ncreate_box 1 b\nmass 1 1.0\ncreate_atoms 1 single 0 0 0\npair_style eff/cut 1.0\npair_coeff * *\nrun 0\n'))
      .rejects.toThrow(/requires atom attributes/);
    await expect(run('units metal\natom_style electron\nregion b block 0 2 0 2 0 2\ncreate_box 1 b\nmass 1 1.0\npair_style eff/cut 1.0\npair_coeff * *\nrun 0\n'))
      .rejects.toThrow(/Check your units/);
  });

  it('minimizes the electron radii with cg and refuses the damped dynamics styles (native: per-atom DOF)', async () => {
    const { session, events } = newSession();
    session.addFile('h2.data', H2_DATA);
    await session.execute(`${HEAD}read_data h2.data\npair_style eff/cut 1.0\npair_coeff * *\nthermo_style custom step pe\nminimize 0 1e-6 5 20\n`);
    expect(thermoRows(events).length).toBeGreaterThan(1);
    const second = newSession();
    second.session.addFile('h2.data', H2_DATA);
    await expect(second.session.execute(`${HEAD}read_data h2.data\npair_style eff/cut 1.0\npair_coeff * *\nmin_style quickmin\nminimize 0 1e-6 5 20\n`))
      .rejects.toThrow(/damped dynamics min style with per-atom DOF/);
    const withData = () => { const n = newSession(); n.session.addFile('h2.data', H2_DATA); return n.session; };
    // Measured with native LAMMPS: fix nve/eff does not read the words after its style name
    await expect(withData().execute(`${HEAD}read_data h2.data\nfix f all nve/eff 1\n`)).resolves.toBeUndefined();
    // As native LAMMPS does (src/EFF/compute_temp_eff.cpp at b332f28): the constructor reads no arguments after the style name
    await expect(withData().execute(`${HEAD}read_data h2.data\ncompute t all temp/eff 1\n`)).resolves.toBeUndefined();
  });

  it('refuses fix nve/eff and compute temp/eff without atom_style electron', async () => {
    await expect(newSession().session.execute('units lj\natom_style atomic\nregion b block 0 2 0 2 0 2\ncreate_box 1 b\nmass 1 1.0\ncreate_atoms 1 single 0 0 0\nfix f all nve/eff\nrun 0\n'))
      .rejects.toThrow(/requires atom style electron/);
    await expect(newSession().session.execute('units lj\natom_style atomic\nregion b block 0 2 0 2 0 2\ncreate_box 1 b\nmass 1 1.0\ncreate_atoms 1 single 0 0 0\ncompute t all temp/eff\n'))
      .rejects.toThrow(/requires atom style electron/);
  });

  it('ke/atom/eff sums to the kinetic energy of compute ke/eff', async () => {
    const { session, events } = newSession();
    session.addFile('h2.data', H2_DATA);
    await session.execute(`${HEAD}read_data h2.data\ncompute k all ke/eff\ncompute ka all ke/atom/eff\ncompute kr all reduce sum c_ka\nvariable a equal c_k\nvariable b equal c_kr\nthermo_style custom step v_a v_b\nrun 0\n`);
    const row = thermoRows(events).at(-1)!;
    expect(row.v_b).toBeCloseTo(row.v_a, 12);
  });

  it('refuses the eFF thermostat and rescale fixes without their arguments, and nvt/eff without electrons', async () => {
    const withData = () => { const n = newSession(); n.session.addFile('h2.data', H2_DATA); return n.session; };
    await expect(withData().execute(`${HEAD}read_data h2.data\nfix f all nvt/eff temp 300 300 0.1 iso 1 1 1\n`)).rejects.toThrow(/Pressure control can not be used with fix nvt\/eff/);
    await expect(withData().execute(`${HEAD}read_data h2.data\nfix f all nvt/eff tchain 0 temp 300 300 0.1\n`)).rejects.toThrow(/Invalid fix nvt\/eff tchain/);
    await expect(withData().execute(`${HEAD}read_data h2.data\nfix f all temp/rescale/eff 10 0 300\n`)).rejects.toThrow(/Illegal fix temp\/rescale\/eff/);
    await expect(newSession().session.execute('units lj\natom_style atomic\nregion b block 0 2 0 2 0 2\ncreate_box 1 b\nmass 1 1.0\ncreate_atoms 1 single 0 0 0\nfix f all nvt/eff temp 1 1 1\n'))
      .rejects.toThrow(/requires atom style electron/);
  });
});
