import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { PairTIP4PBase } from '../src/engine/force/pair/tip4p';
import { buildOwnedSites, tip4pAlpha } from '../src/engine/force/pair/tip4p_sites';
import { Geometry, makeBox } from '../src/engine/domain';
import { StyleError } from '../src/engine/force/types';

/*
 * TIP4P M-site geometry, force projection and argument errors (pair styles
 * lj/cut/tip4p/cut, lj/cut/tip4p/long, tip4p/cut, tip4p/long; kspace pppm/tip4p).
 * The geometry and the projection weights were measured with native LAMMPS
 * (black box); see the comments in src/engine/force/pair/tip4p_sites.ts.
 *
 */

const C = 332.06371;

const run = async (script: string, files: Record<string, string> = {}) => {
  const events: { kind: string; message?: string }[] = [];
  const written = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e as { kind: string; message?: string }),
    writeFile: (n, t, ap) => written.set(n, (ap ? written.get(n) ?? '' : '') + t),
  });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  let error = '';
  try {
    await session.execute(script);
  } catch (e) {
    error = String(e);
  }
  const ev = events.find((e) => e.kind === 'error');
  return { error: error || (ev?.message ?? ''), written };
};

/** One water (O at the origin, H on the bisector-symmetric equilibrium geometry) and an ion above it. */
const waterIonData = (ion: [number, number, number]) => [
  'water and ion', '', '4 atoms', '2 bonds', '1 angles', '3 atom types', '2 bond types', '1 angle types', '',
  '-30 30 xlo xhi', '-30 30 ylo yhi', '-30 30 zlo zhi', '', 'Masses', '', '1 15.9994', '2 1.008', '3 1.0', '',
  'Atoms # full', '',
  `1 1 1 -1.04 0.0 0.0 0.0`,
  `2 1 2 0.0 -0.7569503 0.5858823 0.0`,
  `3 1 2 0.0 0.7569503 0.5858823 0.0`,
  `4 2 3 1.0 ${ion.join(' ')}`,
  '', 'Bonds', '', '1 1 1 2', '2 1 1 3', '', 'Angles', '', '1 1 2 1 3', '',
].join('\n');

const bondedScript = [
  'bond_style harmonic', 'bond_coeff * 0.0 0.9572', 'angle_style harmonic', 'angle_coeff * 0.0 104.52',
].join('\n');

describe('TIP4P M-site geometry', () => {
  it('alpha = qdist / (r0 cos(theta0/2)) (measured with native LAMMPS)', () => {
    // r0 = 1.0, theta0 = 110 degrees, qdist = 0.15 gave alpha 0.26151701934 natively
    expect(tip4pAlpha(0.15, 1.0, 110)).toBeCloseTo(0.2615170193, 9);
    expect(tip4pAlpha(0.1546, 0.9572, 104.52)).toBeCloseTo(0.2638755364, 9);
  });

  it('places M on the HOH bisector at qdist from O for the equilibrium geometry', () => {
    const geom = new Geometry(makeBox({ lo: [-30, -30, -30], hi: [30, 30, 30], periodic: [false, false, false] } as never));
    const qdist = 0.15, r0 = 0.9572, th = 104.52;
    const alpha = tip4pAlpha(qdist, r0, th);
    const x = new Float64Array([0, 0, 0, -0.7569503, 0.5858823, 0, 0.7569503, 0.5858823, 0]);
    const type = new Int32Array([1, 2, 2]);
    const ids = new Int32Array([1, 2, 3]);
    const sites = buildOwnedSites(3, x, type, ids, geom, { otype: 1, htype: 2, alpha });
    // M = (0, qdist, 0) for this symmetric geometry
    expect(sites.M[0]).toBeCloseTo(0, 12);
    expect(sites.M[1]).toBeCloseTo(qdist, 6);
    expect(sites.M[2]).toBeCloseTo(0, 12);
    expect(sites.h1[0]).toBe(1);
    expect(sites.h2[0]).toBe(2);
    expect(sites.h1[1]).toBe(-1);
  });

  it('for a distorted water the M offset is (alpha/2)|d1 + d2| (measured: 0.16766 A for alpha 0.256023)', () => {
    const geom = new Geometry(makeBox({ lo: [-30, -30, -30], hi: [30, 30, 30], periodic: [false, false, false] } as never));
    const alpha = 0.256023;
    const x = new Float64Array([0, 0, 0, -0.8, 0.62, 0, 0.55, 0.66, 0.12]);
    const sites = buildOwnedSites(3, x, new Int32Array([1, 2, 2]), new Int32Array([1, 2, 3]), geom, { otype: 1, htype: 2, alpha });
    const d = Math.hypot(-0.25, 1.28, 0.12);
    expect(Math.hypot(sites.M[0], sites.M[1], sites.M[2])).toBeCloseTo((alpha / 2) * d, 9);
  });

  it('rejects an oxygen without its two H atoms (IDs O+1 and O+2)', () => {
    const geom = new Geometry(makeBox({ lo: [-30, -30, -30], hi: [30, 30, 30], periodic: [false, false, false] } as never));
    expect(() => buildOwnedSites(2, new Float64Array(6), new Int32Array([1, 2]), new Int32Array([1, 2]), geom, { otype: 1, htype: 2, alpha: 0.2 }))
      .toThrow(StyleError);
  });
});

describe('TIP4P force projection and Coulomb cutoff', () => {
  it('projects the M-site force onto O (1 - alpha) and each H (alpha / 2), and the total is conserved', async () => {
    const ion: [number, number, number] = [0, 8, 0];
    const script = [
      'units real', 'atom_style full', 'boundary f f f', 'read_data w.data',
      'pair_style lj/cut/tip4p/cut 1 2 1 1 0.15 12.0', 'pair_coeff * * 0.0 1.0', 'pair_coeff 3 3 0.0 1.0', bondedScript,
      'run 0', 'write_dump all custom t4.dump id fx fy fz modify format float %.15g sort id',
    ].join('\n');
    const r = await run(script, { 'w.data': waterIonData(ion) });
    expect(r.error).toBe('');
    const rows = (r.written.get('t4.dump') ?? '').split('ITEM: ATOMS')[1].trim().split('\n').slice(1).map((l) => l.split(/\s+/).map(Number));
    const f = new Map(rows.map((w) => [w[0], [w[1], w[2], w[3]]]));
    const alpha = tip4pAlpha(0.15, 0.9572, 104.52);
    const fM = f.get(4)!.map((v) => -v); // force on the massless site is minus the ion's force
    // ion force magnitude = C qO qI / (ion - M)^2 with |M - O| = 0.15 on the bisector
    expect(Math.hypot(...fM)).toBeCloseTo((C * 1.04) / (8 - 0.15) ** 2, 6);
    for (let c = 0; c < 3; c++) {
      expect(f.get(1)![c]).toBeCloseTo((1 - alpha) * fM[c], 6);
      expect(f.get(2)![c] + f.get(3)![c]).toBeCloseTo(alpha * fM[c], 6);
    }
    // the total force on the water equals the force on M (no net force created)
    for (let c = 0; c < 3; c++) {
      expect(f.get(1)![c] + f.get(2)![c] + f.get(3)![c]).toBeCloseTo(fM[c], 6);
    }
  });

  it('tip4p/cut takes no cutoff2 and no LJ coefficients: argument errors name the usage', async () => {
    const bad1 = await run('units real\natom_style full\nboundary f f f\nread_data w.data\npair_style tip4p/cut 1 2 1 1 0.15 8.0 9.0', { 'w.data': waterIonData([0, 8, 0]) });
    expect(bad1.error).toMatch(/usage: pair_style tip4p\/cut/);
    const bad2 = await run('units real\natom_style full\nboundary f f f\nread_data w.data\npair_style lj/cut/tip4p/cut 1 2 1 1 0 8.0', { 'w.data': waterIonData([0, 8, 0]) });
    expect(bad2.error).toMatch(/qdist must be > 0/);
    const bad3 = await run('units real\natom_style full\nboundary f f f\nread_data w.data\npair_style lj/cut/tip4p/cut 1 2 1 1 0.15', { 'w.data': waterIonData([0, 8, 0]) });
    expect(bad3.error).toMatch(/usage: pair_style lj\/cut\/tip4p\/cut/);
  });

  it('refuses to run without the bond and angle equilibria (the force-field hook)', () => {
    // the pair style cannot place M without r0 and theta0; the error names the missing piece
    const pair = new (PairTIP4PBase as unknown as new () => PairTIP4PBase)();
    pair.settings(['1', '2', '1', '1', '0.15', '8.0']);
    pair.allocate(3);
    expect(() => pair.init({ s: null, readFile: () => '', log: () => {} } as never)).toThrow(/bond and angle equilibria/);
  });
});
