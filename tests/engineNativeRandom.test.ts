import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { ensureStyles, nativeRandom } from '../src/engine/styles';
import { StyleError } from '../src/engine/force/types';
import { RanPark } from '../src/engine-gpl/random_park';
import { RanMars } from '../src/engine-gpl/random_mars';

/*
 * The per-atom random streams of native LAMMPS (src/engine-gpl/, reached through nativeRandom.ts): the
 * coordinate hash of RanPark::reset(int, double *), the Marsaglia subset selection, and the commands that use
 * them. The expected displacements are native's (black box): displace_atoms all random 1 1 1 4321 units box on
 * the five atoms below, with the output written at %.17g, so each engine value must equal it bit for bit.
 */

// the coordinates as the data file writes them (read_data parses the same text the same way)
const ATOMS: [string, string, string][] = [
  ['0', '0', '0'],
  ['-1.25', '0.5', '-3.75'],
  ['2.5', '-7.125', '0.1'],
  ['-12.3456789012345', '0.001', '9.5'],
  ['-0.3', '-0.3', '-0.3'],
];

// native displace_atoms random 1 1 1 4321 (x, y, z after the displacement; %.17g)
const NATIVE_DISPLACED: [string, string, string][] = [
  ['-0.11793529108070544', '-0.13843719341719396', '-0.71390976277827733'],
  ['-1.6668728428971362', '0.1181294278326116', '-3.848706417297342'],
  ['3.101159653440658', '-7.434704622863654', '0.89440353056155308'],
  ['-12.977670840682316', '0.11247370054920836', '9.0384851305459097'],
  ['-0.72778189733055498', '-0.030348434639325628', '-0.26614098314481821'],
];

const DATA = ['hash probe', '', `${ATOMS.length} atoms`, '1 atom types', '', '-20 20 xlo xhi', '-20 20 ylo yhi',
  '-20 20 zlo zhi', '', 'Atoms # atomic', '', ...ATOMS.map(([x, y, z], i) => `${i + 1} 1 ${x} ${y} ${z}`)].join('\n');

const xyzOf = ([x, y, z]: [string, string, string]): [number, number, number] => [Number(x), Number(y), Number(z)];

describe('native random streams (before the lazy group is loaded)', () => {
  it('a hook used before its group is loaded is an error naming the command word, never another stream', () => {
    expect(() => nativeRandom('geom')).toThrow(StyleError);
    expect(() => nativeRandom('geom')).toThrow(/geom: the native random streams are not loaded/);
  });
});

describe('RanPark (native port)', () => {
  it('refuses a seed <= 0 in the constructor and in reset', () => {
    expect(() => new RanPark(0)).toThrow('Invalid seed for Park random # generator');
    expect(() => new RanPark(-3)).toThrow('Invalid seed for Park random # generator');
    expect(() => new RanPark(1).reset(0)).toThrow('Invalid seed for Park random # generator');
  });

  it('gives uniform numbers in (0, 1) and a fresh stream after each reset', () => {
    const rp = new RanPark(1);
    const first = [rp.uniform(), rp.uniform(), rp.uniform()];
    for (const u of first) expect(u > 0 && u < 1).toBe(true);
    rp.reset(1);
    expect([rp.uniform(), rp.uniform(), rp.uniform()]).toEqual(first);
  });

  it('hashes negative, zero and sign-only coordinates into different streams (signed bytes)', () => {
    const a = new RanPark(1);
    a.resetAtom(4321, 0, 0, 0);
    const b = new RanPark(1);
    b.resetAtom(4321, -0, 0, 0);     // only the sign bit of x differs: the hash reads it as byte 0x80 (signed -128)
    const c = new RanPark(1);
    c.resetAtom(4321, -1.25, 0.5, -3.75);
    const d = new RanPark(1);
    d.resetAtom(4321, -1.25, 0.5, -3.75);
    const ua = a.uniform(), ub = b.uniform(), uc = c.uniform();
    expect(ua).not.toBe(ub);
    expect(uc).not.toBe(ua);
    expect(d.uniform()).toBe(uc);     // same bits, same stream
  });

  it('keeps the second value of a gaussian pair for the next call, and reset drops it', () => {
    const rp = new RanPark(7);
    const g1 = rp.gaussian();
    const g2 = rp.gaussian();              // the kept value of the same pair
    expect(Number.isFinite(g1) && Number.isFinite(g2)).toBe(true);
    rp.reset(7);
    expect(rp.gaussian()).toBe(g1);        // reset gives the same first value again
    expect(rp.gaussian()).toBe(g2);
  });
});

describe('RanMars (native port)', () => {
  it('refuses seeds outside 1..900000000 (the constructor of RanMars)', () => {
    expect(() => new RanMars(0)).toThrow(StyleError);
    expect(() => new RanMars(900000001)).toThrow(/Invalid seed/);
  });

  it('selects exactly ntarget items, and none or all at the ends', () => {
    for (const [ntarget, n] of [[0, 50], [50, 50], [17, 50], [94, 256]] as const) {
      const mark = new Uint8Array(n);
      const next = new Int32Array(n);
      new RanMars(4321).selectSubset(ntarget, n, mark, next);
      expect(mark.reduce((s, m) => s + m, 0)).toBe(ntarget);
    }
  });
});

describe('the hooks (after ensureStyles loads the group)', () => {
  it('displace_atoms random matches native bit for bit on the edge-case coordinates (3d)', async () => {
    await ensureStyles(['random']);
    const h = nativeRandom('random');
    const out = new Float64Array(3);
    ATOMS.forEach((p, i) => {
      const [x, y, z] = xyzOf(p);
      h.displaceRandom(4321, x, y, z, 1, 1, 1, 3, out);
      const [nx, ny, nz] = xyzOf(NATIVE_DISPLACED[i]);
      expect(x + out[0]).toBe(nx);
      expect(y + out[1]).toBe(ny);
      expect(z + out[2]).toBe(nz);
    });
  });

  it('2d displacement draws x and y only, the same as the first two draws in 3d', () => {
    const h = nativeRandom('random');
    const o2 = new Float64Array(3), o3 = new Float64Array(3);
    h.displaceRandom(4321, 1.5, -2.25, 0, 0.3, 0.2, 0.1, 2, o2);
    h.displaceRandom(4321, 1.5, -2.25, 0, 0.3, 0.2, 0.1, 3, o3);
    expect(o2[0]).toBe(o3[0]);
    expect(o2[1]).toBe(o3[1]);
    expect(o2[2]).toBe(0);
  });

  it('velocity loop geom draws three deviates in [-0.5, 0.5) (uniform) or finite gaussians', async () => {
    await ensureStyles(['geom']);
    const h = nativeRandom('geom');
    const out = new Float64Array(3);
    h.velocityGeom(87287, 0.8, 0.8, 0, false, out);
    for (const u of out) expect(u >= -0.5 && u < 0.5).toBe(true);
    h.velocityGeom(87287, 0.8, 0.8, 0, true, out);
    for (const g of out) expect(Number.isFinite(g)).toBe(true);
  });

  it('type/fraction is a threshold on the atom stream; subset selection gives ntarget atoms', async () => {
    await ensureStyles(['type/fraction', 'type/subset']);
    const h = nativeRandom('type/fraction');
    expect(h.typeFraction(4321, 1, 2, 3, 0)).toBe(false);   // uniform() > 0 is always true, so no atom changes
    expect(h.typeFraction(4321, 1, 2, 3, 1)).toBe(true);    // uniform() < 1 always
    const flag = nativeRandom('type/subset').selectSubset(4321, 40, 256);
    expect(flag.reduce((s, m) => s + m, 0)).toBe(40);
  });

  it('dipole, spin and quaternion hooks give the documented normalisation', async () => {
    await ensureStyles(['dipole/random', 'spin/atom/random', 'quat/random', 'theta/random']);
    const mu = new Float64Array(3);
    nativeRandom('dipole/random').dipoleRandom(777, 0.4, 1.1, -2.2, 1.7, 3, mu);
    expect(Math.hypot(mu[0], mu[1], mu[2])).toBeCloseTo(1.7, 14);
    const sp = new Float64Array(3);
    nativeRandom('spin/atom/random').spinRandom(777, 0.4, 1.1, -2.2, 2, sp);
    expect(sp[2]).toBe(0);
    expect(Math.hypot(sp[0], sp[1])).toBeCloseTo(1, 14);
    const q = new Float64Array(4);
    nativeRandom('quat/random').quatRandom(999, 0.4, 1.1, -2.2, 3, q);
    expect(Math.hypot(q[0], q[1], q[2], q[3])).toBeCloseTo(1, 14);
    nativeRandom('quat/random').quatRandom(999, 0.4, 1.1, -2.2, 2, q);
    expect(q[1]).toBe(0);
    expect(q[2]).toBe(0);
    const th = nativeRandom('theta/random').thetaRandom(5, 0, 0, 0);
    expect(th >= 0 && th < 2 * Math.PI).toBe(true);
  });
});

describe('through the session (the lazy group loads when a command names its word)', () => {
  it('displace_atoms random on read_data atoms matches native bit for bit', async () => {
    const events: { kind: string }[] = [];
    const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, undefined, 0, 1);
    session.addFile('hash.data', DATA);
    await session.execute('units lj\natom_style atomic\nread_data hash.data\ndisplace_atoms all random 1 1 1 4321 units box\n');
    const x = session.sys.state.x as Float64Array;
    NATIVE_DISPLACED.forEach((p, i) => {
      const [nx, ny, nz] = xyzOf(p);
      expect(x[3 * i]).toBe(nx);
      expect(x[3 * i + 1]).toBe(ny);
      expect(x[3 * i + 2]).toBe(nz);
    });
  });

  it('velocity loop geom and set type/fraction run through the session without a warning', async () => {
    const warnings: string[] = [];
    const session = new Session({ emit: (e) => { if (e.kind === 'log' && /WARNING/.test(e.text)) warnings.push(e.text); }, writeFile: () => {} }, undefined, 0, 1);
    await session.execute([
      'units lj', 'atom_style atomic', 'lattice fcc 0.8442', 'region box block 0 3 0 3 0 3', 'create_box 2 box',
      'create_atoms 1 box', 'mass * 1.0', 'velocity all create 1.44 87287 loop geom',
      'set group all type/fraction 2 0.5 4321', 'pair_style lj/cut 2.5', 'pair_coeff * * 1.0 1.0 2.5', 'run 0', '',
    ].join('\n'));
    expect(warnings.filter((w) => /native LAMMPS/.test(w))).toEqual([]);
    const types = session.sys.state.type as Int32Array;
    expect(types.some((t) => t === 2) && types.some((t) => t === 1)).toBe(true);
  });

  it('every set random keyword stops for a seed above 900000000, as native (set.cpp setrandom builds a RanMars)', async () => {
    // Measured with native LAMMPS: Invalid seed 900000001 for the Marsaglia random # generator; 900000000 is accepted.
    const run = async (style: string, kw: string) => {
      const errors: string[] = [];
      const session = new Session({ emit: (e) => { if (e.kind === 'error') errors.push(e.message); }, writeFile: () => {} }, undefined, 0, 1);
      await session.execute([
        'units lj', `atom_style ${style}`, 'lattice sc 1.0', 'region b block 0 2 0 2 0 2', 'create_box 2 b',
        'create_atoms 1 box', `set group all ${kw}`, '',
      ].join('\n')).catch(() => {});
      return errors.join(' ');
    };
    for (const [style, kw] of [
      ['atomic', 'type/fraction 2 0.5 900000001'], ['atomic', 'type/ratio 2 0.5 900000001'],
      ['hybrid sphere dipole', 'dipole/random 900000001 1.0'], ['ellipsoid', 'quat/random 900000001'],
      ['spin', 'spin/atom/random 900000001 1.0'], ['spin', 'spin/random 900000001 1.0'],
    ]) {
      expect(await run(style, kw), kw).toMatch(/Invalid seed 900000001 for the Marsaglia random # generator/);
    }
    expect(await run('atomic', 'type/fraction 2 0.5 900000000')).toBe('');
  });
});

