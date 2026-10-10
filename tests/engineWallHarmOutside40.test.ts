import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

// fix wall/harmonic/outside (docs.lammps.org/fix_wall.html); the energies below are the native values
// measured for the fix comment in src/engine/fix/walls.ts (one atom, lj units, boundary f f f).
const run = async (x: number, wall: string) => {
  const s = new Session({ emit: () => {}, writeFile: () => {} } as never);
  await s.execute(`units lj
boundary f f f
region b block -5 5 -5 5 -5 5
create_box 1 b
create_atoms 1 single ${x} 0 0
mass 1 1.0
pair_style zero 1.0
pair_coeff * *
fix w all wall/harmonic/outside ${wall} units box
fix_modify w energy yes
thermo_style custom step pe f_w f_w[1]
run 0
`);
  const f = s.sys.fixes.find((q) => q.id === 'w')!;
  return { e: f.computeScalar(), f1: f.computeVector(0) };
};

describe('fix wall/harmonic/outside', () => {
  it('pulls an atom beyond an lo wall back with E = eps d^2, F = 2 eps d', async () => {
    const r = await run(-2, 'xlo 0.0 1.0 1.0 2.5');
    expect(r.e).toBeCloseTo(4, 12);
    expect(r.f1).toBeCloseTo(-4, 12);
  });
  it('ignores atoms on the inner side and beyond the cutoff (exclusive)', async () => {
    expect((await run(0.5, 'xlo 0.0 1.0 1.0 2.5')).e).toBe(0);
    expect((await run(0.0, 'xlo 0.0 1.0 1.0 2.5')).e).toBe(0);
    expect((await run(-2.5, 'xlo 0.0 1.0 1.0 2.5')).e).toBe(0);
    expect((await run(-0.99, 'xlo 0.0 1.0 1.0 1.0')).e).toBeCloseTo(0.9801, 12);
  });
  it('acts on the outside of a hi wall', async () => {
    const r = await run(2, 'xhi 0.0 1.0 1.0 2.5');
    expect(r.e).toBeCloseTo(4, 12);
    expect(r.f1).toBeCloseTo(4, 12);
  });
});
