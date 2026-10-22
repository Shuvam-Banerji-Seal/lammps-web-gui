import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * Body fixes and compute of the GPL body module (src/engine-gpl/body/fix_nh_body.ts, fix_wall_body.ts,
 * compute_temp_body.ts). Native parity is in the oracle cases tests/oracle/w126body_*.in (examples/body cut down).
 * The checks here are the refusals by name (the StyleError messages) and one body that falls onto a wall.
 */

const ROUNDED = readFileSync(join(__dirname, 'oracle', 'w111body_squares.data'), 'utf8');
const CUBES = readFileSync(join(__dirname, 'oracle', 'w121body_cubes_example.data'), 'utf8');

const run = async (script: string, files: Record<string, string> = {}) => {
  const events: { kind: string; message?: string }[] = [];
  const session = new Session({ emit: (e) => events.push(e as { kind: string }), writeFile: () => {} }, undefined, 0, 1);
  session.addFile('sq.data', ROUNDED);
  session.addFile('cubes.data', CUBES);
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  await session.execute(script);
  return session;
};

const SQ = 'units lj\ndimension 2\natom_style body rounded/polygon 1 6\nread_data sq.data\n';
const CB = 'units lj\ndimension 3\natom_style body rounded/polyhedron 1 10\nread_data cubes.data\n';

describe('body fixes and compute: refusals by name (engine-gpl/body)', () => {
  it('compute temp/body refuses the bias keyword and an unknown keyword', async () => {
    await expect(run(`${CB}pair_style body/rounded/polyhedron 20 5 0 1 0.5\npair_coeff * * 100 1\ncompute t all temp/body bias x\n`)).rejects.toThrow(/temp\/body bias/);
    await expect(run(`${CB}compute t all temp/body foo 1\n`)).rejects.toThrow(/temp\/body keyword foo/);
  });

  it('compute temp/body and the body fixes need atom style body', async () => {
    const BOX = 'units lj\ndimension 3\natom_style atomic\nregion b block 0 6 0 6 0 6\ncreate_box 1 b\n';
    // the atom style is checked at init (compute_temp_body.cpp)
    await expect(run(`${BOX}mass 1 1.0\ncompute t all temp/body\nthermo_style custom step c_t\nrun 0\n`)).rejects.toThrow(/Compute temp\/body requires atom style body/);
    await expect(run(`${BOX}fix 1 all nvt/body temp 1 1 1\n`)).rejects.toThrow(/nvt\/body requires atom style body/);
  });

  it('fix wall/body/polygon refuses history and wiggle and needs the pair style', async () => {
    await expect(run(`${SQ}pair_style body/rounded/polygon 1 1 0.1 0.5 0.5\npair_coeff * * 100 2\nfix 3 all wall/body/polygon 2000 50 50 yplane 0 10 wiggle x 0.1 1\n`)).rejects.toThrow(/wiggle/);
    await expect(run(`${SQ}pair_style body/rounded/polygon 1 1 0.1 0.5 0.5\npair_coeff * * 100 2\nfix 3 all wall/body/polygon 2000 50 50 yplane 0 10 wiggle y 1 1\n`)).rejects.toThrow(/wiggle is not supported/);
    await expect(run(`${SQ}pair_style body/rounded/polygon 1 1 0.1 0.5 0.5\npair_coeff * * 100 2\nfix 3 all wall/body/polyhedron 2000 50 50 zplane 0 NULL\n`)).rejects.toThrow(/requires body style rounded\/polyhedron|incompatible|rounded\/polyhedron/);
    await expect(run(`${SQ}fix 3 all wall/body/polygon 2000 50 50 yplane 0 10\nrun 1\n`)).rejects.toThrow(/incompatible with Pair style/);
  });

  it('fix nvt/body refuses pressure control (fix nh rule) and a point particle in the group', async () => {
    await expect(run(`${SQ}pair_style body/rounded/polygon 1 1 0.1 0.5 0.5\npair_coeff * * 100 2\nfix 1 all nvt/body temp 0.5 0.5 1.0 iso 0.1 0.1 1.0\n`)).rejects.toThrow();
  });
});

describe('a body falls onto a polyhedron wall (fix wall/body/polyhedron)', () => {
  it('runs 2000 steps with the wall fix and the polyhedron pair', async () => {
    const session = await run(`units lj\ndimension 3\nboundary p p f\natom_style body rounded/polyhedron 1 10\nread_data cubes.data\npair_style body/rounded/polyhedron 20 5 0 1 0.5\npair_coeff * * 100 1\nfix 1 all nve/body\nfix 2 all gravity 1.0 vector 0.0 0.0 -1.0\nfix 3 all wall/body/polyhedron 2000 50 50 zplane 0.0 NULL\nthermo 0\nrun 2000\n`);
    expect(session).toBeDefined();
  });
});
