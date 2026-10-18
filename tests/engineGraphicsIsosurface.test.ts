import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix graphics/isosurface (GPL port of src/GRAPHICS/fix_graphics_isosurface.cpp, src/engine-gpl/graphics): the binary
 * STL files, the triangles drawn by dump image and the argument errors. The STL text files agree with native LAMMPS
 * byte for byte in tests/oracle/w213iso_*.in.
 */

const run = async (input: string): Promise<{ files: Map<string, string>; error?: string }> => {
  const files = new Map<string, string>();
  const events: EngineEvent[] = [];
  const s = new Session({ emit: (e) => events.push(e), writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t) });
  try {
    await s.execute(input);
  } catch (e) {
    return { files, error: e instanceof Error ? e.message : String(e) };
  }
  const err = events.find((e) => e.kind === 'error');
  return { files, error: err && 'message' in err ? String(err.message) : undefined };
};

const bytesOf = (url: string): Buffer => Buffer.from(url.slice(url.indexOf(',') + 1), 'base64');

/** The cluster of tests/oracle/w213iso_cluster.in (56 LJ atoms of two types across the periodic x and y faces). */
const cluster = (fixes: string, extra = '', runSteps = 10): string => `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 7 0 7 0 7
create_box 2 box
region ball sphere 0.2 0.3 3.5 1.9
create_atoms 1 region ball
set type 1 type/fraction 2 0.3 12345
mass 1 1.0
mass 2 2.0
velocity all create 1.0 87287 loop geom
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0 2.5
neighbor 0.3 bin
neigh_modify every 1 delay 0 check no
group light type 1
fix 1 all nve
${fixes}
${extra}
run ${runSteps}
`;

describe('fix graphics/isosurface', () => {
  it('writes binary STL files byte for byte as native LAMMPS (binary yes, pad)', async () => {
    const r = await run(cluster([
      'fix iso light graphics/isosurface 5 0.5 1.0 quality min filename bin_a.*.stl binary yes',
      'fix iso2 all graphics/isosurface 10 1.2 0.8 quality low property mass filename bin_b.*.stl binary on pad 3',
    ].join('\n')));
    expect(r.error).toBeUndefined();
    // native files written by b332f28 from the same input (tests/fixtures/stl)
    for (const [mine, ref] of [['bin_a.0.stl', 'w213iso_bin_a.0.stl'], ['bin_b.010.stl', 'w213iso_bin_b.010.stl']]) {
      const url = r.files.get(mine)!;
      expect(url.startsWith('data:model/stl;base64,')).toBe(true);
      const got = bytesOf(url);
      const want = readFileSync(join(__dirname, 'fixtures', 'stl', ref));
      expect(Buffer.compare(got, want)).toBe(0);
    }
    expect([...r.files.keys()].filter((k) => k.endsWith('.stl')).sort()).toEqual(
      ['bin_a.0.stl', 'bin_a.10.stl', 'bin_a.5.stl', 'bin_b.000.stl', 'bin_b.010.stl']);
  });

  it('draws its triangles through the dump image fix keyword', async () => {
    const r = await run(cluster('fix iso light graphics/isosurface 5 0.5 1.0 quality min',
      'dump im all image 5 out.*.ppm type type size 48 48 view 60 30 box no 0 fix iso const 1 0\ndump_modify im acolor * white adiam * 0.01', 5));
    expect(r.error).toBeUndefined();
    const img = bytesOf(r.files.get('out.5.ppm')!);
    // P6 header "P6\n48 48\n255\n", then 48 x 48 RGB pixels: the atoms are drawn tiny, so the lit pixels are the surface
    const rgb = img.subarray(img.length - 48 * 48 * 3);
    let lit = 0;
    for (let k = 0; k < rgb.length; k += 3) if (rgb[k] + rgb[k + 1] + rgb[k + 2] > 0) lit++;
    expect(lit).toBeGreaterThan(100);
  });

  it('refuses arguments as the native constructor does', async () => {
    const err = async (fix: string): Promise<string | undefined> => (await run(cluster(fix, '', 0))).error;
    expect(await err('fix iso all graphics/isosurface 5 0.5')).toMatch(/missing argument/);
    expect(await err('fix iso all graphics/isosurface 0 0.5 1.0')).toMatch(/Illegal fix graphics\/isosurface nevery value 0/);
    expect(await err('fix iso all graphics/isosurface 5 0.5 -1.0')).toMatch(/Illegal fix graphics\/isosurface radius value -2/);
    expect(await err('fix iso all graphics/isosurface 5 0.5 1.0 quality ultra')).toMatch(/Unknown fix graphics\/isosurface quality setting ultra/);
    expect(await err('fix iso all graphics/isosurface 5 0.5 1.0 property c_nope')).toMatch(/Compute ID nope for fix graphics\/isosurface does not exist/);
    expect(await err('fix iso all graphics/isosurface 5 0.5 1.0 property c_thermo_temp')).toMatch(/does not calculate per-atom values/);
    expect(await err('variable e equal 1.0\nfix iso all graphics/isosurface 5 0.5 1.0 property v_e')).toMatch(/is not atom-style variable/);
    expect(await err('fix iso all graphics/isosurface 5 0.5 1.0 property f_nope')).toMatch(/Fix ID nope for fix graphics\/isosurface does not exist/);
    expect(await err('fix iso all graphics/isosurface 5 0.5 1.0 filename iso.stl')).toMatch(/Output to STL file requires file name with '\*'/);
    expect(await err('fix iso all graphics/isosurface 5 0.5 1.0 filename iso.*.stl.gz')).toMatch(/compressed STL output/);
    expect(await err('fix iso all graphics/isosurface 5 0.5 1.0 colour red')).toMatch(/Unknown fix graphics\/isosurface keyword colour/);
  });

  it('is refused for triclinic and 2d boxes, as natively', async () => {
    const tri = await run(`units lj
region box prism 0 5 0 5 0 5 1 0 0
create_box 1 box
create_atoms 1 single 1 1 1
mass 1 1.0
fix iso all graphics/isosurface 5 0.5 1.0
`);
    expect(tri.error).toMatch(/not compatible with triclinic cells/);
    const two = await run(`units lj
dimension 2
region box block 0 5 0 5 -0.5 0.5
create_box 1 box
create_atoms 1 single 1 1 0
mass 1 1.0
fix iso all graphics/isosurface 5 0.5 1.0
`);
    expect(two.error).toMatch(/not compatible with 2d systems/);
  });
});
