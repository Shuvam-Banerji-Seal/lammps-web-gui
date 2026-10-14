import { describe, expect, it } from 'vitest';
import { parseSurfCoeff } from '../src/engine-gpl/gransurf/pair_surf_granular';
import { overlapSphereLine, overlapSphereTri, pointConnection2d, surfContactForces, contactLess, FLAT, NONFLAT, SAME_SIDE, OPPOSITE_SIDE, EXTERNAL, INTERNAL, UNCONNECTED } from '../src/engine-gpl/gransurf/surf_contact';
import { buildConnect2d, lineEnds } from '../src/engine-gpl/gransurf/connect2d';
import { buildConnect3d, normOf } from '../src/engine-gpl/gransurf/connect3d';
import { StyleError } from '../src/engine/force/types';
import { FIXES } from '../src/engine-gpl/registry/gransurf';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import { parseMoleculeFile } from '../src/engine/molecule';

describe('pair_coeff keywords (pair_granular.html)', () => {
  it('accepts hooke, tangential linear_history and damping mass_velocity', () => {
    const c = parseSurfCoeff(['hooke', '4000.0', '100.0', 'tangential', 'linear_history', '1000.0', '1.0', '0.5', 'damping', 'mass_velocity']);
    expect(c).toMatchObject({ k: 4000, damp: 100, kt: 1000, xt: 1, mu: 0.5 });
  });
  it('refuses the granular models that are not ported', () => {
    expect(() => parseSurfCoeff(['hooke', '1', '1', 'tangential', 'linear_nohistory', '1', '1', 'damping', 'mass_velocity'])).toThrow(StyleError);
    expect(() => parseSurfCoeff(['hooke', '1', '1', 'tangential', 'linear_history', '1', '1', '1', 'damping', 'viscoelastic'])).toThrow(StyleError);
    expect(() => parseSurfCoeff(['hertz', '1', '1', 'tangential', 'linear_history', '1', '1', '1', 'damping', 'mass_velocity'])).toThrow(StyleError);
    expect(() => parseSurfCoeff(['hooke', '1', '1', 'rolling', 'sds', '1', '1', '1'])).toThrow(StyleError);
  });
});

describe('sphere-surface overlap', () => {
  const p1 = [-5, 0, 0], p2 = [5, 0, 0];
  it('reports the interior of a segment and the end points', () => {
    expect(overlapSphereLine([0, 0.3, 0], 0, 0.5, p1, p2)?.flag).toBe(1);
    expect(overlapSphereLine([5.2, 0.3, 0], 0, 0.5, p1, p2)?.flag).toBe(-2);
    expect(overlapSphereLine([-5.2, 0.3, 0], 0, 0.5, p1, p2)?.flag).toBe(-1);
  });
  it('reports no contact beyond the radius', () => {
    expect(overlapSphereLine([0, 0.6, 0], 0, 0.5, p1, p2)).toBeNull();
  });
  const a = [-3, -3, 0], b = [3, -3, 0], c = [-3, 3, 0], n = [0, 0, 1];
  it('classifies a triangle face, its edges and its corners', () => {
    expect(overlapSphereTri([-1, -1, 0.3], 0, 0.5, a, b, c, n)?.flag).toBe(1);
    expect(overlapSphereTri([0.3, -3.1, 0.2], 0, 0.5, a, b, c, n)?.flag).toBe(-1);
    expect(overlapSphereTri([0.3, 0.3, 0.2], 0, 0.5, a, b, c, n)?.flag).toBe(-2);
    expect(overlapSphereTri([3.3, -3.2, 0.1], 0, 0.5, a, b, c, n)?.flag).toBe(-5);
    expect(overlapSphereTri([-3.3, -3.3, 0.1], 0, 0.5, a, b, c, n)?.flag).toBe(-4);
  });
});

describe('connection flags (fix_surface.cpp)', () => {
  it('gives a FLAT, SAME-side pair for two collinear lines sharing a point', () => {
    const f = pointConnection2d([0, 1, 0], [0, 1, 0], 1, 0, 0.00015);
    expect(f.fflag).toBe(FLAT);
    expect(f.nside).toBe(SAME_SIDE);
  });
  it('gives NONFLAT for a bent pair and the same end point on both', () => {
    expect(pointConnection2d([0, 1, 0], [0.6, 0.8, 0], 0, 0, 0.00015).fflag).toBe(NONFLAT);
    expect(pointConnection2d([0, 1, 0], [0, 1, 0], 0, 0, 0.00015).nside).toBe(OPPOSITE_SIDE);
  });
});

describe('2d connectivity', () => {
  it('marks a single line as unconnected at both ends and a bent joint as external', () => {
    const x = new Float64Array([0, 0, 0, 0, 0, 0]);
    const bonus = new Float64Array([10, 0, 10, 0]);
    // one line
    const one = buildConnect2d(x.subarray(0, 3), [0], bonus.subarray(0, 2), [1], 0.00015);
    expect(one.get(1)!.external).toEqual([UNCONNECTED, UNCONNECTED]);
    // two lines: (-5,0)-(0,0) and (0,0)-(5,3) (a bent joint at the origin)
    const xs = new Float64Array([-2.5, 0, 0, 2.5, 1.5, 0]);
    const bs = new Float64Array([5, 0, 5.830951894845301, Math.atan2(3, 5)]);
    const two = buildConnect2d(xs, [0, 1], bs, [1, 2], 0.00015);
    expect(two.get(1)!.p[1].map((nb) => nb.tag)).toEqual([2]);
    expect(two.get(1)!.external[1]).toBe(EXTERNAL);
    expect(two.get(1)!.external[0]).toBe(UNCONNECTED);
    expect(two.get(2)!.p[0][0].fflag).toBe(NONFLAT);
    expect(INTERNAL).toBe(0);
  });
  it('computes end points and the unit normal of a segment', () => {
    const e = lineEnds(new Float64Array([0, 0, 0]), 0, 10, 0);
    expect(e.e0).toEqual([-5, 0, 0]);
    expect(e.e1).toEqual([5, 0, 0]);
    expect(e.norm[1]).toBeCloseTo(1, 12);
  });
});

describe('3d connectivity and normals', () => {
  it('finds the edge neighbour of two triangles sharing an edge', () => {
    const quat = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0]);
    // body corners in the principal frame equal to the space corners (identity quaternion)
    const bonus = new Float64Array(24);
    const x = new Float64Array([0, 0, 0, 0, 0, 0]);
    const corners = [[[-3, -3, 0], [3, -3, 0], [-3, 3, 0]], [[3, -3, 0], [3, 3, 0], [-3, 3, 0]]];
    corners.forEach((tri, i) => {
      tri.forEach((p, k) => {
        bonus[12 * i + 3 * k] = p[0];
        bonus[12 * i + 3 * k + 1] = p[1];
        bonus[12 * i + 3 * k + 2] = p[2];
      });
    });
    const conn = buildConnect3d(x, [0, 1], quat, bonus, 12, new Float64Array([4.5, 4.5]), [1, 2], 0.00015);
    expect(conn.get(1)!.e.flat().map((nb) => nb.tag)).toEqual([2]);
    expect(conn.get(1)!.externalEdge.filter((v) => v === EXTERNAL).length).toBe(0);
    expect(conn.get(1)!.externalCor.every((v) => v === INTERNAL || v === UNCONNECTED)).toBe(true);
  });
  it('gives the unit normal (p2-p1) x (p3-p1)', () => {
    const nrm = normOf([[0, 0, 0], [1, 0, 0], [0, 1, 0]]);
    expect(nrm).toEqual([0, 0, 1]);
  });
});

describe('contact model (granular_model.cpp, hooke + linear_history + mass_velocity)', () => {
  const coeffs = { k: 4000, damp: 100, kt: 1000, xt: 1, mu: 0.5 };
  it('gives the hooke normal and mass-velocity damping at rest tangentially', () => {
    const hist = new Float64Array(3);
    const out = { forces: [0, 0, 0] as [number, number, number], torque: [0, 0, 0] as [number, number, number] };
    // sphere radius 0.5 at distance 0.49 along +z from the contact point, approaching with vz = -1 (meff 1)
    surfContactForces(coeffs, {
      dx: [0, 0, 0.49], radi: 0.5, vi: [0, 0, -1], omegai: [0, 0, 0], vj: [0, 0, 0], meff: 1, dt: 0.001, historyUpdate: false,
    }, hist, out);
    // delta = 0.01: Fn = k delta + damp * meff * (-vn) = 40 + 100
    expect(out.forces[2]).toBeCloseTo(40 + 100, 9);
    expect(out.forces[0]).toBe(0);
    expect(out.torque.every((t) => t === 0)).toBe(true);
  });
  it('keeps the tangential force inside the Coulomb cone', () => {
    const hist = new Float64Array(3);
    const out = { forces: [0, 0, 0] as [number, number, number], torque: [0, 0, 0] as [number, number, number] };
    for (let step = 0; step < 50; step++) {
      surfContactForces(coeffs, {
        dx: [0, 0, 0.49], radi: 0.5, vi: [5, 0, 0], omegai: [0, 0, 0], vj: [0, 0, 0], meff: 1, dt: 0.001, historyUpdate: true,
      }, hist, out);
    }
    const fn = out.forces[2];
    const ft = Math.hypot(out.forces[0], out.forces[1]);
    expect(ft).toBeLessThanOrEqual(0.5 * Math.abs(fn) * (1 + 1e-9));
  });
});

describe('contact ordering (fix_surface.cpp: contact_compare)', () => {
  const base = { index: 1, tag: 1, type: 2, flag: 1, rmag: 0, nside: 0, external: 0, priority: 2, convexIndex: -1, overlap: 0.1,
    rsqCom: 1, weight: 1, dr: [0, 0, 1] as [number, number, number], surfNorm: [0, 0, 1] as [number, number, number],
    drForce: [0, 0, 1] as [number, number, number], cindex: [], caflag: [], cwhich: [], ends: [[0, 0, 0], [0, 0, 0]] as never,
    corners: [[0, 0, 0], [0, 0, 0], [0, 0, 0]] as never, ck1: -1, ck2: -1, caflag1: -1, caflag2: -1 };
  it('prefers larger overlap, then the interior priority', () => {
    expect(contactLess({ ...base, overlap: 0.2 }, { ...base, overlap: 0.1 }, false)).toBe(true);
    expect(contactLess({ ...base, priority: 2 }, { ...base, priority: 1 }, false)).toBe(true);
  });
});

describe('style registry', () => {
  it('registers fix surface/global and fix surface/local', () => {
    expect(typeof FIXES['surface/global']).toBe('function');
    expect(typeof FIXES['surface/local']).toBe('function');
  });
});

describe('fix surface/global refusals (fix_surface_global.cpp keywords not ported)', () => {
  const run = async (text: string): Promise<string | null> => {
    let err: string | null = null;
    const s = new Session({ emit: (e: { kind: string; message?: string }) => { if (e.kind === 'error') err ??= e.message ?? ''; }, writeFile: () => {} } as never);
    s.addFile('w127gsg_surf.tri.stl', readFileSync(join(__dirname, 'oracle', 'w127gsg_surf.tri.stl'), 'utf8'));
    try { await s.execute(text); } catch (e) { err ??= String(e); }
    return err;
  };
  const head = `units lj
atom_style sphere
dimension 3
region box block -7 7 -7 7 0 9 units box
create_box 2 box
pair_style granular
pair_coeff * * hooke 4000.0 100.0 tangential linear_history 1000.0 1.0 0.5 damping mass_velocity
`;
  const fix = (extra: string): string => `fix 20 all surface/global input stl 2 1 w127gsg_surf.tri.stl model * * granular hooke 4000.0 100.0 tangential linear_history 1000.0 1.0 0.5 damping mass_velocity ${extra}
`;
  it('builds from an STL file and a molecule-free model', async () => {
    expect(await run(head + fix(''))).toBeNull();
  });
  it('names the keywords it does not support', async () => {
    expect(await run(head + fix('flat 10'))).toMatch(/keyword 'flat' is not supported/);
    expect(await run(head + fix('smaxtype 3'))).toMatch(/keyword 'smaxtype' is not supported/);
  });
  it('refuses a move style other than rotate', async () => {
    expect(await run(head + fix('') + 'fix_modify 20 move * linear 1 0 0\n')).toMatch(/move style 'linear' is not supported/);
  });
});

describe('pair surf/granular boundaries', () => {
  it('refuses a periodic dimension in the surface plane (connectivity is not periodic yet), accepts f f p in 2d', async () => {
    const dir = join(__dirname, 'oracle');
    const input = readFileSync(join(dir, 'w64surf_line_head.in'), 'utf8');
    const data = readFileSync(join(dir, 'w64surf_line_head.data'), 'utf8');
    const run = async (text: string): Promise<string | null> => {
      let err: string | null = null;
      const s = new Session({ emit: (e: { kind: string; message?: string }) => { if (e.kind === 'error') err ??= e.message ?? ''; }, writeFile: () => {} } as never);
      s.addFile('w64surf_line_head.data', data);
      try { await s.execute(text.replace(/^run\s+\d+/m, 'run 0')); } catch (e) { err ??= String(e); }
      return err;
    };
    expect(await run(input)).toBeNull();
    expect(await run(input.replace(/^boundary\s+f f p/m, 'boundary p f p'))).toMatch(/periodic boundaries is not supported/);
  });
});

describe('molecule surface templates (molecule.html: Lines and Triangles sections)', () => {
  const O = { toff: 0, boff: 0, aoff: 0, doff: 0, ioff: 0, scale: 1 };
  it('reads a lines template without atoms (line syntax ID molecule-ID type x1 y1 x2 y2)', () => {
    const t = parseMoleculeFile('lines', 'surf.line', 'LAMMPS molecule file\n\n2 lines\n\nLines\n\n1 1 2 7 0 10 10\n2 1 2 10 10 0 7\n', { ...O, toff: 1, scale: 2 });
    expect(t.natoms).toBe(0);
    expect(t.lines).toEqual([
      { mol: 1, type: 3, p: [14, 0, 20, 20] },
      { mol: 1, type: 3, p: [20, 20, 0, 14] },
    ]);
    expect(t.tris).toBeNull();
  });
  it('reads a triangles template (line syntax ID molecule-ID type x1 y1 z1 x2 y2 z2 x3 y3 z3)', () => {
    const t = parseMoleculeFile('tris', 'surf.tri', 'LAMMPS data file\n\n1 triangles\n\nTriangles\n\n1 1 2 -6 -6 0 6 -6 0 0 0 0\n', O);
    expect(t.tris).toEqual([{ mol: 1, type: 2, p: [-6, -6, 0, 6, -6, 0, 0, 0, 0] }]);
    expect(t.lines).toBeNull();
  });
  it('refuses a missing entry, a bad molecule-ID and a file with both atoms and lines', () => {
    expect(() => parseMoleculeFile('m', 'f', '#\n2 lines\n\nLines\n\n1 1 2 0 0 1 1\n1 1 2 0 0 1 1\n', O)).toThrow(/line 2 missing/);
    expect(() => parseMoleculeFile('m', 'f', '#\n1 lines\n\nLines\n\n1 -1 2 0 0 1 1\n', O)).toThrow(/invalid molecule ID/);
    expect(() => parseMoleculeFile('m', 'f', '#\n1 atoms\n1 lines\n\nLines\n\n1 1 2 0 0 1 1\n', O)).toThrow(/only one/);
  });
});

describe('fix surface/local input mol (fix_surface_local.html: input mol)', () => {
  const run = async (text: string): Promise<{ error: string | null; atoms: number | null }> => {
    let error: string | null = null;
    let atoms: number | null = null;
    const s = new Session({
      emit: (e: { kind: string; message?: string; row?: { atoms?: number } }) => {
        if (e.kind === 'error') error ??= e.message ?? '';
        if (e.kind === 'thermo' && e.row?.atoms !== undefined) atoms = e.row.atoms;
      },
      writeFile: () => {},
    } as never);
    s.addFile('w106gs_surf.line', readFileSync(join(__dirname, 'oracle', 'w106gs_surf.line'), 'utf8'));
    s.addFile('w127gsg_surf.tri.stl', readFileSync(join(__dirname, 'oracle', 'w127gsg_surf.tri.stl'), 'utf8'));
    s.addFile('w151gs_second.stl', 'solid two\n facet normal 0 0 0\n  outer loop\n   vertex -1 -1 4\n   vertex 1 -1 4\n   vertex 0 1 4\n  endloop\n endfacet\n facet normal 0 0 0\n  outer loop\n   vertex 0 0 4\n   vertex 0 0 4\n   vertex 1 1 4\n  endloop\n endfacet\nendsolid two\n');
    try { await s.execute(text); } catch (e) { error ??= String(e); }
    return { error, atoms };
  };
  const head = `units lj
dimension 2
atom_style line
boundary f f p
region box block -15 15 -15 15 -0.5 0.5 units box
create_box 2 box
molecule lines w106gs_surf.line
`;
  const head3 = `units lj
atom_style tri
dimension 3
boundary f f f
region box block -7 7 -7 7 0 9 units box
create_box 3 box
`;
  it('creates one line particle per Lines entry, numbered after the atoms that exist', async () => {
    const r = await run(`${head}fix 20 all surface/local input mol lines\nthermo_style custom step atoms\nrun 0\n`);
    expect(r.error).toBeNull();
    expect(r.atoms).toBe(9);
  });
  it('refuses an STL source in 2d, a zero surface type, a bad flat value and an unknown keyword by name', async () => {
    expect((await run(`${head}fix 20 all surface/local input stl 1 1 w127gsg_surf.tri.stl\n`)).error).toMatch(/STL file for 2d simulations/);
    expect((await run(`${head}fix 20 all surface/local input mol lines flat 91\n`)).error).toMatch(/Invalid value 91 for fix surface\/local flat/);
    expect((await run(`${head}fix 20 all surface/local input mol lines smaxtype 3\n`)).error).toMatch(/keyword 'smaxtype' is not supported/);
    expect((await run(`${head}fix 20 all surface/local input mol nope\n`)).error).toMatch(/does not exist/);
    expect((await run(`${head3}fix 20 all surface/local input stl 0 1 w127gsg_surf.tri.stl\n`)).error).toMatch(/STL surface type must be >= 1/);
    expect((await run(`${head3}fix 20 all surface/local input stl 2 1\n`)).error).toMatch(/needs type, molecule-ID and file name/);
  });
  it('accepts atom_style hybrid tri sphere, and its tri particles come from fix surface/local (in.screwfeeder.local)', async () => {
    const r = await run(`units lj
atom_style hybrid tri sphere 1
dimension 3
boundary f f f
region box block -7 7 -7 7 0 9 units box
create_box 3 box
fix 20 all surface/local input stl 2 1 w127gsg_surf.tri.stl
thermo_style custom step atoms
run 0
`);
    expect(r.error).toBeNull();
    expect(r.atoms).toBe(4);
  });
  it('accepts the flat keyword with input mol', async () => {
    const r = await run(`${head}fix 20 all surface/local input mol lines flat 10\nthermo_style custom step atoms\nrun 0\n`);
    expect(r.error).toBeNull();
    expect(r.atoms).toBe(9);
  });
  it('creates one tri particle per kept STL triangle (degenerate triangles skipped), from several input keywords', async () => {
    // w127gsg_surf.tri.stl: 4 triangles in the plane z = 0; w151gs_second.stl: one triangle at z = 4 and one degenerate
    const r = await run(`${head3}fix 20 all surface/local input stl 2 1 w127gsg_surf.tri.stl input stl 3 2 w151gs_second.stl flat 10\nthermo_style custom step atoms\nrun 0\n`);
    expect(r.error).toBeNull();
    expect(r.atoms).toBe(5);
  });
});
