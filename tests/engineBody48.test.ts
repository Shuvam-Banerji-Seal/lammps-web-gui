import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { BODY_RECORD, eigenSym3, principalFrame, rotationToQuat, bodyRecordStride } from '../src/engine/bodies';
import { quatToMatrix } from '../src/engine/compute/asphere';

/*
 * atom_style body nparticle, read_data Bodies, pair_style body/nparticle and fix nve/body
 * (src/engine/bodies.ts, output/data.ts, force/pair/body_nparticle.ts, fix/nve_body.ts). Native parity on
 * free and interacting bodies and on examples/body/in.body (cut down) is in tests/oracle/w48bd_*.in.
 * The checks here: the inertia-tensor conventions measured with native LAMMPS (black box), the data-file
 * parsing and its refusals, the free rotation of one body, and the refused styles, each by name.
 */

// one body (3 sub-particles) and one point particle; the Bodies entry lists ixx iyy izz ixy ixz iyz first
const ONE = `one body and one point particle

2 atoms
1 bodies
2 atom types

-10 10 xlo xhi
-10 10 ylo yhi
-10 10 zlo zhi

Atoms

1 1 1 2.0 0.0 0.0 0.0 0 0 0
2 2 0 1.0 2.0 0.5 0.3 0 0 0

Bodies

1 1 15
3
1.0 2.0 1.5 0.3 0.2 0.1
0.6 0.4 0.2
-0.5 0.3 -0.4
0.1 -0.7 0.5
`;

const newSession = () => {
  const session = new Session({ emit: () => {}, writeFile: () => {} });
  return session;
};

/** Runs a script with the given data files; resolves to the session. */
const run = async (script: string, files: Record<string, string> = { 'one.data': ONE }) => {
  const session = newSession();
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  await session.execute(script);
  return session;
};

const HEAD = 'units lj\ndimension 3\natom_style body nparticle 1 4\nread_data one.data\n';

describe('principal frame of a body (bodies.ts)', () => {
  it('eigen-decomposition reconstructs the inertia tensor and gives a proper rotation', () => {
    const I = [[1.0, 0.3, 0.2], [0.3, 2.0, 0.1], [0.2, 0.1, 1.5]];
    const M = [I[0][0], I[0][1], I[0][2], I[1][0], I[1][1], I[1][2], I[2][0], I[2][1], I[2][2]];
    const { values, vectors } = eigenSym3(M);
    const E = vectors;
    const det = E[0] * (E[4] * E[8] - E[5] * E[7]) - E[1] * (E[3] * E[8] - E[5] * E[6]) + E[2] * (E[3] * E[7] - E[4] * E[6]);
    expect(det).toBeCloseTo(1, 12);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += E[3 * r + k] * values[k] * E[3 * c + k];
      expect(s).toBeCloseTo(M[3 * r + c], 11);
    }
  });

  it('the quaternion of the principal axes rebuilds the rotation matrix', () => {
    const I6 = [1.0, 2.0, 1.5, 0.3, 0.2, 0.1];
    const pf = principalFrame(I6);
    const q = pf.quat;
    const R = quatToMatrix(q[0], q[1], q[2], q[3]);
    for (let k = 0; k < 9; k++) expect(R[k]).toBeCloseTo(pf.E[k], 11);
    const back = rotationToQuat(R);
    // quaternions are equal up to sign
    const dot = back.reduce((a, v, k) => a + v * q[k], 0);
    expect(Math.abs(dot)).toBeCloseTo(1, 11);
  });

  it('a rod (zero moment about its axis, rounded to 1e-7) is accepted, a clearly negative moment is refused', () => {
    expect(() => principalFrame([0.5, 0.5, 1.0, 0, 0, 0])).not.toThrow();
    expect(() => principalFrame([0.5, 0.5, 1.0, 1e-7, 0, 0])).not.toThrow();
    expect(() => principalFrame([-1.0, 2.0, 1.0, 0, 0, 0])).toThrow(/negative principal moment/);
  });

  it('the record stride is 4 + 3 Nmax', () => {
    expect(bodyRecordStride(6)).toBe(22);
  });
});

describe('read_data Bodies (output/data.ts)', () => {
  it('stores the principal moments, the quaternion and the body-frame sub-particle displacements', async () => {
    const s = (await run(`${HEAD}`)).sys.state;
    expect(s.n).toBe(2);
    expect(s.rmass![0]).toBe(2.0);
    expect(s.rmass![1]).toBe(1.0);
    const rec = s.custom.get(BODY_RECORD)!;
    expect(rec.data[0]).toBe(3); // nsub of the body
    expect(rec.data[rec.cols]).toBe(0); // point particle
    const pf = principalFrame([1.0, 2.0, 1.5, 0.3, 0.2, 0.1]);
    for (let d = 0; d < 3; d++) expect(rec.data[1 + d]).toBeCloseTo(pf.moments[d], 12);
    // with the stored quaternion, the space displacements are the ones of the data file
    const R = quatToMatrix(s.quat![0], s.quat![1], s.quat![2], s.quat![3]);
    const want = [[0.6, 0.4, 0.2], [-0.5, 0.3, -0.4], [0.1, -0.7, 0.5]];
    for (let q = 0; q < 3; q++) {
      const b = [rec.data[4 + 3 * q], rec.data[5 + 3 * q], rec.data[6 + 3 * q]];
      for (let r = 0; r < 3; r++) {
        const sp = R[3 * r] * b[0] + R[3 * r + 1] * b[1] + R[3 * r + 2] * b[2];
        expect(sp).toBeCloseTo(want[q][r], 11);
      }
    }
  });

  it('refuses a body with the wrong number of integers or doubles, and a missing Bodies entry', async () => {
    const bad1 = ONE.replace('1 1 15\n3\n', '1 1 14\n3\n').replace('0.1 -0.7 0.5\n', '0.1 -0.7\n');
    await expect(run(HEAD, { 'one.data': bad1 })).rejects.toThrow(/needs 6 \+ 3N = 15 doubles/);
    const bad2 = ONE.replace('1 1 15\n3\n', '1 2 15\n3 3\n');
    await expect(run(HEAD, { 'one.data': bad2 })).rejects.toThrow(/Ninteger|one integer/);
    // five sub-particles with Nmax 4: native accepts it (Nmin and Nmax only size arrays), so the engine widens its record
    const five = ONE.slice(0, ONE.indexOf('Bodies')) + `Bodies

1 1 21
5
1.0 2.0 1.5 0.3 0.2 0.1
0.6 0.4 0.2
-0.5 0.3 -0.4
0.1 -0.7 0.5
0.2 0.2 0.2
0.3 0.1 -0.1
`;
    const s5 = (await run(HEAD, { 'one.data': five })).sys.state;
    const rec5 = s5.custom.get(BODY_RECORD)!;
    expect(rec5.cols).toBe(bodyRecordStride(5));
    expect(rec5.data[0]).toBe(5);
    // the Bodies entry names atom 1 only: without it the body atom has no record
    const noBodies = ONE.replace(/\nBodies[\s\S]*$/, '\n');
    await expect(run(HEAD, { 'one.data': noBodies.replace('1 bodies\n', '') })).rejects.toThrow(/bodyflag 1 but no Bodies entry|Bodies/);
  });

  it('refuses a bodies header with an atom style that has no bodies', async () => {
    await expect(run('units lj\natom_style atomic\nread_data one.data\n', { 'one.data': ONE })).rejects.toThrow(/No bodies allowed with this atom style/);
  });
});

describe('atom_style body and the refused styles', () => {
  it('names each refused body style', async () => {
    await expect(newSession().execute('atom_style body rounded/polygon 2 6\n')).resolves.toBeUndefined(); // accepted since wave 49 (engineBodyPoly49.test.ts)
    await expect(newSession().execute('atom_style body rounded/polyhedron 2 6\n')).resolves.toBeUndefined(); // accepted (engineBodyPolyhedron.test.ts)
    await expect(newSession().execute('atom_style body dummy 2 6\n')).rejects.toThrow(/body dummy/);
    // As native LAMMPS does (src/BODY/body_nparticle.cpp at b332f28): Nmin must be > 0 and <= Nmax
    await expect(newSession().execute('atom_style body nparticle -1 6\n')).rejects.toThrow(/Invalid body nparticle command/);
    await expect(newSession().execute('atom_style body nparticle 2 bogus\n')).rejects.toThrow("Expected integer parameter instead of 'bogus' in input script or data file");
    await expect(newSession().execute('atom_style hybrid charge body nparticle 2 6\n')).rejects.toThrow(/body/);
  });

  it('refuses create_atoms and write_data for body particles', async () => {
    await expect(run('units lj\natom_style body nparticle 1 4\nregion b block 0 5 0 5 0 5\ncreate_box 1 b\ncreate_atoms 1 single 1 1 1 units box\n', {}))
      .rejects.toThrow(/create_atoms is not supported for atom_style body/);
    await expect(run(`${HEAD}write_data out.data\n`)).rejects.toThrow(/write_data is not supported for atom_style body/);
  });

  it('refuses unsupported body computes and fixes by their name (unknown style)', async () => {
    await expect(run(`${HEAD}pair_style body/nparticle 4.0\npair_coeff * * 1 1\ncompute b all body/nonexistent\n`)).rejects.toThrow(/body\/nonexistent/);
    await expect(run(`${HEAD}pair_style body/nparticle 4.0\npair_coeff * * 1 1\ncompute t all temp/body bias x\nrun 0\n`)).rejects.toThrow(/Compute temp\/body requires bodies/);
    await expect(run(`${HEAD}pair_style body/nparticle 4.0\npair_coeff * * 1 1\nfix 1 all wall/body/polygon 2000 50 50 yplane -5 5\n`)).rejects.toThrow(/wall\/body\/polygon/);
  });

  it('accepts pair_modify shift and refuses a point particle in the nve/body group', async () => {
    await expect(run(`${HEAD}pair_style body/nparticle 4.0\npair_coeff * * 1 1\npair_modify shift yes\nrun 0\n`)).resolves.toBeDefined();
    await expect(run(`${HEAD}pair_style body/nparticle 4.0\npair_coeff * * 1 1\nfix 1 all nve/body\nrun 0\n`)).rejects.toThrow(/Fix nve\/body requires bodies/);
  });
});

describe('free rotation of one body with fix nve/body', () => {
  it('keeps the angular momentum and the translation of a torque-free body', async () => {
    const ONE_BODY = ONE.replace('2 atoms', '1 atoms').replace('2 atom types', '1 atom types').replace('2 2 0 1.0 2.0 0.5 0.3 0 0 0\n', '');
    const script = `${HEAD}set atom 1 angmom 0.4 -0.3 0.7
velocity all set 0.1 -0.05 0.2
pair_style body/nparticle 4.0
pair_coeff * * 1.0 1.0
fix 1 all nve/body
run 200
`;
    const session = await run(script, { 'one.data': ONE_BODY });
    const s = session.sys.state;
    // no torque: L is constant (the body has no other particle); x moves with v (no force)
    const L = Array.from(s.angmom!.subarray(0, 3));
    expect(L[0]).toBeCloseTo(0.4, 10);
    expect(L[1]).toBeCloseTo(-0.3, 10);
    expect(L[2]).toBeCloseTo(0.7, 10);
    expect(s.x[0]).toBeCloseTo(0.1 * 200 * 0.005, 9);
    expect(s.x[1]).toBeCloseTo(-0.05 * 200 * 0.005, 9);
    expect(Math.hypot(s.quat![0], s.quat![1], s.quat![2], s.quat![3])).toBeCloseTo(1, 12);
  });
});
