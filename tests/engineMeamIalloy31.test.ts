import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { meamPe, oracleFiles, relErr, worstForceError, type MeamSetup } from './meamHarness';

/*
 * pair_style meam, ialloy = 0, 1 and 2 (docs.lammps.org/pair_meam.html): "ialloy = integer flag to use alternative
 * averaging rule for t parameters"; "0 = standard averaging (matches ialloy=0 in DYNAMO)", "1 = alternative averaging
 * (matches ialloy=1 in DYNAMO)", "2 = no averaging of t (use single-element values)", "default = 0".
 *
 * The geometry is the one of tests/oracle/w31meamialloy_cluster1.in (isolated dimers, trimers and an 8-atom mixed
 * cluster in a 45 A periodic box, the create_atoms lines are read from that input), the entries are
 * w31meamialloy_lib.meam with the parameter file of cluster1 (ialloy 1) and the same file with ialloy 0 and 2.
 * Measured with native LAMMPS (black box, run 0 of cluster1 with ialloy 0, 1 and 2): pe -16.1003058670605,
 * -16.1403366002971 (the oracle fixture) and -15.8654413666918 eV.
 */

const CLUSTER = readFileSync(join(__dirname, 'oracle', 'w31meamialloy_cluster1.in'), 'utf8');

/** Atom types and coordinates of the create_atoms lines of the cluster input (in input order). */
const atomsOf = (script: string): { types: number[]; x: number[] } => {
  const types: number[] = [];
  const x: number[] = [];
  for (const line of script.split('\n')) {
    const m = /^create_atoms\s+(\d+)\s+single\s+(\S+)\s+(\S+)\s+(\S+)/.exec(line);
    if (!m) continue;
    types.push(Number(m[1]));
    x.push(Number(m[2]), Number(m[3]), Number(m[4]));
  }
  return { types, x };
};

const ATOMS = atomsOf(CLUSTER);
const LIB = oracleFiles('w31meamialloy_lib.meam');
const PAR1 = oracleFiles('w31meamialloy_1.meam')['w31meamialloy_1.meam'];

/** The cluster1 geometry with the parameter file of the given ialloy (the file keeps the name par.meam). */
const cluster = (ialloy: number): MeamSetup => ({
  files: {
    'lib.meam': LIB['w31meamialloy_lib.meam'],
    'par.meam': PAR1.replace(/ialloy = \d+/, `ialloy = ${ialloy}`),
  },
  coeff: 'lib.meam A B par.meam A B',
  types: ATOMS.types,
  x: ATOMS.x,
  box: 45,
  offset: 0,
  ntypes: 2,
});

describe('pair_style meam: ialloy 0, 1 and 2 (the cluster1 geometry)', () => {
  const NATIVE: Array<[number, number]> = [
    [0, -16.1003058670605],
    [1, -16.1403366002971],
    [2, -15.8654413666918],
  ];
  for (const [ialloy, pe] of NATIVE) {
    it(`ialloy = ${ialloy} matches native pe to 1e-9 relative`, async () => {
      expect(relErr(await meamPe(cluster(ialloy)), pe)).toBeLessThan(1e-9);
    });
  }

  it('the three averaging rules give different energies for the mixed cluster', async () => {
    const e = await Promise.all([0, 1, 2].map((k) => meamPe(cluster(k))));
    expect(new Set(e).size).toBe(3);
  });

  it('ialloy = 1 cluster: forces equal -grad E to 1e-5 (finite differences of the engine energy)', async () => {
    expect(await worstForceError(cluster(1), 1e-5, 1e-2)).toBeLessThan(1e-5);
  }, 120000);
});
