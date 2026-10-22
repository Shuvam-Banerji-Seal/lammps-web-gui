import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { StyleError } from '../src/engine/force/types';
import { readAireboFile } from '../src/engine-gpl/airebo/tables';

// pair_style rebo, airebo and airebo/morse (ported from LAMMPS b332f28, see src/engine-gpl/). The parity cases
// live in tests/oracle/w52airebo_*; here are the coefficient-file parsing, the element-map errors and the energy
// of an isolated C-C dimer, measured with the native LAMMPS binary run as a black box (metal units, run 0,
// in.dimer: create_atoms single at 0 and at r, boundary p p p, box 20 A, C type, pair_coeff * * CH.airebo C).

const POT = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const potential = (name: string): string => readFileSync(join(POT, name), 'utf8');

/** Runs a script in a fresh session with the potential files available; returns the thermo rows and the error messages. */
const run = async (script: string, files: string[] = []): Promise<{ rows: Record<string, number>[]; errors: string[] }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} } as never);
  for (const f of files) session.addFile(f, potential(f));
  try {
    await session.execute(script);
  } catch {
    // the error is also an event
  }
  return {
    rows: events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as unknown as Record<string, number>),
    errors: events.filter((e): e is Extract<EngineEvent, { kind: 'error' }> => e.kind === 'error').map((e) => ('message' in e ? String(e.message) : '')),
  };
};

const dimer = (style: string, pot: string, r: number, element = 'C'): string => `units metal
atom_style atomic
boundary p p p
region box block -10 10 -10 10 -10 10
create_box 1 box
create_atoms 1 single 0.0 0.0 0.0 units box
create_atoms 1 single ${r} 0.0 0.0 units box
mass 1 12.011
pair_style ${style}
pair_coeff * * ${pot} ${element}
thermo_style custom step pe
thermo_modify format float %.15g
run 0
`;

describe('AIREBO coefficient files (read_file)', () => {
  it('parses the CH.rebo global parameters, element pairs and spline domains', () => {
    const { params, tables } = readAireboFile(potential('CH.rebo'), 'CH.rebo', 'rebo', false);
    // element pairs are indexed [ii*2+jj]: 0 = C, 1 = H (the file lists CC, CH, HH)
    expect([params.rcmax[0], params.rcmax[1], params.rcmax[3]]).toEqual([2.0, 1.8, 1.7]);
    expect([params.rcmin[0], params.rcmin[1], params.rcmin[3]]).toEqual([1.7, 1.3, 1.1]);
    expect(params.rcmaxp[3]).toBe(1.7);
    expect(params.Q[0]).toBe(0.3134602960833);
    expect(params.alpha[1]).toBe(4.10254983);
    expect(params.A[3]).toBe(32.817355747);
    expect(params.BIJc[0]).toBe(12388.79197798);
    expect(params.Beta[(1 * 2 + 1) * 3]).toBe(1.71589217);
    expect(params.epsilonT[0]).toBe(0.3078851086);
    expect(params.sigma[0]).toBe(3.4);
    expect(params.NCmin).toBe(3.2);
    expect(params.NCmax).toBe(3.7);
    // symmetric element pairs: CH and HC share their values
    expect(params.rcmax[2]).toBe(params.rcmax[1]);
    // thmin and thmax are fixed in read_file
    expect([params.thmin, params.thmax]).toEqual([-1.0, -0.995]);
    // the gC domain and the pCC / piCC domain tables
    expect(tables.gCdom[0]).toBe(-1.0);
    expect(tables.gCdom[4]).toBe(1.0);
    expect(Array.from(tables.pCCdom)).toEqual([0, 4, 0, 4]);
    expect(Array.from(tables.piCCdom)).toEqual([0, 4, 0, 4, 0, 9]);
    // the Morse parameters are not read for the plain variants
    expect(params.epsilonM[0]).toBe(0);
  });

  it('parses the AIREBO-M Morse parameters', () => {
    const { params } = readAireboFile(potential('CH.airebo-m'), 'CH.airebo-m', 'airebo/morse', true);
    expect(params.epsilonM[0]).toBe(0.0028437);
    expect(params.alphaM[0]).toBe(1.8168084);
    expect(params.reqM[0]).toBe(3.6891063);
  });

  it('refuses a potential file of another variant', () => {
    expect(() => readAireboFile(potential('CH.airebo'), 'CH.airebo', 'rebo', false)).toThrow(StyleError);
    expect(() => readAireboFile(potential('CH.airebo'), 'CH.airebo', 'rebo', false)).toThrow(/does not match/);
    expect(() => readAireboFile(potential('CH.rebo'), 'CH.rebo', 'airebo', false)).toThrow(/does not match AIREBO/);
  });
});

describe('AIREBO pair style errors', () => {
  it('names an element that is not C or H', async () => {
    const { errors } = await run(dimer('airebo 3.0 1 1', 'CH.airebo', 1.3, 'N'), ['CH.airebo']);
    expect(errors.join('\n')).toMatch(/Element N not supported by pair style airebo/);
  });

  it('accepts NULL in the element map (docs.lammps.org/pair_airebo.html: "If a mapping value is specified as NULL, the mapping is not performed")', async () => {
    const script = `units metal\natom_style atomic\nregion box block 0 10 0 10 0 10\ncreate_box 2 box\nmass * 12.0\npair_style rebo\npair_coeff * * CH.rebo C NULL\n`;
    const { errors } = await run(script, ['CH.rebo']);
    expect(errors).toEqual([]);
  });

  it('needs one element per atom type', async () => {
    const script = `units metal\natom_style atomic\nregion box block 0 10 0 10 0 10\ncreate_box 2 box\nmass * 12.0\npair_style rebo\npair_coeff * * CH.rebo C\n`;
    const { errors } = await run(script, ['CH.rebo']);
    expect(errors.join('\n')).toMatch(/Incorrect number of args for pair coefficient/);
  });

  it('rejects the wrong number of pair_style arguments', async () => {
    const { errors } = await run('units metal\npair_style airebo 3.0 1\n');
    expect(errors.join('\n')).toMatch(/Illegal pair_style airebo command/);
  });

  it('rejects arguments of pair_style rebo', async () => {
    const { errors } = await run('units metal\npair_style rebo 3.0\n');
    expect(errors.join('\n')).toMatch(/Illegal pair_style command/);
  });

  it('airebo/bc needs the bond-centric knots after the standard data', async () => {
    const { errors } = await run(dimer('airebo/bc 3.0 1 1', 'CH.airebo', 1.3), ['CH.airebo']);
    expect(errors.join('\n')).toMatch(/reading derived-class extra parameters/);
  });
});

describe('AIREBO energy of an isolated C-C dimer (native LAMMPS, black box)', () => {
  it('airebo 3.0 1 1 at r = 1.3 A', async () => {
    const { rows, errors } = await run(dimer('airebo 3.0 1 1', 'CH.airebo', 1.3), ['CH.airebo']);
    expect(errors).toEqual([]);
    expect(rows[0].pe).toBeCloseTo(-6.18860434523531, 9);
  });

  it('airebo 3.0 1 1 at r = 1.5 A', async () => {
    const { rows, errors } = await run(dimer('airebo 3.0 1 1', 'CH.airebo', 1.5), ['CH.airebo']);
    expect(errors).toEqual([]);
    expect(rows[0].pe).toBeCloseTo(-5.61616545471541, 9);
  });

  it('rebo at r = 1.3 A', async () => {
    const { rows, errors } = await run(dimer('rebo', 'CH.rebo', 1.3), ['CH.rebo']);
    expect(errors).toEqual([]);
    expect(rows[0].pe).toBeCloseTo(-6.18860434600699, 9);
  });

  it('is zero beyond the REBO C-C cutoff (r = 2.0 A)', async () => {
    const { rows, errors } = await run(dimer('airebo 3.0 1 1', 'CH.airebo', 2.0), ['CH.airebo']);
    expect(errors).toEqual([]);
    expect(rows[0].pe).toBe(0);
  });
});
