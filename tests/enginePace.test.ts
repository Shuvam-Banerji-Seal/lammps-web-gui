/**
 * pair_style pace (ML-PACE, src/engine-gpl/pace/): the refusals and the messages of the settings and coeff steps, as
 * native pair_pace.cpp gives them (run natively for the keyword messages), and the .ace loader's counts for the
 * Cu-PBE-core-rep.ace file. The native parity of the energies and forces is the oracle cases w257pace_*.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { CTildeBasis } from '../src/engine-gpl/pace/cbasis';

const POT = join(__dirname, '..', 'third_party', 'lammps', 'potentials');

/** Runs a small Cu cell with the given pair_style, pair_coeff and extra lines; returns the error message ('' if none). */
const errorOf = async (pairStyle: string, pairCoeff: string, opts: { units?: string; extra?: string[] } = {}): Promise<string> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} } as never);
  session.addFile('Cu-PBE-core-rep.ace', readFileSync(join(POT, 'Cu-PBE-core-rep.ace'), 'utf8'));
  const script = [
    `units ${opts.units ?? 'metal'}`,
    'atom_style atomic',
    'lattice fcc 3.597',
    'region box block 0 2 0 2 0 2',
    'create_box 1 box',
    'create_atoms 1 box',
    'mass 1 63.546',
    ...(opts.extra ?? []),
    `pair_style ${pairStyle}`,
    pairCoeff,
    'run 0',
  ].join('\n');
  try {
    await session.execute(script);
  } catch (e) {
    const err = events.find((ev) => ev.kind === 'error');
    return err && 'message' in err ? String(err.message) : String(e);
  }
  const err = events.find((ev) => ev.kind === 'error');
  return err && 'message' in err ? String(err.message) : '';
};

describe('pair_style pace keywords (native messages)', () => {
  it('refuses an unknown keyword as native does', async () => {
    expect(await errorOf('pace foo', 'pair_coeff * * Cu-PBE-core-rep.ace Cu')).toMatch(/Unknown pair_style pace keyword: foo/);
  });

  it('refuses an unknown neigh value', async () => {
    expect(await errorOf('pace neigh bogus', 'pair_coeff * * Cu-PBE-core-rep.ace Cu')).toMatch(/Unknown pair_style pace neigh keyword: bogus/);
  });

  it('refuses a chunksize without a number', async () => {
    expect(await errorOf('pace chunksize', 'pair_coeff * * Cu-PBE-core-rep.ace Cu')).toMatch(/Expected integer parameter/);
  });

  it('accepts product, recursive, chunksize and neigh', async () => {
    expect(await errorOf('pace product chunksize 2048 neigh auto', 'pair_coeff * * Cu-PBE-core-rep.ace Cu')).toBe('');
    expect(await errorOf('pace recursive', 'pair_coeff * * Cu-PBE-core-rep.ace Cu')).toBe('');
  });

  it('requires metal units', async () => {
    expect(await errorOf('pace', 'pair_coeff * * Cu-PBE-core-rep.ace Cu', { units: 'lj' })).toMatch(/ACE potentials require 'metal' units/);
  });
});

describe('pair_style pace coeff rules', () => {
  it('refuses a missing potential file as native does (the found path is empty)', async () => {
    // Measured with native LAMMPS: a missing Cu.yaml or Cu.txt gives Unrecognized file type of ''. Accept only .ace or .yace
    expect(await errorOf('pace', 'pair_coeff * * Cu.yaml Cu')).toMatch(/Unrecognized file type of ''. Accept only .ace or .yace/);
    expect(await errorOf('pace', 'pair_coeff * * Cu.txt Cu')).toMatch(/Unrecognized file type of ''. Accept only .ace or .yace/);
  });

  it('refuses an element the potential does not contain', async () => {
    expect(await errorOf('pace', 'pair_coeff * * Cu-PBE-core-rep.ace Ni')).toMatch(/Element Ni is not supported by ACE-potential from file Cu-PBE-core-rep.ace/);
  });

  it('refuses an element that is not a chemical symbol', async () => {
    expect(await errorOf('pace', 'pair_coeff * * Cu-PBE-core-rep.ace Xx')).toMatch(/'Xx' is not a valid element/);
  });

  it('refuses pair_coeff without * *', async () => {
    expect(await errorOf('pace', 'pair_coeff 1 * Cu-PBE-core-rep.ace Cu')).toMatch(/Incorrect args for pair coefficients/);
  });
});

describe('compute pace arguments (native compute_pace.cpp messages)', () => {
  const computeError = async (line: string): Promise<string> => errorOf('pace', 'pair_coeff * * Cu-PBE-core-rep.ace Cu', { extra: [line] });

  it('accepts bikflag and dgradflag', async () => {
    expect(await computeError('compute d all pace Cu-PBE-core-rep.ace 0 0')).toBe('');
  });

  it('needs the two flags as integers', async () => {
    expect(await computeError('compute d all pace Cu-PBE-core-rep.ace')).toMatch(/Expected integer parameter instead of NULL or empty string/);
    expect(await computeError('compute d all pace Cu-PBE-core-rep.ace 1 x')).toMatch(/Expected integer parameter instead of 'x'/);
  });

  it('refuses dgradflag without bikflag', async () => {
    expect(await computeError('compute d all pace Cu-PBE-core-rep.ace 0 1')).toMatch(/Illegal compute pace command: dgradflag=1 requires bikflag=1/);
  });

  it('refuses a file that is not .ace or .yace', async () => {
    expect(await computeError('compute d all pace Cu.txt 0 0')).toMatch(/Unrecognized file type of 'Cu.txt'. Accept only .ace or .yace/);
  });
});

describe('.ace loader (Cu-PBE-core-rep.ace)', () => {
  it('reads the counts and radial settings of the file', () => {
    const text = readFileSync(join(POT, 'Cu-PBE-core-rep.ace'), 'utf8');
    const b = CTildeBasis.fromText(text, 'Cu-PBE-core-rep.ace');
    expect(b.nelements).toBe(1);
    expect(b.elementsName).toEqual(['Cu']);
    expect(b.lmax).toBe(6);
    expect(b.nradbase).toBe(16);
    expect(b.nradmax).toBe(5);
    expect(b.rankmax).toBe(6);
    expect(b.ndensitymax).toBe(2);
    expect(b.totalBasisSizeRank1).toEqual([16]);
    expect(b.totalBasisSize).toEqual([726]);
    expect(b.radial.cut[0]).toBeCloseTo(7.4, 12);
    expect(b.radial.radbasename).toEqual(['ChebPow']);
    expect(b.embedding[0].npoti).toBe('FinnisSinclairShiftedScaled');
  });
});
