/**
 * pair_style pace/extrapolation and the B-basis .yaml format (src/engine-gpl/pace/: bbasis.ts, couplings.ts,
 * bevaluator.ts, npz.ts, pair_pace_extrapolation.ts): the messages of settings, coeff and fix pair as native LAMMPS
 * b332f28 (built with PKG_ML-PACE) gives them, and the Clebsch-Gordan coefficients against their closed forms. The
 * native parity of energies, forces and grades is the oracle cases pace_extrapolation_linear, pace_extrapolation_full and
 * pace_yaml_multi (self-made potentials from scripts/oracle/pace-extrapolation-fixtures.py).
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { clebschGordan, generateCouplingTree, generateMsCgList, wigner3j } from '../src/engine-gpl/pace/couplings';
import { BBasis } from '../src/engine-gpl/pace/bbasis';

const DIR = join(__dirname, 'oracle');
const FILES = ['pace_al_cu.yaml', 'pace_al_cu.asi', 'pace_al_cuni.yaml', 'pace_al_cuni.asi'];

/** Runs a 2x2x2 Cu cell with the given lines; returns the first error message ('' if none). */
const errorOf = async (units: string, pairStyle: string, pairCoeff: string, extra = ''): Promise<string> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} } as never);
  for (const f of FILES) session.addFile(f, readFileSync(join(DIR, f), f.endsWith('.asi') ? 'latin1' : 'utf8'));
  const script = [`units ${units}`, 'atom_style atomic', 'lattice fcc 3.6', 'region box block 0 2 0 2 0 2', 'create_box 1 box',
    'create_atoms 1 box', 'mass 1 63.546', `pair_style ${pairStyle}`, pairCoeff, extra, 'run 0'].join('\n');
  try {
    await session.execute(script);
  } catch (e) {
    const err = events.find((ev) => ev.kind === 'error');
    return (err && 'message' in err ? String(err.message) : String(e)).replace(/^line \d+: /, '');
  }
  const err = events.find((ev) => ev.kind === 'error');
  return err && 'message' in err ? String(err.message) : '';
};

describe('pair_style pace/extrapolation messages (as native)', () => {
  const coeff = 'pair_coeff * * pace_al_cu.yaml pace_al_cu.asi Cu';
  it.each([
    ['metal', 'pace/extrapolation', 'pair_coeff * * pace_al_cu.yaml Cu', '', 'Illegal pair_coeff command: missing argument(s)'],
    ['metal', 'pace/extrapolation bogus', coeff, '', 'Unknown pair_style pace keyword: bogus'],
    ['metal', 'pace/extrapolation neigh foo', coeff, '', 'Unknown pair_style pace/extrapolation neigh keyword: foo'],
    ['metal', 'pace/extrapolation neigh', coeff, '', 'Illegal pair_style pace/extrapolation neigh command: missing argument(s)'],
    ['real', 'pace/extrapolation', coeff, '', "ACE potentials require 'metal' units"],
    ['metal', 'pace/extrapolation', coeff, 'fix g all pair 1 pace/extrapolation gamma 0', 'Fix pair cannot extract property gamma from pair style'],
    ['metal', 'pace/extrapolation', coeff, 'fix g all pair 1 pace/extrapolation bogus 1', 'Fix pair pair style cannot extract bogus_flag'],
    ['metal', 'pace/extrapolation', coeff, 'fix g all pair 1 pace/extrapolation bogus 0', 'Fix pair cannot extract property bogus from pair style'],
    ['metal', 'pace/extrapolation', 'pair_coeff * * pace_al_cu.yaml pace_al_cuni.asi Cu', '',
      'Number of species types in ASI `pace_al_cuni.asi` (2)not equal to number of species in ACEBBassiSet (1)'],
    ['metal', 'pace/extrapolation', 'pair_coeff * * pace_al_cu.yaml pace_al_cu.asi Ni', '', 'Element Ni is not supported by ACE-potential from file pace_al_cu.yaml'],
    ['metal', 'pace/extrapolation', 'pair_coeff * * pace_al_cu.yaml pace_al_cu.asi Xx', '', "'Xx' is not a valid element"],
    ['metal', 'pace', 'pair_coeff * * pace_al_cu.yaml Cu', 'fix g all pair 1 pace gamma 1', 'Fix pair pair style cannot extract gamma_flag'],
    // a missing file: the found path is '', which the library's readers refuse
    ['metal', 'pace/extrapolation', 'pair_coeff * * Cu.yaml Cu.asi Cu', '', "Potential file  doesn't exists"],
    ['metal', 'pace/extrapolation', 'pair_coeff * * pace_al_cu.yaml nosuch.asi Cu', '', 'npz_load: Error! Unable to open file !'],
  ])('%s %s / %s / %s', async (units, style, pc, extra, msg) => {
    expect((await errorOf(units, style, pc, extra)).split('\n')[0]).toBe(msg);
  });

  it('runs with the grade and the core-repulsion factor of fix pair, and pair pace serves corerep', async () => {
    expect(await errorOf('metal', 'pace/extrapolation', coeff, 'fix g all pair 1 pace/extrapolation gamma 1 corerep 1')).toBe('');
    expect(await errorOf('metal', 'pace', 'pair_coeff * * pace_al_cu.yaml Cu', 'fix g all pair 1 pace corerep 1')).toBe('');
  });
});

describe('B-basis construction', () => {
  it('3j symbols and Clebsch-Gordan coefficients match their closed forms', () => {
    // (1 1 0; 0 0 0) = -1/sqrt(3), (2 2 2; 0 0 0) = -sqrt(2/35), <1 1; 1 -1 | 0 0> = 1/sqrt(3)
    expect(wigner3j(1, 0, 1, 0, 0, 0)).toBeCloseTo(-1 / Math.sqrt(3), 15);
    expect(wigner3j(2, 0, 2, 0, 2, 0)).toBeCloseTo(-Math.sqrt(2 / 35), 15);
    expect(clebschGordan(1, 1, 1, -1, 0, 0)).toBeCloseTo(1 / Math.sqrt(3), 15);
    // orthonormality: sum over m1, m2 of C(j1 m1 j2 m2 | J M) C(j1 m1 j2 m2 | J' M) = delta(J, J')
    for (const [j1, j2] of [[1, 1], [2, 1], [3, 2]]) {
      for (let J = Math.abs(j1 - j2); J <= j1 + j2; J++) {
        for (let Jp = Math.abs(j1 - j2); Jp <= j1 + j2; Jp++) {
          let s = 0;
          for (let m1 = -j1; m1 <= j1; m1++) for (let m2 = -j2; m2 <= j2; m2++) {
            if (Math.abs(m1 + m2) > Math.min(J, Jp)) continue;
            s += clebschGordan(j1, m1, j2, m2, J, m1 + m2) * clebschGordan(j1, m1, j2, m2, Jp, m1 + m2);
          }
          expect(s).toBeCloseTo(J === Jp ? 2 * J + 1 : 0, 12);
        }
      }
    }
  });

  it('builds the coupling trees and the half-basis m-combinations', () => {
    expect(Array.from(generateCouplingTree(4))).toEqual([0, 1, 4, 2, 3, 5, 4, 5, -1]);
    expect(Array.from(generateCouplingTree(5))).toEqual([0, 1, 5, 2, 3, 6, 5, 6, 7, 7, 4, -1]);
    // rank 2, l = 1: m = (0, 0) once and (1, -1) doubled for its negative partner
    const list = generateMsCgList(2, [1, 1], [], true).map((p) => [Array.from(p.ms), p.c]);
    expect(list).toEqual([[[1, -1], 2 * clebschGordan(1, 1, 1, -1, 0, 0)], [[0, 0], clebschGordan(1, 0, 1, 0, 0, 0)]]);
    expect(() => generateMsCgList(3, [1, 1, 1], [1], true)).toThrow('sum of ls is not even ls = (1 1 1 )');
  });

  it('keeps one B function per nbody entry and sums them into C-tilde functions', () => {
    const b = BBasis.fromYaml(readFileSync(join(DIR, 'pace_al_cuni.yaml'), 'utf8'), 'pace_al_cuni.yaml');
    expect(b.elementsName).toEqual(['Cu', 'Ni']);
    // every nbody entry of a central element is one function (the .asi of the case is 2 x 37 square per element)
    expect(b.totalBasisSizeRank1[0] + b.totalBasisSize[0]).toBe(37);
    expect(b.totalBasisSizeRank1[1] + b.totalBasisSize[1]).toBe(37);
    const c = b.toCTilde();
    for (let mu = 0; mu < 2; mu++) expect(c.totalBasisSize[mu]).toBeLessThanOrEqual(b.totalBasisSize[mu]);
  });

  it('refuses an inconsistent .yaml as the library does', () => {
    const bad = readFileSync(join(DIR, 'pace_al_cu.yaml'), 'utf8').replace('type: Cu Cu Cu, nr: [1, 1], nl: [0, 0]', 'type: Cu Cu Cu, nr: [1, 1], nl: [0, 1]');
    expect(() => BBasis.fromYaml(bad, 'bad.yaml')).toThrow('All elements of ls ( 0  1 ) should be equal');
  });
});
