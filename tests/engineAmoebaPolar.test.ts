/**
 * Unit checks of the AMOEBA polarization pieces (src/engine-gpl/amoeba/polar.ts, params.ts). Native parity of the whole
 * term is the oracle cases w119amoeba_polar, w119amoeba_dimer and w119amoeba_hexamer; these tests pin the parsing, the
 * polarization groups and the refusals by name.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseAmoebaKey, parseAmoebaPrm } from '../src/engine-gpl/amoeba/params';
import { assignPolarGroups } from '../src/engine-gpl/amoeba/polar';

const ROOT = join(__dirname, 'oracle');
const prmText = readFileSync(join(ROOT, 'w119amoeba_polar.prm'), 'utf8');

describe('AMOEBA polarization parsing (amoeba_file.cpp file_dippolar, polar scales)', () => {
  it('reads the polarize lines with the Thole width and the polar group partner types', () => {
    const prm = parseAmoebaPrm(prmText);
    expect(prm.hasPolarity).toBe(true);
    expect(prm.polar.get(1)).toEqual({ alpha: 0.837, thole: 0.39, group: [2] });
    expect(prm.polar.get(2)).toEqual({ alpha: 0.496, thole: 0.39, group: [1] });
    expect(prm.poltyp).toBe('mutual');
  });

  it('keeps the polarization scale factors of the PRM file, with the index 0 entry equal to 1', () => {
    const prm = parseAmoebaPrm(prmText);
    expect(prm.polarScale.p[1]).toBe(0);
    expect(prm.polarScale.pi[3]).toBe(0.5);
    expect(prm.polarScale.direct11).toBe(0);
    expect(prm.polarScale.mutual11).toBe(1);
    expect(prm.polarScale.p[0]).toBe(1);
  });

  it('reads the solver keys of the key file (polar-eps, usolve-diag, cutoff)', () => {
    const prm = parseAmoebaPrm(prmText);
    const key = parseAmoebaKey('cutoff 10\ntaper 8\npolar-eps 1e-5\nusolve-diag 1.0\n', prm);
    expect(key.ewaldcut).toBe(10);
    expect(key.poleps).toBe(1e-5);
    expect(key.udiag).toBe(1.0);
    expect(key.usolvcut).toBe(4.5);
    expect(key.politer).toBe(100);
    expect(key.pcgprec).toBe(true);
  });

  it('refuses polar-predict and the non-mutual polarization of the key file, by name', () => {
    const prm = parseAmoebaPrm(prmText);
    expect(() => parseAmoebaKey('polar-predict gear\n', prm)).toThrow(/polar-predict/);
    expect(() => parseAmoebaKey('polarization direct\n', prm)).toThrow(/polarization direct/);
  });

  it('refuses an unknown PRM polarization type', () => {
    expect(() => parseAmoebaPrm('vdwtype buffered-14-7\npolarization bogus\n')).toThrow(/bogus/);
  });
});

describe('AMOEBA polarization groups (pair_amoeba.cpp assign_groups)', () => {
  it('puts each water molecule (O bonded to its H atoms) in its own group', () => {
    // the water dimer of w119amoeba_dimer.data: atoms 0..5 (local), bonds 0-1, 0-2, 3-4, 3-5; types 1 2 2 1 2 2
    const oneTwo = [[1, 2], [0], [0], [4, 5], [3], [3]];
    const types = [1, 2, 2, 1, 2, 2];
    const polgroup = (t: number): number[] => (t === 1 ? [2] : [1]);
    const g = assignPolarGroups(6, oneTwo, types, polgroup);
    expect(g[0]).toBe(g[1]);
    expect(g[0]).toBe(g[2]);
    expect(g[3]).toBe(g[4]);
    expect(g[3]).toBe(g[5]);
    expect(g[0]).not.toBe(g[3]);
  });

  it('does not join atoms whose partner type is not in the polarize line', () => {
    const g = assignPolarGroups(2, [[1], [0]], [1, 3], (t) => (t === 1 ? [2] : []));
    expect(g[0]).not.toBe(g[1]);
  });
});
