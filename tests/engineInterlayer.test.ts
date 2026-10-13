import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { calcDTap, calcTap } from '../src/engine-gpl/interlayer/taper';

/*
 * INTERLAYER pair styles ported from LAMMPS (src/engine-gpl/interlayer): the taper function and the error paths.
 * Numbers are checked against native LAMMPS in tests/oracle/w102il_*.in (engineOracle.test.ts).
 */
const ROOT = join(import.meta.dirname, '..');
const POT = join(ROOT, 'third_party/lammps/potentials');
const CASE = join(ROOT, 'tests/oracle');

/** Runs a script; returns the error message it stops with, or null. */
const errorOf = async (script: string, files: Record<string, string>): Promise<string | null> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, undefined, 0, 1);
  for (const [name, text] of Object.entries(files)) session.addFile(name, text);
  try {
    await session.execute(script);
    return null;
  } catch (e) {
    const err = events.find((ev) => ev.kind === 'error');
    return err && 'message' in err ? err.message : String(e);
  }
};

const twoAtoms = readFileSync(join(CASE, 'w102il_2particles.dat'), 'utf8');
const cc = readFileSync(join(POT, 'CC.KC'), 'utf8');
const lebCC = readFileSync(join(POT, 'CC.Lebedeva'), 'utf8');
const ilp = readFileSync(join(POT, 'BNCH.ILP'), 'utf8');

describe('taper function (interlayer_taper.h)', () => {
  it('is 1 at zero distance, 0 at the cutoff, and its derivative matches a finite difference', () => {
    const rc = 14;
    expect(calcTap(0, rc)).toBeCloseTo(1, 12);
    expect(calcTap(rc, rc)).toBe(0);
    expect(calcDTap(rc, rc)).toBe(0);
    for (const f of [0.2, 0.5, 0.8]) {
      const r = f * rc, h = 1e-6;
      const fd = (calcTap(r + h, rc) - calcTap(r - h, rc)) / (2 * h);
      expect(Math.abs(calcDTap(r, rc) - fd)).toBeLessThan(1e-7);
    }
  });
});

describe('INTERLAYER pair style errors', () => {
  it('kolmogorov/crespi/z needs pair_modify shift yes (LAMMPS sets the energy offset)', async () => {
    const err = await errorOf(
      'units metal\natom_style atomic\nread_data two.dat\nmass * 12\npair_style hybrid/overlay kolmogorov/crespi/z 20\npair_coeff * * none\npair_coeff 1 2 kolmogorov/crespi/z CC.KC C C\npair_modify shift no\nrun 0\n',
      { 'two.dat': twoAtoms, 'CC.KC': cc },
    );
    expect(err).toMatch(/shift yes/);
  });

  it('lebedeva/z reports an element with no entry in the potential file', async () => {
    const err = await errorOf(
      'units metal\natom_style atomic\nread_data two.dat\nmass * 12\npair_style hybrid/overlay lebedeva/z 20\npair_coeff * * none\npair_coeff 1 2 lebedeva/z CC.Lebedeva C H\nrun 0\n',
      { 'two.dat': twoAtoms, 'CC.Lebedeva': lebCC },
    );
    expect(err).toMatch(/missing an entry for: C H/);
  });

  it('ilp/graphene/hbn requires atom attribute molecule (atom_style atomic here)', async () => {
    const err = await errorOf(
      'units metal\natom_style atomic\nread_data two.dat\nmass * 12\npair_style hybrid/overlay ilp/graphene/hbn 16.0\npair_coeff * * ilp/graphene/hbn BNCH.ILP C C\nrun 0\n',
      { 'two.dat': twoAtoms, 'BNCH.ILP': ilp },
    );
    expect(err).toMatch(/requires atom attribute molecule/);
  });

  it('the INTERLAYER styles that native lets only inside hybrid/overlay are refused alone (native settings checks)', async () => {
    for (const style of ['ilp/graphene/hbn 16.0', 'kolmogorov/crespi/full 16.0', 'ilp/tmd 16.0', 'saip/metal 16.0', 'saip/metal/tmd 16.0', 'aip/water/2dm 16.0', 'drip']) {
      const err = await errorOf(`units metal\natom_style atomic\nread_data two.dat\nmass * 12\npair_style ${style}\n`, { 'two.dat': twoAtoms });
      expect(err, style).toContain(`Pair style ${style.split(' ')[0]} must be used as sub-style with hybrid/overlay`);
    }
  });

  it('styles that are not ported yet stay refused by name', async () => {
    for (const style of ['comb3']) {
      const err = await errorOf(`units metal\natom_style atomic\nread_data two.dat\nmass * 12\npair_style hybrid/overlay ${style} 16.0\n`, { 'two.dat': twoAtoms });
      expect(err, style).toMatch(new RegExp(style.replace('/', '\\/')));
    }
  });
});
