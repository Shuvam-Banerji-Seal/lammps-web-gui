import { describe, expect, it } from 'vitest';
import { runMeam, meamPe, fccCellSetup, relErr, type MeamSetup } from './meamHarness';

/*
 * pair_style meam, task meam24: the G(Gamma) forms (the ibar column of the library file) and the ZBL blend (the
 * zbl keyword), through the engine. Docs (docs.lammps.org/pair_meam.html): "ibar = 0 => G = sqrt(1+Gamma),
 * 1 => G = exp(Gamma/2), 3 => G = 2/(1+exp(-Gamma)), 4 => G = sqrt(1+Gamma) with the gsmooth switch, -5 =>
 * G = sign(1+Gamma) sqrt(|1+Gamma|)"; "zbl = 1 blends the pair potential into the ZBL potential at short range".
 * Every G form is 1 at Gamma = 0, so a perfect fcc crystal has energy -Ec per atom at its nearest-neighbour
 * distance re for each supported ibar. The displaced-crystal agreement with native LAMMPS is in
 * tests/oracle/w24meam_*.in (engineOracle.test.ts).
 */

const EC = 3.54;
const RE = 2.55;
const LIB = (ibar: number) => `# ibar ${ibar}\n'Cu' 'fcc' 12 29 63.546 4.95 2 1 2 4 3.615 ${EC} 1 1 1 1 1 1 ${ibar}\n`;
const PAR = `rc = 4.0\ndelr = 0.1\nEc(1,1) = ${EC}\nre(1,1) = ${RE}\nalpha(1,1) = 4.95\nlattce(1,1) = fcc\nCmin(1,1,1) = 2.0\nCmax(1,1,1) = 2.8\nzbl(1,1) = 0\n`;

/** Perfect fcc crystal (2x2x2 cells, 32 atoms) at the nearest-neighbour distance re, in a periodic cell. */
const fccCrystal = (files: Record<string, string>): MeamSetup => {
  const c = fccCellSetup(RE * Math.SQRT2, 2, 0, 0);
  return { files, coeff: 'lib.meam Cu par.meam Cu', types: c.types, x: c.x, box: c.box, offset: 0 };
};

const dimer = (files: Record<string, string>, r: number): MeamSetup => ({
  files,
  coeff: 'lib.meam Cu par.meam Cu',
  types: [1, 1],
  x: [0, 0, 0, r, 0, 0],
});

describe('pair_style meam: G(Gamma) forms (library ibar column)', () => {
  for (const ibar of [0, 1, 3, 4, -5]) {
    it(`perfect fcc crystal with ibar = ${ibar}: energy per atom equals -Ec at re`, async () => {
      const { pe } = await runMeam(fccCrystal({ 'lib.meam': LIB(ibar), 'par.meam': PAR }));
      expect(relErr(pe / 32, -EC)).toBeLessThan(1e-9);
    });
  }

  it('ibar = 2 stops with native\'s MEAM library error 1 (G_gam has no branch for it)', async () => {
    await expect(runMeam(dimer({ 'lib.meam': LIB(2), 'par.meam': PAR }, 2.5))).rejects.toThrow(/MEAM library error 1/);
  });
});

describe('pair_style meam: ZBL blend (zbl keyword)', () => {
  const r = 2.5;
  const short = 0.9;
  it('the default zbl blend leaves a dimer at 2.5 A unchanged (same energy as zbl 0)', async () => {
    const noKey = { 'lib.meam': LIB(0), 'par.meam': PAR.replace('zbl(1,1) = 0\n', '') };
    const zbl0 = { 'lib.meam': LIB(0), 'par.meam': PAR };
    expect(await meamPe(dimer(noKey, r))).toBe(await meamPe(dimer(zbl0, r)));
  });

  it('the default (no zbl line) equals an explicit zbl(1,1) = 1, and both differ from zbl(1,1) = 0 at short range', async () => {
    const noKey = { 'lib.meam': LIB(0), 'par.meam': PAR.replace('zbl(1,1) = 0\n', '') };
    const zbl1 = { 'lib.meam': LIB(0), 'par.meam': PAR.replace('zbl(1,1) = 0', 'zbl(1,1) = 1') };
    const zbl0 = { 'lib.meam': LIB(0), 'par.meam': PAR };
    const eDefault = await meamPe(dimer(noKey, short));
    expect(eDefault).toBe(await meamPe(dimer(zbl1, short)));
    expect(eDefault).not.toBe(await meamPe(dimer(zbl0, short)));
    expect(Number.isFinite(eDefault)).toBe(true);
  });
});
