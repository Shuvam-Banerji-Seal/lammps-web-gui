import { describe, expect, it } from 'vitest';
import { meamPe, oracleFiles, relErr, runMeam, worstForceError, jittered, type MeamSetup } from './meamHarness';

/*
 * pair_style meam, meam42 scope: bcc and hcp element references, and the ibar = 3 G(Gamma) form in an alloy,
 * through the engine. The alloy entries are the synthetic w42mb files of tests/oracle (native-measured cases
 * w42mb_*.in, checked by tests/engineOracle*.test.ts). Docs (docs.lammps.org/pair_meam.html): "3 => G = 2/(1+exp(-Gamma))",
 * "2 => not implemented".
 */

const IBAR3 = oracleFiles('w42mb_ibar3_lib.meam', 'w42mb_ibar3.meam');
const IBAR3_CMD = 'w42mb_ibar3_lib.meam A B w42mb_ibar3.meam A B';
const HCP = oracleFiles('w42mb_hcp_lib.meam', 'w42mb_hcp.meam');
const HCP_CMD = 'w42mb_hcp_lib.meam A B w42mb_hcp.meam A B';

/** A 4-atom mixed cluster (types A B A B) with a deterministic displacement. */
const mixed = (files: Record<string, string>, coeff: string): MeamSetup => ({
  files,
  coeff,
  types: [1, 2, 1, 2],
  x: jittered([[0, 0, 0], [2.4, 0.2, 0], [1.2, 2.2, 0.3], [1.0, 0.9, 2.4]], 0.05, 17),
});

/** Single-element bcc W crystal (w15meam_bcc parameters) with the ibar column set to the given value. */
const bccW = (ibar: number): Record<string, string> => ({
  'lib.meam': oracleFiles('w15meam_bcc_lib.meam')['w15meam_bcc_lib.meam'].replace(/ 0\n$/m, ` ${ibar}\n`),
  'par.meam': oracleFiles('w15meam_bcc.meam')['w15meam_bcc.meam'],
});

describe('pair_style meam, ibar = 3 (meam42)', () => {
  it('perfect bcc crystal with ibar = 3: energy per atom equals -Ec at re (G = 1 at Gamma = 0)', async () => {
    // 2x2x2 bcc cells (16 atoms) at the nearest-neighbour distance re = 2.73664 A, Ec = 8.9 eV
    const a = (2 * 2.73664) / Math.sqrt(3);
    const basis = [[0, 0, 0], [0.5, 0.5, 0.5]];
    const x: number[] = [];
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) for (let k = 0; k < 2; k++) for (const b of basis) x.push((i + b[0]) * a, (j + b[1]) * a, (k + b[2]) * a);
    const { pe } = await runMeam({ files: bccW(3), coeff: 'lib.meam W par.meam W', types: Array.from({ length: 16 }, () => 1), x, box: 2 * a, offset: 0 });
    expect(relErr(pe / 16, -8.9)).toBeLessThan(1e-9);
  });

  it('an alloy with ibar = 3 (bcc A, fcc B): forces equal -grad E to 1e-6 relative', async () => {
    expect(await worstForceError(mixed(IBAR3, IBAR3_CMD), 1e-5, 1e-2)).toBeLessThan(1e-6);
  });

  it('ibar = 2 stops with native\'s MEAM library error 1 (G_gam has no branch for it)', async () => {
    const files = { ...IBAR3, 'w42mb_ibar2_lib.meam': IBAR3['w42mb_ibar3_lib.meam'].replace(/ 3\n/g, ' 2\n') };
    await expect(runMeam(mixed(files, 'w42mb_ibar2_lib.meam A B w42mb_ibar3.meam A B'))).rejects.toThrow(/MEAM library error 1/);
  });
});

describe('pair_style meam, bcc and hcp elements in an alloy (meam42)', () => {
  it('hcp element (A) with a B partner: forces equal -grad E to 1e-6 relative', async () => {
    expect(await worstForceError(mixed(HCP, HCP_CMD), 1e-5, 1e-2)).toBeLessThan(1e-6);
  });

  it('the hcp A-B cluster energy is finite and differs from the same cluster with the bcc A element', async () => {
    const eHcp = await meamPe(mixed(HCP, HCP_CMD));
    const eIbar = await meamPe(mixed(IBAR3, IBAR3_CMD));
    expect(Number.isFinite(eHcp)).toBe(true);
    expect(eHcp).not.toBe(eIbar);
  });
});
