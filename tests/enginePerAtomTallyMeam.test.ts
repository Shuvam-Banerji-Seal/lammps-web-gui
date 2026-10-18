import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Per-atom tallies of pair_style meam and meam/ms (src/engine-gpl/meam, ported from src/MEAM/meam_force.cpp: the
 * vatom blocks of the pair term and of the three-body screening term). Two sum rules on one force evaluation of
 * listed oracle cases (the same cases as tests/oracle/*.in, run with the appended computes below):
 *   (A) the sum of compute pe/atom is the thermo pe;
 *   (B) the sum of compute stress/atom NULL virial, component k, is -(compute pressure NULL virial, component k) * vol.
 * The energy and the global virial are right without the per-atom tallies (the virial is the fdotr sum), so (B) is the
 * check that catches a missing per-atom virial, which compute stress/atom would otherwise sum to 0.
 */

const CASES = join(__dirname, 'oracle');
const POTENTIALS = join(__dirname, '..', 'third_party', 'lammps', 'potentials');

/** Covers pair meam (single and multi element, second neighbours, periodic images) and meam/ms (single cell, cloud). */
const SAMPLE = [
  'w107meam_fetibr', // pair meam, Fe-Ti hcp, multi-element with an alloy pair
  'w15meam_bcc', // pair meam, single-element bcc, displaced, fix nve
  'w136meam_hcp2nn', // pair meam, hcp with the partly screened second shell (nn2 = 1)
  'w24meam_ibar3_bcc', // pair meam, ibar = 3 (the WL entry)
  'w31meamialloy_cluster1', // pair meam, ialloy = 1, A-B dimers and trimers
  'w199vo_kmc_edge', // pair meam under fix npt and fix neighbor/swap
  'w128ms_msmeam', // pair meam/ms, H and Ga4 with the HGa L12 pair
  'w128ms_msmeam_cloud', // pair meam/ms, periodic images inside the cutoff
];

/** Appended to each case: the per-atom computes, their sums, and the thermo row of this evaluation. */
const APPEND = `compute zzpa all pe/atom
compute zzsa all stress/atom NULL virial
compute zzpv all pressure NULL virial
compute zzrp all reduce sum c_zzpa
compute zzrs all reduce sum c_zzsa[1] c_zzsa[2] c_zzsa[3] c_zzsa[4] c_zzsa[5] c_zzsa[6]
thermo_style custom step pe c_zzrp vol c_zzpv[1] c_zzpv[2] c_zzpv[3] c_zzpv[4] c_zzpv[5] c_zzpv[6] c_zzrs[1] c_zzrs[2] c_zzrs[3] c_zzrs[4] c_zzrs[5] c_zzrs[6]
thermo_modify format float %.15g norm no
run 0
`;

const directive = (text: string, key: string): string[] => {
  const m = new RegExp(`^#\\s*${key}:\\s*(.*)$`, 'm').exec(text);
  return m ? m[1].trim().split(/\s+/).filter(Boolean) : [];
};

/** The sum rules of one case: the relative errors of (A) and (B). */
async function sumRules(name: string): Promise<{ dA: number; dB: number }> {
  const text = readFileSync(join(CASES, `${name}.in`), 'utf8');
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, undefined, 0, 1);
  for (const f of directive(text, 'oracle-inputs')) session.addFile(f, readFileSync(join(CASES, f), 'utf8'));
  for (const f of directive(text, 'oracle-potentials')) session.addFile(f, readFileSync(join(POTENTIALS, f), 'utf8'));
  await session.execute(`${text}\n${APPEND}`);
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  const r = rows[rows.length - 1];
  expect(r, `${name}: thermo row`).toBeDefined();
  const pe = r.pe as number, sumPe = r.c_zzrp as number, vol = r.vol as number;
  const dA = Math.abs(sumPe - pe) / Math.max(Math.abs(pe), Math.abs(sumPe), 1e-12);
  const ks = [1, 2, 3, 4, 5, 6];
  const expected = ks.map((k) => -(r[`c_zzpv[${k}]`] as number) * vol);
  const sums = ks.map((k) => r[`c_zzrs[${k}]`] as number);
  let scale = 0;
  for (const v of expected) scale = Math.max(scale, Math.abs(v));
  let dB = 0;
  for (let k = 0; k < 6; k++) dB = Math.max(dB, Math.abs(sums[k] - expected[k]) / Math.max(scale, 1e-12));
  return { dA, dB };
}

describe('pair meam and meam/ms: per-atom energy and virial sum to the global values', () => {
  for (const name of SAMPLE) {
    it(name, { timeout: 60000 }, async () => {
      const { dA, dB } = await sumRules(name);
      expect(dA, `${name}: sum pe/atom vs thermo pe`).toBeLessThan(1e-10);
      expect(dB, `${name}: sum stress/atom vs -pressure virial * vol`).toBeLessThan(1e-10);
    });
  }
});
