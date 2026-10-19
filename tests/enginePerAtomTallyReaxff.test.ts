import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * The ReaxFF pair style (src/engine-gpl/reaxff/: the per-atom tallies of tally.ts at the native tally calls of
 * src/REAXFF/reaxff_*.cpp): per-atom energy and virial (tests/perAtomTally.ts checks A and B). Native LAMMPS passes
 * both on CASES. The global virial stays fdotr in native and is not changed here.
 */

const CASES = [
  'w122m_reax_qeq_shielded_rdx',
  'w123q_reax_shielded',
  'w123q_reax_shielded_tight',
  'w124rx_tatb',
  'w124rx_tatb_cut',
  'w149cp_rxatom',
  'w149cp_rxatom_md',
  'w55reax_ch4',
  'w55reax_h2',
  'w55reax_in_rdx',
  'w55reax_in_tatb',
  'w55reax_nh3',
  'w55reax_qeq_rdx',
  'w55reax_qeq_rdx_tight',
  'w55reax_qeq_water',
  'w55reax_rdx_default',
  'w55reax_rdx_tab',
  'w55reax_water',
  'w55reax_water20',
  'w55reax_water_cho',
  'w65reaxq_acks2',
  'w65reaxq_acks2_field',
  'w65reaxq_acks2_field_run10',
  'w65reaxq_acks2_field_run10_tight',
  'w65reaxq_acks2_field_tight',
  'w65reaxq_acks2_tight',
  'w65reaxq_qeq_field',
  'w65reaxq_qeq_field_run10',
  'w65reaxq_qeq_field_run10_tight',
  'w65reaxq_qeq_field_tight',
  'w65reaxq_qeq_min',
  'w65reaxq_qeqr',
  'w65reaxq_qeqr_field',
  'w65reaxq_qeqr_field_run10',
  'w65reaxq_qeqr_field_run10_tight',
  'w65reaxq_qeqr_field_tight',
  'w65reaxq_qeqr_tight',
  'w65reaxq_qtpie',
  'w65reaxq_qtpie_field',
  'w65reaxq_qtpie_field_run10',
  'w65reaxq_qtpie_field_run10_tight',
  'w65reaxq_qtpie_field_tight',
  'w65reaxq_qtpie_min',
  'w65reaxq_qtpie_tight',
];

describe('ReaxFF pair style: per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
