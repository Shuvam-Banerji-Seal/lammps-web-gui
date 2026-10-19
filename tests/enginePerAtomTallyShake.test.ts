import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * The SHAKE and RATTLE fixes (src/engine/fix/shake.ts): per-atom virial of the constraint forces. As native LAMMPS does
 * (src/RIGID/fix_shake.cpp shake() at b332f28, Fix::v_tally), each atom of a cluster takes 1/n of the cluster's virial.
 * Native LAMMPS passes the per-atom tally check (tests/perAtomTally.ts checks A and B) on these cases; before the fix,
 * compute stress/atom threw for fix shake and fix rattle.
 *
 * Left out: the four Drude ethanol cases (w72drude_ethanol_lang, _nh, _tgnpt, w81nhpress_drude_ethanol). Their pe/atom
 * matches native, but stress/atom does not: with pair thole removed (shake kept) the same run-0 comparison agrees to
 * 1e-13, so the gap is the per-atom virial of pair thole, not of shake. Those cases are to be re-added when thole's
 * per-atom virial is fixed.
 */

const CASES = [
  'rattle_clusters',
  'shake_clusters',
  'w102il_aip_water',
  'w109tip_fep',
  'w109tip_hybrid',
  'w109tip_overlay',
  'w117cf_tip4p_pppm_water',
  'w12tip4p_cut',
  'w12tip4p_long',
  'w130pr_e3b_tip4p2005',
  'w143cf_shake_min',
  'w208dt_dipole_tip4p',
  'w21bhyb_shake',
  'w26depmol_shake',
  'w26shake_molb',
  'w26shake_press',
  'w29ehex_hex_constrain',
  'w29ehex_nevery_constrain',
  'w29ehex_rattle_constrain',
  'w29ehex_shake_constrain',
  'w31ehexcom_hex',
  'w31ehexcom_nevery',
  'w31ehexcom_rattle',
  'w31ehexcom_shake',
  'w37gcmc_shake_ins',
  'w41sm_shake_store',
  'w42gap_shake_velocity',
  'w48sh_loose',
  'w48sh_run0',
  'w48sh_run100',
  'w48sh_run20',
  'w48sh_tol5',
  'w48sh_tol7',
  'w50rt_nve20',
  'w50rt_run0',
  'w51km_tip4p_noeffect',
  'w73shake_eth',
  'w73shake_gil',
];

describe('SHAKE and RATTLE: per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
