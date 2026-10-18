import { describe, expect, it } from 'vitest';
import { StyleError } from '../src/engine/force/types';
import { PairLJLongCoulLong } from '../src/engine/force/pair/lj_long';
import { PairLJLongTIP4PLong } from '../src/engine-gpl/tip4pdisp/pair_lj_long_tip4p_long';
import { KSpacePPPMDispTIP4P } from '../src/engine-gpl/tip4pdisp/pppm_disp_tip4p';
import { KSPACES } from '../src/engine-gpl/registry/pppm_disp';
import { PAIRS } from '../src/engine-gpl/registry/lj_long_disp';

// The TIP4P variants of lj/long/coul/long and pppm/disp (src/engine-gpl/tip4pdisp, LAMMPS commit b332f28).
describe('pppm/disp/tip4p and lj/long/tip4p/long (GPL port)', () => {
  it('is registered under the native style names', () => {
    expect(Object.keys(KSPACES)).toContain('pppm/disp/tip4p');
    expect(Object.keys(PAIRS)).toContain('lj/long/tip4p/long');
    expect(KSPACES['pppm/disp/tip4p']().name).toBe('pppm/disp/tip4p');
    expect(PAIRS['lj/long/tip4p/long']().name).toBe('lj/long/tip4p/long');
  });

  it('takes flag_lj flag_coul otype htype btype atype qdist cutoff (cutoff2) and needs Coulomb long', () => {
    const p = new PairLJLongTIP4PLong();
    p.settings(['long', 'long', '1', '2', '1', '1', '0.1546', '8.5']);
    expect(p.cutCoul).toBe(8.5);
    expect(p.ljLong).toBe(true);
    const q = new PairLJLongTIP4PLong();
    q.settings(['cut', 'long', '1', '2', '1', '1', '0.1546', '8.5', '7.0']);
    expect(q.cutGlobal).toBe(8.5);
    expect(q.cutCoul).toBe(7.0);
    expect(q.ljLong).toBe(false);
    expect(() => new PairLJLongTIP4PLong().settings(['long', 'off', '1', '2', '1', '1', '0.1546', '8.5']))
      .toThrow(new StyleError('Coulomb cut not supported in pair_style lj/long/tip4p/long'));
    expect(() => new PairLJLongTIP4PLong().settings(['long', 'long', '1', '2', '1', '1', '0.1546']))
      .toThrow(StyleError);
  });

  it('sets the M-site alpha from the bond and angle equilibria as native init_style does', () => {
    const p = new PairLJLongTIP4PLong();
    p.settings(['long', 'long', '1', '2', '1', '1', '0.1546', '8.5']);
    expect(p.siteModel()).toBeNull();
    p.linkBonded({ bond: () => 0.9572, angle: () => 104.52 });
    p.initStyle();
    const m = p.siteModel();
    expect(m).not.toBeNull();
    // alpha = qdist / (cos(theta/2) r0) with theta = 104.52 degrees, r0 = 0.9572 A (measured in the oracle geometry)
    expect(m!.alpha).toBeCloseTo(0.2638755363830926, 12);
    expect(m!.otype).toBe(1);
    expect(m!.htype).toBe(2);
  });

  it('refuses a bond style that is missing', () => {
    const p = new PairLJLongTIP4PLong();
    p.settings(['long', 'long', '1', '2', '1', '1', '0.1546', '8.5']);
    expect(() => p.initStyle()).toThrow('Must use a bond style with TIP4P potential');
  });

  it('needs the lj/long/tip4p/long pair style for kspace pppm/disp/tip4p', () => {
    const k = new KSpacePPPMDispTIP4P();
    k.linkPair(new PairLJLongCoulLong());
    expect(() => k.init({} as never, {} as never, 8.5, 332.06371, {} as never))
      .toThrow('kspace_style pppm/disp/tip4p needs pair style lj/long/tip4p/long');
  });

  it('refuses per-atom energy and virial by name', () => {
    const k = new KSpacePPPMDispTIP4P();
    expect(() => k.compute({ eatom: new Float64Array(3), vatom: null } as never))
      .toThrow(/pe\/atom or stress\/atom with kspace_style pppm\/disp/);
  });
});
