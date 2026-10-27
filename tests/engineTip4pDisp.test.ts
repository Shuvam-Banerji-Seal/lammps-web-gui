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
    // the type words are read at pair_coeff, as native coeff does ("set atom types from pair_style command")
    p.allocate(2);
    p.coeff(['*', '*', '0.0', '1.0'], { s: { ntypes: 2, topo: { nbondtypes: 1, nangletypes: 1 } } } as never);
    p.linkBonded({ bond: () => 0.9572, angle: () => (104.52 * Math.PI) / 180 }); // radians, as Angle::equilibrium_angle
    p.initStyle({} as never);
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
    expect(() => p.initStyle({} as never)).toThrow('Must use a bond style with TIP4P potential');
  });

  it('needs the lj/long/tip4p/long pair style for kspace pppm/disp/tip4p', () => {
    const k = new KSpacePPPMDispTIP4P();
    k.linkPair(new PairLJLongCoulLong());
    expect(() => k.init({} as never, {} as never, 8.5, 332.06371, {} as never))
      .toThrow('kspace_style pppm/disp/tip4p needs pair style lj/long/tip4p/long');
  });

  it('splits the per-atom energy and virial of an O M site: 1 - alpha to O, alpha/2 to each H (parity in w271pd_tip4p_peratom)', () => {
    const k = new KSpacePPPMDispTIP4P() as unknown as {
      model: { otype: number; alpha: number }; types: Int32Array; h1: Int32Array; h2: Int32Array;
      addCoulombPeratom(kc: { eatom: Float64Array; vatom: Float64Array }, i: number, o: Float64Array, q: number): void;
    };
    k.model = { otype: 1, alpha: 0.2 };
    k.types = Int32Array.from([1, 2, 2, 3]);
    k.h1 = Int32Array.from([1, 0, 0, 0]);
    k.h2 = Int32Array.from([2, 0, 0, 0]);
    const kc = { eatom: new Float64Array(4), vatom: new Float64Array(24) };
    const o = Float64Array.from([10, 1, 2, 3, 4, 5, 6]);
    k.addCoulombPeratom(kc, 0, o, 0.5);
    k.addCoulombPeratom(kc, 3, o, 0.5);
    expect(Array.from(kc.eatom)).toEqual([0.5 * 10 * 0.8, 0.5 * 10 * 0.1, 0.5 * 10 * 0.1, 5]);
    expect(kc.vatom[0]).toBeCloseTo(1 * 0.8 * 0.5, 15);
    expect(kc.vatom[6 + 5]).toBeCloseTo(6 * 0.1 * 0.5, 15);
    expect(kc.vatom[18 + 2]).toBe(1.5);
  });
});
