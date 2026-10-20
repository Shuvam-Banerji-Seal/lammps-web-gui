import { describe, expect, it } from 'vitest';
import {
  AngleCosine, AngleCosineSquared, AngleCosinePeriodic, AngleCosineShift, AngleCosineDelta, AngleCharmm,
  AngleQuartic, AngleFourier, AngleFourierSimple, AngleZero,
} from '../src/engine/force/angle/styles';
import { AngleHarmonic } from '../src/engine/force/angle/harmonic';
import { AngleClass2, AngleClass2P6 } from '../src/engine/force/angle/class2';
import { AngleSpica } from '../src/engine/force/angle/spica51';
import { AngleCosineBuck6d } from '../src/engine/force/angle/buck6d';
import { AngleCosineSquaredRestricted, AngleMm3 } from '../src/engine/force/angle/misc18';
import { AngleLepton } from '../src/engine/force/angle/lepton';
import { AngleGaussian } from '../src/engine/force/angle/gaussian';
import { BondedHybrid } from '../src/engine/force/bonded_hybrid';
import { StyleError, type Bonded, type StyleContext } from '../src/engine/force/types';
import { AngleAmoeba } from '../src/engine-gpl/amoeba/angle_amoeba';
import { AngleClass2xe } from '../src/engine-gpl/class2xe/angle_class2xe';
import { AngleCross } from '../src/engine-gpl/yaff/angle_cross';
import { AngleCosineShiftExp } from '../src/engine-gpl/xmolecule/angle_cosine_shift_exp';
import { AngleMWLC } from '../src/engine-gpl/xmolecule/angle_mwlc';
import { AngleMesoCNT } from '../src/engine-gpl/mesont/angle_mesocnt';
import { AngleDipole } from '../src/engine-gpl/miscx/angle_dipole';

/*
 * The equilibrium angle in radians, as native Angle::equilibrium_angle(i) returns it (angle_*.cpp at b332f28). Each
 * expected value is written with the same expression native uses in that style's coeff() or equilibrium_angle().
 * Doc pages: docs.lammps.org/angle_style.html ("theta0 ... degrees ... converted to radians internally").
 */
const MY_PI = 3.14159265358979323846;
const DEG2RAD = MY_PI / 180.0;
const RAD2DEG = 180.0 / MY_PI;
const CTX: StyleContext = { s: null, readFile: () => '', log: () => {} };

/** allocate, then coeff for all types, as a script's angle_coeff * line does (no init: some styles need atom data). */
const setup = <T extends Bonded>(style: T, coeff: string[]): T => {
  style.allocate(1);
  style.coeff(['*', ...coeff], CTX);
  return style;
};

describe('angle equilibrium_angle in radians (native value, bit for bit)', () => {
  it('degree-stored styles return the stored radian value (native DEG2RAD * theta0_one)', () => {
    const cases: [Bonded, string[], number][] = [
      [new AngleHarmonic(), ['30', '104.52'], DEG2RAD * 104.52],
      [new AngleCosineSquared(), ['30', '109.47'], DEG2RAD * 109.47],
      [new AngleCharmm(), ['30', '111', '5', '2.0'], DEG2RAD * 111],
      [new AngleCosineSquaredRestricted(), ['30', '109.5'], DEG2RAD * 109.5],
      [new AngleCosineDelta(), ['30', '109.5'], DEG2RAD * 109.5],
      [new AngleLepton(), ['120', 'theta^2'], DEG2RAD * 120],
    ];
    for (const [style, coeff, expected] of cases) {
      setup(style, coeff);
      expect(style.equilibriumRadians(1), style.name).toBe(expected);
      expect(style.equilibrium(1), style.name).toBeCloseTo(expected * RAD2DEG, 12);
    }
  });

  it('class 2, Mm3, spica, quartic and amoeba store theta0 / 180 * MY_PI', () => {
    const v = 104.52;
    const expected = (v / 180.0) * MY_PI;
    expect(setup(new AngleClass2(), [String(v), '50', '1', '2']).equilibriumRadians(1)).toBe(expected);
    expect(setup(new AngleClass2P6(), [String(v), '50', '1', '2', '3', '4']).equilibriumRadians(1)).toBe(expected);
    expect(setup(new AngleMm3(), ['1', String(v)]).equilibriumRadians(1)).toBe(expected);
    expect(setup(new AngleSpica('spica'), ['1', String(v)]).equilibriumRadians(1)).toBe(expected);
    expect(setup(new AngleQuartic(), [String(v), '40', '-25', '-14']).equilibriumRadians(1)).toBe(expected);
    expect(setup(new AngleClass2xe(), [String(v), '50', '1', '2']).equilibriumRadians(1)).toBe(expected);
  });

  it('cosine/shift stores theta0 * MY_PI / 180 (native theta[i])', () => {
    const st = setup(new AngleCosineShift(), ['10', '45']);
    expect(st.equilibriumRadians(1)).toBe((45 * MY_PI) / 180.0);
    expect(st.equilibrium(1)).toBeCloseTo(45, 12);
  });

  it('cosine/shift/exp stores (theta0 * MY_PI) / 180 (native theta0[i])', () => {
    const st = setup(new AngleCosineShiftExp(), ['10', '60', '2.0']);
    expect(st.equilibriumRadians(1)).toBe((60 * MY_PI) / 180.0);
  });

  it('cross stores theta0 * MY_PI / 180 (native theta0[i])', () => {
    const st = setup(new AngleCross(), ['1', '2', '3', '1.5', '1.6', '104.52']);
    expect(st.equilibriumRadians(1)).toBe((104.52 * MY_PI) / 180.0);
  });

  it('amoeba stores theta0 / 180.0 * MY_PI', () => {
    const st = setup(new AngleAmoeba(), ['0', '0', '104.52', '50', '1', '2', '3', '4']);
    expect(st.equilibriumRadians(1)).toBe((104.52 / 180.0) * MY_PI);
  });

  it('gaussian returns theta0[i][0], the first term centre (theta * MY_PI / 180.0)', () => {
    const st = setup(new AngleGaussian(), ['300', '2', '1.0', '0.3', '120', '0.5', '0.2', '60']);
    expect(st.equilibriumRadians(1)).toBe((120 * MY_PI) / 180.0);
  });

  it('dipole stores gamma0 / 180.0 * MY_PI (native gamma0[i]); the MY_PI-valued styles return MY_PI', () => {
    expect(setup(new AngleDipole(), ['10', '30']).equilibriumRadians(1)).toBe((30 / 180.0) * MY_PI);
    expect(setup(new AngleCosine(), ['1']).equilibriumRadians(1)).toBe(MY_PI);
    expect(setup(new AngleCosineBuck6d(), ['1', '2', '90']).equilibriumRadians(1)).toBe(MY_PI);
    expect(setup(new AngleMWLC(), ['1', '2', '3', '4']).equilibriumRadians(1)).toBe(MY_PI);
  });

  it('cosine/periodic returns MY_PI * (1 - (B > 0 ? 0 : 1 / n))', () => {
    expect(setup(new AngleCosinePeriodic(), ['10', '1', '3']).equilibriumRadians(1)).toBe(MY_PI * (1.0 - 0.0));
    expect(setup(new AngleCosinePeriodic(), ['10', '-1', '3']).equilibriumRadians(1)).toBe(MY_PI * (1.0 - 1.0 / 3));
  });

  it('fourier returns acos(-C1 / 4 / C2) when in range, else the ratio; MY_PI when C2 is 0', () => {
    expect(setup(new AngleFourier(), ['1', '0', '-1', '0.5']).equilibriumRadians(1)).toBeCloseTo(MY_PI / 3, 14);
    expect(setup(new AngleFourier(), ['1', '0', '1', '0']).equilibriumRadians(1)).toBe(MY_PI);
    // |C1 / 4 / C2| > 1 is returned unchanged (native: "if (fabs(ret) <= 1.0) ret = acos(-ret)")
    expect(setup(new AngleFourier(), ['1', '0', '-10', '0.5']).equilibriumRadians(1)).toBe(-10 / 4 / 0.5);
  });

  it('fourier/simple returns MY_PI / N', () => {
    expect(setup(new AngleFourierSimple(), ['1', '1', '3']).equilibriumRadians(1)).toBe(MY_PI / 3);
  });

  it('zero stores DEG2RAD * theta0 and writes RAD2DEG * theta0 back (native angle_zero.cpp)', () => {
    const st = setup(new AngleZero(), ['120']);
    expect(st.equilibriumRadians(1)).toBe(DEG2RAD * 120);
    expect(st.dataCoeffs()).toEqual(['1 120']);
  });

  it('mesocnt returns the literal 180.0 that native equilibrium_angle returns (not radians)', () => {
    expect(new AngleMesoCNT().equilibriumRadians(1)).toBe(180.0);
  });

  it('styles without an equilibrium return NaN (the base default)', () => {
    const st = new AngleGaussian();
    st.allocate(1);
    expect(Number.isNaN(st.equilibriumRadians(1))).toBe(true);
  });

  it('a hybrid angle style answers with its sub-style', () => {
    const hy = new BondedHybrid('angle', { harmonic: () => new AngleHarmonic(), cosine: () => new AngleCosine() });
    hy.settings(['harmonic', 'cosine']);
    hy.allocate(2);
    hy.coeff(['1', 'harmonic', '50', '104.52']);
    hy.coeff(['2', 'cosine', '10']);
    expect(hy.equilibriumRadians(1)).toBe(DEG2RAD * 104.52);
    expect(hy.equilibriumRadians(2)).toBe(MY_PI);
    expect(hy.equilibrium(1)).toBeCloseTo(104.52, 12);
    hy.coeff(['2', 'none']);
    expect(() => hy.equilibriumRadians(2)).toThrow(StyleError);
  });
});
