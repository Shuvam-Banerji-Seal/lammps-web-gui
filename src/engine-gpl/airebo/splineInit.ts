/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_airebo.cpp (PairAIREBO::spline_init) and
 * src/MANYBODY/pair_rebo.cpp (PairREBO::spline_init), at commit b332f28.
 * The knot-value statements are generated mechanically from the C++ text (array
 * index expressions turned into flat offsets); the loops are written by hand.
 * Authors credited in the C++ file headers: Ase Henry (MIT); bugfixes and optimizations:
 * Marcel Fallet and Steve Stuart (Clemson), Axel Kohlmeyer (Temple U), Markus Hoehnerbach
 * (RWTH Aachen), Cyril Falvo (Universite Paris Sud); AIREBO-M: Thomas C. O'Connor (JHU) 2014.
 * Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { spbicubicPatchCoeffs, sptricubicPatchCoeffs } from './splines';
import type { SplineTables } from './tables';

/** Knot values (C++ PCCf ... Tdfdz). Flat index: 2D [5][5], piXX [5][5][11], Tf [5][5][10]. */
export interface Knots {
  PCCf: Float64Array; PCCdfdx: Float64Array; PCCdfdy: Float64Array;
  PCHf: Float64Array; PCHdfdx: Float64Array; PCHdfdy: Float64Array;
  piCCf: Float64Array; piCCdfdx: Float64Array; piCCdfdy: Float64Array; piCCdfdz: Float64Array;
  piCHf: Float64Array; piCHdfdx: Float64Array; piCHdfdy: Float64Array; piCHdfdz: Float64Array;
  piHHf: Float64Array; piHHdfdx: Float64Array; piHHdfdy: Float64Array; piHHdfdz: Float64Array;
  Tf: Float64Array; Tdfdx: Float64Array; Tdfdy: Float64Array; Tdfdz: Float64Array;
}

export const newKnots = (): Knots => {
  const a2 = () => new Float64Array(25);
  const a3 = () => new Float64Array(275);
  const at = () => new Float64Array(250);
  return {
    PCCf: a2(), PCCdfdx: a2(), PCCdfdy: a2(), PCHf: a2(), PCHdfdx: a2(), PCHdfdy: a2(),
    piCCf: a3(), piCCdfdx: a3(), piCCdfdy: a3(), piCCdfdz: a3(),
    piCHf: a3(), piCHdfdx: a3(), piCHdfdy: a3(), piCHdfdz: a3(),
    piHHf: a3(), piHHdfdx: a3(), piHHdfdy: a3(), piHHdfdz: a3(),
    Tf: at(), Tdfdx: at(), Tdfdy: at(), Tdfdz: at(),
  };
};

/** The bicubic P_CC/P_CH patches (C++ first nH/nC loop of spline_init). */
const bicubicPatches = (K: Knots, T: SplineTables): void => {
  const y = new Float64Array(4), y1 = new Float64Array(4), y2 = new Float64Array(4);
  for (let nH = 0; nH < 4; nH++) {
    for (let nC = 0; nC < 4; nC++) {
      y[0] = K.PCCf[nC * 5 + nH];
      y[1] = K.PCCf[nC * 5 + nH + 1];
      y[2] = K.PCCf[(nC + 1) * 5 + nH];
      y[3] = K.PCCf[(nC + 1) * 5 + nH + 1];
      spbicubicPatchCoeffs(nC, nC + 1, nH, nH + 1, y, y1, y2, T.pCC, (nC * 4 + nH) * 16);
      y[0] = K.PCHf[nC * 5 + nH];
      y[1] = K.PCHf[nC * 5 + nH + 1];
      y[2] = K.PCHf[(nC + 1) * 5 + nH];
      y[3] = K.PCHf[(nC + 1) * 5 + nH + 1];
      spbicubicPatchCoeffs(nC, nC + 1, nH, nH + 1, y, y1, y2, T.pCH, (nC * 4 + nH) * 16);
    }
  }
};

/** PairAIREBO::spline_init: knot values, then the bicubic and tricubic patch coefficients. */
export const splineInit = (K: Knots, T: SplineTables): void => {
  const { PCCf, PCCdfdx, PCCdfdy, PCHf, PCHdfdx, PCHdfdy } = K;
  const { piCCf, piCCdfdx, piCCdfdy, piCCdfdz, piCHf, piCHdfdx, piCHdfdy, piCHdfdz } = K;
  const { piHHf, piHHdfdx, piHHdfdy, piHHdfdz, Tf, Tdfdx, Tdfdy, Tdfdz } = K;
  let i = 0, j = 0, k = 0;
for (i = 0; i < 5; i++) {
for (j = 0; j < 5; j++) {
PCCf[(i)*5 + (j)] = 0.0;
PCCdfdx[(i)*5 + (j)] = 0.0;
PCCdfdy[(i)*5 + (j)] = 0.0;
PCHf[(i)*5 + (j)] = 0.0;
PCHdfdx[(i)*5 + (j)] = 0.0;
PCHdfdy[(i)*5 + (j)] = 0.0;
}
}
PCCf[(0)*5 + (2)] = -0.00050;
PCCf[(0)*5 + (3)] = 0.0161253646;
PCCf[(1)*5 + (1)] = -0.010960;
PCCf[(1)*5 + (2)] = 0.00632624824;
PCCf[(2)*5 + (0)] = -0.0276030;
PCCf[(2)*5 + (1)] = 0.00317953083;
PCHf[(0)*5 + (1)] = 0.2093367328250380;
PCHf[(0)*5 + (2)] = -0.064449615432525;
PCHf[(0)*5 + (3)] = -0.303927546346162;
PCHf[(1)*5 + (0)] = 0.010;
PCHf[(1)*5 + (1)] = -0.1251234006287090;
PCHf[(1)*5 + (2)] = -0.298905245783;
PCHf[(2)*5 + (0)] = -0.1220421462782555;
PCHf[(2)*5 + (1)] = -0.3005291724067579;
PCHf[(3)*5 + (0)] = -0.307584705066;
for (i = 0; i < 5; i++) {
for (j = 0; j < 5; j++) {
for (k = 0; k < 10; k++) {
piCCf[(i)*55 + (j)*11 + (k)] = 0.0;
piCCdfdx[(i)*55 + (j)*11 + (k)] = 0.0;
piCCdfdy[(i)*55 + (j)*11 + (k)] = 0.0;
piCCdfdz[(i)*55 + (j)*11 + (k)] = 0.0;
piCHf[(i)*55 + (j)*11 + (k)] = 0.0;
piCHdfdx[(i)*55 + (j)*11 + (k)] = 0.0;
piCHdfdy[(i)*55 + (j)*11 + (k)] = 0.0;
piCHdfdz[(i)*55 + (j)*11 + (k)] = 0.0;
piHHf[(i)*55 + (j)*11 + (k)] = 0.0;
piHHdfdx[(i)*55 + (j)*11 + (k)] = 0.0;
piHHdfdy[(i)*55 + (j)*11 + (k)] = 0.0;
piHHdfdz[(i)*55 + (j)*11 + (k)] = 0.0;
Tf[(i)*50 + (j)*10 + (k)] = 0.0;
Tdfdx[(i)*50 + (j)*10 + (k)] = 0.0;
Tdfdy[(i)*50 + (j)*10 + (k)] = 0.0;
Tdfdz[(i)*50 + (j)*10 + (k)] = 0.0;
}
}
}
for (i = 3; i < 10; i++) piCCf[(0)*55 + (0)*11 + (i)] = 0.0049586079;
piCCf[(1)*55 + (0)*11 + (1)] = 0.021693495;
piCCf[(0)*55 + (1)*11 + (1)] = 0.021693495;
for (i = 2; i < 10; i++) piCCf[(1)*55 + (0)*11 + (i)] = 0.0049586079;
for (i = 2; i < 10; i++) piCCf[(0)*55 + (1)*11 + (i)] = 0.0049586079;
piCCf[(1)*55 + (1)*11 + (1)] = 0.05250;
piCCf[(1)*55 + (1)*11 + (2)] = -0.002088750;
for (i = 3; i < 10; i++) piCCf[(1)*55 + (1)*11 + (i)] = -0.00804280;
piCCf[(2)*55 + (0)*11 + (1)] = 0.024698831850;
piCCf[(0)*55 + (2)*11 + (1)] = 0.024698831850;
piCCf[(2)*55 + (0)*11 + (2)] = -0.00597133450;
piCCf[(0)*55 + (2)*11 + (2)] = -0.00597133450;
for (i = 3; i < 10; i++) piCCf[(2)*55 + (0)*11 + (i)] = 0.0049586079;
for (i = 3; i < 10; i++) piCCf[(0)*55 + (2)*11 + (i)] = 0.0049586079;
piCCf[(2)*55 + (1)*11 + (1)] = 0.00482478490;
piCCf[(1)*55 + (2)*11 + (1)] = 0.00482478490;
piCCf[(2)*55 + (1)*11 + (2)] = 0.0150;
piCCf[(1)*55 + (2)*11 + (2)] = 0.0150;
piCCf[(2)*55 + (1)*11 + (3)] = -0.010;
piCCf[(1)*55 + (2)*11 + (3)] = -0.010;
piCCf[(2)*55 + (1)*11 + (4)] = -0.01168893870;
piCCf[(1)*55 + (2)*11 + (4)] = -0.01168893870;
piCCf[(2)*55 + (1)*11 + (5)] = -0.013377877400;
piCCf[(1)*55 + (2)*11 + (5)] = -0.013377877400;
piCCf[(2)*55 + (1)*11 + (6)] = -0.015066816000;
piCCf[(1)*55 + (2)*11 + (6)] = -0.015066816000;
for (i = 7; i < 10; i++) piCCf[(2)*55 + (1)*11 + (i)] = -0.015066816000;
for (i = 7; i < 10; i++) piCCf[(1)*55 + (2)*11 + (i)] = -0.015066816000;
piCCf[(2)*55 + (2)*11 + (1)] = 0.0472247850;
piCCf[(2)*55 + (2)*11 + (2)] = 0.0110;
piCCf[(2)*55 + (2)*11 + (3)] = 0.0198529350;
piCCf[(2)*55 + (2)*11 + (4)] = 0.01654411250;
piCCf[(2)*55 + (2)*11 + (5)] = 0.013235290;
piCCf[(2)*55 + (2)*11 + (6)] = 0.00992646749999 ;
piCCf[(2)*55 + (2)*11 + (7)] = 0.006617644999;
piCCf[(2)*55 + (2)*11 + (8)] = 0.00330882250;
piCCf[(3)*55 + (0)*11 + (1)] = -0.05989946750;
piCCf[(0)*55 + (3)*11 + (1)] = -0.05989946750;
piCCf[(3)*55 + (0)*11 + (2)] = -0.05989946750;
piCCf[(0)*55 + (3)*11 + (2)] = -0.05989946750;
for (i = 3; i < 10; i++) piCCf[(3)*55 + (0)*11 + (i)] = 0.0049586079;
for (i = 3; i < 10; i++) piCCf[(0)*55 + (3)*11 + (i)] = 0.0049586079;
piCCf[(3)*55 + (1)*11 + (2)] = -0.0624183760;
piCCf[(1)*55 + (3)*11 + (2)] = -0.0624183760;
for (i = 3; i < 10; i++) piCCf[(3)*55 + (1)*11 + (i)] = -0.0624183760;
for (i = 3; i < 10; i++) piCCf[(1)*55 + (3)*11 + (i)] = -0.0624183760;
piCCf[(3)*55 + (2)*11 + (1)] = -0.02235469150;
piCCf[(2)*55 + (3)*11 + (1)] = -0.02235469150;
for (i = 2; i < 10; i++) piCCf[(3)*55 + (2)*11 + (i)] = -0.02235469150;
for (i = 2; i < 10; i++) piCCf[(2)*55 + (3)*11 + (i)] = -0.02235469150;
piCCdfdx[(2)*55 + (1)*11 + (1)] = -0.026250;
piCCdfdx[(2)*55 + (1)*11 + (5)] = -0.0271880;
piCCdfdx[(2)*55 + (1)*11 + (6)] = -0.0271880;
for (i = 7; i < 10; i++) piCCdfdx[(2)*55 + (1)*11 + (i)] = -0.0271880;
piCCdfdx[(1)*55 + (3)*11 + (2)] = 0.0187723882;
for (i = 2; i < 10; i++) piCCdfdx[(2)*55 + (3)*11 + (i)] = 0.031209;
piCCdfdy[(1)*55 + (2)*11 + (1)] = -0.026250;
piCCdfdy[(1)*55 + (2)*11 + (5)] = -0.0271880;
piCCdfdy[(1)*55 + (2)*11 + (6)] = -0.0271880;
for (i = 7; i < 10; i++) piCCdfdy[(1)*55 + (2)*11 + (i)] = -0.0271880;
piCCdfdy[(3)*55 + (1)*11 + (2)] = 0.0187723882;
for (i = 2; i < 10; i++) piCCdfdy[(3)*55 + (2)*11 + (i)] = 0.031209;
piCCdfdz[(1)*55 + (1)*11 + (2)] = -0.0302715;
piCCdfdz[(2)*55 + (1)*11 + (4)] = -0.0100220;
piCCdfdz[(1)*55 + (2)*11 + (4)] = -0.0100220;
piCCdfdz[(2)*55 + (1)*11 + (5)] = -0.0100220;
piCCdfdz[(1)*55 + (2)*11 + (5)] = -0.0100220;
for (i = 4; i < 9; i++) piCCdfdz[(2)*55 + (2)*11 + (i)] = -0.0033090;
i = 4;
for (j = 0; j < 4; j++) {
for (k = 1; k < 11; k++) {
piCCf[(i)*55 + (j)*11 + (k)] = piCCf[(i-1)*55 + (j)*11 + (k)];
}
}
for (i = 0; i < 4; i++) {
for (j = i+1; j < 5; j++) {
for (k = 1; k < 11; k++) {
piCCf[(i)*55 + (j)*11 + (k)] = piCCf[(j)*55 + (i)*11 + (k)];
}
}
}
for (k = 1; k < 11; k++) piCCf[(4)*55 + (4)*11 + (k)] = piCCf[(3)*55 + (4)*11 + (k)];
k = 10;
for (i = 0; i < 5; i++) {
for (j = 0; j < 5; j++) {
piCCf[(i)*55 + (j)*11 + (k)] = piCCf[(i)*55 + (j)*11 + (k-1)];
}
}
piCHf[(1)*55 + (1)*11 + (1)] = -0.050;
piCHf[(1)*55 + (1)*11 + (2)] = -0.050;
piCHf[(1)*55 + (1)*11 + (3)] = -0.30;
for (i = 4; i < 10; i++) piCHf[(1)*55 + (1)*11 + (i)] = -0.050;
for (i = 5; i < 10; i++) piCHf[(2)*55 + (0)*11 + (i)] = -0.004523893758064;
for (i = 5; i < 10; i++) piCHf[(0)*55 + (2)*11 + (i)] = -0.004523893758064;
piCHf[(2)*55 + (1)*11 + (2)] = -0.250;
piCHf[(1)*55 + (2)*11 + (2)] = -0.250;
piCHf[(2)*55 + (1)*11 + (3)] = -0.250;
piCHf[(1)*55 + (2)*11 + (3)] = -0.250;
piCHf[(3)*55 + (1)*11 + (1)] = -0.10;
piCHf[(1)*55 + (3)*11 + (1)] = -0.10;
piCHf[(3)*55 + (1)*11 + (2)] = -0.125;
piCHf[(1)*55 + (3)*11 + (2)] = -0.125;
piCHf[(3)*55 + (1)*11 + (3)] = -0.125;
piCHf[(1)*55 + (3)*11 + (3)] = -0.125;
for (i = 4; i < 10; i++) piCHf[(3)*55 + (1)*11 + (i)] = -0.10;
for (i = 4; i < 10; i++) piCHf[(1)*55 + (3)*11 + (i)] = -0.10;
i = 4;
for (j = 0; j < 4; j++) {
for (k = 1; k < 11; k++) {
piCHf[(i)*55 + (j)*11 + (k)] = piCHf[(i-1)*55 + (j)*11 + (k)];
}
}
for (i = 0; i < 4; i++) {
for (j = i+1; j < 5; j++) {
for (k = 1; k < 11; k++) {
piCHf[(i)*55 + (j)*11 + (k)] = piCHf[(j)*55 + (i)*11 + (k)];
}
}
}
for (k = 1; k < 11; k++) piCHf[(4)*55 + (4)*11 + (k)] = piCHf[(3)*55 + (4)*11 + (k)];
k = 10;
for (i = 0; i < 5; i++) {
for (j = 0; j < 5; j++) {
piCHf[(i)*55 + (j)*11 + (k)] = piCHf[(i)*55 + (j)*11 + (k-1)];
}
}
piHHf[(1)*55 + (1)*11 + (1)] = 0.124915958;
Tf[(2)*50 + (2)*10 + (1)] = -0.035140;
for (i = 2; i < 10; i++) Tf[(2)*50 + (2)*10 + (i)] = -0.0040480;
  bicubicPatches(K, T);
  // tricubic patches: the 8 corners of each (nC, nH, nConj) cell
  const y = new Float64Array(8), y1 = new Float64Array(8), y2 = new Float64Array(8), y3 = new Float64Array(8);
  const fill = (dest: Float64Array, src: Float64Array, stride2: number, stride1: number, nC: number, nH: number, nConj: number): void => {
    const at = (a: number, b: number, c: number) => src[a * stride2 + b * stride1 + c];
    dest[0] = at(nC, nH, nConj);
    dest[1] = at(nC, nH, nConj + 1);
    dest[2] = at(nC, nH + 1, nConj);
    dest[3] = at(nC, nH + 1, nConj + 1);
    dest[4] = at(nC + 1, nH, nConj);
    dest[5] = at(nC + 1, nH, nConj + 1);
    dest[6] = at(nC + 1, nH + 1, nConj);
    dest[7] = at(nC + 1, nH + 1, nConj + 1);
  };
  for (let nH = 0; nH < 4; nH++) {
    for (let nC = 0; nC < 4; nC++) {
      // Knot values exist up to "10", but are never used because they are clamped down to 9.
      for (let nConj = 0; nConj < 9; nConj++) {
        const off = ((nC * 4 + nH) * 9 + nConj) * 64;
        fill(y, piCCf, 55, 11, nC, nH, nConj); fill(y1, piCCdfdx, 55, 11, nC, nH, nConj);
        fill(y2, piCCdfdy, 55, 11, nC, nH, nConj); fill(y3, piCCdfdz, 55, 11, nC, nH, nConj);
        sptricubicPatchCoeffs(nC, nC + 1, nH, nH + 1, nConj, nConj + 1, y, y1, y2, y3, T.piCC, off);
        fill(y, piCHf, 55, 11, nC, nH, nConj); fill(y1, piCHdfdx, 55, 11, nC, nH, nConj);
        fill(y2, piCHdfdy, 55, 11, nC, nH, nConj); fill(y3, piCHdfdz, 55, 11, nC, nH, nConj);
        sptricubicPatchCoeffs(nC, nC + 1, nH, nH + 1, nConj, nConj + 1, y, y1, y2, y3, T.piCH, off);
        fill(y, piHHf, 55, 11, nC, nH, nConj); fill(y1, piHHdfdx, 55, 11, nC, nH, nConj);
        fill(y2, piHHdfdy, 55, 11, nC, nH, nConj); fill(y3, piHHdfdz, 55, 11, nC, nH, nConj);
        sptricubicPatchCoeffs(nC, nC + 1, nH, nH + 1, nConj, nConj + 1, y, y1, y2, y3, T.piHH, off);
        fill(y, Tf, 50, 10, nC, nH, nConj); fill(y1, Tdfdx, 50, 10, nC, nH, nConj);
        fill(y2, Tdfdy, 50, 10, nC, nH, nConj); fill(y3, Tdfdz, 50, 10, nC, nH, nConj);
        sptricubicPatchCoeffs(nC, nC + 1, nH, nH + 1, nConj, nConj + 1, y, y1, y2, y3, T.Tijc, off);
      }
    }
  }
};

/**
 * PairREBO::spline_init: the REBO-only P_CC and P_CH knot values, applied after the AIREBO knots;
 * the bicubic patches of P_CC and P_CH are rebuilt from them.
 */
export const splineInitRebo = (K: Knots, T: SplineTables): void => {
  K.PCCf[0 * 5 + 2] = 0.007860700254745;
  K.PCCf[0 * 5 + 3] = 0.016125364564267;
  K.PCCf[1 * 5 + 1] = 0.003026697473481;
  K.PCCf[1 * 5 + 2] = 0.006326248241119;
  K.PCCf[2 * 5 + 0] = 0.0;
  K.PCCf[2 * 5 + 1] = 0.003179530830731;
  bicubicPatches(K, T);
};
