/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the PTM library (Polyhedral Template Matching, PM Larsen, MIT licence), src/PTM/ptm_deformation_gradient.cpp at LAMMPS commit b332f28 (https://github.com/lammps/lammps). Copyright (c) 2016 PM Larsen (MIT, see LICENSE.ptm). The LAMMPS compute that uses it (src/PTM/compute_ptm_atom.cpp) is GPL-2.0; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */

// The penrose_* tables of ptm_deformation_gradient.h live in ptm_data.ts; the
// caller passes the chosen table as `penrose`, so they are not imported here.
// ptm_deformation_gradient.h has no inline functions to translate.

// F is the 3x3 deformation gradient, flat (F[3*i+j]); res receives the three
// squared residuals per component.
export function calculate_deformation_gradient(num_points: number, ideal_points: number[][], mapping: number[], normalized: number[][], penrose: number[][], F: number[], res: number[]): void {
        for (let i = 0; i < 3; i++) {
                for (let j = 0; j < 3; j++) {
                        let acc = 0.0;
                        for (let k = 0; k < num_points; k++) {
                                acc += penrose[k][j] * normalized[mapping[k]][i];
                        }

                        F[i * 3 + j] = acc;
                }
        }

        res[0] = 0;
        res[1] = 0;
        res[2] = 0;

        for (let k = 0; k < num_points; k++) {
                for (let i = 0; i < 3; i++) {
                        let acc = 0.0;
                        for (let j = 0; j < 3; j++) {
                                acc += F[i * 3 + j] * ideal_points[k][j];
                        }

                        const delta = acc - normalized[mapping[k]][i];
                        res[i] += delta * delta;
                }
        }
}
