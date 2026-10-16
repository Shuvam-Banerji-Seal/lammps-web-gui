/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the PTM library (Polyhedral Template Matching, PM Larsen, MIT licence), src/PTM/ptm_normalize_vertices.cpp at LAMMPS commit b332f28 (https://github.com/lammps/lammps). Copyright (c) 2016 PM Larsen (MIT, see LICENSE.ptm). The LAMMPS compute that uses it (src/PTM/compute_ptm_atom.cpp) is GPL-2.0; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */

export function subtract_barycentre(num: number, points: number[][], normalized: number[][]): void {
        //calculate barycentre
        const sum = [0, 0, 0];
        for (let i = 0; i < num; i++) {
                sum[0] += points[i][0];
                sum[1] += points[i][1];
                sum[2] += points[i][2];
        }

        sum[0] /= num;
        sum[1] /= num;
        sum[2] /= num;

        //subtract barycentre
        for (let i = 0; i < num; i++) {
                normalized[i][0] = points[i][0] - sum[0];
                normalized[i][1] = points[i][1] - sum[1];
                normalized[i][2] = points[i][2] - sum[2];
        }
}

export function normalize_vertices(num: number, points: number[][], normalized: number[][]): number {
        subtract_barycentre(num, points, normalized);

        //calculate mean length
        let scale = 0.0;
        for (let i = 1; i < num; i++) {
                const x = normalized[i][0];
                const y = normalized[i][1];
                const z = normalized[i][2];

                const norm = Math.sqrt(x * x + y * y + z * z);
                scale += norm;
        }
        scale /= num;

        //scale vertices such that mean length is 1
        for (let i = 0; i < num; i++) {
                normalized[i][0] /= scale;
                normalized[i][1] /= scale;
                normalized[i][2] /= scale;
        }

        return scale;
}
