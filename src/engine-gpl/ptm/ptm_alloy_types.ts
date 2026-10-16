/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the PTM library (Polyhedral Template Matching, PM Larsen, MIT licence), src/PTM/ptm_alloy_types.cpp at LAMMPS commit b332f28 (https://github.com/lammps/lammps). Copyright (c) 2016 PM Larsen (MIT, see LICENSE.ptm). The LAMMPS compute that uses it (src/PTM/compute_ptm_atom.cpp) is GPL-2.0; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */

import { PTM_ALLOY_B2, PTM_ALLOY_BN, PTM_ALLOY_L10, PTM_ALLOY_L12_AU, PTM_ALLOY_L12_CU, PTM_ALLOY_NONE, PTM_ALLOY_PURE, PTM_ALLOY_SIC, PTM_MATCH_BCC, PTM_MATCH_DCUB, PTM_MATCH_DHEX, PTM_MATCH_FCC, PTM_MATCH_GRAPHENE, PTM_MAX_POINTS } from './ptm_constants';
import { RefData } from './ptm_types';

const NUM_ALLOY_TYPES = 3;
const typedata: number[][] = [
  [PTM_MATCH_FCC, PTM_ALLOY_L10, 0x00000db6],
  [PTM_MATCH_FCC, PTM_ALLOY_L12_CU, 0x00000492],
  [PTM_MATCH_FCC, PTM_ALLOY_L12_AU, 0x00001ffe],
];

function test_pure(num_nbrs: number, numbers: number[]): boolean {
  for (let i = 1; i < num_nbrs + 1; i++) if (numbers[i] !== numbers[0]) return false;
  return true;
}

function test_binary(num_nbrs: number, numbers: number[]): boolean {
  const a = numbers[0];
  let b = -1;
  for (let i = 1; i < num_nbrs + 1; i++) {
    if (numbers[i] !== a) {
      if (b === -1) b = numbers[i];
      else if (numbers[i] !== b) return false;
    }
  }

  return true;
}

function test_shell_structure(ref: RefData, mapping: number[], numbers: number[], num_inner: number): boolean {
  const binary = new Array<number>(PTM_MAX_POINTS).fill(0);
  for (let i = 0; i < ref.num_nbrs + 1; i++) binary[i] = numbers[mapping[i]] === numbers[0] ? 0 : 1;

  for (let i = 1; i < num_inner + 1; i++) if (binary[i] === binary[0]) return false;

  for (let i = num_inner + 1; i < ref.num_nbrs + 1; i++) if (binary[i] !== binary[0]) return false;

  return true;
}

/** Returns the C int32_t value: the uint32 minimum code converted to signed (0xFFFFFFFF becomes -1). */
function canonical_alloy_representation(ref: RefData, mapping: number[], numbers: number[]): number {
  const binary = new Array<number>(PTM_MAX_POINTS).fill(0);
  for (let i = 0; i < ref.num_nbrs + 1; i++) binary[i] = numbers[mapping[i]] === numbers[0] ? 0 : 1;

  const temp = new Array<number>(PTM_MAX_POINTS).fill(0);
  let best = 0xffffffff;
  for (let j = 0; j < ref.num_mappings; j++) {
    for (let i = 0; i < ref.num_nbrs + 1; i++) temp[ref.mapping![j][i]] = binary[i];

    let code = 0;
    for (let i = 0; i < ref.num_nbrs + 1; i++) code |= temp[i] << i;

    best = Math.min(best, code >>> 0);
  }

  return best | 0;
}

export function find_alloy_type(ref: RefData, mapping: number[], numbers: number[]): number {
  for (let i = 0; i < ref.num_nbrs + 1; i++) if (numbers[i] === -1) return PTM_ALLOY_NONE;

  if (test_pure(ref.num_nbrs, numbers)) return PTM_ALLOY_PURE;

  if (!test_binary(ref.num_nbrs, numbers)) return PTM_ALLOY_NONE;

  const code = canonical_alloy_representation(ref, mapping, numbers) >>> 0;
  for (let i = 0; i < NUM_ALLOY_TYPES; i++)
    if (ref.type === typedata[i][0] && code === typedata[i][2]) return typedata[i][1];

  if (ref.type === PTM_MATCH_BCC) if (test_shell_structure(ref, mapping, numbers, 8)) return PTM_ALLOY_B2;

  if (ref.type === PTM_MATCH_DCUB || ref.type === PTM_MATCH_DHEX) if (test_shell_structure(ref, mapping, numbers, 4)) return PTM_ALLOY_SIC;

  if (ref.type === PTM_MATCH_GRAPHENE) if (test_shell_structure(ref, mapping, numbers, 3)) return PTM_ALLOY_BN;

  return PTM_ALLOY_NONE;
}
