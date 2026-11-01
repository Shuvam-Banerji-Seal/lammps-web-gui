import type { Exclusion } from '../neighbor';

/*
 * neigh_modify exclude and include on the GPU path (the styles kernel, webgpuForces.ts KERNEL_STYLES). neigh_modify.html:
 * "type M N = exclude if one atom in pair is type M, other is type N"; "group group1-ID group2-ID = exclude if one atom
 * is in 1st group, other in 2nd"; "molecule/intra group-ID = exclude if both atoms are in the same molecule and in
 * group"; "molecule/inter group-ID = exclude if both atoms are in different molecules and in group"; "The include
 * option limits the building of pairwise neighbor lists to atoms in the specified group." An excluded pair is not in the
 * neighbor list at all, so it has no pair term, no special weight and no kspace correction.
 *
 * The type exclusions become zero cutoffs in the packed coefficients (both orders of the type pair). The other rules
 * and the include group go to the kernel as a short rule list in its uniform parameters, with each slot's group mask
 * and molecule ID in the ints buffer; the kernel tests a pair before anything else, as the CPU neighbor list does
 * (neighbor.ts excludeFn). Native applies the rules when it builds a list, with the masks of that moment: groups that
 * change during a run (dynamic groups) are therefore left to the CPU path.
 */

/** The rules of the kernel beyond the type pairs: [kind, a, b], kind 1 group (bits a, b), 2 molecule/intra, 3 molecule/inter (bit a). */
export type GpuExclusionRule = [number, number, number];

export interface GpuExclusions {
  /** type pairs excluded in both orders */
  typePairs: Array<[number, number]>;
  rules: GpuExclusionRule[];
  /** neigh_modify include group bit (0: none) */
  include: number;
  key: string;
}

/** The rule list's length in the kernel's uniform parameters. */
export const MAX_GPU_EXCLUSION_RULES = 16;

/** The exclusions of a neighbor setup, or null without any. */
export const gpuExclusions = (excludes: readonly Exclusion[], includeBit: number): GpuExclusions | null => {
  if (!excludes.length && !includeBit) return null;
  const typePairs: Array<[number, number]> = [];
  const rules: GpuExclusionRule[] = [];
  for (const e of excludes) {
    if (e.kind === 'type') typePairs.push([e.a, e.b]);
    else if (e.kind === 'group') rules.push([1, e.a >>> 0, e.b >>> 0]);
    else rules.push([e.kind === 'molecule/intra' ? 2 : 3, e.a >>> 0, 0]);
  }
  return { typePairs, rules, include: includeBit >>> 0, key: JSON.stringify([typePairs, rules, includeBit]) };
};

/** The group bits the exclusions read (to keep dynamic groups off this path). */
export const exclusionGroupBits = (ex: GpuExclusions): number =>
  ex.rules.reduce((m, r) => m | r[1] | r[2], ex.include) >>> 0;

/** Whether the pair (i, j) is excluded by the rules and the include group (the type pairs are in the coefficients). */
export const excludedPair = (ex: GpuExclusions, mask: ArrayLike<number>, mol: ArrayLike<number> | null, i: number, j: number): boolean => {
  const mi = mask[i] >>> 0, mj = mask[j] >>> 0;
  if (ex.include && (!(mi & ex.include) || !(mj & ex.include))) return true;
  for (const [kind, a, b] of ex.rules) {
    if (kind === 1) {
      if (((mi & a) && (mj & b)) || ((mi & b) && (mj & a))) return true;
    } else if ((mi & a) && (mj & a)) {
      const same = (mol ? mol[i] : 0) === (mol ? mol[j] : 0);
      if ((kind === 2 && same) || (kind === 3 && !same)) return true;
    }
  }
  return false;
};

/** excludedPair in WGSL, for a kernel whose Params has exclBase, nrules, incl and rules, and whose ints holds mask, molecule per slot. */
export const WGSL_EXCLUDED = /* wgsl */ `
// neigh_modify exclude/include (gpu/exclusions.ts): the slots i, j are not a pair
fn excludedPair(i : u32, j : u32) -> bool {
  let mi = ints[P.exclBase + 2u * i];
  let mj = ints[P.exclBase + 2u * j];
  if (P.incl != 0u && ((mi & P.incl) == 0u || (mj & P.incl) == 0u)) { return true; }
  for (var r = 0u; r < P.nrules; r = r + 1u) {
    let e = P.rules[r];
    if (e.x == 1u) {
      if (((mi & e.y) != 0u && (mj & e.z) != 0u) || ((mi & e.z) != 0u && (mj & e.y) != 0u)) { return true; }
    } else if ((mi & e.y) != 0u && (mj & e.y) != 0u) {
      let same = ints[P.exclBase + 2u * i + 1u] == ints[P.exclBase + 2u * j + 1u];
      if ((e.x == 2u && same) || (e.x == 3u && !same)) { return true; }
    }
  }
  return false;
}
`;
