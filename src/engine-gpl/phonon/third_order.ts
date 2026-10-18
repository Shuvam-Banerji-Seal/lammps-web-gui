/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/PHONON/third_order.cpp and
 * third_order.h (class ThirdOrder, command third_order group-ID regular|eskm delta keyword values: the third-order
 * force constant tensor by central finite differences of the forces, four displaced configurations per
 * (i, alpha, j, beta), written as text). Contributing author: Charlie Sievers (UC Davis).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Documented behaviour (docs.lammps.org/third_order.html): "Calculate the third order force constant tensor by finite
 * difference of the selected group", "The three elements correspond to the three gamma elements for a specific
 * i/alpha/j/beta/k. The initial five numbers are i, alpha, j, beta, and k respectively.", and "If the style eskm is
 * selected, the tensor will be using energy units of 10 J/mol."
 *
 * Operation order follows the C++: displacements are applied to the owned atom and to its periodic images as
 * "x += delta * magnitude" in the native sequence (+1, +1 / -2 / +1, -2 / +1 / -2 ...), the four force sums are added
 * to the tensor in native order, and the neighbour sets are those of a full neighbour list with cutoff
 * "cut + skin" per type pair (getNeighbortags: the second shell of each atom).
 *
 * Browser scope: the binary and gzip outputs are native-only and are refused by name, as are neigh_modify
 * exclude/include and bonded topology (special pairs change the neighbour sets). The output numbers use the
 * "%17.8f" format of native fmt (round-half-even on the exact binary value).
 *
 * Native quirk kept (writeMatrix in third_order.cpp): for a group that is not the whole system, the third index of a
 * line is groupmap[k] + 1 with groupmap indexed by atom ID, so it is 0 for an atom that is not in the group.
 */
import type { Handler } from '../../engine/commands/args';
import { StyleError } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';

/** conv_energy, conv_mass, conv_distance of each unit style (third_order.cpp convert_units). */
const CONVERT: Record<string, [number, number, number]> = {
  real: [418.4, 1, 1],
  metal: [9648.5, 1, 1],
  si: [6.022E22, 6.022E26, 1E-10],
  cgs: [6.022E12, 6.022E23, 1E-7],
  electron: [262550, 1, 0.529177249],
  micro: [6.022E10, 6.022E11, 1E-4],
  nano: [6.022E4, 6.022E5, 0.1],
};

/** utils::logical of native LAMMPS (yes/on/true/1 and no/off/false/0). */
const logical = (w: string): boolean => {
  if (w === 'yes' || w === 'on' || w === 'true' || w === '1') return true;
  if (w === 'no' || w === 'off' || w === 'false' || w === '0') return false;
  throw new StyleError(`Expected boolean parameter instead of '${w}' in input script or data file`);
};

/**
 * C printf / fmt "%.8f" of a double, exact: the binary value is written as mant * 2^e2 and rounded half to even at
 * the eighth decimal, so no decimal rounding of the float's shortest form is involved. The sign is printed for
 * negative values and for -0 (as printf does).
 */
export const fixed8 = (v: number): string => {
  if (Number.isNaN(v)) return 'nan';
  if (!Number.isFinite(v)) return v < 0 ? '-inf' : 'inf';
  const neg = v < 0 || Object.is(v, -0);
  const dv = new DataView(new ArrayBuffer(8));
  dv.setFloat64(0, Math.abs(v));
  const hi = dv.getUint32(0), lo = dv.getUint32(4);
  const be = (hi >>> 20) & 0x7ff;
  let mant = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let e2: number;
  if (be === 0) e2 = -1074;
  else { mant |= 1n << 52n; e2 = be - 1075; }
  const scale = 100000000n;
  let n: bigint;
  if (e2 >= 0) {
    n = (mant << BigInt(e2)) * scale;
  } else {
    const num = mant * scale;
    const den = 1n << BigInt(-e2);
    n = num / den;
    const twice = 2n * (num % den);
    if (twice > den || (twice === den && (n & 1n) === 1n)) n += 1n;
  }
  const digits = n.toString().padStart(9, '0');
  return `${neg ? '-' : ''}${digits.slice(0, -8)}.${digits.slice(-8)}`;
};

export const thirdOrderCmd: Handler = ({ sys }, a) => {
  // command(): narg < 2, then the group, then the style, then the options, then delta
  if (a.length < 2) throw new StyleError('Illegal third_order command');
  const [group, style] = a;
  let bit: number;
  try { bit = sys.groups.bit(group); } catch { throw new StyleError('Could not find third_order group ID'); }
  if (style !== 'regular' && style !== 'eskm') throw new StyleError('Illegal Dynamical Matrix command');
  // options(narg - 3, ...): a missing delta is the "narg < 0" case
  if (a.length < 3) throw new StyleError('Illegal Third Order command');
  let file = 'third_order.dat';
  let folded = false;
  for (let k = 3; k < a.length;) {
    const key = a[k];
    const val = a[k + 1];
    if (key === 'binary') {
      if (val === undefined) throw new StyleError('Illegal Third Order command');
      if (val === 'gzip') throw new StyleError('third_order binary gzip output is native-only; the browser engine writes text only');
      if (logical(val)) throw new StyleError('third_order binary yes output is native-only; the browser engine writes text only (binary no)');
    } else if (key === 'file') {
      if (val === undefined) throw new StyleError('Illegal third_order command');
      file = val;
    } else if (key === 'fold') {
      if (val === undefined) throw new StyleError('Illegal Third Order command');
      if (val !== 'yes' && val !== 'no') throw new StyleError('Illegal input for Third Order fold option');
      folded = val === 'yes';
    } else {
      throw new StyleError('Illegal Third Order command');
    }
    k += 2;
  }
  const del = parseNum(a[2], 'third_order delta');

  // conversion: 1 for regular; eskm divides by conv_distance twice (convert_units, command())
  let conversion = 1;
  if (style === 'eskm') {
    const ust = sys.units.style;
    if (ust === 'lj') throw new StyleError('Conversion Not Set');
    const c = CONVERT[ust];
    if (!c) throw new StyleError('Units Type Conversion Not Found');
    conversion = c[0] / c[2] / c[2];
  }

  const ss = sys.state;
  if (sys.nb.excludes.length || sys.nb.includeBit) {
    throw new StyleError('third_order with neigh_modify exclude or include is not ported (the neighbour sets would differ)');
  }
  if (ss.topo.bonds.n > 0) {
    throw new StyleError('third_order with bonded topology is not ported (special pairs change the neighbour sets)');
  }
  sys.setupNeighbors();
  const s = sys.state;
  const natoms = s.n;
  const nb = sys.nb;

  // tag (1-based) -> owned index; native indexes its arrays by tag, so the IDs must be 1..N
  const idxOfTag = new Int32Array(natoms).fill(-1);
  for (let i = 0; i < natoms; i++) {
    const t = s.id[i];
    if (!Number.isInteger(t) || t < 1 || t > natoms || idxOfTag[t - 1] >= 0) {
      throw new StyleError('third_order requires atom IDs 1 to N, each used once');
    }
    idxOfTag[t - 1] = i;
  }

  // groupmap (create_groupmap): indexed by tag - 1, the rank of the atom among the group atoms in ID order, or -1
  const gm = new Int32Array(natoms).fill(-1);
  let gcount = 0;
  for (let t = 0; t < natoms; t++) {
    if (s.mask[idxOfTag[t]] & bit) gm[t] = gcount++;
  }
  const dynlen = gcount * 3;
  const dynlenb = folded ? natoms * 3 : dynlen;

  // getNeighbortags: near[t] is the first shell of atom t (its full-list neighbours and itself, by tag - 1), built
  // with the cutoff cut + skin of its type pair, "rsq <= cutneighsq"
  const nt = s.ntypes + 1;
  const cut = sys.ff.pair?.cut;
  const cutsq = new Float64Array(nt * nt);
  for (let k = 0; k < nt * nt; k++) {
    const c = cut ? cut[k] : 0;
    if (c > 0) { const cn = c + nb.skin; cutsq[k] = cn * cn; }
  }
  const xa = nb.xall, owner = nb.owner, ta = nb.typeall, nall = nb.nall;
  const near: Uint8Array[] = [];
  for (let t = 0; t < natoms; t++) {
    const ii = idxOfTag[t];
    const row = new Uint8Array(natoms);
    row[t] = 1;
    const xi = xa[3 * ii], yi = xa[3 * ii + 1], zi = xa[3 * ii + 2];
    const tyi = ta[ii] * nt;
    for (let k = 0; k < nall; k++) {
      const oi = owner[k];
      if (oi === ii) continue;
      const dx = xi - xa[3 * k], dy = yi - xa[3 * k + 1], dz = zi - xa[3 * k + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      if (rsq <= cutsq[tyi + ta[k]]) row[s.id[oi] - 1] = 1;
    }
    near.push(row);
  }
  // the second shell: the atom itself, its first shell and the first shell of each of those atoms
  const shell: number[][] = [];
  for (let t = 0; t < natoms; t++) {
    const row = new Uint8Array(natoms);
    row.set(near[t]);
    for (let u = 0; u < natoms; u++) {
      if (!near[t][u]) continue;
      const nu = near[u];
      for (let w = 0; w < natoms; w++) if (nu[w]) row[w] = 1;
    }
    const list: number[] = [];
    for (let w = 0; w < natoms; w++) if (row[w]) list.push(w);
    shell.push(list);
  }

  // displace_atom: the owned atom moves, its periodic images follow (refreshed from it before each force call)
  const displace = (li: number, dir: number, magnitude: number): void => {
    if (li < 0) return;
    sys.state.x[3 * li + dir] += del * magnitude;
  };
  const force = (): Float64Array => {
    sys.nb.forwardComm(sys.state, sys.geom);
    sys.computeForcesInRun(sys.computeFlags());
    return sys.state.f;
  };

  const dynmat = new Float64Array(dynlenb);
  const lines: string[] = [];
  // writeMatrix: one line per (i, alpha, j, beta, k) whose three-vector has norm > 1e-16
  const writeMatrix = (i: number, a0: number, j: number, b0: number): void => {
    if (folded) {
      for (let k = 0; k < natoms; k++) {
        const x0 = dynmat[k * 3], x1 = dynmat[k * 3 + 1], x2 = dynmat[k * 3 + 2];
        const norm = x0 * x0 + x1 * x1 + x2 * x2;
        if (norm > 1.0e-16) {
          lines.push(`${i + 1} ${a0 + 1} ${j + 1} ${b0 + 1} ${k + 1} ${fixed8(x0 * conversion).padStart(17)} ${fixed8(x1 * conversion).padStart(17)} ${fixed8(x2 * conversion).padStart(17)}\n`);
        }
      }
    } else {
      for (let k = 0; k < gcount; k++) {
        const x0 = dynmat[k * 3], x1 = dynmat[k * 3 + 1], x2 = dynmat[k * 3 + 2];
        const norm = x0 * x0 + x1 * x1 + x2 * x2;
        if (norm > 1.0e-16) {
          // groupmap[k] + 1 (native indexes groupmap by tag - 1; see the quirk in the header)
          lines.push(`${i + 1} ${a0 + 1} ${j + 1} ${b0 + 1} ${gm[k] + 1} ${fixed8(x0 * conversion).padStart(17)} ${fixed8(x1 * conversion).padStart(17)} ${fixed8(x2 * conversion).padStart(17)}\n`);
        }
      }
    }
  };

  for (let i = 1; i <= natoms; i++) {
    if (gm[i - 1] < 0) continue;
    const li = idxOfTag[i - 1];
    for (let alpha = 0; alpha < 3; alpha++) {
      for (const j0 of shell[i - 1]) {
        if (gm[j0] < 0 && !folded) continue;
        const lj = idxOfTag[j0];
        for (let beta = 0; beta < 3; beta++) {
          // the four displaced configurations: (+,+), (+,-), (-,+), (-,-) with the native sign pattern
          const accum = (f: Float64Array, subtract: boolean, divide: boolean): void => {
            for (let k = 0; k < natoms; k++) {
              const lk = idxOfTag[k];
              for (let gamma = 0; gamma < 3; gamma++) {
                if (li >= 0 && lj >= 0 && lk >= 0 && ((gm[j0] >= 0 && gm[k] >= 0) || folded)) {
                  const e = folded ? k * 3 + gamma : gm[k] * 3 + gamma;
                  if (subtract) dynmat[e] -= f[3 * lk + gamma];
                  else dynmat[e] += f[3 * lk + gamma];
                  if (divide) dynmat[e] /= (4 * del * del);
                }
              }
            }
          };
          displace(li, alpha, 1);
          displace(lj, beta, 1);
          accum(force(), false, false);
          displace(lj, beta, -2);
          accum(force(), true, false);
          displace(lj, beta, 1);
          displace(li, alpha, -2);
          displace(lj, beta, 1);
          accum(force(), true, false);
          displace(lj, beta, -2);
          accum(force(), false, true);
          displace(lj, beta, 1);
          displace(li, alpha, 1);
          writeMatrix(gm[i - 1], alpha, folded ? j0 : gm[j0], beta);
          dynmat.fill(0);
        }
      }
    }
  }
  // the displaced positions were returned to their place (the positions change, so bump), but native keeps the forces
  // of the last update_force call, made at the last displaced configuration: those stored forces stay current here too
  // (as hyper and neb do after computeForcesInRun); every run recomputes its forces at setup
  sys.bump();
  sys.forcesCurrent();
  sys.writeFile(file, lines.join(''), false);
};
