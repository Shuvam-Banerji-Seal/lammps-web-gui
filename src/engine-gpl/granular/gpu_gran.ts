/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRANULAR/pair_gran_hooke.cpp, src/GRANULAR/pair_gran_hooke_history.cpp and
 * src/GRANULAR/pair_gran_hertz_history.cpp (the compute() functions) at commit b332f28. Contributing authors (from the
 * sources): Leo Silbert (SNL), Gary Grest (SNL). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the
 * GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair gran/hooke, gran/hooke/history and gran/hertz/history as a GPU atom kernel (src/engine/gpu/atomKernel.ts): each
 * atom walks the cells around its own and evaluates every contact it is in. Both atoms of a contact evaluate it the
 * same way, in the frame of the lower atom ID (lo, hi: del = x_lo - x_hi, the frame the CPU style keeps the shear in,
 * pair_gran_native.ts): their inputs are then the same values and their results bit-identical, so the forces are
 * equal and opposite and the two copies of a contact's shear stay the same. Each atom adds the force and torque the
 * C++ loop gives its side, and half of the contact's virial. The one rule that depends on the frame: a contact of two
 * atoms of the fix freeze group takes the mass of the lower ID's atom as m_eff (the C++ "if (mask[i] & freeze_group_bit)
 * meff = mj; if (mask[j] & freeze_group_bit) meff = mi;" takes the one of whichever atom its list put first); fix freeze
 * zeroes both atoms' forces, so only that contact's shear and virial follow it.
 *
 * The shear history lives in the CPU pair (PairGranNative.historyContacts): each evaluation hands every atom a table of
 * its contacts (the partner's ID and the shear, K entries), and finish() takes the contacts back from the tables of the
 * lower IDs. A table that overflows makes the evaluation run again with a larger K (again()). Atom IDs are kept as f32,
 * so they must be below 2^24.
 *
 * ext per atom (3 + K vec4): (v, radius), (omega, contact mass), (ID, frozen, table length, 0), then the table entries
 * (partner ID, shear). outv per atom (16 + 4 K): force, torque, half virial (xx yy zz xy xz yz), the new table length,
 * the contacts the table had no room for, then the new table.
 */
import { ATOM_KERNEL_WORKGROUP, WGSL_ATOM_KERNEL_HEAD, type GpuAtomKernel } from '../../engine/gpu/atomKernel';
import { NO_IMAGE } from '../../engine/gpu/cellSort';
import { contactMassArray } from '../../engine/atoms';
import type { System } from '../../engine/system';
import type { SimState } from '../../engine/types';
import type { PairGranNative } from './pair_gran_native';

/** The per-atom values ahead of the table (ext vec4s, outv values). */
const EXT_HEAD = 3;
const OUT_HEAD = 16;
/** The first table length; a table grows in steps of 4. */
const K_START = 12;

/** The run constants of the kernel (gpuParams of the pair). */
interface GranParams {
  kn: number; kt: number; gamman: number; gammat: number; xmu: number; limitDamping: boolean;
  history: boolean; hertz: boolean; freezeBit: number;
}

/** One contact in the frame of the lower ID, as the C++ loop computes it with i = lo, j = hi (fp64). */
interface ContactIn {
  delx: number; dely: number; delz: number; rsq: number; rlo: number; rhi: number;
  vr1: number; vr2: number; vr3: number;
  wlo: [number, number, number]; whi: [number, number, number];
  mlo: number; mhi: number; flo: boolean; fhi: boolean;
  shear: [number, number, number];
}

interface ContactOut { fx: number; fy: number; fz: number; tor1: number; tor2: number; tor3: number }

/**
 * pair_gran_hooke.cpp / pair_gran_hooke_history.cpp / pair_gran_hertz_history.cpp compute, one touching pair (fp64, the
 * C++ statement order); updates c.shear in place for the history styles.
 */
const contactFp64 = (P: GranParams, c: ContactIn, dt: number, shearupdate: boolean, o: ContactOut): void => {
  const { delx, dely, delz, rsq, rlo: radi, rhi: radj } = c;
  const radsum = radi + radj;
  const r = Math.sqrt(rsq);
  const rinv = 1.0 / r;
  const rsqinv = 1.0 / rsq;
  const { vr1, vr2, vr3 } = c;
  const vnnr = vr1 * delx + vr2 * dely + vr3 * delz;
  const vn1 = delx * vnnr * rsqinv, vn2 = dely * vnnr * rsqinv, vn3 = delz * vnnr * rsqinv;
  const vt1 = vr1 - vn1, vt2 = vr2 - vn2, vt3 = vr3 - vn3;
  const wr1 = (radi * c.wlo[0] + radj * c.whi[0]) * rinv;
  const wr2 = (radi * c.wlo[1] + radj * c.whi[1]) * rinv;
  const wr3 = (radi * c.wlo[2] + radj * c.whi[2]) * rinv;
  const mi = c.mlo, mj = c.mhi;
  let meff = mi * mj / (mi + mj);
  if (c.flo) meff = mj;
  if (c.fhi) meff = mi;
  const damp = meff * P.gamman * vnnr * rsqinv;
  let ccel = P.kn * (radsum - r) * rinv - damp;
  let polyhertz = 1.0;
  if (P.hertz) {
    polyhertz = Math.sqrt((radsum - r) * radi * radj / radsum);
    ccel *= polyhertz;
  }
  if (P.limitDamping && ccel < 0.0) ccel = 0.0;
  const vtr1 = vt1 - (delz * wr2 - dely * wr3);
  const vtr2 = vt2 - (delx * wr3 - delz * wr1);
  const vtr3 = vt3 - (dely * wr1 - delx * wr2);
  let fs1: number, fs2: number, fs3: number;
  if (!P.history) {
    // pair_gran_hooke.cpp: tangential force due to tangential velocity damping, capped by the friction
    let vrel = vtr1 * vtr1 + vtr2 * vtr2 + vtr3 * vtr3;
    vrel = Math.sqrt(vrel);
    const fn = P.xmu * Math.abs(ccel * r);
    const fs = meff * P.gammat * vrel;
    const ft = vrel !== 0.0 ? Math.min(fn, fs) / vrel : 0.0;
    fs1 = -ft * vtr1; fs2 = -ft * vtr2; fs3 = -ft * vtr3;
  } else {
    const shear = c.shear;
    if (shearupdate) {
      shear[0] += vtr1 * dt;
      shear[1] += vtr2 * dt;
      shear[2] += vtr3 * dt;
    }
    const shrmag = Math.sqrt(shear[0] * shear[0] + shear[1] * shear[1] + shear[2] * shear[2]);
    if (shearupdate) {
      let rsht = shear[0] * delx + shear[1] * dely + shear[2] * delz;
      rsht *= rsqinv;
      shear[0] -= rsht * delx;
      shear[1] -= rsht * dely;
      shear[2] -= rsht * delz;
    }
    const kt = P.kt, gammat = P.gammat;
    fs1 = -polyhertz * (kt * shear[0] + meff * gammat * vtr1);
    fs2 = -polyhertz * (kt * shear[1] + meff * gammat * vtr2);
    fs3 = -polyhertz * (kt * shear[2] + meff * gammat * vtr3);
    const fs = Math.sqrt(fs1 * fs1 + fs2 * fs2 + fs3 * fs3);
    const fn = P.xmu * Math.abs(ccel * r);
    if (fs > fn) {
      if (shrmag !== 0.0) {
        shear[0] = (fn / fs) * (shear[0] + meff * gammat * vtr1 / kt) - meff * gammat * vtr1 / kt;
        shear[1] = (fn / fs) * (shear[1] + meff * gammat * vtr2 / kt) - meff * gammat * vtr2 / kt;
        shear[2] = (fn / fs) * (shear[2] + meff * gammat * vtr3 / kt) - meff * gammat * vtr3 / kt;
        fs1 *= fn / fs;
        fs2 *= fn / fs;
        fs3 *= fn / fs;
      } else fs1 = fs2 = fs3 = 0.0;
    }
  }
  o.fx = delx * ccel + fs1;
  o.fy = dely * ccel + fs2;
  o.fz = delz * ccel + fs3;
  o.tor1 = rinv * (dely * fs3 - delz * fs2);
  o.tor2 = rinv * (delz * fs1 - delx * fs3);
  o.tor3 = rinv * (delx * fs2 - dely * fs1);
};

/** The WGSL of the kernel for a table of K entries (K = 0: gran/hooke, no history). */
const kernelCode = (K: number, history: boolean, hertz: boolean): string => {
  const EW = EXT_HEAD + K, OW = OUT_HEAD + 4 * K;
  return /* wgsl */ `${WGSL_ATOM_KERNEL_HEAD}
const EW : u32 = ${EW}u;
const OW : u32 = ${OW}u;
const KMAX : u32 = ${K}u;

@compute @workgroup_size(${ATOM_KERNEL_WORKGROUP})
fn force(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pi = pos[i];
  let bi = EW * i;
  let a0 = ext[bi]; let a1 = ext[bi + 1u]; let a2 = ext[bi + 2u];
  let radi = a0.w; let idi = a2.x;
  let nh = u32(a2.z);
  let kn = coef[0]; let kt = coef[1]; let gamman = coef[2]; let gammat = coef[3]; let xmu = coef[4];
  let limitDamping = coef[5] != 0.0;
  let dt = P.u0.x;
  let shearupdate = P.u0.y != 0.0;
  var f = vec3<f32>(0.0);
  var tq = vec3<f32>(0.0);
  var vd = vec3<f32>(0.0);   // xx yy zz
  var vo = vec3<f32>(0.0);   // xy xz yz
  var nout = 0u;
  var over = 0u;
  let ob = OW * i;
  let nc = P.nc;
  let c = ints[P.cellOfBase + i];
  let cx = i32(c % nc.x);
  let cy = i32((c / nc.x) % nc.y);
  let cz = i32(c / (nc.x * nc.y));
  let lo = vec3<i32>(select(-1, 0, nc.x < 3u), select(-1, 0, nc.y < 3u), select(-1, 0, nc.z < 3u));
  let hi = vec3<i32>(select(1, i32(nc.x) - 1, nc.x < 3u), select(1, i32(nc.y) - 1, nc.y < 3u), select(1, i32(nc.z) - 1, nc.z < 3u));
  for (var dzc = lo.z; dzc <= hi.z; dzc = dzc + 1) {
    for (var dyc = lo.y; dyc <= hi.y; dyc = dyc + 1) {
      for (var dxc = lo.x; dxc <= hi.x; dxc = dxc + 1) {
        let c2 = (wrapCell(cz + dzc, nc.z) * nc.y + wrapCell(cy + dyc, nc.y)) * nc.x + wrapCell(cx + dxc, nc.x);
        let jEnd = ints[c2 + 1u];
        for (var j = ints[c2]; j < jEnd; j = j + 1u) {
          if (j == i) { continue; }
          let pj = pos[j];
          // x_j - x_i by the minimum image (the negation of the partner's)
          let d = minImage(pi.xyz, pj.xyz);
          let rsq = d.x * d.x + d.y * d.y + d.z * d.z;
          let bj = EW * j;
          let b0 = ext[bj];
          let radj = b0.w;
          let radsum = radi + radj;
          if (rsq >= radsum * radsum) { continue; }
          let b1 = ext[bj + 1u]; let b2 = ext[bj + 2u];
          let idj = b2.x;
          // the contact in the frame of the lower ID: del = x_lo - x_hi
          let iLo = idi < idj;
          let del = select(d, -d, iLo);
          let rlo = select(radj, radi, iLo); let rhi = select(radi, radj, iLo);
          let vr = select(b0.xyz, a0.xyz, iLo) - select(a0.xyz, b0.xyz, iLo);
          let wlo = select(b1.xyz, a1.xyz, iLo); let whi = select(a1.xyz, b1.xyz, iLo);
          let mlo = select(b1.w, a1.w, iLo); let mhi = select(a1.w, b1.w, iLo);
          let flo = select(b2.y, a2.y, iLo) != 0.0; let fhi = select(a2.y, b2.y, iLo) != 0.0;
          var shear = vec3<f32>(0.0);
${history ? `          for (var m = 0u; m < nh; m = m + 1u) {
            let h = ext[bi + ${EXT_HEAD}u + m];
            if (h.x == idj) { shear = h.yzw; break; }
          }
` : ''}          let r = sqrt(rsq);
          let rinv = 1.0 / r;
          let rsqinv = 1.0 / rsq;
          let vnnr = vr.x * del.x + vr.y * del.y + vr.z * del.z;
          let vn = vec3<f32>(del.x * vnnr * rsqinv, del.y * vnnr * rsqinv, del.z * vnnr * rsqinv);
          let vt = vr - vn;
          let wr = vec3<f32>((rlo * wlo.x + rhi * whi.x) * rinv, (rlo * wlo.y + rhi * whi.y) * rinv, (rlo * wlo.z + rhi * whi.z) * rinv);
          var meff = mlo * mhi / (mlo + mhi);
          if (flo) { meff = mhi; }
          if (fhi) { meff = mlo; }
          let damp = meff * gamman * vnnr * rsqinv;
          var ccel = kn * (radsum - r) * rinv - damp;
          var polyhertz = 1.0;
${hertz ? `          polyhertz = sqrt((radsum - r) * rlo * rhi / radsum);
          ccel = ccel * polyhertz;
` : ''}          if (limitDamping && ccel < 0.0) { ccel = 0.0; }
          let vtr = vec3<f32>(vt.x - (del.z * wr.y - del.y * wr.z), vt.y - (del.x * wr.z - del.z * wr.x), vt.z - (del.y * wr.x - del.x * wr.y));
          var fs3 = vec3<f32>(0.0);
${history ? `          if (shearupdate) { shear = shear + vtr * dt; }
          let shrmag = sqrt(shear.x * shear.x + shear.y * shear.y + shear.z * shear.z);
          if (shearupdate) {
            let rsht = (shear.x * del.x + shear.y * del.y + shear.z * del.z) * rsqinv;
            shear = shear - rsht * del;
          }
          fs3 = -polyhertz * (kt * shear + meff * gammat * vtr);
          let fsm = sqrt(fs3.x * fs3.x + fs3.y * fs3.y + fs3.z * fs3.z);
          let fnorm = xmu * abs(ccel * r);
          if (fsm > fnorm) {
            if (shrmag != 0.0) {
              shear = (fnorm / fsm) * (shear + meff * gammat * vtr / kt) - meff * gammat * vtr / kt;
              fs3 = fs3 * (fnorm / fsm);
            } else {
              fs3 = vec3<f32>(0.0);
            }
          }
` : `          let vrel = sqrt(vtr.x * vtr.x + vtr.y * vtr.y + vtr.z * vtr.z);
          let fnorm = xmu * abs(ccel * r);
          let fsd = meff * gammat * vrel;
          var ft = 0.0;
          if (vrel != 0.0) { ft = min(fnorm, fsd) / vrel; }
          fs3 = -ft * vtr;
`}          let F = del * ccel + fs3;
          let tor = rinv * vec3<f32>(del.y * fs3.z - del.z * fs3.y, del.z * fs3.x - del.x * fs3.z, del.x * fs3.y - del.y * fs3.x);
          if (iLo) {
            f = f + F;
            tq = tq - rlo * tor;
          } else {
            f = f - F;
            tq = tq - rhi * tor;
          }
          vd = vd + 0.5 * del * F;
          vo = vo + 0.5 * vec3<f32>(del.x * F.y, del.x * F.z, del.y * F.z);
${history ? `          if (nout < KMAX) {
            let o = ob + ${OUT_HEAD}u + 4u * nout;
            outv[o] = idj; outv[o + 1u] = shear.x; outv[o + 2u] = shear.y; outv[o + 3u] = shear.z;
            nout = nout + 1u;
          } else {
            over = over + 1u;
          }
` : ''}        }
      }
    }
  }
  outv[ob] = f.x; outv[ob + 1u] = f.y; outv[ob + 2u] = f.z;
  outv[ob + 3u] = tq.x; outv[ob + 4u] = tq.y; outv[ob + 5u] = tq.z;
  outv[ob + 6u] = vd.x; outv[ob + 7u] = vd.y; outv[ob + 8u] = vd.z;
  outv[ob + 9u] = vo.x; outv[ob + 10u] = vo.y; outv[ob + 11u] = vo.z;
  outv[ob + 12u] = f32(nout); outv[ob + 13u] = f32(over); outv[ob + 14u] = 0.0; outv[ob + 15u] = 0.0;
}
`;
};

/**
 * The GPU kernel of pair gran/hooke, gran/hooke/history or gran/hertz/history for the run (run/accel.ts through the
 * gplHook of the style name), or why it cannot run there. The pair is initialized (accelerator() runs init first).
 */
export const gpuGranKernel = (pair: PairGranNative, sys: System): GpuAtomKernel | string => {
  const s0 = sys.state;
  if (!s0.radius || !s0.rmass || !s0.omega || !s0.torque) return `pair ${pair.name} without the per-atom radius, mass, omega and torque of atom_style sphere`;
  const P: GranParams = pair.gpuParams();
  const coef = Float64Array.from([P.kn, P.kt, P.gamman, P.gammat, P.xmu, P.limitDamping ? 1 : 0]);
  const codes = new Map<number, string>();
  let K = P.history ? K_START : 0;
  let shearupdate = false;
  // the cells are at least the largest contact distance (twice the largest radius, which fix pour may raise)
  const maxContact = (st: SimState): number => {
    let m = 0;
    const rad = st.radius!;
    for (let i = 0; i < st.n; i++) if (rad[i] > m) m = rad[i];
    if (m > 0) return 2 * m;
    // no particle with a radius: any cell size will do, the kernel finds no contact
    const b = st.box;
    return 0.25 * Math.min(b.hi[0] - b.lo[0], b.hi[1] - b.lo[1]);
  };
  let maxCut = maxContact(s0);
  const tooLargeId = (st: SimState): boolean => {
    for (let i = 0; i < st.n; i++) if (st.id[i] >= 16777216) return true;
    return false;
  };
  if (tooLargeId(s0)) return 'atom IDs of 2^24 or more (the kernel keeps the contact partners as f32)';

  // the contact tables of this evaluation (atom order): length, partner IDs and shear
  let tabN = new Int32Array(0), tabId = new Float64Array(0), tabSh = new Float64Array(0);
  let index = new Int32Array(0);
  const buildTables = (st: SimState): void => {
    const n = st.n;
    if (tabN.length < n) tabN = new Int32Array(n);
    tabN.fill(0, 0, n);
    if (!P.history) return;
    const c = pair.historyContacts();
    // the atom of each ID (IDs below 2^24)
    let maxId = 0;
    for (let i = 0; i < n; i++) if (st.id[i] > maxId) maxId = st.id[i];
    if (index.length < maxId + 1) index = new Int32Array(maxId + 1);
    index.fill(-1, 0, maxId + 1);
    for (let i = 0; i < n; i++) index[st.id[i]] = i;
    const at = (id: number): number => (id <= maxId ? index[id] : -1);
    // the longest table
    let longest = 0;
    for (let k = 0; k < c.n; k++) {
      const a = at(c.lo[k]), b = at(c.hi[k]);
      if (a < 0 || b < 0) continue;
      if (++tabN[a] > longest) longest = tabN[a];
      if (++tabN[b] > longest) longest = tabN[b];
    }
    if (longest + 4 > K) K = 4 * Math.ceil((longest + 4) / 4);
    if (tabId.length < K * n) { tabId = new Float64Array(K * n); tabSh = new Float64Array(3 * K * n); }
    tabN.fill(0, 0, n);
    const put = (a: number, partner: number, k: number): void => {
      const m = tabN[a]++;
      tabId[K * a + m] = partner;
      const o = 3 * (K * a + m);
      tabSh[o] = c.shear[3 * k]; tabSh[o + 1] = c.shear[3 * k + 1]; tabSh[o + 2] = c.shear[3 * k + 2];
    };
    for (let k = 0; k < c.n; k++) {
      const a = at(c.lo[k]), b = at(c.hi[k]);
      if (a < 0 || b < 0) continue;
      put(a, c.hi[k], k);
      put(b, c.lo[k], k);
    }
  };

  const outWidth = (): number => OUT_HEAD + 4 * K;

  /** The kernel's arithmetic in fp64 over every pair (the minimum image of cellSort.ts), outWidth values per atom. */
  const fp64 = (st: SimState): Float64Array => {
    const n = st.n, x = st.x, v = st.v, om = st.omega!, rad = st.radius!, id = st.id, mask = st.mask;
    const mass = contactMassArray(st);
    const OW = outWidth();
    const out = new Float64Array(OW * n);
    const box = st.box;
    const L = [0, 1, 2].map((d) => (box.periodic[d] || (d === 2 && st.dimension === 2) ? box.hi[d] - box.lo[d] : NO_IMAGE));
    const [xy, xz, yz] = box.triclinic ? box.tilt : [0, 0, 0];
    const two = st.dimension === 2;
    const frozen = (i: number): boolean => P.freezeBit !== 0 && (mask[i] & P.freezeBit) !== 0;
    const o: ContactOut = { fx: 0, fy: 0, fz: 0, tor1: 0, tor2: 0, tor3: 0 };
    for (let i = 0; i < n; i++) {
      let fx = 0, fy = 0, fz = 0, tx = 0, ty = 0, tz = 0;
      let vxx = 0, vyy = 0, vzz = 0, vxy = 0, vxz = 0, vyz = 0;
      let nout = 0, over = 0;
      const ob = OW * i;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        let ddx = x[3 * j] - x[3 * i], ddy = x[3 * j + 1] - x[3 * i + 1], ddz = two ? 0 : x[3 * j + 2] - x[3 * i + 2];
        const fzc = ddz / L[2];
        const fyc = (ddy - yz * fzc) / L[1];
        const fxc = (ddx - xy * fyc - xz * fzc) / L[0];
        const kx = Math.round(fxc), ky = Math.round(fyc), kz = Math.round(fzc);
        ddx -= kx * L[0] + ky * xy + kz * xz; ddy -= ky * L[1] + kz * yz; ddz -= kz * L[2];
        const rsq = ddx * ddx + ddy * ddy + ddz * ddz;
        const radsum = rad[i] + rad[j];
        if (rsq >= radsum * radsum) continue;
        const iLo = id[i] < id[j];
        const lo = iLo ? i : j, hi = iLo ? j : i;
        const sg = iLo ? -1 : 1;
        const shear: [number, number, number] = [0, 0, 0];
        if (P.history) {
          for (let m = 0; m < tabN[i]; m++) {
            if (tabId[K * i + m] === id[j]) {
              const q = 3 * (K * i + m);
              shear[0] = tabSh[q]; shear[1] = tabSh[q + 1]; shear[2] = tabSh[q + 2];
              break;
            }
          }
        }
        const delx = sg * ddx, dely = sg * ddy, delz = sg * ddz;
        contactFp64(P, {
          delx, dely, delz, rsq, rlo: rad[lo], rhi: rad[hi],
          vr1: v[3 * lo] - v[3 * hi], vr2: v[3 * lo + 1] - v[3 * hi + 1], vr3: v[3 * lo + 2] - v[3 * hi + 2],
          wlo: [om[3 * lo], om[3 * lo + 1], om[3 * lo + 2]], whi: [om[3 * hi], om[3 * hi + 1], om[3 * hi + 2]],
          mlo: mass[lo], mhi: mass[hi], flo: frozen(lo), fhi: frozen(hi), shear,
        }, st.dt, shearupdate, o);
        if (iLo) {
          fx += o.fx; fy += o.fy; fz += o.fz;
          tx -= rad[lo] * o.tor1; ty -= rad[lo] * o.tor2; tz -= rad[lo] * o.tor3;
        } else {
          fx -= o.fx; fy -= o.fy; fz -= o.fz;
          tx -= rad[hi] * o.tor1; ty -= rad[hi] * o.tor2; tz -= rad[hi] * o.tor3;
        }
        vxx += 0.5 * delx * o.fx; vyy += 0.5 * dely * o.fy; vzz += 0.5 * delz * o.fz;
        vxy += 0.5 * delx * o.fy; vxz += 0.5 * delx * o.fz; vyz += 0.5 * dely * o.fz;
        if (P.history) {
          if (nout < K) {
            const q = ob + OUT_HEAD + 4 * nout;
            out[q] = id[j]; out[q + 1] = shear[0]; out[q + 2] = shear[1]; out[q + 3] = shear[2];
            nout++;
          } else over++;
        }
      }
      out[ob] = fx; out[ob + 1] = fy; out[ob + 2] = fz;
      out[ob + 3] = tx; out[ob + 4] = ty; out[ob + 5] = tz;
      out[ob + 6] = vxx; out[ob + 7] = vyy; out[ob + 8] = vzz; out[ob + 9] = vxy; out[ob + 10] = vxz; out[ob + 11] = vyz;
      out[ob + 12] = nout; out[ob + 13] = over;
    }
    return out;
  };

  return {
    name: pair.name,
    get code(): string {
      let c = codes.get(K);
      if (!c) codes.set(K, (c = kernelCode(K, P.history, P.hertz)));
      return c;
    },
    key: `gran:${Array.from(coef).join(',')}`,
    coef,
    stride: s0.ntypes + 1,
    get maxCutoff(): number { return maxCut; },
    get extWidth(): number { return EXT_HEAD + K; },
    get outWidth(): number { return outWidth(); },
    begin: (st, step) => {
      // As native LAMMPS does (pair_gran_hooke_history.cpp compute at b332f28): "if (update->setupflag) shearupdate = 0"
      shearupdate = step;
      maxCut = maxContact(st);
      if (tooLargeId(st)) throw new Error('pair gran on the GPU: an atom ID of 2^24 or more (the kernel keeps the contact partners as f32)');
      buildTables(st);
    },
    scalars: (st) => [st.dt, shearupdate ? 1 : 0, 0, 0, 0, 0, 0, 0],
    extra: (st, out) => {
      const n = st.n, v = st.v, om = st.omega!, rad = st.radius!, id = st.id, mask = st.mask;
      const mass = contactMassArray(st);
      const EW4 = 4 * (EXT_HEAD + K);
      for (let i = 0; i < n; i++) {
        const b = EW4 * i;
        out[b] = v[3 * i]; out[b + 1] = v[3 * i + 1]; out[b + 2] = v[3 * i + 2]; out[b + 3] = rad[i];
        out[b + 4] = om[3 * i]; out[b + 5] = om[3 * i + 1]; out[b + 6] = om[3 * i + 2]; out[b + 7] = mass[i];
        out[b + 8] = id[i]; out[b + 9] = P.freezeBit !== 0 && (mask[i] & P.freezeBit) !== 0 ? 1 : 0; out[b + 10] = tabN[i]; out[b + 11] = 0;
        for (let m = 0; m < tabN[i]; m++) {
          const q = b + 4 * (EXT_HEAD + m), t = 3 * (K * i + m);
          out[q] = tabId[K * i + m]; out[q + 1] = tabSh[t]; out[q + 2] = tabSh[t + 1]; out[q + 3] = tabSh[t + 2];
        }
      }
    },
    again: (st, out) => {
      if (!P.history) return false;
      const OW = outWidth();
      let over = 0;
      for (let i = 0; i < st.n; i++) over = Math.max(over, out[OW * i + 13]);
      if (over === 0) return false;
      // a table with no room for every contact of its atom: run again with room for them
      K = 4 * Math.ceil((K + over + 4) / 4);
      if (tabId.length < K * st.n) { tabId = new Float64Array(K * st.n); tabSh = new Float64Array(3 * K * st.n); }
      buildTables(st);
      return true;
    },
    finish: (st, out) => {
      const n = st.n, f = st.f, tq = st.torque!, id = st.id;
      const OW = outWidth();
      const vt = [0, 0, 0, 0, 0, 0];
      for (let i = 0; i < n; i++) {
        const o = OW * i;
        f[3 * i] = out[o]; f[3 * i + 1] = out[o + 1]; f[3 * i + 2] = out[o + 2];
        tq[3 * i] = out[o + 3]; tq[3 * i + 1] = out[o + 4]; tq[3 * i + 2] = out[o + 5];
        for (let c = 0; c < 6; c++) vt[c] += out[o + 6 + c];
      }
      if (P.history) {
        // the contacts from the tables of their lower IDs (the higher ID's copy is the same values)
        let nc = 0;
        for (let i = 0; i < n; i++) {
          const m = out[OW * i + 12];
          for (let q = 0; q < m; q++) if (id[i] < out[OW * i + OUT_HEAD + 4 * q]) nc++;
        }
        const lo = new Int32Array(nc), hi = new Int32Array(nc), sh = new Float64Array(3 * nc);
        let k = 0;
        for (let i = 0; i < n; i++) {
          const m = out[OW * i + 12];
          for (let q = 0; q < m; q++) {
            const e = OW * i + OUT_HEAD + 4 * q;
            if (!(id[i] < out[e])) continue;
            lo[k] = id[i]; hi[k] = out[e];
            sh[3 * k] = out[e + 1]; sh[3 * k + 1] = out[e + 2]; sh[3 * k + 2] = out[e + 3];
            k++;
          }
        }
        pair.setHistoryContacts(lo, hi, sh);
      }
      return { pe: 0, virial: vt[0] + vt[1] + vt[2], virialTensor: vt, evdwl: 0, ecoul: 0 };
    },
    fp64,
  };
};
