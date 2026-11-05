/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRANULAR/pair_granular.cpp (compute), src/GRANULAR/granular_model.cpp (GranularModel::calculate_forces),
 * src/GRANULAR/gran_sub_mod_normal.cpp, gran_sub_mod_damping.cpp, gran_sub_mod_tangential.cpp and gran_sub_mod.cpp
 * (rotate_rescale_vec) at commit b332f28. Contributing authors (from the sources): Dan Bolintineanu (SNL), Ishan
 * Srivastava (SNL), Jeremy Lechman (SNL), Leo Silbert (SNL), Gary Grest (SNL), Joel Clemmer (SNL). LAMMPS is Copyright
 * (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair granular as a GPU atom kernel (src/engine/gpu/atomKernel.ts), for the normal models hooke, hertz, hertz/material
 * and dmt, every damping model, and the tangential models linear_nohistory, linear_history, mindlin, mindlin/force,
 * mindlin_rescale and mindlin_rescale/force (PairGranular.gpuModels lists what keeps a run on the CPU). Each atom walks
 * the cells around its own and evaluates every contact it is in, in the frame of the lower atom ID (lo, hi:
 * n = (x_lo - x_hi) / r, the frame the CPU style keeps the tangential history in). Swapping the two atoms negates every
 * input of the contact exactly, so both atoms of a contact compute the same values: the forces are equal and opposite
 * and the two copies of the history stay the same. Each atom adds the force and torque its side of the contact gets
 * (the torque -(n x F_t)(R - delta/2) on both atoms, granular_model.cpp "torquesi" and "torquesj") and half of the
 * contact's virial. The one rule that depends on the frame: a contact of two atoms of the fix freeze group takes the
 * mass of the higher ID's atom as m_eff (the CPU loop takes the partner of whichever atom its list put first); fix
 * freeze zeroes both atoms' forces, so only that contact's history and virial follow it.
 *
 * The history lives in the CPU pair (PairGranular.historyContacts): each evaluation hands every atom a table of its
 * contacts (the partner's ID, the tangential history and the last contact radius, K entries), and finish() takes the
 * contacts back from the tables of the lower IDs. A table that overflows makes the evaluation run again with a larger K
 * (again()). Atom IDs are kept as f32, so they must be below 2^24.
 *
 * coef per type pair (NC values, index itype * stride + jtype): present, normal, k_n, k (4/3 E_eff), cohesion, damping,
 * eta, damp constant, tangential, k_t, x_gamma,t, mu_t, limit_damping.
 * ext per atom (3 + 2 K vec4): (v, radius), (omega, rmass), (ID, frozen, table length, 0), then per table entry
 * (partner ID, h0, h1, h2) and (h3, 0, 0, 0). outv per atom (16 + 5 K): force, torque, half virial (xx yy zz xy xz yz),
 * the new table length, the contacts the table had no room for, then the new table (partner ID, h0 .. h3).
 */
import { ATOM_KERNEL_WORKGROUP, WGSL_ATOM_KERNEL_HEAD, type GpuAtomKernel } from '../../engine/gpu/atomKernel';
import { NO_IMAGE } from '../../engine/gpu/cellSort';
import { granularContact, newContactOut, type ContactIn, type Pm } from '../../engine/force/pair/granular';
import type { System } from '../../engine/system';
import type { SimState } from '../../engine/types';

/** The per-atom values ahead of the table (ext vec4s, outv values), the vec4s and values of a table entry. */
const EXT_HEAD = 3;
const OUT_HEAD = 16;
const EXT_ENTRY = 2;
const OUT_ENTRY = 5;
/** The first table length; a table grows in steps of 4. */
const K_START = 12;
/** Values per type pair in coef. */
const NC = 16;

const NORMAL: Record<string, number> = { hooke: 0, hertz: 1, 'hertz/material': 2, dmt: 3 };
const DAMP: Record<string, number> = { velocity: 0, mass_velocity: 1, viscoelastic: 2, tsuji: 3, coeff_restitution: 3 };
const TANG: Record<string, number> = {
  linear_nohistory: 0, linear_history: 1, mindlin: 2, 'mindlin/force': 3, mindlin_rescale: 4, 'mindlin_rescale/force': 5,
};

/** The pair the kernel serves (src/engine/force/pair/granular.ts PairGranular). */
export interface GranularPair {
  readonly name: string;
  gpuModels(): { pm: (Pm | null)[]; stride: number; freezeBit: number } | string;
  historyContacts(): { lo: Int32Array; hi: Int32Array; h: Float64Array; n: number };
  setHistoryContacts(lo: ArrayLike<number>, hi: ArrayLike<number>, h: ArrayLike<number>): void;
}

/** The coefficient rows of the type pairs. */
const coefOf = (pm: (Pm | null)[]): Float64Array => {
  const c = new Float64Array(NC * pm.length);
  pm.forEach((p, t) => {
    if (!p) return;
    const o = NC * t;
    c[o] = 1; c[o + 1] = NORMAL[p.normal]; c[o + 2] = p.kn; c[o + 3] = p.kHM; c[o + 4] = p.gamma;
    c[o + 5] = DAMP[p.damp]; c[o + 6] = p.eta; c[o + 7] = p.dampC;
    c[o + 8] = TANG[p.tang]; c[o + 9] = p.kt; c[o + 10] = p.xgt; c[o + 11] = p.mu; c[o + 12] = p.limit ? 1 : 0;
  });
  return c;
};

/** The WGSL of the kernel for a table of K entries (K = 0: no type pair keeps a history). */
const kernelCode = (K: number): string => {
  const EW = EXT_HEAD + EXT_ENTRY * K, OW = OUT_HEAD + OUT_ENTRY * K;
  const history = K > 0;
  return /* wgsl */ `${WGSL_ATOM_KERNEL_HEAD}
const EW : u32 = ${EW}u;
const OW : u32 = ${OW}u;
const KMAX : u32 = ${K}u;
const NC : u32 = ${NC}u;

fn len3(v : vec3<f32>) -> f32 { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

// GranSubMod::rotate_rescale_vec: v without its component along n, rescaled to its previous length
fn rotateRescale(v : vec3<f32>, n : vec3<f32>) -> vec3<f32> {
  let rsht = v.x * n.x + v.y * n.y + v.z * n.z;
  let shrmag = len3(v);
  let p = v - rsht * n;
  let prjmag = len3(p);
  var t = 0.0;
  if (prjmag > 0.0) { t = shrmag / prjmag; }
  return t * p;
}

@compute @workgroup_size(${ATOM_KERNEL_WORKGROUP})
fn force(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pi = pos[i];
  let bi = EW * i;
  let a0 = ext[bi]; let a1 = ext[bi + 1u]; let a2 = ext[bi + 2u];
  let radi = a0.w; let idi = a2.x;
  let nh = u32(a2.z);
  let ti = u32(pi.w);
  let dt = P.u0.x;
  let update = P.u0.y != 0.0;
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
          let cb = NC * (ti * P.stride + u32(pj.w));
          if (coef[cb] == 0.0) { continue; }
          let normal = u32(coef[cb + 1u]);
          let kn = coef[cb + 2u]; let kHM = coef[cb + 3u]; let cohesion = coef[cb + 4u];
          let damp = u32(coef[cb + 5u]); let eta = coef[cb + 6u]; let dampC = coef[cb + 7u];
          let tang = u32(coef[cb + 8u]); let kt = coef[cb + 9u]; let xgt = coef[cb + 10u]; let mu = coef[cb + 11u];
          let limit = coef[cb + 12u] != 0.0;
          let b1 = ext[bj + 1u]; let b2 = ext[bj + 2u];
          let idj = b2.x;
          // the contact in the frame of the lower ID: dx = x_lo - x_hi
          let iLo = idi < idj;
          let dx = select(d, -d, iLo);
          let rlo = select(radj, radi, iLo); let rhi = select(radi, radj, iLo);
          let vr = select(b0.xyz, a0.xyz, iLo) - select(a0.xyz, b0.xyz, iLo);
          let wlo = select(b1.xyz, a1.xyz, iLo); let whi = select(a1.xyz, b1.xyz, iLo);
          let mlo = select(b1.w, a1.w, iLo); let mhi = select(a1.w, b1.w, iLo);
          let flo = select(b2.y, a2.y, iLo) != 0.0; let fhi = select(a2.y, b2.y, iLo) != 0.0;
          let r = sqrt(rsq);
          let rinv = 1.0 / r;
          let n = dx * rinv;
          let delta = radsum - r;
          let Rf = rlo * rhi / radsum;
          var meff = mlo * mhi / (mlo + mhi);
          if (flo) { meff = mhi; } else if (fhi) { meff = mlo; }
          // relative velocities: vn = (vr . n) n, vt = vr - vn - (R_lo w_lo + R_hi w_hi) x n
          let vnn = vr.x * n.x + vr.y * n.y + vr.z * n.z;
          let A = rlo * wlo + rhi * whi;
          let wn = vec3<f32>(A.y * n.z - A.z * n.y, A.z * n.x - A.x * n.z, A.x * n.y - A.y * n.x);
          let vt = vr - vnn * n - wn;
          let vrel = len3(vt);
          // normal force (repulsive positive): elastic part, then the damping prefactor
          var fPull = 0.0;
          if (normal == 3u) { fPull = 4.0 * 3.14159265358979 * cohesion * Rf; }
          let aC = sqrt(delta * Rf);
          var fne = kn * delta;
          if (normal == 1u) { fne = kn * aC * delta; } else if (normal >= 2u) { fne = kHM * aC * delta; }
          if (normal == 3u) { fne = fne - fPull; }
          var etaN = eta;
          if (damp == 1u) { etaN = eta * meff; }
          else if (damp == 2u) { etaN = eta * meff * aC; }
          else if (damp == 3u) {
            var s1 = 0.0;
            if (delta > 0.0) { s1 = max(0.0, meff * fne / delta); }
            etaN = dampC * sqrt(s1);
          }
          // fn is a WGSL keyword: fnTot is the normal force with damping (Fntot)
          var fnTot = fne - etaN * vnn;
          if (limit && fnTot < 0.0) { fnTot = 0.0; }
          var fnAbs = abs(fnTot);
          if (normal == 3u) { fnAbs = abs(fne + 2.0 * fPull); }
          let etaT = xgt * etaN;
          let fcap = fnAbs * mu;
          var ft = vec3<f32>(0.0);
          var h = vec3<f32>(0.0);
          var h3 = 0.0;
          if (tang == 0u) {
            let fsmag = etaT * vrel;
            var Ft = 0.0;
            if (vrel != 0.0) { Ft = min(mu * fnAbs, fsmag) / vrel; }
            ft = -Ft * vt;
          } else {
${history ? `            for (var m = 0u; m < nh; m = m + 1u) {
              let e = ext[bi + ${EXT_HEAD}u + ${EXT_ENTRY}u * m];
              if (e.x == idj) { h = e.yzw; h3 = ext[bi + ${EXT_HEAD + 1}u + ${EXT_ENTRY}u * m].x; break; }
            }
` : ''}            let forceVariant = tang == 3u || tang == 5u;
            let rescale = tang == 4u || tang == 5u;
            var keff = kt * aC;
            if (tang == 1u) { keff = kt; }
            if (rescale && aC < h3) { h = h * (aC / h3); }
            if (update) {
              let rsht = h.x * n.x + h.y * n.y + h.z * n.z;
              var kTest = keff;
              if (forceVariant) { kTest = 1.0; }
              if (abs(rsht) * kTest > 1e-10 * fcap) { h = rotateRescale(h, n); }
              if (forceVariant) { h = h + (-keff * dt) * vt; } else { h = h + dt * vt; }
            }
            if (forceVariant) { ft = h; } else { ft = -keff * h; }
            let dmp = -etaT * vt;
            ft = ft + dmp;
            let magfs = len3(ft);
            if (magfs > fcap) {
              if (len3(h) != 0.0) {
                ft = (fcap / magfs) * ft;
                h = ft - dmp;
                if (!forceVariant) { h = (-1.0 / keff) * h; }
              } else {
                ft = vec3<f32>(0.0);
              }
            }
            if (rescale) {
              if (update) { h3 = aC; }
            } else {
              h3 = aC;
            }
          }
          let F = fnTot * n + ft;
          let cr = vec3<f32>(n.y * ft.z - n.z * ft.y, n.z * ft.x - n.x * ft.z, n.x * ft.y - n.y * ft.x);
          if (iLo) {
            f = f + F;
            tq = tq - cr * (rlo - 0.5 * delta);
          } else {
            f = f - F;
            tq = tq - cr * (rhi - 0.5 * delta);
          }
          vd = vd + 0.5 * dx * F;
          vo = vo + 0.5 * vec3<f32>(dx.x * F.y, dx.x * F.z, dx.y * F.z);
${history ? `          if (tang != 0u) {
            if (nout < KMAX) {
              let o = ob + ${OUT_HEAD}u + ${OUT_ENTRY}u * nout;
              outv[o] = idj; outv[o + 1u] = h.x; outv[o + 2u] = h.y; outv[o + 3u] = h.z; outv[o + 4u] = h3;
              nout = nout + 1u;
            } else {
              over = over + 1u;
            }
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
 * The GPU kernel of pair granular for the run (run/accel.ts through the gplHook 'granular'), or why it cannot run
 * there. The pair is initialized (accelerator() runs init first).
 */
export const gpuGranularKernel = (pair: GranularPair, sys: System): GpuAtomKernel | string => {
  const s0 = sys.state;
  if (!s0.radius || !s0.rmass || !s0.omega || !s0.torque) return 'pair granular without the per-atom radius, mass, omega and torque of atom_style sphere';
  if (s0.rigidBodyMass && s0.rigidBodyMass.size > 0) return 'pair granular with atoms in rigid bodies';
  const models = pair.gpuModels();
  if (typeof models === 'string') return models;
  const { pm, stride, freezeBit } = models;
  const coef = coefOf(pm);
  const history = pm.some((p) => p !== null && p.tang !== 'linear_nohistory');
  const codes = new Map<number, string>();
  let K = history ? K_START : 0;
  let update = false;
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
  const frozen = (st: SimState, i: number): boolean => freezeBit !== 0 && (st.mask[i] & freezeBit) !== 0;

  // the contact tables of this evaluation (atom order): length, partner IDs and the four history values
  let tabN = new Int32Array(0), tabId = new Float64Array(0), tabH = new Float64Array(0);
  let index = new Int32Array(0);
  const buildTables = (st: SimState): void => {
    const n = st.n;
    if (tabN.length < n) tabN = new Int32Array(n);
    tabN.fill(0, 0, n);
    if (!history) return;
    const c = pair.historyContacts();
    // the atom of each ID (IDs below 2^24)
    let maxId = 0;
    for (let i = 0; i < n; i++) if (st.id[i] > maxId) maxId = st.id[i];
    if (index.length < maxId + 1) index = new Int32Array(maxId + 1);
    index.fill(-1, 0, maxId + 1);
    for (let i = 0; i < n; i++) index[st.id[i]] = i;
    const at = (id: number): number => (id <= maxId ? index[id] : -1);
    let longest = 0;
    for (let k = 0; k < c.n; k++) {
      const a = at(c.lo[k]), b = at(c.hi[k]);
      if (a < 0 || b < 0) continue;
      if (++tabN[a] > longest) longest = tabN[a];
      if (++tabN[b] > longest) longest = tabN[b];
    }
    if (longest + 4 > K) K = 4 * Math.ceil((longest + 4) / 4);
    if (tabId.length < K * n) { tabId = new Float64Array(K * n); tabH = new Float64Array(4 * K * n); }
    tabN.fill(0, 0, n);
    const put = (a: number, partner: number, k: number): void => {
      const m = tabN[a]++;
      tabId[K * a + m] = partner;
      const o = 4 * (K * a + m);
      for (let q = 0; q < 4; q++) tabH[o + q] = c.h[4 * k + q];
    };
    for (let k = 0; k < c.n; k++) {
      const a = at(c.lo[k]), b = at(c.hi[k]);
      if (a < 0 || b < 0) continue;
      put(a, c.hi[k], k);
      put(b, c.lo[k], k);
    }
  };

  const outWidth = (): number => OUT_HEAD + OUT_ENTRY * K;

  /**
   * The kernel's arithmetic in fp64 over every pair (the minimum image of cellSort.ts), outWidth values per atom: each
   * contact through the CPU style's granularContact in the frame of the lower ID.
   */
  const fp64 = (st: SimState): Float64Array => {
    const n = st.n, x = st.x, v = st.v, om = st.omega!, rad = st.radius!, id = st.id, rm = st.rmass!, type = st.type;
    const OW = outWidth();
    const out = new Float64Array(OW * n);
    const box = st.box;
    const L = [0, 1, 2].map((d) => (box.periodic[d] || (d === 2 && st.dimension === 2) ? box.hi[d] - box.lo[d] : NO_IMAGE));
    const [xy, xz, yz] = box.triclinic ? box.tilt : [0, 0, 0];
    const two = st.dimension === 2;
    const o = newContactOut();
    const sh = new Float64Array(11);
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
        const p = pm[type[i] * stride + type[j]];
        if (!p) continue;
        const iLo = id[i] < id[j];
        const lo = iLo ? i : j, hi = iLo ? j : i;
        const sg = iLo ? -1 : 1;
        sh.fill(0);
        let found = false;
        for (let m = 0; m < tabN[i]; m++) {
          if (tabId[K * i + m] === id[j]) {
            const q = 4 * (K * i + m);
            for (let c = 0; c < 4; c++) sh[c] = tabH[q + c];
            found = true;
            break;
          }
        }
        const dx = sg * ddx, dy = sg * ddy, dz = sg * ddz;
        const r = Math.sqrt(rsq), rinv = 1 / r;
        let meff = (rm[lo] * rm[hi]) / (rm[lo] + rm[hi]);
        if (frozen(st, lo)) meff = rm[hi];
        else if (frozen(st, hi)) meff = rm[lo];
        const cin: ContactIn = {
          pm: p, nx: dx * rinv, ny: dy * rinv, nz: dz * rinv, r, delta: radsum - r, Rf: (rad[lo] * rad[hi]) / radsum,
          ri: rad[lo], rj: rad[hi], meff,
          vrx: v[3 * lo] - v[3 * hi], vry: v[3 * lo + 1] - v[3 * hi + 1], vrz: v[3 * lo + 2] - v[3 * hi + 2],
          oix: om[3 * lo], oiy: om[3 * lo + 1], oiz: om[3 * lo + 2], ojx: om[3 * hi], ojy: om[3 * hi + 1], ojz: om[3 * hi + 2],
          dt: st.dt, update, rsq, radsum, dx, dy, dz, sg: 1,
        };
        if (!granularContact(cin, sh, o, found)) continue;
        if (iLo) {
          fx += o.fx; fy += o.fy; fz += o.fz;
          tx += o.tix; ty += o.tiy; tz += o.tiz;
        } else {
          fx -= o.fx; fy -= o.fy; fz -= o.fz;
          tx += o.tjx; ty += o.tjy; tz += o.tjz;
        }
        vxx += 0.5 * dx * o.fx; vyy += 0.5 * dy * o.fy; vzz += 0.5 * dz * o.fz;
        vxy += 0.5 * dx * o.fy; vxz += 0.5 * dx * o.fz; vyz += 0.5 * dy * o.fz;
        if (o.hist) {
          if (nout < K) {
            const q = ob + OUT_HEAD + OUT_ENTRY * nout;
            out[q] = id[j];
            for (let c = 0; c < 4; c++) out[q + 1 + c] = sh[c];
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
      if (!c) codes.set(K, (c = kernelCode(K)));
      return c;
    },
    key: `granular:${Array.from(coef).join(',')}`,
    coef,
    stride,
    get maxCutoff(): number { return maxCut; },
    get extWidth(): number { return EXT_HEAD + EXT_ENTRY * K; },
    get outWidth(): number { return outWidth(); },
    begin: (st, step) => {
      // As native LAMMPS does (pair_granular.cpp compute at b332f28): "if (update->setupflag) history_update = 0"
      update = step;
      maxCut = maxContact(st);
      if (tooLargeId(st)) throw new Error('pair granular on the GPU: an atom ID of 2^24 or more (the kernel keeps the contact partners as f32)');
      buildTables(st);
    },
    scalars: (st) => [st.dt, update ? 1 : 0, 0, 0, 0, 0, 0, 0],
    extra: (st, out) => {
      const n = st.n, v = st.v, om = st.omega!, rad = st.radius!, id = st.id, rm = st.rmass!;
      const EW4 = 4 * (EXT_HEAD + EXT_ENTRY * K);
      for (let i = 0; i < n; i++) {
        const b = EW4 * i;
        out[b] = v[3 * i]; out[b + 1] = v[3 * i + 1]; out[b + 2] = v[3 * i + 2]; out[b + 3] = rad[i];
        out[b + 4] = om[3 * i]; out[b + 5] = om[3 * i + 1]; out[b + 6] = om[3 * i + 2]; out[b + 7] = rm[i];
        out[b + 8] = id[i]; out[b + 9] = frozen(st, i) ? 1 : 0; out[b + 10] = tabN[i]; out[b + 11] = 0;
        for (let m = 0; m < tabN[i]; m++) {
          const q = b + 4 * (EXT_HEAD + EXT_ENTRY * m), t = 4 * (K * i + m);
          out[q] = tabId[K * i + m]; out[q + 1] = tabH[t]; out[q + 2] = tabH[t + 1]; out[q + 3] = tabH[t + 2];
          out[q + 4] = tabH[t + 3]; out[q + 5] = 0; out[q + 6] = 0; out[q + 7] = 0;
        }
      }
    },
    again: (st, out) => {
      if (!history) return false;
      const OW = outWidth();
      let over = 0;
      for (let i = 0; i < st.n; i++) over = Math.max(over, out[OW * i + 13]);
      if (over === 0) return false;
      // a table with no room for every contact of its atom: run again with room for them
      K = 4 * Math.ceil((K + over + 4) / 4);
      if (tabId.length < K * st.n) { tabId = new Float64Array(K * st.n); tabH = new Float64Array(4 * K * st.n); }
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
      if (history) {
        // the contacts from the tables of their lower IDs (the higher ID's copy is the same values)
        let nc = 0;
        for (let i = 0; i < n; i++) {
          const m = out[OW * i + 12];
          for (let q = 0; q < m; q++) if (id[i] < out[OW * i + OUT_HEAD + OUT_ENTRY * q]) nc++;
        }
        const lo = new Int32Array(nc), hi = new Int32Array(nc), h = new Float64Array(4 * nc);
        let k = 0;
        for (let i = 0; i < n; i++) {
          const m = out[OW * i + 12];
          for (let q = 0; q < m; q++) {
            const e = OW * i + OUT_HEAD + OUT_ENTRY * q;
            if (!(id[i] < out[e])) continue;
            lo[k] = id[i]; hi[k] = out[e];
            for (let c = 0; c < 4; c++) h[4 * k + c] = out[e + 1 + c];
            k++;
          }
        }
        pair.setHistoryContacts(lo, hi, h);
      }
      return { pe: 0, virial: vt[0] + vt[1] + vt[2], virialTensor: vt, evdwl: 0, ecoul: 0 };
    },
    fp64,
  };
};
