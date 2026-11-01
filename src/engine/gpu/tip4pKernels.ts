import { WGSL_BOX_IMAGE } from './cellSort';
/*
 * The TIP4P pair styles in WGSL (gpu/tip4pSpec.ts packs the coefficients and is the fp64 statement of this arithmetic,
 * tip4pGather; webgpuForces.ts computeTip4p builds the cells, charges, special list and site offsets and spreads the
 * site forces). One invocation per atom (cell-sorted slot) scans the cells around its own (every cell once along an edge
 * of fewer than 3 cells, where the minimum image finds the one copy of a partner) and writes
 *   outv[10 slot ..] = fA.xyz (Lennard-Jones force on the atom), fS.xyz (Coulomb force on its charge site), evdwl,
 *   ecoul and the virial, then the virial tensor xx yy zz xy xz yz (outv[16 slot + 10 ..]), each pair's energies and
 *   virial halved at both of its atoms.
 * site[slot].xyz is the site offset (M - x for an oxygen, 0 otherwise).
 */
export const TIP4P_WORKGROUP = 64;

export const KERNEL_TIP4P = /* wgsl */ `
struct Params {
  L : vec3<f32>,
  n : u32,
  nc : vec3<u32>,
  stride : u32,
  two : u32,
  cellOfBase : u32,
  qBase : u32,
  specBase : u32,   // special bonds: each slot's first entry in ints at specBase + slot (0: none)
  qqrd2e : f32,
  gewald : f32,     // G-ewald of the long styles (0: the bare Coulomb term)
  cutCsq : f32,
  cutCsqPlus : f32,
  hasLJ : u32,
  pad0 : u32,
  pad1 : u32,
  pad2 : u32,
  wlj : vec4<f32>,  // special_bonds lj weights by order (1-2, 1-3, 1-4 at 1..3)
  wc : vec4<f32>,   // and coul weights
  tilt : vec4<f32>,  // xy xz yz: the minimum image of a triclinic box (cellSort.ts WGSL_BOX_IMAGE)
};

${WGSL_BOX_IMAGE}

@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> ints : array<u32>;
@group(0) @binding(3) var<storage, read> coef : array<f32>;
@group(0) @binding(4) var<storage, read> site : array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> outv : array<f32>;

fn wrapCell(c : i32, n : u32) -> u32 {
  let m = i32(n);
  return u32(((c % m) + m) % m);
}

@compute @workgroup_size(${TIP4P_WORKGROUP})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pi = pos[i];
  let ti = u32(pi.w);
  let qi = bitcast<f32>(ints[P.qBase + i]);
  let si = site[i].xyz;
  let nc = P.nc;
  let c = ints[P.cellOfBase + i];
  let cx = i32(c % nc.x);
  let cy = i32((c / nc.x) % nc.y);
  let cz = i32(c / (nc.x * nc.y));
  let lo = vec3<i32>(select(-1, 0, nc.x < 3u), select(-1, 0, nc.y < 3u), select(-1, 0, nc.z < 3u));
  let hi = vec3<i32>(select(1, i32(nc.x) - 1, nc.x < 3u), select(1, i32(nc.y) - 1, nc.y < 3u), select(1, i32(nc.z) - 1, nc.z < 3u));
  var fA = vec3<f32>(0.0, 0.0, 0.0);
  var fS = vec3<f32>(0.0, 0.0, 0.0);
  var ev = 0.0;
  var ec = 0.0;
  var w = 0.0;
  var vd = vec3<f32>(0.0, 0.0, 0.0); // the virial tensor xx yy zz, then xy xz yz (half of each pair's at each atom)
  var vo = vec3<f32>(0.0, 0.0, 0.0);
  for (var dz = lo.z; dz <= hi.z; dz = dz + 1) {
    for (var dy = lo.y; dy <= hi.y; dy = dy + 1) {
      for (var dx = lo.x; dx <= hi.x; dx = dx + 1) {
        let c2 = (wrapCell(cz + dz, nc.z) * nc.y + wrapCell(cy + dy, nc.y)) * nc.x + wrapCell(cx + dx, nc.x);
        let jEnd = ints[c2 + 1u];
        for (var j = ints[c2]; j < jEnd; j = j + 1u) {
          if (j == i) { continue; }
          let pj = pos[j];
          var d = pi.xyz - pj.xyz;
          d = boxImage(d);
          if (P.two == 1u) { d.z = 0.0; }
          let r2 = dot(d, d);
          let k = 8u * (ti * P.stride + u32(pj.w));
          let cutLJ = select(0.0, coef[k], P.hasLJ == 1u);
          if (r2 >= max(cutLJ, P.cutCsqPlus)) { continue; }
          // a special pair: its weights, and no pair at all when both are 0 without kspace (with it the pair stays, for
          // the Coulomb correction below)
          var flj = 1.0;
          var fc = 1.0;
          if (P.specBase > 0u) {
            let t1 = ints[P.specBase + i + 1u];
            for (var t = ints[P.specBase + i]; t < t1; t = t + 1u) {
              let e = ints[t];
              if ((e >> 2u) == j) {
                flj = P.wlj[e & 3u];
                fc = P.wc[e & 3u];
                break;
              }
            }
            if (flj == 0.0 && fc == 0.0 && P.gewald == 0.0) { continue; }
          }
          if (r2 < cutLJ) {
            let r2inv = 1.0 / r2;
            let r6inv = r2inv * r2inv * r2inv;
            let fpair = flj * r6inv * (coef[k + 1u] * r6inv - coef[k + 2u]) * r2inv;
            fA = fA + fpair * d;
            ev = ev + 0.5 * flj * (r6inv * (coef[k + 3u] * r6inv - coef[k + 4u]) - coef[k + 5u]);
            w = w + 0.5 * fpair * r2;
            vd = vd + (0.5 * fpair) * d * d;
            vo = vo + (0.5 * fpair) * vec3<f32>(d.x * d.y, d.x * d.z, d.y * d.z);
          }
          let qj = bitcast<f32>(ints[P.qBase + j]);
          if (qi == 0.0 || qj == 0.0 || r2 >= P.cutCsqPlus) { continue; }
          // the sites' separation: the atoms' minimum image plus the offsets
          var e = d + si - site[j].xyz;
          if (P.two == 1u) { e.z = 0.0; }
          let rc = dot(e, e);
          if (rc >= P.cutCsq) { continue; }
          let r = sqrt(rc);
          let pre = P.qqrd2e * qi * qj / r;
          var cf = 0.0;
          if (P.gewald > 0.0) {
            // the Ewald real-space term, erfc by the polynomial fit (erfc.ts erfcPoly), less (1 - w) of the bare term
            // for a special pair
            let gr = P.gewald * r;
            let ex = exp(-gr * gr);
            let t = 1.0 / (1.0 + 0.3275911 * gr);
            let erfc = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * ex;
            cf = pre * (erfc + 1.12837917 * gr * ex - (1.0 - fc)) / rc;
            ec = ec + 0.5 * pre * (erfc - (1.0 - fc));
          } else {
            cf = fc * pre / rc;
            ec = ec + 0.5 * fc * pre;
          }
          fS = fS + cf * e;
          w = w + 0.5 * cf * rc;
          vd = vd + (0.5 * cf) * e * e;
          vo = vo + (0.5 * cf) * vec3<f32>(e.x * e.y, e.x * e.z, e.y * e.z);
        }
      }
    }
  }
  let o = 16u * i;
  outv[o] = fA.x;
  outv[o + 1u] = fA.y;
  outv[o + 2u] = fA.z;
  outv[o + 3u] = fS.x;
  outv[o + 4u] = fS.y;
  outv[o + 5u] = fS.z;
  outv[o + 6u] = ev;
  outv[o + 7u] = ec;
  outv[o + 8u] = w;
  outv[o + 9u] = 0.0;
  outv[o + 10u] = vd.x;
  outv[o + 11u] = vd.y;
  outv[o + 12u] = vd.z;
  outv[o + 13u] = vo.x;
  outv[o + 14u] = vo.y;
  outv[o + 15u] = vo.z;
}
`;
