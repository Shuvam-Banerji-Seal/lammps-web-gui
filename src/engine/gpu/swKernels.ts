import { WGSL_STYLE_LAWS } from './styleLaws';

/*
 * pair sw and sw/mod in WGSL (gpu/swSpec.ts packs the parameters and is the fp64 statement of this arithmetic, swGather;
 * webgpuForces.ts computeSW runs the two passes in one submission):
 *   nbr:   each atom's neighbors within its type pair's cutoff, from the cells around its own (every cell once along an
 *          edge of fewer than 3 cells, where the minimum image finds the one copy of a partner), written after the cell
 *          data in ints: the count at countBase + i, the slots at listBase + i * maxn (a count above maxn is kept, and
 *          the force pass flags the atom so the backend can run again with longer lists);
 *   force: the forces on the atom from its pairs, from the triplets it centers and from those centered on a neighbor
 *          in which it is a leg; half of each pair's energy and virial, and a triplet's energy and virial at its center:
 *          outv[6 slot ..] = fx fy fz e flag w (flag -1: a neighbor list was cut short).
 */
export const SW_WORKGROUP = 64;

export const KERNEL_SW = /* wgsl */ `
struct Params {
  L : vec3<f32>,
  n : u32,
  nc : vec3<u32>,
  stride : u32,
  two : u32,
  cellOfBase : u32,
  maxn : u32,
  countBase : u32,
  listBase : u32,
  tripBase : u32,
  three : u32,
  swmod : u32,     // sw/mod (WGSL reserves the word mod)
  delta1 : f32,
  delta2 : f32,
  pad0 : f32,
  pad1 : f32,
};

@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> ints : array<u32>;
@group(0) @binding(3) var<storage, read> coef : array<f32>;
@group(0) @binding(4) var<storage, read_write> outv : array<f32>;

${WGSL_STYLE_LAWS}

fn wrapCell(c : i32, n : u32) -> u32 {
  let m = i32(n);
  return u32(((c % m) + m) % m);
}

fn minImage(a : vec3<f32>, b : vec3<f32>) -> vec3<f32> {
  var d = b - a;
  d = d - P.L * round(d / P.L);
  if (P.two == 1u) { d.z = 0.0; }
  return d;
}

fn pairBase(ti : u32, tj : u32) -> u32 {
  return 12u * (ti * P.stride + tj);
}

@compute @workgroup_size(${SW_WORKGROUP})
fn nbr(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pi = pos[i];
  let ti = u32(pi.w);
  let nc = P.nc;
  let c = ints[P.cellOfBase + i];
  let cx = i32(c % nc.x);
  let cy = i32((c / nc.x) % nc.y);
  let cz = i32(c / (nc.x * nc.y));
  let lo = vec3<i32>(select(-1, 0, nc.x < 3u), select(-1, 0, nc.y < 3u), select(-1, 0, nc.z < 3u));
  let hi = vec3<i32>(select(1, i32(nc.x) - 1, nc.x < 3u), select(1, i32(nc.y) - 1, nc.y < 3u), select(1, i32(nc.z) - 1, nc.z < 3u));
  var count = 0u;
  for (var dz = lo.z; dz <= hi.z; dz = dz + 1) {
    for (var dy = lo.y; dy <= hi.y; dy = dy + 1) {
      for (var dx = lo.x; dx <= hi.x; dx = dx + 1) {
        let c2 = (wrapCell(cz + dz, nc.z) * nc.y + wrapCell(cy + dy, nc.y)) * nc.x + wrapCell(cx + dx, nc.x);
        let jEnd = ints[c2 + 1u];
        for (var j = ints[c2]; j < jEnd; j = j + 1u) {
          if (j == i) { continue; }
          let pj = pos[j];
          let d = minImage(pi.xyz, pj.xyz);
          if (dot(d, d) >= coef[pairBase(ti, u32(pj.w))]) { continue; }
          if (count < P.maxn) { ints[P.listBase + i * P.maxn + count] = j; }
          count = count + 1u;
        }
      }
    }
  }
  ints[P.countBase + i] = count;
}

struct Leg { d : vec3<f32>, r : f32, es : f32, w : f32, ok : bool };

// the leg center -> j of a triplet: inside the leg cutoff of the type pair, its screening factor and d(ln es)/dr
fn legOf(ctr : u32, j : u32) -> Leg {
  let o = pairBase(u32(pos[ctr].w), u32(pos[j].w));
  let d = minImage(pos[ctr].xyz, pos[j].xyz);
  let r = sqrt(dot(d, d));
  let lc = coef[o + 2u];
  if (!(r < lc)) { return Leg(d, r, 0.0, 0.0, false); }
  let den = r - lc;
  let gs = coef[o + 3u];
  return Leg(d, r, exp(gs / den), gs / (den * den), true);
}

struct Trip { f1 : vec3<f32>, f2 : vec3<f32>, e3 : f32, w : f32, ok : bool };

// the triplet centered at ctr with legs to j (first) and k: the forces on j and k, its energy and virial
fn tripletOf(ctr : u32, j : u32, k : u32) -> Trip {
  let none = Trip(vec3<f32>(0.0), vec3<f32>(0.0), 0.0, 0.0, false);
  let l1 = legOf(ctr, j);
  if (!l1.ok) { return none; }
  let l2 = legOf(ctr, k);
  if (!l2.ok) { return none; }
  let t3 = (u32(pos[ctr].w) * P.stride + u32(pos[j].w)) * P.stride + u32(pos[k].w);
  let lam = coef[P.tripBase + 2u * t3];
  if (lam == 0.0) { return none; }
  let cs = dot(l1.d, l2.d) / (l1.r * l2.r);
  let dc = cs - coef[P.tripBase + 2u * t3 + 1u];
  // sw/mod's switching factor f_C(delta) and its derivative (sinCos: WGSL's sin and cos may err by 2^-11)
  var fc = 1.0;
  var fcd = 0.0;
  if (P.swmod == 1u) {
    let ad = abs(dc);
    if (ad >= P.delta2) { fc = 0.0; } else if (ad > P.delta1) {
      let wd = P.delta2 - P.delta1;
      let sc = sinCos(3.141592653589793 * (ad - P.delta1) / wd);
      fc = 0.5 + 0.5 * sc.y;
      let der = -(3.141592653589793 / (2.0 * wd)) * sc.x;
      fcd = select(der, -der, dc < 0.0);
    }
  }
  let zeta = fc * dc;
  let aE = lam * zeta * zeta;
  let aD = lam * 2.0 * zeta * (fc + dc * fcd);
  let esPair = l1.es * l2.es;
  let e3 = aE * esPair;
  let dEdc = aD * esPair;
  let bk = dEdc / (l1.r * l2.r);
  let a1 = dEdc * cs / (l1.r * l1.r) + e3 * l1.w / l1.r;
  let a2 = dEdc * cs / (l2.r * l2.r) + e3 * l2.w / l2.r;
  let f1 = a1 * l1.d - bk * l2.d;
  let f2 = a2 * l2.d - bk * l1.d;
  return Trip(f1, f2, e3, dot(f1, l1.d) + dot(f2, l2.d), true);
}

@compute @workgroup_size(${SW_WORKGROUP})
fn force(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pi = pos[i];
  let ti = u32(pi.w);
  let full = ints[P.countBase + i];
  let cnt = min(full, P.maxn);
  var bad = full > P.maxn;
  var f = vec3<f32>(0.0, 0.0, 0.0);
  var ev = 0.0;
  var w = 0.0;
  for (var a = 0u; a < cnt; a = a + 1u) {
    let j = ints[P.listBase + i * P.maxn + a];
    let o = pairBase(ti, u32(pos[j].w));
    let d = minImage(pi.xyz, pos[j].xyz);
    let r2 = dot(d, d);
    let r = sqrt(r2);
    let den = r - coef[o + 9u];
    // inside the cutoff by r^2 and by r: in fp32, sqrt(r^2) can land an ulp past a sigma while r^2 is still below the
    // rounded cutoff squared, and exp(sigma / den) of a tiny positive den is infinite (a run went to inf at step 45)
    if (r2 < coef[o + 1u] && den < 0.0) {
      // phi2 = A eps (B u^p - u^q) exp(sigma / (r - a sigma)), u = sigma / r
      let sig = coef[o + 8u];
      let u = sig / r;
      let es = exp(sig / den);
      let B = coef[o + 5u];
      let p = coef[o + 6u];
      let q = coef[o + 7u];
      let f2 = B * pow(u, p) - pow(u, q);
      var dfdu = 0.0;
      if (p > 0.0) { dfdu = dfdu + B * p * pow(u, p - 1.0); }
      if (q > 0.0) { dfdu = dfdu - q * pow(u, q - 1.0); }
      let dphi = coef[o + 4u] * es * (-dfdu * sig / r2 - f2 * sig / (den * den));
      let fpair = -dphi / r;
      f = f - fpair * d;
      ev = ev + 0.5 * coef[o + 4u] * f2 * es;
      w = w + 0.5 * fpair * r2;
    }
  }
  if (P.three == 1u) {
    // the triplets centered at i, each unordered pair of neighbors once
    for (var a = 0u; a < cnt; a = a + 1u) {
      let j = ints[P.listBase + i * P.maxn + a];
      for (var b = a + 1u; b < cnt; b = b + 1u) {
        let k = ints[P.listBase + i * P.maxn + b];
        let t = tripletOf(i, j, k);
        if (!t.ok) { continue; }
        f = f - (t.f1 + t.f2);
        ev = ev + t.e3;
        w = w + t.w;
      }
    }
    // the triplets centered at a neighbor j in which i is a leg
    for (var a = 0u; a < cnt; a = a + 1u) {
      let j = ints[P.listBase + i * P.maxn + a];
      let fullj = ints[P.countBase + j];
      if (fullj > P.maxn) { bad = true; }
      let cntj = min(fullj, P.maxn);
      for (var b = 0u; b < cntj; b = b + 1u) {
        let k = ints[P.listBase + j * P.maxn + b];
        if (k == i) { continue; }
        let t = tripletOf(j, i, k);
        if (!t.ok) { continue; }
        f = f + t.f1;
      }
    }
  }
  let o = 6u * i;
  outv[o] = f.x;
  outv[o + 1u] = f.y;
  outv[o + 2u] = f.z;
  outv[o + 3u] = ev;
  outv[o + 4u] = select(0.0, -1.0, bad);
  outv[o + 5u] = w;
}
`;
