import { WGSL_BOX_IMAGE } from './cellSort';
import { WGSL_STYLE_LAWS } from './styleLaws';

/*
 * pair tersoff in WGSL (gpu/tersoffSpec.ts packs the tables and is the fp64 statement of this arithmetic,
 * tersoffGather; webgpuForces.ts computeTersoff runs the three passes in one submission):
 *   nbr:   each atom's neighbors within the (symmetric) list cutoff of its type pair, from the cells around its own (every
 *          cell once along an edge of fewer than 3 cells, where the minimum image finds the one copy of a partner),
 *          written after the cell data in ints: the count at countBase + i, the slots at listBase + i * maxn (a count
 *          above maxn is kept, and the force pass flags the atom so the backend can run again with longer lists);
 *   bond:  per slot of an atom's list, the bond's b_ij and C_ij = 1/2 f_C f_A db/dzeta (b = -1: outside the pair cutoff);
 *   force: the forces on the atom from its own bonds, its neighbors' bonds toward it and their bonds in whose zeta it is
 *          the third atom; half of each of its own bonds' energy, and their virial:
 *          outv[12 slot ..] = fx fy fz e flag w, then the virial tensor xx yy zz xy xz yz (flag -1: a neighbor list was cut
 *          short).
 */
export const TERSOFF_WORKGROUP = 64;

export const KERNEL_TERSOFF = /* wgsl */ `
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
  shift : f32,
  pad0 : u32,
  tilt : vec4<f32>,  // xy xz yz: the minimum image of a triclinic box (cellSort.ts WGSL_BOX_IMAGE)
};

${WGSL_BOX_IMAGE}

@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> ints : array<u32>;
@group(0) @binding(3) var<storage, read> coef : array<f32>;
@group(0) @binding(4) var<storage, read_write> bondv : array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> outv : array<f32>;

${WGSL_STYLE_LAWS}

fn wrapCell(c : i32, n : u32) -> u32 {
  let m = i32(n);
  return u32(((c % m) + m) % m);
}

fn minImage(a : vec3<f32>, b : vec3<f32>) -> vec3<f32> {
  var d = b - a;
  d = boxImage(d);
  if (P.two == 1u) { d.z = 0.0; }
  return d;
}

fn pairBase(ti : u32, tj : u32) -> u32 {
  return 12u * (ti * P.stride + tj);
}

fn tripBaseOf(ti : u32, tj : u32, tk : u32) -> u32 {
  return P.tripBase + 12u * ((ti * P.stride + tj) * P.stride + tk);
}

// f_C(r) and d f_C/dr with cutoff R, D (sinCos: WGSL's sin and cos may err by 2^-11)
fn cutoffFn(r : f32, R : f32, D : f32) -> vec2<f32> {
  if (r < R - D) { return vec2<f32>(1.0, 0.0); }
  if (r > R + D) { return vec2<f32>(0.0, 0.0); }
  let sc = sinCos(1.5707963267948966 * (r - R) / D);
  return vec2<f32>(0.5 - 0.5 * sc.x, -(3.141592653589793 / (4.0 * D)) * sc.y);
}

struct ZT { H : f32, drij : f32, drik : f32, ddc : f32, e2 : vec3<f32>, d2 : vec3<f32>, r2 : f32, ok : bool };

// the term of atom k in zeta of the bond ctr -> j (unit vector e1, length r1) and its partial derivatives
fn zterm(ctr : u32, tc : u32, tj : u32, k : u32, e1 : vec3<f32>, r1 : f32) -> ZT {
  var z : ZT;
  let pk = pos[k];
  let o = tripBaseOf(tc, tj, u32(pk.w));
  let d2 = minImage(pos[ctr].xyz, pk.xyz);
  let r2 = sqrt(dot(d2, d2));
  if (!(r2 < coef[o]) || r2 == 0.0) { return z; }
  let e2 = d2 / r2;
  let fc = cutoffFn(r2 + P.shift, coef[o + 1u], coef[o + 2u]);
  let cth = dot(e1, e2);
  let q = cth - coef[o + 6u];
  let c2 = coef[o + 4u];
  let dd = coef[o + 5u];
  let den = dd + q * q;
  let gam = coef[o + 3u];
  // g = gamma (1 + c^2/d^2 - c^2/(d^2 + q^2)) without the difference of two large numbers (tersoffSpec.ts)
  let g = gam * (1.0 + c2 * q * q / (dd * den));
  let dgdc = 2.0 * gam * c2 * q / (den * den);
  let dr = r1 - r2;
  let lp = coef[o + 7u];
  var efil = 1.0;
  var dphi = 0.0;
  if (lp != 0.0) {
    // the exponent capped as native does (tersoffSpec.ts)
    if (coef[o + 8u] == 3.0) {
      efil = exp(min(lp * dr * dr * dr, 69.0776));
      dphi = 3.0 * lp * dr * dr;
    } else {
      efil = exp(min(lp * dr, 69.0776));
      dphi = lp;
    }
  }
  let H = fc.x * g * efil;
  z.H = H;
  z.drij = H * dphi;
  z.drik = g * efil * fc.y - H * dphi;
  z.ddc = fc.x * efil * dgdc;
  z.e2 = e2;
  z.d2 = d2;
  z.r2 = r2;
  z.ok = true;
  return z;
}

// the force on the bond's far atom j (first) and on the third atom k of a zeta term with prefactor C
fn zforceJ(C : f32, e1 : vec3<f32>, r1 : f32, z : ZT) -> vec3<f32> {
  let cth = dot(e1, z.e2);
  return -C * (z.drij * e1 + z.ddc * (z.e2 - cth * e1) / r1);
}

fn zforceK(C : f32, e1 : vec3<f32>, z : ZT) -> vec3<f32> {
  let cth = dot(e1, z.e2);
  return -C * (z.drik * z.e2 + z.ddc * (e1 - cth * z.e2) / z.r2);
}

// V_ij and dV_ij/dr of the type pair at distance r for a fixed b
fn radial(ti : u32, tj : u32, r : f32, b : f32) -> vec2<f32> {
  let o = pairBase(ti, tj);
  let rs = r + P.shift;
  let fc = cutoffFn(rs, coef[o + 2u], coef[o + 3u]);
  let fr = coef[o + 4u] * exp(-coef[o + 5u] * rs);
  let fa = -coef[o + 6u] * exp(-coef[o + 7u] * rs);
  return vec2<f32>(fc.x * (fr + b * fa), fc.y * (fr + b * fa) + fc.x * (-coef[o + 5u] * fr - b * coef[o + 7u] * fa));
}

@compute @workgroup_size(${TERSOFF_WORKGROUP})
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
          let r2 = dot(d, d);
          if (!(r2 > 0.0) || r2 >= coef[pairBase(ti, u32(pj.w))]) { continue; }
          if (count < P.maxn) { ints[P.listBase + i * P.maxn + count] = j; }
          count = count + 1u;
        }
      }
    }
  }
  ints[P.countBase + i] = count;
}

@compute @workgroup_size(${TERSOFF_WORKGROUP})
fn bond(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pi = pos[i];
  let ti = u32(pi.w);
  let cnt = min(ints[P.countBase + i], P.maxn);
  for (var a = 0u; a < cnt; a = a + 1u) {
    let j = ints[P.listBase + i * P.maxn + a];
    let tj = u32(pos[j].w);
    let o = pairBase(ti, tj);
    let d1 = minImage(pi.xyz, pos[j].xyz);
    let r1 = sqrt(dot(d1, d1));
    var out = vec2<f32>(-1.0, 0.0);
    if (r1 < coef[o + 1u]) {
      let e1 = d1 / r1;
      var zeta = 0.0;
      for (var b = 0u; b < cnt; b = b + 1u) {
        if (b == a) { continue; }
        let z = zterm(i, ti, tj, ints[P.listBase + i * P.maxn + b], e1, r1);
        if (z.ok) { zeta = zeta + z.H; }
      }
      var bo = 1.0;
      var dbdz = 0.0;
      if (zeta > 0.0) {
        // from u = ln((beta zeta)^n), without (beta zeta)^n itself, which overflows fp32 (tersoffSpec.ts)
        let nn = coef[o + 9u];
        let u = coef[o + 8u] + nn * log(zeta);
        var sfac = 0.0;
        if (u <= 0.0) {
          let t = exp(u);
          bo = pow(1.0 + t, -1.0 / (2.0 * nn));
          sfac = t / (1.0 + t);
        } else {
          let t = exp(-u);
          bo = exp(-u / (2.0 * nn)) * pow(1.0 + t, -1.0 / (2.0 * nn));
          sfac = 1.0 / (1.0 + t);
        }
        dbdz = -0.5 * bo * sfac / zeta;
      }
      let rs = r1 + P.shift;
      let fa = -coef[o + 6u] * exp(-coef[o + 7u] * rs);
      out = vec2<f32>(bo, 0.5 * cutoffFn(rs, coef[o + 2u], coef[o + 3u]).x * fa * dbdz);
    }
    bondv[i * P.maxn + a] = out;
  }
}

@compute @workgroup_size(${TERSOFF_WORKGROUP})
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
  var vd = vec3<f32>(0.0, 0.0, 0.0); // the virial tensor xx yy zz, then xy xz yz
  var vo = vec3<f32>(0.0, 0.0, 0.0);
  // the bonds of i: the radial term, and the zeta terms it centers
  for (var a = 0u; a < cnt; a = a + 1u) {
    let bo = bondv[i * P.maxn + a];
    if (bo.x < 0.0) { continue; }
    let j = ints[P.listBase + i * P.maxn + a];
    let tj = u32(pos[j].w);
    let d1 = minImage(pi.xyz, pos[j].xyz);
    let r1 = sqrt(dot(d1, d1));
    let e1 = d1 / r1;
    let rad = radial(ti, tj, r1, bo.x);
    f = f + 0.5 * rad.y * e1;
    ev = ev + 0.5 * rad.x;
    w = w - 0.5 * rad.y * r1;
    // d1 (x) F_j with F_j = -1/2 dV/dr e1
    vd = vd - (0.5 * rad.y / r1) * d1 * d1;
    vo = vo - (0.5 * rad.y / r1) * vec3<f32>(d1.x * d1.y, d1.x * d1.z, d1.y * d1.z);
    if (bo.y == 0.0) { continue; }
    for (var b = 0u; b < cnt; b = b + 1u) {
      if (b == a) { continue; }
      let z = zterm(i, ti, tj, ints[P.listBase + i * P.maxn + b], e1, r1);
      if (!z.ok) { continue; }
      let fj = zforceJ(bo.y, e1, r1, z);
      let fk = zforceK(bo.y, e1, z);
      f = f - (fj + fk);
      w = w + dot(d1, fj) + dot(z.d2, fk);
      vd = vd + d1 * fj + z.d2 * fk;
      vo = vo + vec3<f32>(d1.x * fj.y + z.d2.x * fk.y, d1.x * fj.z + z.d2.x * fk.z, d1.y * fj.z + z.d2.y * fk.z);
    }
  }
  // the bonds of each neighbor j: toward i (its radial term and the zeta terms of j -> i), and to another atom l (the
  // zeta term in which i is the third atom)
  for (var a = 0u; a < cnt; a = a + 1u) {
    let j = ints[P.listBase + i * P.maxn + a];
    let pj = pos[j];
    let tj = u32(pj.w);
    let fullj = ints[P.countBase + j];
    if (fullj > P.maxn) { bad = true; }
    let cntj = min(fullj, P.maxn);
    for (var c = 0u; c < cntj; c = c + 1u) {
      let bo = bondv[j * P.maxn + c];
      if (bo.x < 0.0) { continue; }
      let l = ints[P.listBase + j * P.maxn + c];
      if (l != i && bo.y == 0.0) { continue; }
      let tl = u32(pos[l].w);
      let dl = minImage(pj.xyz, pos[l].xyz);
      let rl = sqrt(dot(dl, dl));
      let el = dl / rl;
      if (l == i) {
        f = f - 0.5 * radial(tj, ti, rl, bo.x).y * el;
        if (bo.y == 0.0) { continue; }
        for (var c2 = 0u; c2 < cntj; c2 = c2 + 1u) {
          if (c2 == c) { continue; }
          let z = zterm(j, tj, ti, ints[P.listBase + j * P.maxn + c2], el, rl);
          if (!z.ok) { continue; }
          f = f + zforceJ(bo.y, el, rl, z);
        }
      } else {
        let z = zterm(j, tj, tl, i, el, rl);
        if (z.ok) { f = f + zforceK(bo.y, el, z); }
      }
    }
  }
  let o = 12u * i;
  outv[o] = f.x;
  outv[o + 1u] = f.y;
  outv[o + 2u] = f.z;
  outv[o + 3u] = ev;
  outv[o + 4u] = select(0.0, -1.0, bad);
  outv[o + 5u] = w;
  outv[o + 6u] = vd.x;
  outv[o + 7u] = vd.y;
  outv[o + 8u] = vd.z;
  outv[o + 9u] = vo.x;
  outv[o + 10u] = vo.y;
  outv[o + 11u] = vo.z;
}
`;
