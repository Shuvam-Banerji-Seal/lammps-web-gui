/*
 * pair eam, eam/alloy and eam/fs in WGSL (gpu/eamSpec.ts packs the tables; webgpuForces.ts computeEAM runs the step).
 * One invocation per atom slot scans the cells around its own (every cell once along an edge of fewer than 3 cells,
 * where the minimum image finds the one copy of a partner inside the cutoff) and accumulates its own sums, so no
 * atomics are needed:
 *   rho:   the density of the slot's atom, outv[slot];
 *   force: with F' of every atom in ints at fpBase (f32 bits, written by the CPU after its embedding step), the force,
 *          half of each pair's energy and half of its virial, outv[6 slot .. 6 slot + 5] = fx fy fz e 0 w.
 * The table arithmetic is eamSpec.ts eamDensity and eamPair (eamKernel.ts, as native LAMMPS's pair_eam.cpp).
 */
export const EAM_WORKGROUP = 64;

export const KERNEL_EAM = /* wgsl */ `
struct Params {
  L : vec3<f32>,
  n : u32,
  nc : vec3<u32>,
  stride : u32,
  two : u32,
  cellOfBase : u32,
  fpBase : u32,
  nr : u32,
  rdr : f32,
  pad0 : u32,
  pad1 : u32,
  pad2 : u32,
};

@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> ints : array<u32>;
@group(0) @binding(3) var<storage, read> coef : array<f32>;
@group(0) @binding(4) var<storage, read_write> outv : array<f32>;

fn wrapCell(c : i32, n : u32) -> u32 {
  let m = i32(n);
  return u32(((c % m) + m) % m);
}

struct Grid { o : u32, p : f32 };

// the table interval and fraction of distance r: p = r rdr + 1, m <= nr - 1, p - m <= 1 (eamKernel.ts)
fn gridOf(r : f32) -> Grid {
  var p = r * P.rdr + 1.0;
  var m = u32(p);
  if (m > P.nr - 1u) { m = P.nr - 1u; }
  p = p - f32(m);
  if (p > 1.0) { p = 1.0; }
  return Grid(7u * (m - 1u), p);
}

fn value(k : u32, p : f32) -> f32 {
  return ((coef[k + 3u] * p + coef[k + 4u]) * p + coef[k + 5u]) * p + coef[k + 6u];
}

fn slope(k : u32, p : f32) -> f32 {
  return (coef[k] * p + coef[k + 1u]) * p + coef[k + 2u];
}

fn cellOf(i : u32) -> vec3<i32> {
  let c = ints[P.cellOfBase + i];
  let nc = P.nc;
  return vec3<i32>(i32(c % nc.x), i32((c / nc.x) % nc.y), i32(c / (nc.x * nc.y)));
}

fn scanLo() -> vec3<i32> {
  let nc = P.nc;
  return vec3<i32>(select(-1, 0, nc.x < 3u), select(-1, 0, nc.y < 3u), select(-1, 0, nc.z < 3u));
}

fn scanHi() -> vec3<i32> {
  let nc = P.nc;
  return vec3<i32>(select(1, i32(nc.x) - 1, nc.x < 3u), select(1, i32(nc.y) - 1, nc.y < 3u), select(1, i32(nc.z) - 1, nc.z < 3u));
}

@compute @workgroup_size(${EAM_WORKGROUP})
fn rho(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pi = pos[i];
  let ti = u32(pi.w);
  let L = P.L;
  let nc = P.nc;
  let c = cellOf(i);
  let lo = scanLo();
  let hi = scanHi();
  var rhoSum = 0.0;
  for (var dz = lo.z; dz <= hi.z; dz = dz + 1) {
    for (var dy = lo.y; dy <= hi.y; dy = dy + 1) {
      for (var dx = lo.x; dx <= hi.x; dx = dx + 1) {
        let c2 = (wrapCell(c.z + dz, nc.z) * nc.y + wrapCell(c.y + dy, nc.y)) * nc.x + wrapCell(c.x + dx, nc.x);
        let jEnd = ints[c2 + 1u];
        for (var j = ints[c2]; j < jEnd; j = j + 1u) {
          if (j == i) { continue; }
          let pj = pos[j];
          var d = pi.xyz - pj.xyz;
          d = d - L * round(d / L);
          if (P.two == 1u) { d.z = 0.0; }
          let r2 = dot(d, d);
          let h = 4u * (ti * P.stride + u32(pj.w));
          if (r2 >= coef[h + 2u]) { continue; }
          let g = gridOf(sqrt(r2));
          rhoSum = rhoSum + value(u32(coef[h]) + g.o, g.p);
        }
      }
    }
  }
  outv[i] = rhoSum;
}

@compute @workgroup_size(${EAM_WORKGROUP})
fn force(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pi = pos[i];
  let ti = u32(pi.w);
  let fpi = bitcast<f32>(ints[P.fpBase + i]);
  let L = P.L;
  let nc = P.nc;
  let c = cellOf(i);
  let lo = scanLo();
  let hi = scanHi();
  var f = vec3<f32>(0.0, 0.0, 0.0);
  var ev = 0.0;
  var w = 0.0;
  for (var dz = lo.z; dz <= hi.z; dz = dz + 1) {
    for (var dy = lo.y; dy <= hi.y; dy = dy + 1) {
      for (var dx = lo.x; dx <= hi.x; dx = dx + 1) {
        let c2 = (wrapCell(c.z + dz, nc.z) * nc.y + wrapCell(c.y + dy, nc.y)) * nc.x + wrapCell(c.x + dx, nc.x);
        let jEnd = ints[c2 + 1u];
        for (var j = ints[c2]; j < jEnd; j = j + 1u) {
          if (j == i) { continue; }
          let pj = pos[j];
          var d = pi.xyz - pj.xyz;
          d = d - L * round(d / L);
          if (P.two == 1u) { d.z = 0.0; }
          let r2 = dot(d, d);
          let tj = u32(pj.w);
          let h = 4u * (ti * P.stride + tj);
          if (r2 >= coef[h + 2u]) { continue; }
          let hji = 4u * (tj * P.stride + ti);
          let r = sqrt(r2);
          let g = gridOf(r);
          // rhojp: the density at i due to j (table ij); rhoip: at j due to i (table ji)
          let rhojp = slope(u32(coef[h]) + g.o, g.p);
          let rhoip = slope(u32(coef[hji]) + g.o, g.p);
          let kz = u32(coef[h + 1u]) + g.o;
          let z2p = slope(kz, g.p);
          let z2 = value(kz, g.p);
          let recip = 1.0 / r;
          let phi = z2 * recip;
          let phip = z2p * recip - phi * recip;
          let fpj = bitcast<f32>(ints[P.fpBase + j]);
          let psip = fpi * rhojp + fpj * rhoip + phip;
          let sc = coef[h + 3u];
          let fpair = -sc * psip * recip;
          f = f + fpair * d;
          ev = ev + 0.5 * sc * phi;
          w = w + 0.5 * fpair * r2;
        }
      }
    }
  }
  let o = 6u * i;
  outv[o] = f.x;
  outv[o + 1u] = f.y;
  outv[o + 2u] = f.z;
  outv[o + 3u] = ev;
  outv[o + 4u] = 0.0;
  outv[o + 5u] = w;
}
`;
