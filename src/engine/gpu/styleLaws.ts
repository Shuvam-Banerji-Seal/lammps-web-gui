/*
 * The short-range force laws of the pair styles of gpu/pairSpec.ts in WGSL, shared by the per-step kernel
 * (webgpuForces.ts KERNEL_STYLES) and the whole-step kernel (resident.ts): shortRange returns F/r and the pair energy
 * of one pair inside its short-range cutoff, from the coefficients [cutsqA cutsqC a b] [c d eshift cscale] [e f _ _]
 * of the type pair (pairSpec.ts pairSpecEval is the same arithmetic in fp64).
 */
export const WGSL_STYLE_LAWS = /* wgsl */ `
// sin and cos of x in [0, pi] to fp32 rounding: WGSL's builtins may err by 2^-11 absolute (the WGSL spec's accuracy of
// sin and cos), which pair soft's energy would show. Folded to [0, pi/2], then the Taylor series to x^17 and x^18.
fn sinCos(x : f32) -> vec2<f32> {
  var y = x;
  var sc = 1.0;
  if (y > 1.5707963267948966) { y = 3.141592653589793 - y; sc = -1.0; }
  let y2 = y * y;
  let s = y * (1.0 + y2 * (-1.0 / 6.0 + y2 * (1.0 / 120.0 + y2 * (-1.0 / 5040.0 + y2 * (1.0 / 362880.0
    + y2 * (-1.0 / 39916800.0 + y2 * (1.0 / 6227020800.0 + y2 * (-1.0 / 1307674368000.0 + y2 * (1.0 / 355687428096000.0)))))))));
  let c = 1.0 + y2 * (-0.5 + y2 * (1.0 / 24.0 + y2 * (-1.0 / 720.0 + y2 * (1.0 / 40320.0 + y2 * (-1.0 / 3628800.0
    + y2 * (1.0 / 479001600.0 + y2 * (-1.0 / 87178291200.0 + y2 * (1.0 / 20922789888000.0 + y2 * (-1.0 / 6402373705728000.0)))))))));
  return vec2<f32>(s, sc * c);
}

// F/r and E of one pair of style s (pairSpec.ts GPU_PAIR_STYLES) inside its short-range cutoff
fn shortRange(s : u32, r2 : f32, A : vec4<f32>, B : vec4<f32>, C3 : vec4<f32>) -> vec2<f32> {
  if (s == 13u) {
    // lj/charmm/coul/long: switched past the inner cutoff (C3.x its square, C3.y 1/(outer^2 - inner^2)^3)
    let r2inv = 1.0 / r2;
    let r6inv = r2inv * r2inv * r2inv;
    let u = r6inv * (B.x * r6inv - B.y);
    let fstd = r6inv * (A.z * r6inv - A.w) * r2inv;
    if (r2 <= C3.x) { return vec2<f32>(fstd, u); }
    let tt = A.x - r2;
    let sw = tt * tt * (A.x + 2.0 * r2 - 3.0 * C3.x) * C3.y;
    return vec2<f32>(fstd * sw + 12.0 * u * tt * (r2 - C3.x) * C3.y, u * sw);
  }
  if (s == 1u || s == 8u || s == 10u || s == 12u) {
    if (s != 8u) {
      let r2inv = 1.0 / r2;
      let r6inv = r2inv * r2inv * r2inv;
      return vec2<f32>(r6inv * (A.z * r6inv - A.w) * r2inv, r6inv * (B.x * r6inv - B.y) - B.z);
    }
    let rinv = 1.0 / sqrt(r2);
    let r2inv = rinv * rinv;
    let r3inv = r2inv * rinv;
    let r6inv = r3inv * r3inv;
    let r9inv = r6inv * r3inv;
    return vec2<f32>(r2inv * (A.z * r9inv - A.w * r6inv), r9inv * B.x - r6inv * B.y - B.z);
  }
  let r = sqrt(r2);
  if (s == 3u || s == 4u) {
    let ex = exp(-r * A.w);
    let r2inv = 1.0 / r2;
    return vec2<f32>(A.z * A.w * ex / r - 6.0 * B.x * r2inv * r2inv * r2inv * r2inv, A.z * ex - B.x * (r2inv * r2inv * r2inv) - B.z);
  }
  if (s == 5u) {
    let dr = r - B.x;
    let e1 = exp(-2.0 * A.w * dr);
    let e2 = exp(-A.w * dr);
    return vec2<f32>(2.0 * A.w * A.z * (e1 - e2) / r, A.z * (e1 - 2.0 * e2) - B.z);
  }
  if (s == 6u) {
    // arg = pi r / rc in [0, pi): sinCos, not the builtins (see sinCos)
    let sc2 = sinCos(r * A.w);
    return vec2<f32>(A.z * A.w * sc2.x / r, A.z * (1.0 + sc2.y));
  }
  if (s == 7u) {
    let ir = 1.0 / r;
    let ir2 = 1.0 / r2;
    let ek = exp(-A.w * r);
    return vec2<f32>(A.z * ek * (A.w * ir2 + ir2 * ir), A.z * ir * ek - B.z);
  }
  if (s == 9u) {
    let rinv = 1.0 / r;
    let ex = exp((B.x - r) * A.w);
    let r2inv = 1.0 / r2;
    let r6inv = r2inv * r2inv * r2inv;
    let r8inv = r6inv * r2inv;
    return vec2<f32>(A.z * A.w * ex * rinv - 6.0 * B.y * r8inv + 8.0 * C3.x * r8inv * r2inv, A.z * ex - B.y * r6inv + C3.x * r8inv - B.z);
  }
  return vec2<f32>(0.0, 0.0);
}
`;
