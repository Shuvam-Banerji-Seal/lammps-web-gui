/*
 * Support code for the styles ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the C
 * library log() that the native binary calls. LAMMPS code calls log() from libm, and the parity tests compare against the
 * native binary as it runs here: Ubuntu glibc 2.35 on an x86_64 CPU with FMA and AVX2, where glibc dispatches log to
 * __ieee754_log_fma (sysdeps/x86_64/fpu/multiarch/e_log-fma.c: sysdeps/ieee754/dbl-64/e_log.c compiled with -mfma
 * -mavx2). This file reproduces that function bit for bit: the algorithm and the tables of e_log.c and e_log_data.c
 * (glibc 2.35; written by Szabolcs Nagy, Copyright (C) 2018-2022 Free Software Foundation, Inc.), and the fused
 * multiply-adds exactly where the compiled libm.so.6 has them (read from its disassembly; GCC contracts more than the
 * one explicit __builtin_fma of the source). The table values were read from the installed libm.so.6.
 *
 * glibc is distributed under the GNU Lesser General Public License, version 2.1 or later; this file is used under its
 * section 3, which allows the code to be distributed under the GNU General Public License, version 2 or later, and is
 * GPL-2.0 like the rest of src/engine-gpl (LICENSE). The owner decided on 2026-10-16 that third-party code may come
 * here when native parity needs it.
 *
 * fma() is a correctly rounded fused multiply-add emulated in binary64 (exact Dekker product, then the sum rounded to
 * odd as in S. Boldo and G. Melquiond, "Emulation of FMA and correctly rounded sums: proved algorithms using rounding
 * to odd", IEEE Trans. Computers 57(4), 2008); valid without overflow or underflow of the partial products, which holds
 * for every call below. Checked against the native libm on 5,000,000 inputs (RanMars rsq values, random normal
 * doubles over the whole exponent range, the near-1 interval, subnormals, values in (0, 4)): no difference.
 */

const F = new Float64Array(1);
const U = new Uint32Array(F.buffer); // little-endian: U[0] low word, U[1] high word
const SPLIT = 134217729; // 2^27 + 1

/** Correctly rounded a * b + c (see the header). */
export const fma = (a: number, b: number, c: number): number => {
  const p = a * b;
  let t = SPLIT * a;
  const ah = t - (t - a), al = a - ah;
  t = SPLIT * b;
  const bh = t - (t - b), bl = b - bh;
  const ul = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
  // TwoSum(c, p)
  const th = c + p;
  const bb = th - c;
  const tl = (c - (th - bb)) + (p - bb);
  // round-to-odd sum of the two error terms
  const s = tl + ul;
  const b2 = s - tl;
  const err = (tl - (s - b2)) + (ul - b2);
  if (err === 0) return th + s;
  F[0] = s;
  if ((U[0] & 1) === 0) {
    if ((err > 0) === (s > 0)) {
      if (U[0] === 0xffffffff) { U[0] = 0; U[1] = U[1] + 1; } else U[0] = U[0] + 1;
    } else if (U[0] === 0) { U[0] = 0xffffffff; U[1] = U[1] - 1; } else U[0] = U[0] - 1;
  }
  return th + F[0];
};

/** Veltkamp split of b into bh + bl (26 + 27 bits), for a multiplicand used many times. */
export const splitHi = (b: number): number => { const t = SPLIT * b; return t - (t - b); };

/** fma(a, b, c) with b given with its split (bh, bl): the same correctly rounded result, one split fewer. */
export const fmaS = (a: number, b: number, bh: number, bl: number, c: number): number => {
  const p = a * b;
  const t = SPLIT * a;
  const ah = t - (t - a), al = a - ah;
  const ul = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
  const th = c + p;
  const bb = th - c;
  const tl = (c - (th - bb)) + (p - bb);
  const s = tl + ul;
  const b2 = s - tl;
  const err = (tl - (s - b2)) + (ul - b2);
  if (err === 0) return th + s;
  F[0] = s;
  if ((U[0] & 1) === 0) {
    if ((err > 0) === (s > 0)) {
      if (U[0] === 0xffffffff) { U[0] = 0; U[1] = U[1] + 1; } else U[0] = U[0] + 1;
    } else if (U[0] === 0) { U[0] = 0xffffffff; U[1] = U[1] - 1; } else U[0] = U[0] - 1;
  }
  return th + F[0];
};

// __log_data (e_log_data.c): ln2hi, ln2lo, poly (A), poly1 (B) and tab (invc, logc) for N = 128
const LN2HI = 0.6931471805598903;
const LN2LO = 5.497923018708371e-14;
const A0 = -0.5000000000000001, A1 = 0.33333333331825593, A2 = -0.2499999999622955, A3 = 0.20000304511814496, A4 = -0.16667054827627667;
const B = [-0.5, 0.3333333333333352, -0.24999999999998432, 0.19999999999320328, -0.16666666669929706, 0.14285715076560868, -0.12499997863982555, 0.11110712032936046, -0.10000486757818193, 0.09181994006195467, -0.08328363062289341];
// splits of the constant multiplicands
const SA0 = splitHi(A0), SA2 = splitHi(A2), SA4 = splitHi(A4), SL2H = splitHi(LN2HI), SL2L = splitHi(LN2LO);
const TAB = new Float64Array([
  1.4504249240398293, -0.3718565645633589,
  1.442253508327276, -0.36620682668944937,
  1.4341736174350004, -0.3605888069791945,
  1.426183816329995, -0.3550022171419869,
  1.4182825527052965, -0.34944666968829097,
  1.4104682921759335, -0.3439217713603284,
  1.4027396147468003, -0.3384271921261188,
  1.3950954438932313, -0.332962831494342,
  1.3875338232485754, -0.32752794345742586,
  1.3800539211058593, -0.32212257167088865,
  1.3726542695419708, -0.3167462884799761,
  1.3653332798446802, -0.3113985598928366,
  1.358090204587874, -0.3060794515165526,
  1.3509234892132138, -0.300788424667644,
  1.3438320840699889, -0.2955252968476998,
  1.3368146974742003, -0.29028969275850613,
  1.329870114677736, -0.2850812793277555,
  1.322997339161106, -0.27989987391470095,
  1.316195352741367, -0.27474526621870154,
  1.3094628125672239, -0.2696169863701243,
  1.3027990455471041, -0.26451506180308115,
  1.2962024229438942, -0.2594387762767383,
  1.2896726275815547, -0.2543884090981692,
  1.2832080305745537, -0.24936321635129843,
  1.276807885983376, -0.24436312405975968,
  1.2704714060687552, -0.23938801747897287,
  1.2641976054949482, -0.23443761696705678,
  1.257985357514882, -0.22951151871518505,
  1.2518337750655457, -0.2246094963439873,
  1.2457421919097305, -0.21973149037705753,
  1.2397094966625508, -0.2148770752847895,
  1.2337348463589233, -0.210046029103637,
  1.2278176973028803, -0.20523836373934046,
  1.2219570190618474, -0.20045368751368642,
  1.2161519732977757, -0.1956917537758045,
  1.2104018095009725, -0.19095237845203883,
  1.204705805718973, -0.18623539250290833,
  1.1990631185441964, -0.18154051731551135,
  1.1934733004462308, -0.1768677957431919,
  1.1879350812847385, -0.17221657406412305,
  1.1824481322833125, -0.16758697765942543,
  1.1770114976921955, -0.16297859687290384,
  1.1716248121809465, -0.15839151377804228,
  1.1662869231674715, -0.15382513241456763,
  1.1609977486762766, -0.14927976358922024,
  1.1557563220795803, -0.14475495398119165,
  1.1505619105480347, -0.14025044090817573,
  1.1454138888505974, -0.1357660466685502,
  1.140311877374656, -0.1313018016355727,
  1.1352550225747513, -0.12685731518763532,
  1.1302429094831266, -0.1224325737671279,
  1.1252747693068048, -0.11802724521862729,
  1.1203501571039876, -0.11364127671663482,
  1.1154683327680124, -0.10927434611278386,
  1.1106291463292157, -0.10492665324943573,
  1.1058315813301596, -0.10059761422644442,
  1.1010752177696026, -0.09628717309055901,
  1.0963597137952512, -0.09199534069557558,
  1.0916844827550398, -0.08772190036688698,
  1.0870487291277784, -0.08346643613867855,
  1.082452357388312, -0.07922916827544668,
  1.0778948225025884, -0.0750099004750382,
  1.0733751731601076, -0.07080805133352897,
  1.068893585073351, -0.06662408085151128,
  1.0644491706655506, -0.06245745471915143,
  1.0600414846328305, -0.0583080438042316,
  1.0556701316181605, -0.05417576112313327,
  1.051334750556926, -0.050060547896805474,
  1.0470347288442157, -0.045962101199052086,
  1.0427699229652954, -0.04188056008865715,
  1.0385395013738175, -0.03781540056183985,
  1.034343418940345, -0.03376684757915882,
  1.0301811073173315, -0.029734619131772888,
  1.026052043621297, -0.025718470239212365,
  1.0219561082336197, -0.021718543925430822,
  1.0178926505784922, -0.01773446126981071,
  1.0138614436244586, -0.013766252464051831,
  1.0098620186501341, -0.009813706322574944,
  1.0058938559734134, -0.005876555150052809,
  1.00195696235014, -0.0019550499938532084,
  0.9961089923088509, 0.0038985973556009412,
  0.9884170338185201, 0.011650571286395461,
  0.9808429191005297, 0.019342955478919066,
  0.9733840169987446, 0.0269766014846482,
  0.9660377568876556, 0.034552359728422744,
  0.9588014945307369, 0.04207121767183253,
  0.9516728569073111, 0.049533940950141186,
  0.9446494635965822, 0.056941358295944156,
  0.9377288993026223, 0.06429439168346107,
  0.9309091073790681, 0.0715936354946507,
  0.924187681612722, 0.07884010933776153,
  0.9175626765599192, 0.08603438905970506,
  0.9110320403624034, 0.09317721180013905,
  0.9045935839762024, 0.10026951462748457,
  0.8982456375922825, 0.10731170956330516,
  0.8919860966782501, 0.11430473320717738,
  0.8858131121185129, 0.12124928503033061,
  0.879725075760676, 0.12814583422959913,
  0.8737201372634685, 0.1349951636851756,
  0.8677966405782273, 0.1417978768189414,
  0.8619528050060739, 0.14855476039031146,
  0.8561872354420692, 0.1552661937658968,
  0.8504983927816893, 0.16193275688146969,
  0.8448844572790304, 0.16855539792220497,
  0.8393442741575965, 0.1751343179947753,
  0.8338762249349438, 0.1816702989864325,
  0.8284789320557778, 0.18816387146023317,
  0.8231510800065832, 0.1946155228479256,
  0.8178913903778707, 0.20102572579389744,
  0.8126984007245374, 0.2073952090795501,
  0.8075710029460227, 0.21372429840596396,
  0.8025078881160415, 0.2200135945981856,
  0.7975077379364331, 0.22626374162859975,
  0.792569604966373, 0.23247494747693054,
  0.7876923641254114, 0.23864766620658884,
  0.7828746724940998, 0.24478265647405806,
  0.7781155388790811, 0.25088025827324145,
  0.7734139557869777, 0.2569408552510595,
  0.7687687179914933, 0.26296511155101143,
  0.7641790698041854, 0.2689531327189343,
  0.7596438763692399, 0.27490553924610595,
  0.7551621951078668, 0.2808227248478943,
  0.7507331780216866, 0.286704979267256,
  0.7463557196361751, 0.29255295645509705,
  0.7420289364869653, 0.2983670386142876,
  0.7377521537065876, 0.30414734587282055,
  0.7335242966002608, 0.30989455774829366,
  0.729344777457841, 0.31560871301871884,
]);
/** The fast path's distance from a rounding midpoint that the fused steps cannot cross (see log). */
const FAST_MARGIN = 2.168404344971009e-19; // 2^-62

/** Split high parts of the 1/c values of TAB (the low part is invc - high). */
const INVC_H = new Float64Array(128);
for (let i = 0; i < 128; i++) INVC_H[i] = splitHi(TAB[2 * i]);

/** log(x) exactly as the native binary's glibc computes it (see the header). */
export const log = (x: number): number => {
  F[0] = x;
  let hi = U[1];
  const lo = U[0];
  // close to 1.0: 1 - 0x1p-4 <= x < 1 + 0x1.09p-4
  if (hi >= 0x3fee0000 && hi < 0x3ff10900) {
    if (hi === 0x3ff00000 && lo === 0) return 0;
    const r = x - 1.0;
    const r2 = r * r;
    const r3 = r * r2;
    // Fast path, as in the main range below. The products with 2^27 and with B[0] = -0.5 are exact, so the fused rhi,
    // hi and lo equal the plain ones; the fused polynomial steps and the last two fused products change the sum by at
    // most |r^3| 2^-50 + r^2 2^-80, so a plain y whose exact sum is farther than that (with a factor 4 to spare) from
    // a rounding midpoint is libm's.
    {
      const p0 = (B[1] + r * B[2] + r2 * B[3] + r3 * (B[4] + r * B[5] + r2 * B[6] + r3 * (B[7] + r * B[8] + r2 * B[9] + r3 * B[10])));
      const t = r + r * 134217728;
      const rhi = t - r * 134217728;
      const rlo = r - rhi;
      const w = rhi * rhi * B[0];
      const h = r + w;
      const l = (r - h) + w + B[0] * rlo * (r + rhi);
      const v0 = r3 * p0 + l;
      const y0 = h + v0;
      const bb = y0 - v0;
      const e0 = (v0 - (y0 - bb)) + (h - bb);
      F[0] = y0;
      const ex = (U[1] >>> 20) & 0x7ff;
      if (ex > 60) {
        const pow2 = (U[1] & 0xfffff) === 0 && U[0] === 0;
        U[1] = (ex - 53) << 20;
        U[0] = 0;
        const half = F[0];
        const away = (e0 > 0) === (y0 > 0);
        const dist = (away || !pow2 ? half : half / 2) - Math.abs(e0);
        if (dist > 4 * (Math.abs(r3) * 8.881784197001252e-16 + r2 * 8.271806125530277e-25)) return y0;
      }
    }
    const a1 = fma(r, B[2], B[1]);
    const a2 = fma(r, B[5], B[4]);
    const a3 = fma(r, B[8], B[7]);
    const b1 = fma(r2, B[3], a1);
    const b2 = fma(r2, B[6], a2);
    const b3 = fma(r2, B[9], a3);
    const c3 = fma(r3, B[10], b3);
    const c2 = fma(c3, r3, b2);
    const c1 = fma(c2, r3, b1);
    // rhi = r + r 2^27 - r 2^27, the two steps fused as in libm
    const rhi = fma(-134217728, r, fma(r, 134217728, r));
    const rlo = r - rhi;
    const rhi2 = rhi * rhi;
    const h = fma(rhi2, B[0], r);
    const l = fma(rhi2, B[0], r - h);
    const l2 = fma(B[0] * rlo, r + rhi, l);
    return h + fma(c1, r3, l2);
  }
  const top = hi >>> 16;
  if (((top - 0x0010) >>> 0) >= 0x7fe0) {
    // x < 0x1p-1022, inf or nan
    if ((hi & 0x7fffffff) === 0 && lo === 0) return -Infinity;
    if (hi === 0x7ff00000 && lo === 0) return x;
    if ((top & 0x8000) || (top & 0x7ff0) === 0x7ff0) return NaN;
    // subnormal: normalize (x 2^52, exponent field - 52)
    F[0] = x * 4503599627370496;
    hi = (U[1] - 0x03400000) >>> 0;
  }
  const lo2 = U[0];
  // x = 2^k z, z in [OFF, 2 OFF), OFF = 0x3fe6000000000000; tmp = ix - OFF (the low word has no borrow)
  const tmpHi = (hi - 0x3fe60000) >>> 0;
  const i = (tmpHi >>> 13) & 127;
  const k = (tmpHi | 0) >> 20;
  U[1] = (hi - (tmpHi & 0xfff00000)) >>> 0;
  U[0] = lo2;
  const z = F[0];
  const invc = TAB[2 * i], logc = TAB[2 * i + 1];
  // r = fma(z, invc, -1): z invc is within 2^-8 of 1, so p - 1 is exact (Sterbenz) and (p - 1) + err(z invc) is the
  // correctly rounded z invc - 1, the fused result
  const ih = INVC_H[i], il = invc - ih;
  const pz = z * invc;
  const tz = SPLIT * z;
  const zh = tz - (tz - z), zl = z - zh;
  const r = (pz - 1.0) + (((zh * ih - pz) + zh * il + zl * ih) + zl * il);
  // Fast path. k LN2HI is exact (LN2HI has 11 trailing zero bits and |k| < 2^11), so the fused w equals the plain one.
  // The other fused steps change v below by at most about 2^-68 (the polynomial terms are below 2^-16 for |r| < 2^-8),
  // and |y| >= 0.06 here, so the unfused y is the fused one unless the exact sum lies within 2^-62 of a rounding
  // midpoint; then the fused steps decide, as libm does.
  {
    const w0 = k * LN2HI + logc;
    const h0 = r + w0;
    const r20 = r * r;
    const l0 = k * LN2LO + ((w0 - h0) + r);
    const v0 = (r * r20) * ((r * A4 + A3) * r20 + (r * A2 + A1)) + (r20 * A0 + l0);
    const y0 = v0 + h0;
    const bb = y0 - v0;
    const e0 = (v0 - (y0 - bb)) + (h0 - bb);
    F[0] = y0;
    const ex = (U[1] >>> 20) & 0x7ff;
    const pow2 = (U[1] & 0xfffff) === 0 && U[0] === 0;
    U[1] = (ex - 53) << 20;
    U[0] = 0;
    const half = F[0]; // ulp(y0) / 2
    // the midpoint on the side of the exact value: below a power of two the spacing is half as large
    const away = (e0 > 0) === (y0 > 0);
    const dist = (away || !pow2 ? half : half / 2) - Math.abs(e0);
    if (dist > FAST_MARGIN) return y0;
  }
  const w = fmaS(k, LN2HI, SL2H, LN2HI - SL2H, logc);
  const t1 = fmaS(r, A2, SA2, A2 - SA2, A1);
  const h = r + w;
  const r2 = r * r;
  const l = fmaS(k, LN2LO, SL2L, LN2LO - SL2L, (w - h) + r);
  const r3 = r * r2;
  const t2 = fmaS(r, A4, SA4, A4 - SA4, A3);
  const u = fmaS(r2, A0, SA0, A0 - SA0, l);
  const p = fma(t2, r2, t1);
  return fma(r3, p, u) + h;
};
