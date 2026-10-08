import { StyleError } from '../force/types';
import { parseNum, parseInt_ } from '../force/util';

/*
 * Model files of pair_style mliap (wave 16). Doc (docs.lammps.org/pair_mliap.html, quoted from
 * plans/lammps-docs/pair_mliap.rst): "The top of the model file can contain any number of blank and
 * comment lines (start with #), but follows a strict format after that. The first non-blank
 * non-comment line must contain two integers: nelems = Number of elements, nparams = Number of
 * parameters." For linear and quadratic: "this is followed by one block for each of the nelem
 * elements. Each block consists of nparams parameters, one per line." The quadratic model's
 * parameters follow the pair_style snap layout (1 + K + K(K+1)/2 values, measured with native
 * LAMMPS: the quadratic energy equals pair_style snap with quadraticflag 1 on the same numbers).
 *
 * Measured with native LAMMPS (black box) for the nn model: the second non-blank non-comment line
 * holds "NET ndescriptors nlayers" followed on the same line by one activation name and node count
 * per layer ("NET 5 2 tanh 3 linear 1"); a block per element is ndescriptors scale0 values (the
 * minimum of each descriptor), ndescriptors scale1 values (maximum minus minimum), then nparams
 * values, bias first and then the weights of each node, node by node, layer by layer. The
 * descriptor is normalised as (B - scale0) / scale1. Every layer, the output layer included, applies
 * its activation: a model "sigmoid 4 relu 3 tanh 1" was matched to 1e-15 with this rule. Writing a
 * layer on its own line made the native reader fail, so this engine requires it on the NET line.
 */

export type ModelKind = 'linear' | 'quadratic' | 'nn';

export interface MliapModel {
  readonly kind: ModelKind;
  readonly nelems: number;
  /** Energy of an atom of element e from its descriptor vector Bf; gam receives dE/dBf (length K). */
  energy(e: number, Bf: Float64Array, gam: Float64Array): number;
}

const tokensOf = (text: string): string[] =>
  text.split('\n').map((l) => l.replace(/#.*$/, '').trim()).filter((l) => l.length > 0).join(' ').split(/\s+/);

const linesOf = (text: string): string[] =>
  text.split('\n').map((l) => l.replace(/#.*$/, '').trim()).filter((l) => l.length > 0);

interface Layer {
  act: Act;
  nin: number;
  nout: number;
  /** Offset of the layer's first parameter in the element block (bias of node 0). */
  off: number;
}

type Act = 'linear' | 'tanh' | 'sigmoid' | 'relu';
const ACTS: Act[] = ['linear', 'tanh', 'sigmoid', 'relu'];

const readBlocks = (vals: number[], nelems: number, per: number, filename: string): Float64Array[] => {
  if (vals.length !== nelems * per) {
    throw new StyleError(`mliap model file ${filename}: expected ${nelems * per} parameter values (${nelems} elements x ${per}), found ${vals.length}`);
  }
  const out: Float64Array[] = [];
  for (let e = 0; e < nelems; e++) out.push(Float64Array.from(vals.slice(e * per, (e + 1) * per)));
  return out;
};

const polynomialModel = (kind: 'linear' | 'quadratic', text: string, filename: string, K: number, nelemsDesc: number): MliapModel => {
  const w = tokensOf(text);
  const nelems = parseInt_(w[0], `${filename} nelems`);
  const nparams = parseInt_(w[1], `${filename} nparams`);
  const want = kind === 'linear' ? 1 + K : 1 + K + (K * (K + 1)) / 2;
  if (nelems !== nelemsDesc) throw new StyleError(`mliap model file ${filename}: nelems ${nelems} does not match the descriptor (${nelemsDesc} elements)`);
  if (nparams !== want) {
    throw new StyleError(`mliap model file ${filename}: ${kind} model needs nparams = ${want} for ${K} descriptor components (got ${nparams})`);
  }
  const blocks = readBlocks(w.slice(2).map((v) => parseNum(v, `${filename} parameter`)), nelems, nparams, filename);
  const Q = kind === 'quadratic';
  return {
    kind, nelems,
    energy(e, Bf, gam) {
      const c = blocks[e];
      let E = c[0];
      for (let k = 0; k < K; k++) { E += c[1 + k] * Bf[k]; gam[k] = c[1 + k]; }
      if (Q) {
        let q = 1 + K;
        for (let kk = 0; kk < K; kk++) {
          for (let ll = kk; ll < K; ll++) {
            const cq = c[q++];
            if (kk === ll) {
              E += cq * 0.5 * Bf[kk] * Bf[kk];
              gam[kk] += cq * Bf[kk];
            } else {
              E += cq * Bf[kk] * Bf[ll];
              gam[kk] += cq * Bf[ll];
              gam[ll] += cq * Bf[kk];
            }
          }
        }
      }
      return E;
    },
  };
};

const nnModel = (text: string, filename: string, K: number, nelemsDesc: number): MliapModel => {
  const lines = linesOf(text);
  if (lines.length < 2) throw new StyleError(`mliap nn model file ${filename}: expected the header lines 'nelems nparams' and 'NET ...'`);
  const head = lines[0].split(/\s+/);
  const nelems = parseInt_(head[0], `${filename} nelems`);
  const nparams = parseInt_(head[1], `${filename} nparams`);
  const net = lines[1].split(/\s+/);
  if (net[0] !== 'NET') throw new StyleError(`mliap nn model file ${filename}: the second line must start with NET`);
  if (net.length < 3 || (net.length - 3) % 2 !== 0) {
    throw new StyleError(`mliap nn model file ${filename}: 'NET ndescriptors nlayers' must be followed on the same line by one activation and node count per layer`);
  }
  const ndesc = parseInt_(net[1], `${filename} ndescriptors`);
  const nlayers = parseInt_(net[2], `${filename} nlayers`);
  if (ndesc !== K) throw new StyleError(`mliap nn model file ${filename}: ${ndesc} descriptors do not match the descriptor (${K} components)`);
  if (nlayers !== (net.length - 3) / 2) throw new StyleError(`mliap nn model file ${filename}: nlayers ${nlayers} does not match the layer list`);
  if (nelems !== nelemsDesc) throw new StyleError(`mliap nn model file ${filename}: nelems ${nelems} does not match the descriptor (${nelemsDesc} elements)`);
  const layers: Layer[] = [];
  let nin = ndesc, off = 0;
  for (let l = 0; l < nlayers; l++) {
    const act = net[3 + 2 * l] as Act;
    if (!ACTS.includes(act)) throw new StyleError(`mliap nn model file ${filename}: activation '${act}' is not implemented (linear, tanh, sigmoid, relu)`);
    const nout = parseInt_(net[4 + 2 * l], `${filename} nodes of layer ${l + 1}`);
    if (nout < 1) throw new StyleError(`mliap nn model file ${filename}: layer ${l + 1} needs at least one node`);
    layers.push({ act, nin, nout, off });
    off += nout * (nin + 1);
    nin = nout;
  }
  if (nin !== 1) throw new StyleError(`mliap nn model file ${filename}: the output layer must have one node (the energy)`);
  if (off !== nparams) throw new StyleError(`mliap nn model file ${filename}: the layers need ${off} parameters per element, the header says ${nparams}`);
  // element blocks: scale0 (K), scale1 (K), then nparams values
  const vals = tokensOf(lines.slice(2).join('\n')).map((v) => parseNum(v, `${filename} value`));
  const per = 2 * K + nparams;
  if (vals.length !== nelems * per) {
    throw new StyleError(`mliap nn model file ${filename}: expected ${nelems * per} values (${nelems} elements x (2 x ${K} scale values + ${nparams} parameters)), found ${vals.length}`);
  }
  const scale0: Float64Array[] = [], scale1: Float64Array[] = [], par: Float64Array[] = [];
  for (let e = 0; e < nelems; e++) {
    const b = e * per;
    scale0.push(Float64Array.from(vals.slice(b, b + K)));
    scale1.push(Float64Array.from(vals.slice(b + K, b + 2 * K)));
    par.push(Float64Array.from(vals.slice(b + 2 * K, b + per)));
  }
  // work buffers, reused across calls: node values per layer (hbuf[0] = normalised input), pre-activations, gradients
  let maxN = K;
  for (const l of layers) if (l.nout > maxN) maxN = l.nout;
  const hbuf: Float64Array[] = [new Float64Array(K)];
  for (const l of layers) hbuf.push(new Float64Array(l.nout));
  const zbuf: Float64Array[] = layers.map((l) => new Float64Array(l.nout));
  const gA = new Float64Array(maxN), gB = new Float64Array(maxN);
  return {
    kind: 'nn', nelems,
    energy(e, Bf, gam) {
      const p = par[e], s0 = scale0[e], s1 = scale1[e];
      const h0 = hbuf[0];
      for (let k = 0; k < K; k++) h0[k] = (Bf[k] - s0[k]) / s1[k];
      for (let l = 0; l < layers.length; l++) {
        const L = layers[l];
        const input = hbuf[l], out = hbuf[l + 1], z = zbuf[l];
        for (let j = 0; j < L.nout; j++) {
          const b = L.off + j * (L.nin + 1);
          let v = p[b];
          for (let i = 0; i < L.nin; i++) v += p[b + 1 + i] * input[i];
          z[j] = v;
          out[j] = actValue(L.act, v);
        }
      }
      const E = hbuf[layers.length][0];
      // backward pass: dOut = dE/dh of the current layer's nodes, starting from the energy node
      let dOut = gA, dIn = gB;
      dOut[0] = 1;
      for (let l = layers.length - 1; l >= 0; l--) {
        const L = layers[l];
        const z = zbuf[l], h = hbuf[l + 1];
        for (let i = 0; i < L.nin; i++) dIn[i] = 0;
        for (let j = 0; j < L.nout; j++) {
          const dz = dOut[j] * actDeriv(L.act, z[j], h[j]);
          const b = L.off + j * (L.nin + 1);
          for (let i = 0; i < L.nin; i++) dIn[i] += p[b + 1 + i] * dz;
        }
        const tmp = dOut; dOut = dIn; dIn = tmp;
      }
      // dOut now holds dE/dx with x = (B - scale0) / scale1
      for (let k = 0; k < K; k++) gam[k] = dOut[k] / s1[k];
      return E;
    },
  };
};

const actValue = (act: Act, z: number): number => {
  switch (act) {
    case 'linear': return z;
    case 'tanh': return Math.tanh(z);
    case 'sigmoid': return 1 / (1 + Math.exp(-z));
    default: return z > 0 ? z : 0;
  }
};

/** Derivative of the activation, from the pre-activation z and the activation value h. */
const actDeriv = (act: Act, z: number, h: number): number => {
  switch (act) {
    case 'linear': return 1;
    case 'tanh': return 1 - h * h;
    case 'sigmoid': return h * (1 - h);
    default: return z > 0 ? 1 : 0;
  }
};

/** Parse a model file of pair_style mliap (model style linear, quadratic or nn). */
export const parseMliapModel = (kind: ModelKind, text: string, filename: string, K: number, nelems: number): MliapModel => {
  if (kind === 'nn') return nnModel(text, filename, K, nelems);
  return polynomialModel(kind, text, filename, K, nelems);
};

