import { StyleError } from '../force/types';
import { parseNum, parseInt_ } from '../force/util';
import { buildTriples, rawBispectrum, type Triple, type Cmat } from '../compute/sna';

/*
 * Descriptor "sna" of pair_style mliap and compute mliap (wave 16).
 * Doc (docs.lammps.org/pair_mliap.html, plans/lammps-docs/pair_mliap.rst): "The SNAP descriptor
 * file closely follows the format of the pair_style snap parameter file. The file can contain
 * blank and comment lines (start with #) anywhere. Each non-blank non-comment line must contain one
 * keyword/value pair. The required keywords are rcutfac and twojmax. ... In addition, the SNAP
 * descriptor file must contain the nelems, elems, radelems, and welems keywords."
 * The descriptor is the bispectrum of compute sna/atom with the SNAP conventions of pair_style snap
 * (the same bispectrum, cutoff, switching and normalisation code is used here, see sna.ts).
 *
 * Keywords accepted (measured with native LAMMPS, black box): rfac0, rmin0, switchflag, bzeroflag,
 * bnormflag, wselfallflag and chemflag. Measured to change the energy: bzeroflag 0 and switchflag 0
 * and bnormflag 1. Keywords the native reader refuses for this descriptor and that are reported here
 * as StyleError: quadraticflag (the quadratic terms belong to the model), chunksize, parallelthresh
 * and switchinnerflag 1.
 *
 * chemflag 1 (explicit multi-element bispectrum, shared with pair_style snap in src/engine/compute/snapChem.ts).
 * The doc sentence is in the block comment above the interface. The
 * descriptor then has K N_elem^3 columns per element block (nmodel), and the model file has nparams = 1 + K N_elem^3
 * (measured with native LAMMPS, black box, InP_JCPA2020.mliap.model: 2 241).
 */

/*
 * Doc (docs.lammps.org/pair_snap.html, plans/lammps-docs-live/pair_snap.rst), the chemflag keyword that the sna
 * descriptor shares with pair_style snap: "If *chemflag* is set to 1, then the energy expression is written in terms of explicit multi-element bispectrum components indexed on ordered triplets of elements".
 */
export interface MliapSnaDescriptor {
  elems: string[];
  /** Radius and weight per element (index = element index in the descriptor's elems list). */
  radius: Float64Array;
  weight: Float64Array;
  rcutfac: number;
  twojmax: number;
  rfac0: number;
  rmin0: number;
  switchflag: boolean;
  bzeroflag: boolean;
  bnormflag: boolean;
  /** chemflag 1: the explicit multi-element bispectrum (K N_elem^3 model columns). */
  chem: boolean;
  wselfall: boolean;
  /** Number of model columns per element: K, or K N_elem^3 with chem. */
  nmodel: number;
  triples: Triple[];
  /** Number of bispectrum components K. */
  K: number;
  /** Per-component division (2J+1 with bnormflag, else 1). */
  norm: Float64Array;
  /** Bispectrum of an isolated atom (the B0 subtracted with bzeroflag 1). */
  b0: Float64Array;
}

const dataLines = (text: string): string[][] =>
  text.split('\n').map((l) => l.replace(/#.*$/, '').trim()).filter((l) => l.length > 0).map((l) => l.split(/\s+/));

export const parseMliapSnaDescriptor = (text: string, filename: string): MliapSnaDescriptor => {
  const kv = new Map<string, string[]>();
  for (const w of dataLines(text)) {
    const [kw, ...vals] = w;
    if (vals.length === 0) throw new StyleError(`mliap sna descriptor file ${filename}: keyword '${kw}' has no value`);
    kv.set(kw, vals);
  }
  const one = (kw: string): string | undefined => kv.get(kw)?.[0];
  const flag = (kw: string, dflt: boolean): boolean => {
    const v = one(kw);
    if (v === undefined) return dflt;
    if (v !== '0' && v !== '1') throw new StyleError(`mliap sna descriptor ${filename}: ${kw} must be 0 or 1 (got '${v}')`);
    return v === '1';
  };
  for (const kw of kv.keys()) {
    if (!['nelems', 'elems', 'radelems', 'welems', 'rcutfac', 'twojmax', 'rfac0', 'rmin0', 'switchflag', 'bzeroflag', 'bnormflag', 'wselfallflag', 'chemflag'].includes(kw)) {
      throw new StyleError(`mliap sna descriptor ${filename}: keyword '${kw}' is not implemented in this engine`);
    }
  }
  for (const kw of ['rcutfac', 'twojmax', 'nelems', 'elems', 'radelems', 'welems']) {
    if (!kv.has(kw)) throw new StyleError(`mliap sna descriptor ${filename}: keyword '${kw}' is required`);
  }
  const nelems = parseInt_(one('nelems'), `${filename} nelems`);
  if (nelems < 1) throw new StyleError(`mliap sna descriptor ${filename}: nelems must be at least 1`);
  const elems = kv.get('elems')!;
  const rad = kv.get('radelems')!.map((v) => parseNum(v, `${filename} radelems`));
  const wt = kv.get('welems')!.map((v) => parseNum(v, `${filename} welems`));
  if (elems.length !== nelems || rad.length !== nelems || wt.length !== nelems) {
    throw new StyleError(`mliap sna descriptor ${filename}: elems, radelems and welems must each list nelems = ${nelems} entries`);
  }
  const rcutfac = parseNum(one('rcutfac'), `${filename} rcutfac`);
  if (!(rcutfac > 0)) throw new StyleError(`mliap sna descriptor ${filename}: rcutfac must be positive`);
  const twojmax = parseInt_(one('twojmax'), `${filename} twojmax`);
  if (twojmax < 0) throw new StyleError(`mliap sna descriptor ${filename}: twojmax must be a non-negative integer`);
  const rfac0 = kv.has('rfac0') ? parseNum(one('rfac0'), `${filename} rfac0`) : 0.99363;
  const rmin0 = kv.has('rmin0') ? parseNum(one('rmin0'), `${filename} rmin0`) : 0;
  const switchflag = flag('switchflag', true);
  const bzeroflag = flag('bzeroflag', true);
  const bnormflag = flag('bnormflag', false);
  const chem = flag('chemflag', false);
  const wselfall = flag('wselfallflag', false); // acts only with chem
  const triples = buildTriples(twojmax);
  const K = triples.length;
  const nmodel = chem ? K * nelems * nelems * nelems : K;
  const norm = new Float64Array(K);
  for (let c = 0; c < K; c++) norm[c] = bnormflag ? triples[c].J + 1 : 1;
  const b0 = new Float64Array(K);
  if (bzeroflag) {
    const id: Cmat[] = [];
    for (let J = 0; J <= twojmax; J++) {
      const n = J + 1;
      const re = new Float64Array(n * n), im = new Float64Array(n * n);
      for (let q = 0; q < n; q++) re[q * n + q] = 1;
      id.push({ re, im });
    }
    rawBispectrum(triples, id, b0);
  }
  return {
    elems, radius: Float64Array.from(rad), weight: Float64Array.from(wt), rcutfac, twojmax, rfac0, rmin0,
    switchflag, bzeroflag, bnormflag, chem, wselfall, nmodel, triples, K, norm, b0,
  };
};
