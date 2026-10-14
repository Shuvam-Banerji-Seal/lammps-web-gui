/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ML-UF3/pair_uf3.cpp, function PairUF3::uf3_read_unified_pot_file (the two passes: block sizes, then the
 * knot and coefficient values, with the permutation copies and the fill of the element pairs the file does not
 * list), and the line reading of src/text_file_reader.cpp (TextFileReader::next_line, skip_line, rewind) at commit
 * b332f28. Authors credited in the C++ headers: Ajinkya Hire, Hendrik Krass, Matthias Rupp and Richard Hennig.
 * Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 (src/engine-gpl/LICENSE).
 *
 * File layout (the unified UF3 potential file, docs.lammps.org/pair_uf3.html): each block starts with a line
 * containing "#UF3 POT" and "UNITS:"; the next line is "2B e1 e2 0 3 uk|nk" or "3B e1 e2 e3 0 3 uk|nk".
 */
import { StyleError } from '../../engine/force/types';

/** TextFileReader: lines with ignore_comments off; next_line(n) joins lines until n words, skipping blank lines. */
export class UF3TextReader {
  private pos = 0;
  constructor(private readonly lines: string[]) {}

  rewind(): void { this.pos = 0; }

  /** Next non-blank line joined with following lines until it has at least nparams words, or null at EOF. */
  nextLine(nparams = 1): string | null {
    let text = '';
    let nwords = 0;
    for (;;) {
      if (this.pos >= this.lines.length) {
        if (nwords > 0 && nwords < nparams) throw new StyleError(`Incorrect format in UF3 potential file: ${nwords}/${nparams} parameters`);
        return nwords > 0 ? text : null;
      }
      const piece = this.lines[this.pos++];
      const w = words(piece).length;
      if (w === 0 && nwords === 0) continue;
      text = nwords === 0 ? piece : `${text} ${piece}`;
      nwords += w;
      if (nwords >= nparams) return text;
    }
  }

  /** Reads one raw line and ignores it (native skip_line does not skip blank lines). */
  skipLine(): void {
    if (this.pos >= this.lines.length) throw new StyleError('Missing line in UF3 potential file!');
    this.pos++;
  }
}

const words = (s: string): string[] => s.trim().split(/\s+/).filter((w) => w.length > 0);

/** One 2-body block for an ordered element pair (1-based type indices in the containers below). */
export interface UF3Pair2 {
  /** Knot vector (length maxK2) and coefficient vector (length maxC2). */
  knots: Float64Array;
  coeff: Float64Array;
  nKnots: number;
  nCoeff: number;
  /** knot spacing type: 0 uniform ("uk"), 1 non-uniform ("nk"). */
  spacingType: number;
  spacing: number;
}

/** Parsed unified potential (native fields, same meaning). Type indices are 1-based. */
export interface UF3Potential {
  ntypes: number;
  nbody3: boolean;
  /** Index (i * (ntypes + 1) + j). */
  cut2: Float64Array;
  setflag2: Uint8Array;
  pair2: (UF3Pair2 | null)[];
  maxK2: number;
  maxC2: number;
  /** Index (i * (ntypes + 1) + j). */
  cutList3: Float64Array;
  /** Index ((i * (ntypes + 1) + j) * (ntypes + 1) + k). */
  cut3: Float64Array;
  setflag3: Uint8Array;
  spacingType3: Int32Array;
  /** min_cut_3b [i][j][k][d] at index (ijk) * 3 + d. */
  minCut3: Float64Array;
  /** knot_spacing_3b [i][j][k][d] at index (ijk) * 3 + d. */
  spacing3: Float64Array;
  /** Per key (map_3b): knot vectors of dims jk, ik, ij (index key * 3 + d), sizes and the coefficient cube (stride maxC3). */
  nKnots3: Int32Array;
  sizeC3: Int32Array;
  knots3: Float64Array[];
  coeff3: Float64Array[];
  maxK3: number;
  maxC3: number;
}

/** Index of a triplet (1-based types) in the (ntypes+1)^3 containers. */
export const idx3 = (n: number, i: number, j: number, k: number): number => (i * (n + 1) + j) * (n + 1) + k;
/** map_3b key of the native code: ((i-1) * ntypes + (j-1)) * ntypes + (k-1). */
export const key3 = (n: number, i: number, j: number, k: number): number => ((i - 1) * n + (j - 1)) * n + (k - 1);

const toInt = (w: string): number => {
  const v = Number(w);
  if (!Number.isInteger(v)) throw new StyleError(`UF3: expected an integer, found '${w}'`);
  return v;
};
const toDouble = (w: string): number => {
  const v = Number(w);
  if (Number.isNaN(v)) throw new StyleError(`UF3: expected a number, found '${w}'`);
  return v;
};

/**
 * Reads a unified UF3 file. elementOfType[t] (t = 1..ntypes) is the element name of atom type t, mapOf[t] the index of
 * that element in the list of unique element names (native map[]), and elements that list.
 */
export function readUF3Potential(
  text: string, nbody: 2 | 3, ntypes: number, elementOfType: string[], mapOf: number[], elements: string[], fileName: string,
): UF3Potential {
  const N = ntypes;
  const pot3 = nbody === 3;
  const reader = new UF3TextReader(text.split(/\r?\n/));
  const find = (name: string): number => {
    for (let i = 1; i <= N; i++) if (elements[mapOf[i]] === name) return i;
    return 0;
  };
  const S = N + 1;
  const pIdx = (i: number, j: number) => i * S + j;
  const cut2 = new Float64Array(S * S);
  const setflag2 = new Uint8Array(S * S);
  const cutList3 = new Float64Array(S * S);
  const cut3 = new Float64Array(S * S * S);
  const setflag3 = new Uint8Array(S * S * S);
  const spacingType3 = new Int32Array(S * S * S);
  const minCut3 = new Float64Array(S * S * S * 3);
  const spacing3 = new Float64Array(S * S * S * 3);
  const nKeys = N * N * N;
  const nKnots3 = new Int32Array(nKeys * 3);
  /** Coefficient cube dimensions per key (native n3b_coeff_array_size). */
  const sizeC3 = new Int32Array(nKeys * 3);
  const pair2: (UF3Pair2 | null)[] = new Array(S * S).fill(null);
  // Pass-1 sizes per ordered pair and per key.
  const nK2 = new Int32Array(S * S);
  const nC2 = new Int32Array(S * S);
  let maxK2 = 0, maxC2 = 0, maxK3 = 0, maxC3 = 0;

  // ---- pass 1: block sizes ----
  for (let line = reader.nextLine(1); line !== null; line = reader.nextLine(1)) {
    if (!line.includes('#UF3 POT')) continue;
    const second = reader.nextLine(1);
    if (second === null) throw new StyleError(`UF3: ${fileName} file ends after a block header`);
    if (!line.includes('UNITS:')) throw new StyleError(`UF3: ${fileName} file does not contain the 'UNITS:' metadata in the header`);
    const w = words(second);
    const nb = w[0];
    if (nb === '2B') {
      if (w.length !== 6) throw new StyleError(`UF3: Expected 6 words on a 2B header line of ${fileName} file but found ${w.length} word/s`);
      const itype = find(w[1]), jtype = find(w[2]);
      if (itype !== 0 && jtype !== 0) {
        if (toInt(w[3]) !== 0) throw new StyleError('UF3: Current implementation is throughly tested only for leading_trim=0');
        if (toInt(w[4]) !== 3) throw new StyleError('UF3: Current implementation is throughly tested only for trailing_trim=3');
        const third = reader.nextLine(1);
        const tw = words(third ?? '');
        if (tw.length !== 2) throw new StyleError(`UF3: Expected only 2 words on 3rd line => Rij_CUTOFF NUM_OF_KNOTS. Found ${tw.length} word/s`);
        cut2[pIdx(itype, jtype)] = toDouble(tw[0]);
        cut2[pIdx(jtype, itype)] = cut2[pIdx(itype, jtype)];
        const nk = toInt(tw[1]);
        nK2[pIdx(itype, jtype)] = nk;
        nK2[pIdx(jtype, itype)] = nk;
        maxK2 = Math.max(maxK2, nk);
        reader.skipLine();
        const fifth = reader.nextLine(1);
        const nc = toInt(words(fifth ?? '')[0]);
        if (!(nc > 0)) throw new StyleError(`UF3: 0 or negative number found for num_coeff_2b in ${fileName}`);
        nC2[pIdx(itype, jtype)] = nc;
        nC2[pIdx(jtype, itype)] = nc;
        maxC2 = Math.max(maxC2, nc);
      }
    } else if (nb === '3B' && pot3) {
      if (w.length !== 7) throw new StyleError(`UF3: Expected 7 words on a 3B header line of ${fileName} but found ${w.length} word/s`);
      const itype = find(w[1]), jtype = find(w[2]), ktype = find(w[3]);
      if (itype !== 0 && jtype !== 0 && ktype !== 0) {
        if (toInt(w[4]) !== 0) throw new StyleError('UF3: Current implementation is throughly tested only for leading_trim=0');
        if (toInt(w[5]) !== 3) throw new StyleError('UF3: Current implementation is throughly tested only for trailing_trim=3');
        const third = words(reader.nextLine(6) ?? '');
        if (third.length !== 6) throw new StyleError(`UF3: Expected only 6 numbers on 3rd line of a 3B block, found ${third.length}`);
        const rjk = toDouble(third[0]), rij = toDouble(third[1]), rik = toDouble(third[2]);
        if (rij !== rik) throw new StyleError(`UF3: rij!=rik for ${w[1]}-${w[2]}-${w[3]}. Current implementation only works for rij=rik`);
        if (2 * rik !== rjk) throw new StyleError(`UF3: 2rij=2rik!=rik for ${w[1]}-${w[2]}-${w[3]}. Current implementation only works for 2rij=2rik!=rik`);
        cutList3[pIdx(itype, jtype)] = Math.max(rij, cutList3[pIdx(itype, jtype)]);
        cutList3[pIdx(itype, ktype)] = Math.max(cutList3[pIdx(itype, ktype)], rik);
        cut3[idx3(N, itype, jtype, ktype)] = rij;
        cut3[idx3(N, itype, ktype, jtype)] = rik;
        const nJK = toInt(third[3]), nIK = toInt(third[4]), nIJ = toInt(third[5]);
        const k1 = key3(N, itype, jtype, ktype), k2 = key3(N, itype, ktype, jtype);
        nKnots3[k1 * 3] = nJK; nKnots3[k1 * 3 + 1] = nIK; nKnots3[k1 * 3 + 2] = nIJ;
        nKnots3[k2 * 3] = nJK; nKnots3[k2 * 3 + 1] = nIJ; nKnots3[k2 * 3 + 2] = nIK;
        maxK3 = Math.max(maxK3, nJK, nIK, nIJ);
        reader.skipLine();
        reader.skipLine();
        reader.skipLine();
        const shape = words(reader.nextLine(3) ?? '');
        if (shape.length !== 3) throw new StyleError(`UF3: Expected 3 numbers on the shape line of a 3B block, found ${shape.length}`);
        const d1 = toInt(shape[0]), d2 = toInt(shape[1]), d3 = toInt(shape[2]);
        sizeC3[k1 * 3] = d1; sizeC3[k1 * 3 + 1] = d2; sizeC3[k1 * 3 + 2] = d3;
        sizeC3[k2 * 3] = d2; sizeC3[k2 * 3 + 1] = d1; sizeC3[k2 * 3 + 2] = d3;
        maxC3 = Math.max(maxC3, d1, d2, d3);
      }
    } else if (!(nb === '3B' && !pot3)) {
      throw new StyleError(`UF3: Expected either '2B' or '3B' word in ${fileName} file`);
    }
  }
  if (maxK2 <= 0) throw new StyleError(`UF3: Error reading the size of 2B knot vector\nPossibly no 2B UF3 potential block detected in ${fileName} file`);
  if (maxC2 <= 0) throw new StyleError(`UF3: Error reading the size of 2B coeff vector\nPossibly no 2B UF3 potential block detected in ${fileName} file`);
  if (pot3 && maxK3 <= 0) throw new StyleError(`UF3: Error reading the size of 3B knot vector\nPossibly no 3B UF3 potential block detected in ${fileName} file`);
  if (pot3 && maxC3 <= 0) throw new StyleError(`UF3: Error reading the size of 3B coeff matrices\nPossibly no 3B UF3 potential block detected in ${fileName} file`);

  const knots3: Float64Array[] = [];
  const coeff3: Float64Array[] = [];
  if (pot3) {
    for (let q = 0; q < nKeys * 3; q++) knots3.push(new Float64Array(maxK3));
    for (let q = 0; q < nKeys; q++) coeff3.push(new Float64Array(maxC3 * maxC3 * maxC3));
  }

  // ---- pass 2: values ----
  reader.rewind();
  const fill2 = (i: number, j: number): UF3Pair2 => {
    const p = pIdx(i, j);
    const nk = nK2[p], nc = nC2[p];
    return { knots: new Float64Array(maxK2), coeff: new Float64Array(maxC2), nKnots: nk, nCoeff: nc, spacingType: 0, spacing: 0 };
  };
  for (let line = reader.nextLine(1); line !== null; line = reader.nextLine(1)) {
    if (!line.includes('#UF3 POT')) continue;
    const second = reader.nextLine(1);
    const w = words(second ?? '');
    const nb = w[0];
    if (nb === '2B') {
      const itype = find(w[1]), jtype = find(w[2]);
      const kt = w[5];
      if (kt !== 'uk' && kt !== 'nk') {
        throw new StyleError(`UF3: Expected either 'uk'(uniform-knots) or 'nk'(non-uniform knots). Found ${kt} on the 2nd line of ${w[1]}-${w[2]} interaction block`);
      }
      const type = kt === 'uk' ? 0 : 1;
      if (itype !== 0 && jtype !== 0) {
        reader.skipLine();
        const nk = nK2[pIdx(itype, jtype)];
        const v4 = words(reader.nextLine(nk) ?? '');
        if (v4.length !== nk) throw new StyleError(`UF3: Error reading the 2B potential block for ${w[1]}-${w[2]}: expected ${nk} numbers on the knot line, found ${v4.length}`);
        const a = fill2(itype, jtype), b = fill2(jtype, itype);
        for (let q = 0; q < nk; q++) {
          a.knots[q] = toDouble(v4[q]);
          b.knots[q] = a.knots[q];
        }
        const sp = a.knots[4] - a.knots[3];
        reader.skipLine();
        const nc = nC2[pIdx(itype, jtype)];
        const v6 = words(reader.nextLine(nc) ?? '');
        if (v6.length !== nc) throw new StyleError(`UF3: Error reading the 2B potential block for ${w[1]}-${w[2]}: expected ${nc} numbers on the coefficient line, found ${v6.length}`);
        for (let q = 0; q < nc; q++) {
          a.coeff[q] = toDouble(v6[q]);
          b.coeff[q] = a.coeff[q];
        }
        if (nk !== nc + 4) {
          throw new StyleError(`UF3: ${w[1]}-${w[2]} interaction block has incorrect knot and coeff data nknots (=${nk}) != ncoeffs (=${nc}) + 3 + 1`);
        }
        a.spacingType = type; a.spacing = sp;
        b.spacingType = type; b.spacing = sp;
        pair2[pIdx(itype, jtype)] = a;
        pair2[pIdx(jtype, itype)] = b;
        setflag2[pIdx(itype, jtype)] = 1;
        setflag2[pIdx(jtype, itype)] = 1;
      }
    } else if (nb === '3B' && pot3) {
      const itype = find(w[1]), jtype = find(w[2]), ktype = find(w[3]);
      const kt = w[6];
      if (kt !== 'uk' && kt !== 'nk') {
        throw new StyleError(`UF3: Expected either 'uk'(uniform-knots) or 'nk'(non-uniform knots) Found ${kt} on the 2nd line of ${w[1]}-${w[2]}-${w[3]} interaction block`);
      }
      const type = kt === 'uk' ? 0 : 1;
      if (itype !== 0 && jtype !== 0 && ktype !== 0) {
        const ijk = idx3(N, itype, jtype, ktype), ikj = idx3(N, itype, ktype, jtype);
        spacingType3[ijk] = type;
        spacingType3[ikj] = type;
        reader.skipLine();
        const k1 = key3(N, itype, jtype, ktype), k2 = key3(N, itype, ktype, jtype);
        const nJK = nKnots3[k1 * 3], nIK = nKnots3[k1 * 3 + 1], nIJ = nKnots3[k1 * 3 + 2];
        const K = (kk: number, d: number) => knots3[kk * 3 + d];
        // line 4: jk knots
        const v4 = words(reader.nextLine(nJK) ?? '');
        if (v4.length !== nJK) throw new StyleError(`UF3: Error reading the 3B potential block for ${w[1]}-${w[2]}-${w[3]}: expected ${nJK} numbers on the 4th line, found ${v4.length}`);
        for (let q = 0; q < nJK; q++) {
          K(k1, 0)[q] = toDouble(v4[q]);
          K(k2, 0)[q] = K(k1, 0)[q];
        }
        minCut3[ijk * 3] = K(k1, 0)[0];
        minCut3[ikj * 3] = K(k2, 0)[0];
        spacing3[ijk * 3] = K(k1, 0)[4] - K(k1, 0)[3];
        spacing3[ikj * 3] = spacing3[ijk * 3];
        // line 5: ik knots
        const v5 = words(reader.nextLine(nIK) ?? '');
        if (v5.length !== nIK) throw new StyleError(`UF3: Error reading the 3B potential block for ${w[1]}-${w[2]}-${w[3]}: expected ${nIK} numbers on the 5th line, found ${v5.length}`);
        for (let q = 0; q < nIK; q++) {
          K(k1, 1)[q] = toDouble(v5[q]);
          K(k2, 2)[q] = K(k1, 1)[q];
        }
        minCut3[ijk * 3 + 1] = K(k1, 1)[0];
        minCut3[ikj * 3 + 2] = K(k2, 2)[0];
        spacing3[ijk * 3 + 1] = K(k1, 1)[4] - K(k1, 1)[3];
        spacing3[ikj * 3 + 2] = spacing3[ijk * 3 + 1];
        // line 6: ij knots
        const v6 = words(reader.nextLine(nIJ) ?? '');
        if (v6.length !== nIJ) throw new StyleError(`UF3: Error reading the 3B potential block for ${w[1]}-${w[2]}-${w[3]}: expected ${nIJ} numbers on the 6th line, found ${v6.length}`);
        for (let q = 0; q < nIJ; q++) {
          K(k1, 2)[q] = toDouble(v6[q]);
          K(k2, 1)[q] = K(k1, 2)[q];
        }
        minCut3[ijk * 3 + 2] = K(k1, 2)[0];
        minCut3[ikj * 3 + 1] = K(k2, 1)[0];
        spacing3[ijk * 3 + 2] = K(k1, 2)[4] - K(k1, 2)[3];
        spacing3[ikj * 3 + 1] = spacing3[ijk * 3 + 2];
        reader.skipLine();
        const d1 = sizeC3[k1 * 3], d2 = sizeC3[k1 * 3 + 1], d3 = sizeC3[k1 * 3 + 2];
        if (nJK !== d3 + 4) throw new StyleError(`UF3: ${w[1]}-${w[2]}-${w[3]} interaction block has incorrect knot (NUM_OF_KNOTS_JK) and coeff (coeff_matrix_dim3) data nknots!=ncoeffs + 3 + 1`);
        if (nIK !== d2 + 4) throw new StyleError(`UF3: ${w[1]}-${w[2]}-${w[3]} interaction block has incorrect knot (NUM_OF_KNOTS_IK) and coeff (coeff_matrix_dim2) data nknots!=ncoeffs + 3 + 1`);
        if (nIJ !== d1 + 4) throw new StyleError(`UF3: ${w[1]}-${w[2]}-${w[3]} interaction block has incorrect knot (NUM_OF_KNOTS_IJ) and coeff (coeff_matrix_dim1) data nknots!=ncoeffs + 3 + 1`);
        const M = maxC3;
        const c1 = coeff3[k1], c2 = coeff3[k2];
        for (let i = 0; i < d1; i++) {
          for (let j = 0; j < d2; j++) {
            const row = words(reader.nextLine(d3) ?? '');
            if (row.length !== d3) throw new StyleError(`UF3: Error reading 3B potential block for ${w[1]}-${w[2]}-${w[3]}: expected ${d3} numbers, found ${row.length}`);
            for (let k = 0; k < d3; k++) c1[(i * M + j) * M + k] = toDouble(row[k]);
          }
        }
        // symmetric copy (native: key2 gets [j][i][k]; in place when key1 == key2)
        for (let i = 0; i < d1; i++) {
          for (let j = 0; j < d2; j++) {
            for (let k = 0; k < d3; k++) c2[(j * M + i) * M + k] = c1[(i * M + j) * M + k];
          }
        }
        setflag3[ijk] = 1;
        setflag3[ikj] = 1;
      }
    }
  }

  // ---- fill the element tuples the file does not list (native: map[] of the types) ----
  for (let i = 1; i <= N; i++) {
    for (let j = 1; j <= N; j++) {
      const p = pIdx(i, j);
      if (setflag2[p] === 1) continue;
      const iM = mapOf[i] + 1, jM = mapOf[j] + 1;
      if (iM === i && jM === j) {
        throw new StyleError(`UF3: Potential for interaction ${i}-${j} ie ${elementOfType[i]}-${elementOfType[j]} not found in ${fileName} file`);
      }
      const q = pIdx(iM, jM);
      cut2[p] = cut2[q];
      const src = pair2[q];
      if (!src) throw new StyleError(`UF3: Potential for interaction ${i}-${j} not found in ${fileName} file`);
      const dst: UF3Pair2 = { knots: new Float64Array(maxK2), coeff: new Float64Array(maxC2), nKnots: src.nKnots, nCoeff: src.nCoeff, spacingType: src.spacingType, spacing: src.spacing };
      dst.knots.set(src.knots);
      dst.coeff.set(src.coeff);
      pair2[p] = dst;
      setflag2[p] = 1;
    }
  }
  if (pot3) {
    for (let i = 1; i <= N; i++) {
      for (let j = 1; j <= N; j++) {
        for (let k = 1; k <= N; k++) {
          const ijk = idx3(N, i, j, k);
          if (setflag3[ijk] === 1) continue;
          const iM = mapOf[i] + 1, jM = mapOf[j] + 1, kM = mapOf[k] + 1;
          if (iM === i && jM === j && kM === k) {
            throw new StyleError(`UF3: Potential for interaction ${i}-${j}-${k} ie ${elementOfType[i]}-${elementOfType[j]}-${elementOfType[k]} not found in ${fileName} file`);
          }
          const mq = idx3(N, iM, jM, kM);
          if (setflag3[mq] !== 1) {
            throw new StyleError(`UF3: Interaction ${i}-${j}-${k} was mapped to ${iM}-${jM}-${kM}, but potential interaction for ${iM}-${jM}-${kM} was not found in ${fileName} file`);
          }
          cutList3[pIdx(i, j)] = Math.max(cutList3[pIdx(iM, jM)], cutList3[pIdx(i, j)]);
          cut3[ijk] = cut3[mq];
          spacingType3[ijk] = spacingType3[mq];
          for (let d = 0; d < 3; d++) spacing3[ijk * 3 + d] = spacing3[mq * 3 + d];
          const key = key3(N, i, j, k), mk = key3(N, iM, jM, kM);
          for (let d = 0; d < 3; d++) nKnots3[key * 3 + d] = nKnots3[mk * 3 + d];
          for (let d = 0; d < 3; d++) sizeC3[key * 3 + d] = sizeC3[mk * 3 + d];
          for (let d = 0; d < 3; d++) minCut3[ijk * 3 + d] = minCut3[mq * 3 + d];
          for (let d = 0; d < 3; d++) {
            for (let q = 0; q < nKnots3[key * 3 + d]; q++) knots3[key * 3 + d][q] = knots3[mk * 3 + d][q];
          }
          const M = maxC3;
          for (let a = 0; a < sizeC3[key * 3]; a++) {
            for (let b = 0; b < sizeC3[key * 3 + 1]; b++) {
              for (let c = 0; c < sizeC3[key * 3 + 2]; c++) {
                coeff3[key][(a * M + b) * M + c] = coeff3[mk][(a * M + b) * M + c];
              }
            }
          }
          setflag3[ijk] = 1;
        }
      }
    }
  }
  return {
    ntypes: N, nbody3: pot3, cut2, setflag2, pair2, maxK2, maxC2, cutList3, cut3, setflag3, spacingType3,
    minCut3, spacing3, nKnots3, sizeC3, knots3, coeff3, maxK3, maxC3,
  };
}
