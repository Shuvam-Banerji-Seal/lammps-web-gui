/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/ML-IAP/compute_mliap.cpp
 * (ComputeMLIAP: the constructor, compute_array and dbdotr_compute), with the parts of mliap_data.cpp
 * (generate_neighdata), mliap_model_linear.cpp and mliap_model_quadratic.cpp (compute_gradgrads and
 * compute_force_gradients) and mliap_descriptor_ace.cpp (compute_force_gradients and compute_descriptor_gradients)
 * that the array of compute mliap with descriptor ace uses, at commit b332f28. Authors named in the sources: Aidan
 * Thompson (Sandia National Laboratories) and James Goff (descriptor ace), with the LAMMPS Sandia copyright notice
 * (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * compute ID group-ID mliap model linear|quadratic descriptor ace file [gradgradflag 0|1]
 * (docs.lammps.org/compute_mliap.html). The array has 1 + 3N + 6 rows and nelements * nparams + 1 columns: the
 * energy gradient with respect to the model parameters (row 0), the force gradients of each atom ID (rows
 * 3 (id-1) + 1 + d, ghosts folded onto their atom), and the virial sums x . dF/dparam in Voigt order; the last
 * column holds the potential energy, the forces and the pressure (compute pressure NULL virial). As native, every
 * atom contributes (the group is not used), type t is element t - 1, the neighbours come from an occasional full
 * list at the pair cutoff plus skin, and gradgradflag 1 (the default) sums the sparse double gradients of the model
 * while 0 sums the descriptor gradients.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { NEIGHMASK } from '../../engine/neighbor';
import type { MliapAceDescriptor } from '../../engine/mliap/ace';
import { NativeFullOrder } from '../neighbor/nativeOrder';

export class ComputeMliapAce extends Compute {
  readonly style = 'mliap';
  private readonly quadratic: boolean;
  private readonly gradgradflag: number;
  private readonly natoms: number;
  private readonly nparams: number;
  private readonly natOrder = new NativeFullOrder();

  constructor(sys: System, id: string, group: string, args: string[], private readonly ace: MliapAceDescriptor) {
    super(sys, id, group, args);
    this.arrayFlag = true;
    this.extarray = 0;
    let gradgradflag = 1;
    let model: string | null = null, descriptor = false;
    let iarg = 0;
    while (iarg < args.length) {
      const kw = args[iarg];
      if (kw === 'model') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal compute mliap model command: missing argument(s)');
        const m = args[iarg + 1];
        if (m === 'mliappy') throw new StyleError("Must enable PYTHON package and -DMLIAP_PYTHON setting to use 'mliappy' model");
        if (m !== 'linear' && m !== 'quadratic') throw new StyleError(`Unknown compute mliap model ${m}`);
        model = m;
        iarg += 2;
      } else if (kw === 'descriptor') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal compute mliap descriptor command: missing argument(s)');
        const d = args[iarg + 1];
        if (d !== 'ace') throw new StyleError(`compute mliap: descriptor ${d} together with descriptor ace`);
        if (iarg + 3 > args.length) throw new StyleError('Illegal compute mliap descriptor ace command: missing argument(s)');
        descriptor = true;
        iarg += 3;
      } else if (kw === 'gradgradflag') {
        const v = args[iarg + 1];
        // utils::logical: yes/no, on/off, true/false or 1/0
        if (v === 'yes' || v === 'on' || v === 'true' || v === '1') gradgradflag = 1;
        else if (v === 'no' || v === 'off' || v === 'false' || v === '0') gradgradflag = 0;
        // a missing value is read as NULL by utils::logical (src/compute_mliap.cpp reads arg[narg] past the end)
        else if (v === undefined) throw new StyleError('Expected boolean parameter instead of NULL or empty string in input script or data file');
        else throw new StyleError(`Expected boolean parameter instead of '${v}' in input script or data file`);
        iarg += 2;
      } else {
        throw new StyleError(`Unknown compute mliap keyword ${kw}`);
      }
    }
    if (model === null || !descriptor) throw new StyleError('Illegal compute_style command');
    this.quadratic = model === 'quadratic';
    this.gradgradflag = gradgradflag;
    const K = ace.ndescriptors;
    this.nparams = this.quadratic ? 1 + K + (K * (K + 1)) / 2 : 1 + K;
    this.natoms = sys.state.n;
    this.sizeArrayRows = 1 + 3 * this.natoms + 6;
    this.sizeArrayCols = this.nparams * ace.elements.length + 1;
  }

  init(): void {
    if (!this.sys.ff.pair) throw new StyleError('Compute mliap requires a pair style be defined');
  }

  protected computeArray(): void {
    const sys = this.sys, s = sys.state, nb = sys.nb, ace = this.ace;
    const acc = sys.forces();
    const n = s.n, natoms = this.natoms;
    if (n !== natoms) throw new StyleError(`compute mliap: the atom count changed since the compute was defined (${natoms} then, ${n} now)`);
    const nel = ace.elements.length, K = ace.ndescriptors, nparams = this.nparams;
    const rows = this.sizeArrayRows, cols = this.sizeArrayCols, lastcol = cols - 1;
    const nall = nb.nall, xa = nb.xall, ta = nb.typeall, owner = nb.owner;
    const tag = (k: number): number => {
      const t = s.id[owner[k]];
      if (t < 1 || t > natoms) throw new StyleError(`compute mliap: atom ID ${t} is outside 1..${natoms} (the array is indexed by atom ID)`);
      return t;
    };
    for (let t = 1; t <= s.ntypes; t++) {
      if (t > nel) throw new StyleError(`compute mliap: atom type ${t} has no element (type t is element t-1 of the ${nel} in the descriptor file)`);
    }
    const arr = new Float64Array(rows * cols);
    const yoffset = nparams * nel, zoffset = 2 * yoffset;
    const sizeGrad = 3 * nparams * nel;
    const gradforce = new Float64Array(nall * sizeGrad);
    const egradient = new Float64Array(nel * nparams);
    const list = nb.occasionalList(s, nb.cutneighmax, true, s.ntypes);
    const nbrs = this.natOrder.neighbors(s, nb, sys.geom, list);
    let jl = new Int32Array(64);
    const quad = this.quadratic;
    for (let i = 0; i < n; i++) {
      const ielem = ta[i] - 1;
      const xi = xa[3 * i], yi = xa[3 * i + 1], zi = xa[3 * i + 2];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      if (jl.length < k1 - k0) jl = new Int32Array(k1 - k0);
      // generate_neighdata: the neighbours inside cutsq of the two elements
      let m = 0;
      for (let k = k0; k < k1; k++) {
        const j = nbrs[k] & NEIGHMASK;
        const dx = xa[3 * j] - xi, dy = xa[3 * j + 1] - yi, dz = xa[3 * j + 2] - zi;
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq < ace.cutsq(ielem, ta[j] - 1)) jl[m++] = j;
      }
      const { B, dB } = ace.evaluate(i, xa, ta, m, jl);
      const eo = nparams * ielem;
      const gi = i * sizeGrad;
      for (let jj = 0; jj < m; jj++) {
        const gj = jl[jj] * sizeGrad;
        const add = (l: number, g: number, k: number): void => {
          const o = (k * m + jj) * 3;
          const fx = g * dB[o], fy = g * dB[o + 1], fz = g * dB[o + 2];
          gradforce[gi + l] += fx; gradforce[gi + l + yoffset] += fy; gradforce[gi + l + zoffset] += fz;
          gradforce[gj + l] -= fx; gradforce[gj + l + yoffset] -= fy; gradforce[gj + l + zoffset] -= fz;
        };
        if (this.gradgradflag === 1) {
          // descriptor compute_force_gradients over the model's sparse double gradients (gamma, row l, column k)
          let l = eo + 1;
          for (let k = 0; k < K; k++) add(l++, 1.0, k);
          if (quad) {
            for (let k = 0; k < K; k++) {
              add(l, B[k], k);
              l++;
              for (let k2 = k + 1; k2 < K; k2++) {
                add(l, B[k2], k);
                add(l, B[k], k2);
                l++;
              }
            }
          }
        } else {
          // model compute_force_gradients over the descriptor gradients
          let l = eo + 1;
          for (let k = 0; k < K; k++) {
            const o = (k * m + jj) * 3;
            gradforce[gi + l] += dB[o]; gradforce[gi + l + yoffset] += dB[o + 1]; gradforce[gi + l + zoffset] += dB[o + 2];
            gradforce[gj + l] -= dB[o]; gradforce[gj + l + yoffset] -= dB[o + 1]; gradforce[gj + l + zoffset] -= dB[o + 2];
            l++;
          }
          if (quad) {
            for (let k = 0; k < K; k++) {
              const bi = B[k], ok = (k * m + jj) * 3;
              for (let d = 0; d < 3; d++) {
                const v = dB[ok + d] * bi;
                gradforce[gi + l + d * yoffset] += v;
                gradforce[gj + l + d * yoffset] -= v;
              }
              l++;
              for (let k2 = k + 1; k2 < K; k2++) {
                const bj = B[k2], ok2 = (k2 * m + jj) * 3;
                for (let d = 0; d < 3; d++) {
                  const v = dB[ok + d] * bj + dB[ok2 + d] * bi;
                  gradforce[gi + l + d * yoffset] += v;
                  gradforce[gj + l + d * yoffset] -= v;
                }
                l++;
              }
            }
          }
        }
      }
      // gradient of the energy of atom i with respect to the parameters
      let l = eo;
      egradient[l++] += 1.0;
      for (let k = 0; k < K; k++) egradient[l++] += B[k];
      if (quad) {
        for (let k = 0; k < K; k++) {
          const bi = B[k];
          egradient[l++] += 0.5 * bi * bi;
          for (let k2 = k + 1; k2 < K; k2++) egradient[l++] += bi * B[k2];
        }
      }
    }

    // force gradients of every atom and ghost, folded onto the rows of its atom ID
    const np = nparams * nel;
    for (let e = 0; e < nel; e++) {
      const eo = nparams * e;
      for (let p = 0; p < nparams; p++) {
        for (let i = 0; i < nall; i++) {
          const g = i * sizeGrad + eo + p;
          const r = 3 * (tag(i) - 1) + 1;
          arr[r * cols + p + eo] += gradforce[g];
          arr[(r + 1) * cols + p + eo] += gradforce[g + yoffset];
          arr[(r + 2) * cols + p + eo] += gradforce[g + zoffset];
        }
      }
    }
    for (let i = 0; i < n; i++) {
      const r = 3 * (tag(i) - 1) + 1;
      arr[r * cols + lastcol] = s.f[3 * i];
      arr[(r + 1) * cols + lastcol] = s.f[3 * i + 1];
      arr[(r + 2) * cols + lastcol] = s.f[3 * i + 2];
    }
    // dbdotr_compute: sum over atoms and ghosts of x times the force gradient
    const irow0 = 1 + 3 * natoms;
    for (let i = 0; i < nall; i++) {
      const x0 = xa[3 * i], x1 = xa[3 * i + 1], x2 = xa[3 * i + 2];
      for (let e = 0; e < nel; e++) {
        const eo = nparams * e;
        for (let p = 0; p < nparams; p++) {
          const g = i * sizeGrad + eo + p;
          const dbdx = gradforce[g], dbdy = gradforce[g + yoffset], dbdz = gradforce[g + zoffset];
          const col = p + eo;
          arr[irow0 * cols + col] += dbdx * x0;
          arr[(irow0 + 1) * cols + col] += dbdy * x1;
          arr[(irow0 + 2) * cols + col] += dbdz * x2;
          arr[(irow0 + 3) * cols + col] += dbdz * x1;
          arr[(irow0 + 4) * cols + col] += dbdz * x0;
          arr[(irow0 + 5) * cols + col] += dbdy * x0;
        }
      }
    }
    for (let c = 0; c < np; c++) arr[c] = egradient[c];

    // the reference energy (thermo_pe) and virial (compute pressure NULL virial) in the last column
    arr[lastcol] = acc.evdwl + acc.ecoul + acc.ebond + acc.eangle + acc.edihed + acc.eimp + acc.elong + sys.fixEnergy();
    const vol = sys.geom.volume(sys.dimension);
    const w = new Float64Array(6);
    const addv = (v: Float64Array): void => { for (let c = 0; c < 6; c++) w[c] += v[c]; };
    addv(acc.virial); addv(acc.vbond); addv(acc.vangle); addv(acc.vdihed); addv(acc.vimp); addv(acc.vlong);
    sys.fixVirial(w);
    const tail = sys.ff.ptail(vol);
    const press = new Float64Array(6);
    for (let c = 0; c < 6; c++) {
      let v = w[c] / vol;
      if (c < 3) v += tail;
      press[c] = v * s.units.nktv2p;
    }
    // pressure vector xx yy zz xy xz yz -> Voigt rows xx yy zz yz xz xy
    const voigt = [0, 1, 2, 5, 4, 3];
    for (let c = 0; c < 6; c++) arr[(irow0 + c) * cols + lastcol] = press[voigt[c]];
    this.array = arr;
  }
}
