/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/PHONON/fix_phonon.cpp and fix_phonon.h
 * (class FixPhonon, fix phonon: the force-constant matrix and the dynamical matrix of a crystal from the displacement
 * correlations of its atoms, Phi(q) = kB T G(q)^-1 with G(q) = <R_q R_q^*> - <R>_q <R>_q^*). Contributing author:
 * Ling-Ti Kong (Shanghai Jiao Tong University). LAMMPS is Copyright (2003) Sandia Corporation and distributed under
 * the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Documented behaviour (docs.lammps.org/fix_phonon.html): "fix ID group-ID phonon Nevery Nrepeat Nfreq ..." is
 * replaced here by the arguments of the command as this engine parses them: nevery, nfreq, waitsteps, the map file
 * (a mapping of the lattice indices l1 l2 l3 and the basis index k to atom IDs, or GAMMA), the output prefix, and the
 * keywords sysdim and nasr. The text log prefix.log holds the dynamical matrix after every nfreq measurements.
 *
 * Browser scope: the binary file prefix.bin.N of the force constants is native-only; it is not written, and the
 * fix says so once (a named warning, not a silent skip). Its basis-vector sums are only needed for that file.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { missingArgs, parseInt_ } from '../../engine/force/util';
import type { System } from '../../engine/system';
import type { Compute } from '../../engine/compute/compute';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

/** The map file name that makes every group atom its own cell (the gamma-point cluster). */
const GAMMA_MAP = 'GAMMA';

/** Complex number as a [re, im] pair, the layout the fix uses for Phi and R(q). */
type Cx = [number, number];

const cmul = (a: Cx, b: Cx): Cx => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const cadd = (a: Cx, b: Cx): Cx => [a[0] + b[0], a[1] + b[1]];
const csub = (a: Cx, b: Cx): Cx => [a[0] - b[0], a[1] - b[1]];
const cconj = (a: Cx): Cx => [a[0], -a[1]];
const cnorm = (a: Cx): number => a[0] * a[0] + a[1] * a[1];
const cdiv = (a: Cx, b: Cx): Cx => {
  const d = b[0] * b[0] + b[1] * b[1];
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d];
};

/**
 * 3d DFT of an nx x ny x nz grid (the data index is (ix * ny + iy) * nz + iz), unnormalised, with the kernel
 * exp(+2 pi i k.x / N). fix_phonon.cpp calls its fft3d with FORWARD = -1, which is the backward kernel of that
 * wrapper (measured with native LAMMPS, black box: the imaginary parts of Phi(q) have this sign). Separable 1d transforms.
 */
const dft3 = (data: Cx[], nx: number, ny: number, nz: number): Cx[] => {
  const dft1 = (buf: Cx[], n: number, stride: number, count: number, base: (c: number) => number): void => {
    const tmp: Cx[] = new Array(n);
    for (let c = 0; c < count; c++) {
      const b = base(c);
      for (let k = 0; k < n; k++) {
        let re = 0, im = 0;
        for (let j = 0; j < n; j++) {
          const ph = (2 * Math.PI * ((j * k) % n)) / n;
          const x = buf[b + j * stride];
          // exp(+i ph) times x: the sign of the transform of the fix's fft3d call (FORWARD = -1 there)
          re += x[0] * libmCos(ph) - x[1] * libmSin(ph);
          im += x[1] * libmCos(ph) + x[0] * libmSin(ph);
        }
        tmp[k] = [re, im];
      }
      for (let k = 0; k < n; k++) buf[b + k * stride] = tmp[k];
    }
  };
  const out = data.slice();
  // z (stride 1), then y (stride nz), then x (stride ny nz)
  dft1(out, nz, 1, nx * ny, (c) => c * nz);
  dft1(out, ny, nz, nx * nz, (c) => (Math.floor(c / nz) * ny * nz) + (c % nz));
  dft1(out, nx, ny * nz, ny * nz, (c) => c);
  return out;
};

export class FixPhonon extends Fix {
  readonly style = 'phonon';
  private readonly waitsteps: number;
  private readonly nfreq: number;
  private readonly prefix: string;
  private readonly logfile: string;
  private readonly mapfile: string;
  private sysdim: number;
  private nasr: number;
  private nx = 1;
  private ny = 1;
  private nz = 1;
  private nucell = 1;
  private ntotal = 1;
  private ngroup = 0;
  private fftDim = 0;
  private fftDim2 = 0;
  /** tag -> surface index and back (the surface index is (cell * nucell + basis index) in the native order) */
  private tag2surf = new Map<number, number>();
  private surf2tag = new Map<number, number>();
  private massInvSqrt = new Float64Array(0);
  private prevNstep = 0;
  private neval = 0;
  private ifreq = 0;
  private tempSum: Float64Array;
  private invNTemp = 1;
  private temp: Compute | null = null;
  private tempId = 'thermo_temp';
  private hsum = new Float64Array(6);
  /** Sum of the unwrapped R(r) per surface index and component (fft_dim per cell). */
  private rsum: Float64Array = new Float64Array(0);
  /** Sum of R(q)_a conj(R(q)_b) per q: fftDim2 complex values per cell. */
  private rqsum: Cx[] = [];
  private warnedBinary = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 5) throw new StyleError('Illegal fix phonon command: usage fix ID group phonon nevery nfreq waitsteps mapfile prefix [sysdim N] [nasr N]');
    this.nevery = parseInt_(args[0], 'fix phonon nevery');
    if (this.nevery < 1) throw new StyleError(`Illegal fix phonon nevery value ${this.nevery}`);
    this.nfreq = parseInt_(args[1], 'fix phonon nfreq');
    if (this.nfreq < 1) throw new StyleError(`Illegal fix phonon nfreq value ${this.nfreq}`);
    this.waitsteps = parseInt_(args[2], 'fix phonon waitsteps');
    if (this.waitsteps < 0) throw new StyleError(`Illegal fix phonon waitsteps value ${this.waitsteps}`);
    this.mapfile = args[3];
    this.prefix = args[4];
    this.logfile = `${this.prefix}.log`;
    const dim = sys.state.dimension;
    let sdim: number = dim;
    this.sysdim = dim as number;
    this.nasr = 20;
    for (let k = 5; k < args.length;) {
      const key = args[k];
      const val = args[k + 1];
      // As native LAMMPS does (src/PHONON/fix_phonon.cpp FixPhonon::FixPhonon at b332f28): an unknown keyword is refused
      // before its value is looked for; a known keyword without its value is "Illegal fix phonon {keyword} command: missing argument(s)"
      if (key !== 'sysdim' && key !== 'nasr') throw new StyleError(`Unknown fix phonon keyword ${key}`);
      if (val === undefined) throw missingArgs(`fix phonon ${key}`);
      if (key === 'sysdim') {
        sdim = parseInt_(val, 'fix phonon sysdim');
        if (sdim < 1) throw new StyleError('Illegal fix phonon command: sysdim should not be less than 1');
      } else {
        this.nasr = parseInt_(val, 'fix phonon nasr');
      }
      k += 2;
    }
    if (sdim < this.sysdim) this.sysdim = sdim;
    this.nasr = Math.max(0, this.nasr);
    let ng = 0;
    for (let i = 0; i < sys.state.n; i++) if (sys.state.mask[i] & this.groupBit) ng++;
    if (ng < 1) throw new StyleError(`No atoms in fix phonon group ${group}`);
    this.ngroup = ng;
    this.readmap(sys);
    if (this.nucell === 1) this.nasr = Math.min(1, this.nasr);
    this.getmass(sys);
    this.fftDim = this.nucell * this.sysdim;
    this.fftDim2 = this.fftDim * this.fftDim;
    const ncell = this.nx * this.ny * this.nz;
    this.rsum = new Float64Array(ncell * this.fftDim);
    this.rqsum = Array.from({ length: ncell * this.fftDim2 }, () => [0, 0] as Cx);
    this.tempSum = new Float64Array(this.sysdim);
    // the log: the header and the mapping (the fix writes it when it is defined)
    sys.writeFile(this.logfile, this.headerText(), false);
  }

  private countGroup(c: Compute): number {
    const s = this.sys.state;
    let n = 0;
    for (let i = 0; i < s.n; i++) if (s.mask[i] & c.groupBit) n++;
    return n;
  }

  private headerText(): string {
    const lines: string[] = [];
    const bar = '############################################################';
    lines.push(bar);
    lines.push(`# group name of the atoms under study      : ${this.group}`);
    lines.push(`# total number of atoms in the group       : ${this.ngroup}`);
    lines.push(`# dimension of the system                  : ${this.sysdim} D`);
    lines.push(`# number of atoms per unit cell            : ${this.nucell}`);
    lines.push(`# dimension of the FFT mesh                : ${this.nx} x ${this.ny} x ${this.nz}`);
    lines.push(`# number of wait steps before measurement  : ${this.waitsteps}`);
    lines.push(`# frequency of the measurement             : ${this.nevery}`);
    lines.push(`# output result after this many measurement: ${this.nfreq}`);
    lines.push('# number of processors used by this run    : 1');
    lines.push(bar);
    lines.push('# mapping information between lattice indices and atom id');
    lines.push('# nx ny nz nucell');
    lines.push(`${this.nx} ${this.ny} ${this.nz} ${this.nucell}`);
    lines.push('# l1 l2 l3 k atom_id');
    for (let idx = 0; idx < this.ngroup; idx++) {
      const itag = this.surf2tag.get(idx) ?? 0;
      const iu = idx % this.nucell;
      const iz = Math.floor(idx / this.nucell) % this.nz;
      const iy = Math.floor(idx / (this.nucell * this.nz)) % this.ny;
      const ix = Math.floor(idx / (this.nucell * this.nz * this.ny)) % this.nx;
      lines.push(`${ix} ${iy} ${iz} ${iu} ${itag}`);
    }
    lines.push(bar);
    return lines.join('\n') + '\n';
  }

  /** getmass(): M_inv_sqrt per basis atom, from the mean mass of the group atoms of each basis index. */
  private getmass(sys: System): void {
    const s = sys.state;
    const massOne = new Float64Array(this.nucell);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const idx = this.tag2surf.get(s.id[i]);
      if (idx === undefined) throw new StyleError('The mapping info read is incorrect');
      const iu = idx % this.nucell;
      massOne[iu] += s.rmass ? s.rmass[i] : s.massByType[s.type[i]];
    }
    const invTotal = 1 / this.ntotal;
    this.massInvSqrt = new Float64Array(this.nucell);
    for (let i = 0; i < this.nucell; i++) {
      const m = massOne[i] * invTotal;
      this.massInvSqrt[i] = Math.sqrt(1 / m);
    }
  }

  /** readmap(): the lattice indices of every group atom (fix_phonon.cpp readmap: the header nx ny nz nucell, then a comment). */
  private readmap(sys: System): void {
    if (this.mapfile === GAMMA_MAP) {
      this.nx = this.ny = this.nz = this.ntotal = 1;
      this.nucell = this.ngroup;
      const s = sys.state;
      let k = 0;
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & this.groupBit)) continue;
        this.tag2surf.set(s.id[i], k);
        this.surf2tag.set(k, s.id[i]);
        k++;
      }
      return;
    }
    const lines = sys.readFile(this.mapfile).split('\n');
    const head = (lines[0] ?? '').trim().split(/\s+/).map(Number);
    if (head.length < 4 || head.slice(0, 4).some((v) => !Number.isInteger(v))) throw new StyleError('Incorrect header format in the fix phonon map file');
    [this.nx, this.ny, this.nz, this.nucell] = head;
    this.ntotal = this.nx * this.ny * this.nz;
    if (this.ntotal * this.nucell !== this.ngroup) throw new StyleError('FFT mesh and number of atoms in group mismatch');
    for (let i = 0; i < this.ngroup; i++) {
      const w = (lines[2 + i] ?? '').trim().split(/\s+/).map(Number);
      if (w.length < 5 || w.some((v) => !Number.isFinite(v))) throw new StyleError('Error while reading mapping file');
      const [ix, iy, iz, iu, itag] = w;
      if (ix < 0 || ix >= this.nx || iy < 0 || iy >= this.ny || iz < 0 || iz >= this.nz || iu < 0 || iu >= this.nucell) {
        throw new StyleError('Error while reading mapping file');
      }
      const idx = ((ix * this.ny + iy) * this.nz + iz) * this.nucell + iu;
      this.tag2surf.set(itag, idx);
      this.surf2tag.set(idx, itag);
    }
    if (this.tag2surf.size !== this.ngroup || this.surf2tag.size !== this.ngroup) throw new StyleError('The mapping is incomplete');
  }

  modify(key: string, values: string[]): number {
    if (key === 'temp') {
      const c = this.sys.compute(values[0] ?? '');
      if (!c.tempFlag) throw new StyleError(`Fix_modify temp ID ${values[0]} does not compute temperature`);
      this.temp = c;
      this.tempId = values[0];
      return 1;
    }
    return super.modify(key, values);
  }

  /** The temperature compute (thermo_temp unless fix_modify temp) and its atom count, at the start of each run. */
  init(): void {
    this.temp = this.sys.compute(this.tempId);
    this.invNTemp = 1 / this.countGroup(this.temp);
  }

  /** Accumulators are reset at the start of each run (fix_phonon.cpp setup). */
  setup(): void {
    this.tempSum.fill(0);
    this.rsum.fill(0);
    for (const q of this.rqsum) { q[0] = 0; q[1] = 0; }
    this.hsum.fill(0);
    this.neval = 0;
    this.ifreq = 0;
    this.prevNstep = this.sys.state.step;
  }

  endOfStep(): void {
    const s = this.sys.state;
    if (s.step - this.prevNstep <= this.waitsteps) return;
    const sys = this.sys;
    sys.refreshComputes();
    const tv = this.temp!.vectorValues();
    for (let d = 0; d < this.sysdim; d++) this.tempSum[d] += tv[d];
    // unwrapped positions of the group atoms, stored per surface index (native Rsort) and then per cell and basis
    const ncell = this.nx * this.ny * this.nz;
    const fd = this.fftDim;
    const sort = new Float64Array(this.ngroup * this.sysdim);
    const u: number[] = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const idx = this.tag2surf.get(s.id[i]);
      if (idx === undefined) throw new StyleError('The mapping info read is incorrect');
      sys.geom.unwrap(s.x, s.image, i, u);
      for (let d = 0; d < this.sysdim; d++) sort[idx * this.sysdim + d] = u[d];
    }
    // Rnow[cell][iu * sysdim + d] = Rsort[cell * nucell + iu][d]; Rsum accumulates it
    const rnow = new Float64Array(ncell * fd);
    for (let k = 0; k < ncell * fd; k++) rnow[k] = sort[k];
    for (let k = 0; k < ncell * fd; k++) this.rsum[k] += rnow[k];
    // R(q) of every component: 3d DFT over the cells
    const rq = this.transformComponents(rnow, ncell, fd);
    // accumulate R(q)_a conj(R(q)_b)
    for (let q = 0; q < ncell; q++) {
      let ndim = 0;
      for (let a = 0; a < fd; a++) {
        for (let b = 0; b < fd; b++) {
          this.rqsum[q * this.fftDim2 + ndim] = cadd(this.rqsum[q * this.fftDim2 + ndim], cmul(rq[q * fd + a], cconj(rq[q * fd + b])));
          ndim++;
        }
      }
    }
    // domain->h: xprd, yprd, zprd, yz, xz, xy
    const g = sys.geom, b = s.box;
    const h = [g.lx, g.ly, g.lz, b.tilt[2], b.tilt[1], b.tilt[0]];
    for (let d = 0; d < 6; d++) this.hsum[d] += h[d];
    this.neval++;
    if (++this.ifreq === this.nfreq) this.postprocess();
  }

  /** The 3d transform of each component of a cell-major array: out[q * fd + m] = FFT(in[., m])[q]. */
  private transformComponents(data: Float64Array, ncell: number, fd: number): Cx[] {
    const out: Cx[] = new Array(ncell * fd);
    for (let m = 0; m < fd; m++) {
      const grid: Cx[] = new Array(ncell);
      for (let c = 0; c < ncell; c++) grid[c] = [data[c * fd + m], 0];
      const f = dft3(grid, this.nx, this.ny, this.nz);
      for (let c = 0; c < ncell; c++) out[c * fd + m] = f[c];
    }
    return out;
  }

  /** postprocess(): Phi(q) = kB T G(q)^-1 with G(q) = <R_q R_q*> - <R>_q <R>_q*, then D = M^-1/2 Phi M^-1/2. */
  private postprocess(): void {
    if (this.neval < 1) return;
    this.ifreq = 0;
    const s = this.sys.state;
    const fd = this.fftDim;
    const fd2 = this.fftDim2;
    const ncell = this.nx * this.ny * this.nz;
    const invNeval = 1 / this.neval;
    const boltz = s.units.boltz;
    // Phi(q) = <R_q R_q*>
    const phi: Cx[] = this.rqsum.map((v) => [v[0] * invNeval, v[1] * invNeval] as Cx);
    // <R>_q
    const rAvg = new Float64Array(ncell * fd);
    for (let k = 0; k < ncell * fd; k++) rAvg[k] = this.rsum[k] * invNeval;
    const rqAvg = this.transformComponents(rAvg, ncell, fd);
    // G(q) = <R_q R_q*> - <R>_q <R>_q*
    for (let q = 0; q < ncell; q++) {
      let ndim = 0;
      for (let a = 0; a < fd; a++) {
        for (let b = 0; b < fd; b++) {
          phi[q * fd2 + ndim] = csub(phi[q * fd2 + ndim], cmul(rqAvg[q * fd + a], cconj(rqAvg[q * fd + b])));
          ndim++;
        }
      }
    }
    // temperatures and the kB T scale of each component
    const tempFac = invNeval * this.invNTemp;
    const normFac = tempFac * this.ntotal;
    const kbtsqrt = new Float64Array(this.sysdim);
    let tempAve = 0;
    for (let d = 0; d < this.sysdim; d++) {
      kbtsqrt[d] = Math.sqrt(this.tempSum[d] * normFac);
      tempAve += this.tempSum[d] * tempFac;
    }
    tempAve /= this.sysdim * boltz;
    // Phi = G^-1 scaled by the kB T factors, per q
    for (let q = 0; q < ncell; q++) {
      const mat = phi.slice(q * fd2, q * fd2 + fd2);
      gaussJordan(fd, mat);
      let ndim = 0;
      for (let a = 0; a < fd; a++) {
        for (let b = 0; b < fd; b++) {
          phi[q * fd2 + ndim] = [mat[ndim][0] * kbtsqrt[a % this.sysdim] * kbtsqrt[b % this.sysdim], mat[ndim][1] * kbtsqrt[a % this.sysdim] * kbtsqrt[b % this.sysdim]];
          ndim++;
        }
      }
    }
    // the cell vectors (the mean box over the measurements)
    const hs = this.hsum.map((v) => v * invNeval);
    const basevec = [hs[0] / this.nx, 0, 0, hs[5] / this.ny, hs[1] / this.ny, 0, hs[4] / this.nz, hs[3] / this.nz, hs[2] / this.nz];
    if (!this.warnedBinary) {
      this.warnedBinary = true;
      this.sys.warn(`fix phonon: the binary force-constant file ${this.prefix}.bin.N is native-only and is not written by the browser engine; the text log ${this.logfile} is`);
    }
    // acoustic sum rule at the gamma point, then D = M^-1/2 Phi M^-1/2
    this.enforceASR(phi);
    for (let q = 0; q < ncell; q++) {
      let ndim = 0;
      for (let a = 0; a < fd; a++) {
        for (let b = 0; b < fd; b++) {
          const f = this.massInvSqrt[Math.floor(a / this.sysdim)] * this.massInvSqrt[Math.floor(b / this.sysdim)];
          phi[q * fd2 + ndim] = [phi[q * fd2 + ndim][0] * f, phi[q * fd2 + ndim][1] * f];
          ndim++;
        }
      }
    }
    const out: string[] = [];
    const bar = '############################################################';
    out.push(bar);
    out.push(`# Current time step                      : ${s.step}`);
    out.push(`# Total number of measurements           : ${this.neval}`);
    out.push(`# Average temperature of the measurement : ${tempAve}`);
    out.push(`# Boltzmann constant under current units : ${boltz}`);
    out.push(`# basis vector A1 = [${basevec[0]} ${basevec[1]} ${basevec[2]}]`);
    out.push(`# basis vector A2 = [${basevec[3]} ${basevec[4]} ${basevec[5]}]`);
    out.push(`# basis vector A3 = [${basevec[6]} ${basevec[7]} ${basevec[8]}]`);
    out.push(bar);
    out.push('# qx\t qy \t qz \t\t Phi(q)');
    let q = 0;
    for (let ix = 0; ix < this.nx; ix++) {
      const qx = ix / this.nx;
      for (let iy = 0; iy < this.ny; iy++) {
        const qy = iy / this.ny;
        for (let iz = 0; iz < this.nz; iz++) {
          const qz = iz / this.nz;
          let line = `${qx} ${qy} ${qz}`;
          for (let k = 0; k < fd2; k++) line += ` ${phi[q * fd2 + k][0]} ${phi[q * fd2 + k][1]}`;
          out.push(line);
          q++;
        }
      }
    }
    this.sys.writeFile(this.logfile, out.join('\n') + '\n', true);
  }

  /** EnforceASR(): the simple and symmetric acoustic sum rules on the gamma-point block (real parts). */
  private enforceASR(phi: Cx[]): void {
    if (this.nasr < 1) return;
    const fd = this.fftDim, nc = this.nucell, sd = this.sysdim;
    const at = (k: number, a: number, kp: number, b: number): number => (k * sd + a) * fd + kp * sd + b;
    for (let iit = 0; iit < this.nasr; iit++) {
      for (let a = 0; a < sd; a++) {
        for (let b = 0; b < sd; b++) {
          for (let k = 0; k < nc; k++) {
            let sum = 0;
            for (let kp = 0; kp < nc; kp++) sum += phi[at(k, a, kp, b)][0];
            sum /= nc;
            for (let kp = 0; kp < nc; kp++) phi[at(k, a, kp, b)] = [phi[at(k, a, kp, b)][0] - sum, phi[at(k, a, kp, b)][1]];
          }
        }
      }
      for (let k = 0; k < nc; k++) {
        for (let kp = k; kp < nc; kp++) {
          for (let a = 0; a < sd; a++) {
            for (let b = 0; b < sd; b++) {
              const idx = at(k, a, kp, b), jdx = at(kp, b, k, a);
              const csum = (phi[idx][0] + phi[jdx][0]) * 0.5;
              phi[idx] = [csum, phi[idx][1]];
              phi[jdx] = [csum, phi[jdx][1]];
            }
          }
        }
      }
    }
    for (let a = 0; a < sd; a++) {
      for (let b = 0; b < sd; b++) {
        for (let k = 0; k < nc; k++) {
          let sum = 0;
          for (let kp = 0; kp < nc; kp++) sum += phi[at(k, a, kp, b)][0];
          sum /= nc - k;
          for (let kp = k; kp < nc; kp++) {
            const idx = at(k, a, kp, b), jdx = at(kp, b, k, a);
            phi[idx] = [phi[idx][0] - sum, phi[idx][1]];
            phi[jdx] = [phi[idx][0], phi[jdx][1]];
          }
        }
      }
    }
  }

  postRun(): void {
    if (this.ifreq > 0 && this.ifreq !== this.nfreq) this.postprocess();
  }
}

/**
 * GaussJordan(): the inverse of a complex n x n matrix by Gauss-Jordan elimination with full pivoting (the
 * Numerical Recipes scheme that fix_phonon.cpp uses), in place on mat (row-major, n*n entries).
 */
export const gaussJordan = (n: number, mat: Cx[]): void => {
  const indxc = new Array<number>(n), indxr = new Array<number>(n), ipiv = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    let big = 0;
    let irow = -1, icol = -1;
    for (let j = 0; j < n; j++) {
      if (ipiv[j] === 1) continue;
      for (let k = 0; k < n; k++) {
        if (ipiv[k] === 0) {
          const nmjk = cnorm(mat[j * n + k]);
          if (nmjk >= big) { big = nmjk; irow = j; icol = k; }
        } else if (ipiv[k] > 1) {
          throw new StyleError('Singular matrix in complex GaussJordan');
        }
      }
    }
    if (icol < 0) throw new StyleError('Singular matrix in complex GaussJordan');
    ipiv[icol] += 1;
    if (irow !== icol) {
      for (let l = 0; l < n; l++) {
        const t = mat[irow * n + l];
        mat[irow * n + l] = mat[icol * n + l];
        mat[icol * n + l] = t;
      }
    }
    indxr[i] = irow;
    indxc[i] = icol;
    const piv = mat[icol * n + icol];
    if (piv[0] === 0 && piv[1] === 0) throw new StyleError('Singular matrix in complex GaussJordan');
    const pivinv = cdiv([1, 0], piv);
    mat[icol * n + icol] = [1, 0];
    for (let l = 0; l < n; l++) mat[icol * n + l] = cmul(mat[icol * n + l], pivinv);
    for (let ll = 0; ll < n; ll++) {
      if (ll === icol) continue;
      const dum = mat[ll * n + icol];
      mat[ll * n + icol] = [0, 0];
      for (let l = 0; l < n; l++) mat[ll * n + l] = csub(mat[ll * n + l], cmul(mat[icol * n + l], dum));
    }
  }
  for (let l = n - 1; l >= 0; l--) {
    const rl = indxr[l], cl = indxc[l];
    if (rl !== cl) {
      for (let k = 0; k < n; k++) {
        const t = mat[k * n + rl];
        mat[k * n + rl] = mat[k * n + cl];
        mat[k * n + cl] = t;
      }
    }
  }
};
