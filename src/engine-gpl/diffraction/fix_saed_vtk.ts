/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/DIFFRACTION/fix_saed_vtk.cpp and
 * fix_saed_vtk.h (FixSAEDVTK: fix saed/vtk, which averages the intensities of a compute saed over Nrepeat samples every
 * Nfreq steps and writes them as an image-data VTK file), and the subset of src/vtk_writer.cpp (b332f28) it uses. Contributing
 * author of the fix: Pieter in 't Veld (SNL); incorporating SAED: Shawn Coleman (Arkansas). Copyright Sandia Corporation (GNU
 * General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_saed_vtk.html (verbatim): "fix ID group-ID saed/vtk Nevery Nrepeat Nfreq c_ID keyword values ...";
 * "file = filename, the index of the output is appended to the name"; "ave = one, running or window M"; "start = Nstart";
 * "format = legacy or xml"; "binary = yes or no".
 *
 * Native behaviour kept here:
 *  - the sample of each end-of-step invocation is the compute's vector; the Nrepeat samples are NOT summed. The output
 *    of an Nfreq step is the vector of the last sample divided by Nrepeat (fix_saed_vtk.cpp invoke_vector declares a local
 *    `vector` that shadows the accumulator, so the average is never accumulated). With Nrepeat 1 this is the sample.
 *  - the grid is the bounding box of the selected reciprocal points (Zone 0 0 0: all points with |K| < Kmax); points that
 *    are not selected are written as -1.
 *  - the file is <file>.<index>.vtk (legacy) or .vti (xml), the index counting the outputs of the run; the file named by
 *    the file keyword is created empty when the fix is defined (native opens it with "w").
 *  - values are written as single precision (the VTKWriter default), printed with the shortest form that reads back as the
 *    same float (C++ fmt "{}"); origin and spacing are doubles.
 *
 * Not supported (named StyleErrors): binary yes (the binary VTK payloads and the zlib-compressed XML).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { ComputeSAED } from './compute_saed';
import { reciprocalSpacing } from './compute_xrd';
import { fmtShortest } from '../../engine/cformat';

type AveMode = 'one' | 'running' | 'window';

/** Values as the VTK writer prints them: legacy rows of nine, then a short last row (vtk_writer.cpp write_legacy_values). */
const legacyRows = (values: Float64Array, single: boolean): string => {
  const parts: string[] = [];
  for (let i = 0; i < values.length; i++) {
    parts.push(fmtShortest(values[i], single));
    parts.push((i + 1) % 9 === 0 ? '\n' : ' ');
  }
  if (values.length % 9 !== 0) parts.push('\n');
  return parts.join('');
};

/** XML ASCII values, nine per line with the given indent (vtk_writer.cpp xml_values). */
const xmlRows = (values: Float64Array, single: boolean, indent: number): string => {
  const pad = ' '.repeat(indent);
  const parts: string[] = [];
  for (let i = 0; i < values.length; i++) {
    if (i % 9 === 0) parts.push(pad);
    parts.push(fmtShortest(values[i], single));
    parts.push((i + 1) % 9 === 0 ? '\n' : ' ');
  }
  if (values.length % 9 !== 0) parts.push('\n');
  return parts.join('');
};

/** Parses a non-negative or positive integer keyword value; native utils::inumeric. */
const intArg = (w: string | undefined, what: string): number => {
  const n = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isInteger(n)) throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in ${what}`);
  return n;
};

export class FixSAEDVTK extends Fix {
  readonly style = 'saed/vtk';
  private readonly compute: ComputeSAED;
  private readonly every: number;
  private readonly repeat: number;
  private readonly freq: number;
  private readonly startStep: number;
  private readonly nrows: number;
  private readonly ave: AveMode;
  private readonly nwindow: number;
  private readonly fileName: string | null;
  private readonly xmlFormat: boolean;
  /** scheduling: the next step on which the end-of-step work runs, and the samples taken so far in this Nfreq block */
  private nvalid = 0;
  private irepeat = 0;
  private nOutput = 0;
  private lastSample: Float64Array;
  private vectorTotal: Float64Array;
  private vectorList: Float64Array[] = [];
  private iwindow = 0;
  private windowLimit = false;
  private norm = 0;
  // the output grid (native Knmin, Knmax, Dim, dK, R_Ewald and the scaled Zone)
  private readonly dK: [number, number, number];
  private readonly knmin: [number, number, number] = [0, 0, 0];
  private readonly knmax: [number, number, number] = [0, 0, 0];
  private readonly dim: [number, number, number] = [0, 0, 0];
  private readonly zone: [number, number, number];
  private readonly rEwald: number;
  private readonly dREwald: number;
  private readonly kmax: number;
  private readonly lambda: number;
  private readonly c: [number, number, number];
  private readonly title: string;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const what = 'fix saed/vtk';
    if (args.length < 4) throw new StyleError('Illegal fix saed/vtk command: expected Nevery Nrepeat Nfreq c_ID [keywords]');
    this.every = intArg(args[0], what);
    this.repeat = intArg(args[1], what);
    this.freq = intArg(args[2], what);
    this.globalFreq = this.freq;

    // keywords (native options()), read before the compute is looked up as native does
    let ave: AveMode = 'one';
    let nwindow = 0;
    let startStep = 0;
    let fileName: string | null = null;
    let xml = false;
    for (let k = 4; k < args.length;) {
      const kw = args[k];
      if (kw === 'file') {
        if (k + 2 > args.length) throw new StyleError('Illegal fix saed/vtk command: file needs a file name');
        fileName = args[k + 1];
        k += 2;
      } else if (kw === 'format') {
        if (k + 2 > args.length) throw new StyleError('Illegal fix saed/vtk command: format needs legacy or xml');
        if (args[k + 1] === 'legacy') xml = false;
        else if (args[k + 1] === 'xml') xml = true;
        else throw new StyleError(`Unknown fix saed/vtk format ${args[k + 1]}`);
        k += 2;
      } else if (kw === 'binary') {
        if (k + 2 > args.length) throw new StyleError('Illegal fix saed/vtk command: binary needs yes or no');
        if (args[k + 1] === 'yes' || args[k + 1] === 'on' || args[k + 1] === 'true' || args[k + 1] === '1') {
          throw new StyleError('fix saed/vtk binary yes is not supported (the binary VTK payloads are not written by the engine)');
        }
        k += 2;
      } else if (kw === 'ave') {
        if (k + 2 > args.length) throw new StyleError('Illegal fix saed/vtk command: ave needs one, running or window');
        const m = args[k + 1];
        if (m === 'one') ave = 'one';
        else if (m === 'running') ave = 'running';
        else if (m === 'window') {
          ave = 'window';
          if (k + 3 > args.length) throw new StyleError('Illegal fix saed/vtk command: ave window needs M');
          nwindow = intArg(args[k + 2], what);
          if (nwindow <= 0) throw new StyleError('Illegal fix saed/vtk command: ave window M must be positive');
          k += 1;
        } else throw new StyleError('Illegal fix saed/vtk command');
        k += 2;
      } else if (kw === 'start') {
        if (k + 2 > args.length) throw new StyleError('Illegal fix saed/vtk command: start needs Nstart');
        startStep = intArg(args[k + 1], what);
        k += 2;
      } else throw new StyleError(`Illegal fix saed/vtk command: unknown keyword '${kw}'`);
    }
    this.ave = ave;
    this.nwindow = nwindow;
    this.startStep = startStep;
    this.fileName = fileName;
    this.xmlFormat = xml;

    const cid = args[3];
    if (!cid.startsWith('c_') || cid.includes('[')) throw new StyleError('Illegal fix saed/vtk command: the 4th argument must be c_ID');
    const compute = sys.compute(cid.slice(2));
    if (!(compute instanceof ComputeSAED)) throw new StyleError(`Compute ID ${cid.slice(2)} for fix saed/vtk does not exist or is of incorrect style`);
    this.compute = compute;

    // native reads the settings from the compute's saed_var; the zone is already scaled to the Ewald radius there
    const v = compute.saedVar;
    this.lambda = v.lambda;
    this.kmax = v.Kmax;
    this.zone = [v.zone[0], v.zone[1], v.zone[2]];
    this.c = v.c;
    this.dREwald = v.dRewald;
    const manual = v.manual;
    this.nrows = compute.sizeVector;
    if (!compute.vectorFlag) throw new StyleError('Fix saed/vtk compute does not calculate a vector');

    if (this.every <= 0 || this.repeat <= 0 || this.freq <= 0) throw new StyleError('Illegal fix saed/vtk command');
    if (this.freq % this.every || this.repeat * this.every > this.freq) throw new StyleError('Illegal fix saed/vtk command');

    // grid setup (native constructor, nOutput == 0 branch)
    this.rEwald = 1 / this.lambda;
    if (!(this.zone[0] === 0 && this.zone[1] === 0 && this.zone[2] === 0)) {
      const rnorm = this.rEwald / Math.sqrt(this.zone[0] * this.zone[0] + this.zone[1] * this.zone[1] + this.zone[2] * this.zone[2]);
      this.zone[0] *= rnorm;
      this.zone[1] *= rnorm;
      this.zone[2] *= rnorm;
    }
    const prdInv = reciprocalSpacing(sys, manual, 'Fix saed/vtk');
    this.dK = [prdInv[0] * this.c[0], prdInv[1] * this.c[1], prdInv[2] * this.c[2]];
    const full = this.zone[0] === 0 && this.zone[1] === 0 && this.zone[2] === 0;
    if (full) {
      for (let d = 0; d < 3; d++) {
        this.knmax[d] = Math.ceil(this.kmax / this.dK[d]);
        this.knmin[d] = -this.knmax[d];
      }
    } else {
      this.knmax[0] = this.knmax[1] = this.knmax[2] = -10000;
      this.knmin[0] = this.knmin[1] = this.knmin[2] = 10000;
      const ksearch = [Math.ceil(this.kmax / this.dK[0]), Math.ceil(this.kmax / this.dK[1]), Math.ceil(this.kmax / this.dK[2])];
      const K = [0, 0, 0];
      for (let k = -ksearch[2]; k <= ksearch[2]; k++) {
        for (let j = -ksearch[1]; j <= ksearch[1]; j++) {
          for (let i = -ksearch[0]; i <= ksearch[0]; i++) {
            K[0] = i * this.dK[0];
            K[1] = j * this.dK[1];
            K[2] = k * this.dK[2];
            const dinv2 = K[0] * K[0] + K[1] * K[1] + K[2] * K[2];
            if (dinv2 < this.kmax * this.kmax) {
              const r = this.radius(K);
              if (r > this.rEwald - this.dREwald && r < this.rEwald + this.dREwald) {
                const idx = [i, j, k];
                for (let d = 0; d < 3; d++) {
                  if (idx[d] < this.knmin[d]) this.knmin[d] = idx[d];
                  if (idx[d] > this.knmax[d]) this.knmax[d] = idx[d];
                }
              }
            }
          }
        }
      }
    }
    for (let d = 0; d < 3; d++) this.dim[d] = this.knmax[d] - this.knmin[d] + 1;
    if (this.dim[0] < 1 || this.dim[1] < 1 || this.dim[2] < 1) {
      throw new StyleError(`Fix saed/vtk found no reciprocal space points to output. Check the Kmax, Zone and dR_Ewald settings of compute ${compute.id}`);
    }
    this.title = `Image data set c_${compute.id}`;

    this.lastSample = new Float64Array(this.nrows);
    this.vectorTotal = new Float64Array(this.nrows);
    // native allocates the window buffer as nwindow x 1 (memory->create(vector_list,nwindow,1)) and then stores one value per
    // grid point in each window slot: with more than one point the store overruns the buffer (lmp segfaults in Thermo)
    if (this.ave === 'window' && this.nrows > 1) {
      throw new StyleError('fix saed/vtk ave window is not supported: native allocates the window buffer as nwindow x 1 (fix_saed_vtk.cpp) and overruns it with one value per reciprocal point (lmp crashes); use ave one or running');
    }
    this.vectorFlag = true;
    this.extvector = 0;
    this.sizeVector = this.nrows;
    this.nvalid = this.nextValid();

    // native opens the file at definition (so a bad path fails early); its content is the empty file until output 0
    if (this.fileName !== null) this.sys.writeFile(this.outputName(), '', false);
  }

  /** Distance of K from the zone axis point (native: sqrt of the squared components of K - Zone). */
  private radius(K: number[]): number {
    let r = 0;
    for (let m = 0; m < 3; m++) r += (K[m] - this.zone[m]) * (K[m] - this.zone[m]);
    return Math.sqrt(r);
  }

  /** Native nextvalid(): the next step on which end_of_step does something. */
  private nextValid(): number {
    const step = this.sys.state.step;
    let nv = Math.floor(step / this.freq) * this.freq + this.freq;
    while (nv < this.startStep) nv += this.freq;
    if (nv - this.freq === step && this.repeat === 1) nv = step;
    else nv -= (this.repeat - 1) * this.every;
    if (nv < step) nv += this.freq;
    return nv;
  }

  /** Name of the next output file: <file>.<index>.vtk, or .vti for the xml format (native filecurrent()). */
  private outputName(): string {
    return `${this.fileName}.${this.nOutput}.${this.xmlFormat ? 'vti' : 'vtk'}`;
  }

  init(): void {
    if (this.nvalid < this.sys.state.step) {
      this.irepeat = 0;
      this.nvalid = this.nextValid();
    }
  }

  /** Native setup() calls end_of_step(), so the output of a step that is a multiple of Nfreq is written at run start. */
  setup(): void {
    this.endOfStep();
  }

  endOfStep(): void {
    const step = this.sys.state.step;
    if (step !== this.nvalid) return;
    this.invokeVector(step);
  }

  private invokeVector(step: number): void {
    // one sample: the compute's vector at this step (native invoke_vector; the vector is copied out)
    const sample = this.compute.vectorValues();
    this.lastSample.set(sample.subarray(0, this.nrows));
    this.irepeat++;
    if (this.irepeat < this.repeat) {
      this.nvalid += this.every;
      return;
    }
    this.irepeat = 0;
    this.nvalid = step + this.freq - (this.repeat - 1) * this.every;

    // native divides the vector of the last sample by Nrepeat (see the header)
    const vector = new Float64Array(this.nrows);
    for (let i = 0; i < this.nrows; i++) vector[i] = this.lastSample[i] / this.repeat;

    if (this.ave === 'one') {
      this.vectorTotal.set(vector);
      this.norm = 1;
    } else if (this.ave === 'running') {
      for (let i = 0; i < this.nrows; i++) this.vectorTotal[i] += vector[i];
      this.norm++;
    } else {
      for (let i = 0; i < this.nrows; i++) {
        this.vectorTotal[i] += vector[i];
        if (this.windowLimit) this.vectorTotal[i] -= this.vectorList[this.iwindow][i];
        this.vectorList[this.iwindow][i] = vector[i];
      }
      this.iwindow++;
      if (this.iwindow === this.nwindow) {
        this.iwindow = 0;
        this.windowLimit = true;
      }
      this.norm = this.windowLimit ? this.nwindow : this.iwindow;
    }

    if (this.fileName !== null) this.writeOutput();
    this.nOutput++;
  }

  /** The intensity of every grid point (k outer, i inner, as native loops them), -1 where nothing is selected. */
  private writeOutput(): void {
    const fullVolume = this.zone[0] === 0 && this.zone[1] === 0 && this.zone[2] === 0;
    const intensity = new Float64Array(this.dim[0] * this.dim[1] * this.dim[2]);
    let p = 0;
    let nsel = 0;
    const K = [0, 0, 0];
    for (let k = this.knmin[2]; k <= this.knmax[2]; k++) {
      for (let j = this.knmin[1]; j <= this.knmax[1]; j++) {
        for (let i = this.knmin[0]; i <= this.knmax[0]; i++) {
          K[0] = i * this.dK[0];
          K[1] = j * this.dK[1];
          K[2] = k * this.dK[2];
          const dinv2 = K[0] * K[0] + K[1] * K[1] + K[2] * K[2];
          let selected = dinv2 < this.kmax * this.kmax;
          if (selected && !fullVolume) {
            const r = this.radius(K);
            selected = r > this.rEwald - this.dREwald && r < this.rEwald + this.dREwald;
          }
          if (selected) {
            if (nsel >= this.nrows) throw new StyleError('Fix saed/vtk: the output grid and the compute saed vector disagree');
            // the VTK writer stores the intensities as single precision
            intensity[p] = Math.fround(this.vectorTotal[nsel] / this.norm);
            nsel++;
          } else intensity[p] = -1;
          p++;
        }
      }
    }
    if (nsel !== this.nrows) throw new StyleError('Fix saed/vtk: the output grid and the compute saed vector disagree');
    const origin = [this.knmin[0] * this.dK[0], this.knmin[1] * this.dK[1], this.knmin[2] * this.dK[2]];
    const [nx, ny, nz] = this.dim;
    const spacing = `${fmtShortest(this.dK[0], false)} ${fmtShortest(this.dK[1], false)} ${fmtShortest(this.dK[2], false)}`;
    const org = `${fmtShortest(origin[0], false)} ${fmtShortest(origin[1], false)} ${fmtShortest(origin[2], false)}`;
    const n = nx * ny * nz;
    let text: string;
    if (!this.xmlFormat) {
      text = `# vtk DataFile Version 5.1\n${this.title}\nASCII\nDATASET STRUCTURED_POINTS\n`
        + `DIMENSIONS ${nx} ${ny} ${nz}\nSPACING ${spacing}\nORIGIN ${org}\n`
        + `POINT_DATA ${n}\nSCALARS intensity float\nLOOKUP_TABLE default\n`
        + legacyRows(intensity, true);
    } else {
      const ext = `0 ${nx - 1} 0 ${ny - 1} 0 ${nz - 1}`;
      text = `<?xml version="1.0"?>\n<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" header_type="UInt32">\n`
        + `  <ImageData WholeExtent="${ext}" Origin="${org}" Spacing="${spacing}">\n`
        + `    <Piece Extent="${ext}">\n`
        + `      <PointData Scalars="intensity">\n`
        + `        <DataArray type="Float32" Name="intensity" format="ascii">\n`
        + xmlRows(intensity, true, 10)
        + `        </DataArray>\n      </PointData>\n      <CellData>\n      </CellData>\n    </Piece>\n  </ImageData>\n</VTKFile>\n`;
    }
    this.sys.writeFile(this.outputName(), text, false);
  }

  /** f_ID[i] (native compute_vector): the running, window or single average of the output. */
  computeVector(i: number): number {
    return this.norm ? this.vectorTotal[i] / this.norm : 0;
  }
}
