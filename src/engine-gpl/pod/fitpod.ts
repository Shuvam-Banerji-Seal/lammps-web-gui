/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/ML-POD/fitpod_command.cpp and
 * fitpod_command.h (class FitPOD, the fitpod command; contributing author Ngoc Cuong Nguyen (MIT); copyright Sandia
 * Corporation). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fitpod_command.html (verbatim): "fitpod Ta_param.pod Ta_data.pod"; "Fit a machine-learning interatomic
 * potential (ML-IAP) based on proper orthogonal descriptors (POD)"; the training data are "extended XYZ" files in the
 * directories the data file names, and the command writes "basename_coefficients.pod", "basename_training_errors.pod"
 * and "basename_training_analysis.pod".
 *
 * Native behaviour kept here: the data-file keywords and defaults (datastruct), the extxyz reader (lattice, energy,
 * stress, species, positions, forces; the lattice turned into LAMMPS's triclinic form and the atoms rotated with it), the
 * per-configuration neighbour list over the periodic images (latticecoords, podneighborlist), the least-squares system
 * A += we^2/natom^2 gd gd^T + wf^2 gdd^T gdd, b += we^2/natom^2 E gd + wf^2 gdd^T f, its symmetrization and
 * regularization, the Cholesky solve (DPOSV), the error analysis and the output files with their formats. The BLAS and
 * LAPACK calls are written out in the reference order (native links OpenBLAS: agreement is at rounding level).
 *
 * Fractions of the data sets below 1 with randomize 1 select configurations at random (native shuffles with
 * std::random_device, so no two runs select the same; here Math.random). Not supported (a named StyleError):
 * number_of_environment_clusters > 1 (the environment-adaptive fit with its k-means clustering and PCA).
 */
import type { Handler } from '../../engine/commands/args';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { fmtShortest, formatNumber } from '../../engine/cformat';
import { parsePODCoeffFile, parsePODParams, type PODParams } from './eapod';
import { FitPODModel, type PodNeighbors } from './fitpod_eapod';

const SMALL = 1.0e-10;

/** utils::trim_comment and Tokenizer(..., "\"' \t\n\r\f"): words without quotes. */
const words = (line: string): string[] => {
  const h = line.indexOf('#');
  const t = h >= 0 ? line.slice(0, h) : line;
  return t.split(/["' \t\n\r\f]+/).filter(Boolean);
};

const numeric = (w: string | undefined): number => {
  if (w === undefined) throw new StyleError('Expected floating point parameter instead of NULL or empty string in input script or data file');
  const v = Number(w);
  if (w.trim() === '' || !/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(w) || !Number.isFinite(v)) {
    throw new StyleError(`Expected floating point parameter instead of '${w}' in input script or data file`);
  }
  return v;
};

const inumeric = (w: string | undefined): number => {
  if (w === undefined || !/^[-+]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in input script or data file`);
  return Number(w);
};

/** fmt's "{:g}"-like printing of the data-file echo ({} of a double: shortest form). */
/** fmt's "{}" of a double: the shortest decimal that reads back as v (cformat.ts fmtShortest). */
const fmtDouble = (v: number): string => fmtShortest(v, false);

interface DataStruct {
  data_path: string;
  file_format: string;
  file_extension: string;
  data_files: string[];
  group_names: string[];
  filenames: string[];
  filenametag: string;
  group_weight_type: string;
  num_atom: number[];
  num_atom_cumsum: number[];
  num_atom_each_file: number[];
  num_config: number[];
  num_config_cumsum: number[];
  num_atom_sum: number;
  num_config_sum: number;
  num_atom_min: number;
  num_atom_max: number;
  lattice: Float64Array;
  energy: Float64Array;
  stress: Float64Array;
  position: Float64Array;
  force: Float64Array;
  atomtype: Int32Array;
  we: Float64Array;
  wf: Float64Array;
  training: number;
  normalizeenergy: number;
  training_analysis: number;
  test_analysis: number;
  training_calculation: number;
  test_calculation: number;
  randomize: number;
  precision: number;
  fraction: number;
  fitting_weights: number[];
  we_map: Map<string, number>;
  wf_map: Map<string, number>;
}

/** FitPOD::datastruct::datastruct(): the defaults. */
const newData = (): DataStruct => ({
  data_path: '', file_format: 'extxyz', file_extension: 'xyz', data_files: [], group_names: [], filenames: [],
  filenametag: 'pod', group_weight_type: 'global', num_atom: [], num_atom_cumsum: [], num_atom_each_file: [], num_config: [],
  num_config_cumsum: [], num_atom_sum: 0, num_config_sum: 0, num_atom_min: 0, num_atom_max: 0,
  lattice: new Float64Array(0), energy: new Float64Array(0), stress: new Float64Array(0), position: new Float64Array(0),
  force: new Float64Array(0), atomtype: new Int32Array(0), we: new Float64Array(0), wf: new Float64Array(0),
  training: 1, normalizeenergy: 1, training_analysis: 1, test_analysis: 1, training_calculation: 0, test_calculation: 0,
  randomize: 1, precision: 8, fraction: 1.0,
  fitting_weights: [100.0, 1.0, 0.0, 1, 1, 0, 0, 1, 1, 1, 1, 1e-10],
  we_map: new Map(), wf_map: new Map(),
});

/** datastruct::copydatainfo. */
const copyDataInfo = (from: DataStruct, to: DataStruct): void => {
  to.data_path = from.data_path;
  to.file_format = from.file_format;
  to.file_extension = from.file_extension;
  to.data_files = [...from.data_files];
  to.filenametag = from.filenametag;
  to.filenames = [...from.filenames];
  to.training_analysis = from.training_analysis;
  to.test_analysis = from.test_analysis;
  to.training_calculation = from.training_calculation;
  to.test_calculation = from.test_calculation;
  to.fraction = from.fraction;
  to.randomize = from.randomize;
  to.precision = from.precision;
  to.training = from.training;
  to.normalizeenergy = from.normalizeenergy;
  to.fitting_weights = [...from.fitting_weights];
  to.we_map = new Map(from.we_map);
  to.wf_map = new Map(from.wf_map);
};

/** The bytes of doubles as one character per byte (the engine's binary files, see nativeRestart.ts). */
const doublesToBinary = (a: ArrayLike<number>): string => {
  const f = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) f[i] = a[i];
  const b = new Uint8Array(f.buffer);
  let s = '';
  for (let i = 0; i < b.length; i += 8192) s += String.fromCharCode(...b.subarray(i, i + 8192));
  return s;
};

const podCumsum = (input: number[], length: number): number[] => {
  const out = new Array<number>(length).fill(0);
  for (let j = 1; j < length; j++) out[j] = input[j - 1] + out[j - 1];
  return out;
};

const latticeVolume = (L: Float64Array, o: number): number => {
  const b0 = L[o + 1] * L[o + 5] - L[o + 2] * L[o + 4];
  const b1 = L[o + 2] * L[o + 3] - L[o] * L[o + 5];
  const b2 = L[o] * L[o + 4] - L[o + 1] * L[o + 3];
  return b0 * L[o + 6] + b1 * L[o + 7] + b2 * L[o + 8];
};

class FitPOD {
  private model!: FitPODModel;
  private saveDescriptors = 0;
  private computeDescriptors = 0;
  private traindata = newData();
  private testdata = newData();
  private envdata = newData();
  private nb: PodNeighbors = { y: new Float64Array(0), alist: new Int32Array(0), jlist: new Int32Array(0), pairnumsum: new Int32Array(0) };
  private gd = new Float64Array(0);
  private gdd = new Float64Array(0);
  private bd = new Float64Array(0);
  private A = new Float64Array(0);
  private b = new Float64Array(0);
  private c = new Float64Array(0);

  constructor(private readonly sys: System) {}

  private log(text: string): void { this.sys.log(text); }

  /** utils::logmesg of a text ending in a newline: the lines as written, trailing blanks kept. */
  private logBlock(text: string): void { this.sys.log(text.endsWith('\n') ? text.slice(0, -1) : text); }

  run(args: string[]): void {
    if (args.length < 2) throw new StyleError('Illegal fitpod command: missing argument(s)');
    const podFile = args[0];
    const dataFile = args[1];
    const coeffFile = args.length > 2 ? args[2] : '';
    const params = parsePODParams(this.sys.readFile(podFile, (n) => `Cannot open POD coefficient file ${n}: `));
    const coeffs = coeffFile ? parsePODCoeffFile(this.sys.readFile(coeffFile, (n) => `Cannot open model coefficient file ${n}: `)) : null;
    if (params.nClusters > 1) throw new StyleError('fitpod: number_of_environment_clusters > 1 (environment-adaptive POD models) is not ported to the browser engine');
    // without a coefficient file the model starts with zeros, filled by the fit (mknewcoeff)
    const model = new FitPODModel(params, coeffs ?? { coeff: new Float64Array(0), proj: new Float64Array(0), centroids: new Float64Array(0) });
    this.model = model;
    this.logPodSummary(params);
    const nCoeffAll = model.nCoeffAll;
    this.readDataFiles(dataFile, params.species, params.pbc, params.rcut);
    this.logMemory(params.pbc, params.rcut);
    this.A = new Float64Array(nCoeffAll * nCoeffAll);
    this.b = new Float64Array(nCoeffAll);
    this.c = new Float64Array(nCoeffAll);
    this.gd = new Float64Array(nCoeffAll);
    if (coeffs) this.c.set(model.coeff);
    const pbc = params.pbc, rcut = params.rcut;
    const tr = this.traindata, te = this.testdata;

    if (this.computeDescriptors === 0) {
      if (!coeffFile) {
        this.leastSquaresFit(tr, pbc, rcut);
        let out = `model_coefficients: ${nCoeffAll} 0 0\n`;
        for (let k = 0; k < nCoeffAll; k++) out += `${formatNumber(this.c[k], `%-10.${tr.precision}f`)}\n`;
        this.sys.writeFile(`${tr.filenametag}_coefficients.pod`, out, false);
      }
      if (tr.training_analysis && tr.data_path.length > 1) this.errorAnalysis(tr, pbc, rcut);
      if (tr.training_calculation && tr.data_path.length > 1) this.energyforceCalculation(tr, pbc, rcut);
      if (!(te.data_path === tr.data_path && te.fraction === 1.0 && tr.fraction === 1.0)) {
        if (te.test_analysis && te.data_path.length > 1 && te.fraction > 0) this.errorAnalysis(te, pbc, rcut);
        if (te.test_analysis && te.test_calculation && te.data_path.length > 1 && te.fraction > 0) this.energyforceCalculation(te, pbc, rcut);
      }
    } else {
      this.descriptorsCalculation(tr, pbc, rcut);
      if (!(te.data_path === tr.data_path && te.fraction === 1.0) && te.data_path.length > 1) this.descriptorsCalculation(te, pbc, rcut);
    }
  }

  /** The summary EAPOD::read_pod_file logs (eapod.cpp at b332f28); the engine's POD port has no five- to seven-body terms. */
  private logPodSummary(p: PODParams): void {
    const md = this.model;
    const L = (t: string) => this.log(t);
    L('**************** Begin of POD Potentials ****************');
    L(`species: ${p.species.map((e) => `${e} `).join('')}`);
    L(`periodic boundary conditions: ${p.pbc[0]} ${p.pbc[1]} ${p.pbc[2]}`);
    L(`number of environment clusters: ${p.nClusters}`);
    L(`number of principal compoments: ${p.nComponents}`);
    L(`inner cut-off radius: ${fmtDouble(p.rin)}`);
    L(`outer cut-off radius: ${fmtDouble(p.rcut)}`);
    L(`bessel polynomial degree: ${p.besseldegree}`);
    L(`inverse polynomial degree: ${p.inversedegree}`);
    L(`one-body potential: ${p.onebody}`);
    L(`two-body radial basis functions: ${p.nrbf2}`);
    L(`three-body radial basis functions: ${p.nrbf3}`);
    L(`three-body angular degree: ${p.P3}`);
    // native: the four-body line shows nrbf4 and P4 when P23 (0) < P4, else nrbf23 and P23 (both 0)
    L(`four-body radial basis functions: ${p.P4 > 0 ? p.nrbf4 : 0}`);
    L(`four-body angular degree: ${p.P4 > 0 ? p.P4 : 0}`);
    L(`five-body radial basis functions: ${p.nrbf33}`);
    L(`five-body angular degree: ${p.P33}`);
    L(`six-body radial basis functions: ${p.nrbf34}`);
    L(`six-body angular degree: ${p.P34}`);
    L(`seven-body radial basis functions: ${p.nrbf44}`);
    L(`seven-body angular degree: ${p.P44}`);
    L(`number of local descriptors per element for one-body potential: ${md.nl1}`);
    L(`number of local descriptors per element for two-body potential: ${md.nl2}`);
    L(`number of local descriptors per element for three-body potential: ${md.nl3}`);
    L(`number of local descriptors per element for four-body potential: ${md.nl4}`);
    L('number of local descriptors per element for five-body potential: 0');
    L('number of local descriptors per element for six-body potential: 0');
    L('number of local descriptors per element for seven-body potential: 0');
    L(`number of local descriptors per element for all potentials: ${md.nl1 + md.Mdesc}`);
    L(`number of global descriptors: ${md.nCoeffAll}`);
    this.logBlock('**************** End of POD Potentials ****************\n\n');
  }

  /**
   * The sizes FitPOD's estimate_memory_neighborstruct and estimate_memory_fastpod find for the training and test sets,
   * and the block allocate_memory_descriptorstruct logs (the engine allocates per configuration instead).
   */
  private logMemory(pbc: number[], rcut: number): void {
    let natomMax = 0, sza = 0, szp = 0, szd = 0;
    const nCoeffAll = this.model.nCoeffAll;
    for (const d of [this.traindata, this.testdata]) {
      natomMax = Math.max(natomMax, d.num_atom_max);
      for (let ci = 0; ci < d.num_atom.length; ci++) {
        const natom = d.num_atom[ci];
        const o = 9 * ci;
        const m = pbc[0] === 1 ? Math.ceil(rcut / d.lattice[o]) : 0;
        const n = pbc[1] === 1 ? Math.ceil(rcut / d.lattice[o + 4]) : 0;
        const q = pbc[2] === 1 ? Math.ceil(rcut / d.lattice[o + 8]) : 0;
        const nl = (2 * m + 1) * (2 * n + 1) * (2 * q + 1);
        sza = Math.max(sza, natom * nl);
        szp = Math.max(szp, natom * natom * nl);
        szd = Math.max(szd, 3 * this.neighborList(d, ci, pbc, rcut) * nCoeffAll);
      }
    }
    this.log('**************** Begin of Memory Allocation ****************');
    this.log(`maximum number of atoms in periodic domain: ${natomMax}`);
    this.log(`maximum number of atoms in extended domain: ${sza}`);
    this.log(`maximum number of neighbors in extended domain: ${szp}`);
    this.log(`size of double memory: ${szd}`);
    this.log(`size of descriptor matrix: ${nCoeffAll} x ${nCoeffAll}`);
    this.log('**************** End of Memory Allocation ****************');
  }

  /** FitPOD::read_data_file: the keywords of the data file (returns the coefficient precision). */
  private readDataFile(fw: number[], d: DataStruct, file: string, env: DataStruct, test: DataStruct): number {
    let precision = 8;
    const lines = this.sys.readFile(file, (n) => `Cannot open training data file ${n}: `).split('\n');
    for (let li = 0; li < lines.length; li++) {
      let w = words(lines[li]);
      if (w.length === 0) continue;
      const key = w[0];
      if (w.length !== 2) throw new StyleError('Improper POD data file.');
      const v = w[1];
      switch (key) {
        case 'fitting_weight_energy': fw[0] = numeric(v); break;
        case 'fitting_weight_force': fw[1] = numeric(v); break;
        case 'fitting_weight_stress': fw[2] = numeric(v); break;
        case 'error_analysis_for_training_data_set': fw[3] = numeric(v); break;
        case 'error_analysis_for_test_data_set': fw[4] = numeric(v); break;
        case 'energy_force_calculation_for_training_data_set': fw[5] = numeric(v); break;
        case 'energy_force_calculation_for_test_data_set': fw[6] = numeric(v); break;
        case 'fraction_training_data_set': fw[7] = numeric(v); break;
        case 'fraction_test_data_set': fw[8] = numeric(v); break;
        case 'randomize_training_data_set': fw[9] = numeric(v); break;
        case 'randomize_test_data_set': fw[10] = numeric(v); break;
        case 'fitting_regularization_parameter': fw[11] = numeric(v); break;
        case 'precision_for_pod_coefficients': precision = inumeric(v); break;
        case 'save_pod_descriptors': this.saveDescriptors = inumeric(v); break;
        case 'compute_pod_descriptors': this.computeDescriptors = inumeric(v); break;
        case 'file_format': d.file_format = v; break;
        case 'file_extension': d.file_extension = v; break;
        case 'path_to_training_data_set': d.data_path = v; break;
        case 'path_to_test_data_set': test.data_path = v; break;
        case 'path_to_environment_configuration_set': env.data_path = v; break;
        case 'basename_for_output_files': d.filenametag = v; break;
        default: break;
      }
      if (key === 'group_weights') d.group_weight_type = v;
      if (d.group_weight_type === 'table') {
        // the group weight table: "name we wf" lines until a line of another length
        li++;
        if (li >= lines.length) break;
        w = words(lines[li]);
        while (w.length === 3) {
          d.we_map.set(w[0], numeric(w[1]));
          d.wf_map.set(w[0], numeric(w[2]));
          li++;
          if (li >= lines.length) break;
          w = words(lines[li]);
        }
      }
    }
    this.log('**************** Begin of Data File ****************');
    this.log(`file format: ${d.file_format}`);
    this.log(`file extension: ${d.file_extension}`);
    this.log(`path to training data set: ${d.data_path}`);
    this.log(`path to test data set: ${test.data_path}`);
    this.log(`path to environment configuration set: ${env.data_path}`);
    this.log(`basename for output files: ${d.filenametag}`);
    this.log(`training fraction: ${fmtDouble(fw[7])}`);
    this.log(`test fraction: ${fmtDouble(fw[8])}`);
    this.log(`randomize training data set: ${fmtDouble(fw[9])}`);
    this.log(`randomize test data set: ${fmtDouble(fw[10])}`);
    this.log(`error analysis for training data set: ${fmtDouble(fw[3])}`);
    this.log(`error analysis for test data set: ${fmtDouble(fw[4])}`);
    this.log(`energy/force calculation for training data set: ${fmtDouble(fw[5])}`);
    this.log(`energy/force calculation for test data set: ${fmtDouble(fw[6])}`);
    this.log(`fitting weight for energy: ${fmtDouble(fw[0])}`);
    this.log(`fitting weight for force: ${fmtDouble(fw[1])}`);
    this.log(`fitting weight for stress: ${fmtDouble(fw[2])}`);
    this.log(`save pod descriptors: ${this.saveDescriptors}`);
    this.log(`compute pod descriptors: ${this.computeDescriptors}`);
    this.log('**************** End of Data File ****************');
    return precision;
  }

  /** FitPOD::read_data_files. */
  private readDataFiles(dataFile: string, species: string[], pbc: number[], rcut: number): void {
    const data = newData();
    data.precision = this.readDataFile(data.fitting_weights, data, dataFile, this.envdata, this.testdata);
    const fw = data.fitting_weights;
    data.training_analysis = Math.trunc(fw[3]);
    data.test_analysis = Math.trunc(fw[4]);
    data.training_calculation = Math.trunc(fw[5]);
    data.test_calculation = Math.trunc(fw[6]);
    data.fraction = fw[7];
    data.randomize = Math.trunc(fw[9]);
    const tr = this.traindata;
    copyDataInfo(data, tr);
    if (data.fraction >= 1.0) {
      this.log('**************** Begin of Training Data Set ****************');
      if (tr.data_path.length > 1) this.getData(tr, species);
      else throw new StyleError('data set is not found');
      this.log('**************** End of Training Data Set ****************');
    } else {
      this.log('**************** Begin of Training Data Set ****************');
      if (data.data_path.length > 1) this.getData(data, species);
      else throw new StyleError('data set is not found');
      this.log('**************** End of Training Data Set ****************');
      this.log('**************** Begin of Select Training Data Set ****************');
      this.selectData(tr, data);
      this.log('**************** End of Select Training Data Set ****************');
    }
    const te = this.testdata;
    te.fraction = tr.fitting_weights[8];
    te.test_analysis = tr.test_analysis;
    te.filenametag = tr.filenametag;
    if (te.data_path === tr.data_path && te.fraction === 1.0 && tr.fraction === 1.0) {
      te.data_path = tr.data_path;
    } else if (te.data_path.length > 1 && te.fraction > 0 && te.test_analysis) {
      te.training = 0;
      te.file_format = tr.file_format;
      te.file_extension = tr.file_extension;
      te.training_analysis = tr.training_analysis;
      te.training_calculation = tr.training_calculation;
      te.test_calculation = tr.test_calculation;
      te.randomize = Math.trunc(tr.fitting_weights[10]);
      if (te.fraction >= 1.0) {
        this.log('**************** Begin of Test Data Set ****************');
        this.getData(te, species);
        this.log('**************** End of Test Data Set ****************');
      } else {
        const tm = newData();
        copyDataInfo(te, tm);
        tm.data_path = te.data_path;
        this.log('**************** Begin of Test Data Set ****************');
        this.getData(tm, species);
        this.log('**************** End of Test Data Set ****************');
        this.log('**************** Begin of Select Test Data Set ****************');
        this.selectData(te, tm);
        this.log('**************** End of Select Test Data Set ****************');
      }
    } else {
      te.data_path = tr.data_path;
    }
    void pbc; void rcut;
  }

  /**
   * FitPOD::select_data: with randomize 1 a random subset of each file (native shuffles with std::random_device, so a
   * run does not repeat; here Math.random), else evenly spaced configurations (linspace).
   */
  private selectData(nd: DataStruct, data: DataStruct): void {
    if (data.randomize === 1) this.log(`Select ${fmtDouble(data.fraction)} fraction of the data set at random using shuffle`);
    else this.log(`Select ${fmtDouble(data.fraction)} fraction of the data set deterministically using linspace`);
    const nfiles = data.data_files.length;
    const selected: number[][] = [];
    nd.num_atom = [];
    nd.num_config = new Array(nfiles).fill(0);
    nd.num_atom_each_file = new Array(nfiles).fill(0);
    for (let file = 0; file < nfiles; file++) {
      const nconfigs = data.num_config[file];
      let m = Math.round(nconfigs * data.fraction);
      m = Math.max(m, 1);
      selected[file] = data.randomize === 1 ? shuffle(1, nconfigs, m) : linspace(1, nconfigs, m);
      const ns = selected[file].length;
      nd.num_config[file] = ns;
      let sum = 0;
      for (let ii = 0; ii < ns; ii++) {
        const ci = data.num_config_cumsum[file] + selected[file][ii] - 1;
        nd.num_atom.push(data.num_atom[ci]);
        sum += data.num_atom[ci];
      }
      nd.num_atom_each_file[file] = sum;
    }
    const len = nd.num_atom.length;
    let mn = nd.num_atom[0], mx = nd.num_atom[0];
    for (const a of nd.num_atom) { if (a < mn) mn = a; if (a > mx) mx = a; }
    nd.num_atom_min = mn; nd.num_atom_max = mx;
    nd.num_atom_cumsum = podCumsum(nd.num_atom, len + 1);
    nd.num_atom_sum = nd.num_atom_cumsum[len];
    nd.num_config_cumsum = podCumsum(nd.num_config, nfiles + 1);
    nd.num_config_sum = len;
    const n = len;
    nd.lattice = new Float64Array(9 * n); nd.stress = new Float64Array(9 * n); nd.energy = new Float64Array(n);
    nd.we = new Float64Array(n); nd.wf = new Float64Array(n);
    nd.position = new Float64Array(3 * nd.num_atom_sum); nd.force = new Float64Array(3 * nd.num_atom_sum); nd.atomtype = new Int32Array(nd.num_atom_sum);
    let cn = 0;
    for (let file = 0; file < nfiles; file++) {
      for (const sel of selected[file]) {
        const ci = data.num_config_cumsum[file] + sel - 1;
        const natom = data.num_atom[ci];
        const ac = data.num_atom_cumsum[ci];
        const nac = nd.num_atom_cumsum[cn];
        nd.energy[cn] = data.energy[ci];
        nd.we[cn] = data.we[ci];
        nd.wf[cn] = data.wf[ci];
        for (let j = 0; j < 9; j++) { nd.stress[j + 9 * cn] = data.stress[j + 9 * ci]; nd.lattice[j + 9 * cn] = data.lattice[j + 9 * ci]; }
        for (let na = 0; na < natom; na++) {
          nd.atomtype[na + nac] = data.atomtype[ac + na];
          for (let j = 0; j < 3; j++) {
            nd.position[j + 3 * na + 3 * nac] = data.position[j + 3 * na + 3 * ac];
            nd.force[j + 3 * na + 3 * nac] = data.force[j + 3 * na + 3 * ac];
          }
        }
        cn++;
      }
    }
    const keepFiles = [...data.data_files];
    copyDataInfo(data, nd);
    nd.data_files = keepFiles;
    nd.filenames = data.data_files.map((f) => f.slice(data.data_path.length + 1));
    let maxname = 9;
    for (const f of data.data_files) maxname = Math.max(maxname, f.length);
    maxname -= data.data_path.length + 1;
    const dash = '-'.repeat(maxname + 90);
    this.log(`${dash}\n ${center('data_file', maxname)} | # configs (selected) | # atoms (selected) | # configs (original) | # atoms (original)\n${dash}`);
    for (let i = 0; i < nfiles; i++) {
      const w8 = (v: number) => String(v).padStart(8);
      this.log(` ${nd.filenames[i].padEnd(maxname)} |       ${w8(nd.num_config[i])}       |      ${w8(nd.num_atom_each_file[i])}      |       ${w8(data.num_config[i])}       |     ${w8(data.num_atom_each_file[i])}`);
    }
    this.log(`${dash}\nnumber of files: ${nfiles}`);
    this.log(`number of configurations in all files (selected and original): ${nd.num_config_sum} and ${data.num_config_sum}`);
    this.log(`number of atoms in all files (selected and original: ${nd.num_atom_sum} and ${data.num_atom_sum}`);
  }

  /** FitPOD::get_exyz_files: the files of the directory with the extension, sorted by name. */
  private getExyzFiles(d: DataStruct): void {
    const dir = this.sys.resolveFile(d.data_path);
    const prefix = dir === '' || dir === '.' ? '' : `${dir}/`;
    const names: string[] = [];
    for (const key of this.sys.files.keys()) {
      if (prefix && !key.startsWith(prefix)) continue;
      const rest = key.slice(prefix.length);
      if (rest.includes('/')) continue;
      names.push(rest);
    }
    names.sort();
    const re = new RegExp(`.*\\.${d.file_extension.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`);
    for (const fname of names) {
      if (!re.test(fname)) continue;
      d.data_files.push(`${d.data_path}/${fname}`);
      // native: fname.substr(0, fname.find(extension) - 1)
      const pos = fname.indexOf(d.file_extension) - 1;
      d.group_names.push(fname.slice(0, pos < 0 ? 0 : pos));
    }
  }

  /** FitPOD::get_data: count the atoms, read the configurations, convert the cells to the triclinic form. */
  private getData(d: DataStruct, species: string[]): void {
    this.getExyzFiles(d);
    const texts = d.data_files.map((f) => this.sys.readFile(f, (n) => `Cannot open POD coefficient file ${n}: `));
    for (const t of texts) {
      let nconf = 0, nsum = 0;
      for (const line of t.split('\n')) {
        const w = words(line);
        if (w.length === 1) {
          const na = inumeric(w[0]);
          d.num_atom.push(na);
          nconf++;
          nsum += na;
        }
      }
      d.num_config.push(nconf);
      d.num_atom_each_file.push(nsum);
    }
    d.num_atom_sum = d.num_atom.reduce((a, b) => a + b, 0);
    d.num_config_sum = d.num_atom.length;
    let maxname = 9;
    for (const f of d.data_files) maxname = Math.max(maxname, f.length);
    maxname -= d.data_path.length + 1;
    const sep = '-'.repeat(maxname + 46);
    this.log(`${sep}\n ${center('data file', maxname)} | number of configurations | number of atoms\n${sep}`);
    d.data_files.forEach((f, i) => {
      const name = f.slice(d.data_path.length + 1);
      d.filenames.push(name);
      this.log(` ${name.padEnd(maxname)} |        ${String(d.num_config[i]).padStart(10)}        |    ${String(d.num_atom_each_file[i]).padStart(8)}`);
    });
    this.log(sep);
    this.log(`number of files: ${d.data_files.length}`);
    this.log(`number of configurations in all files: ${d.num_config_sum}`);
    this.log(`number of atoms in all files: ${d.num_atom_sum}`);
    if (d.data_files.length === 0) {
      throw new StyleError('Cannot fit potential without data files. The data paths may not be valid. Please check the data paths in the POD data file.');
    }
    const n = d.num_config_sum;
    d.lattice = new Float64Array(9 * n); d.stress = new Float64Array(9 * n); d.energy = new Float64Array(n);
    d.we = new Float64Array(n); d.wf = new Float64Array(n);
    d.position = new Float64Array(3 * d.num_atom_sum); d.force = new Float64Array(3 * d.num_atom_sum); d.atomtype = new Int32Array(d.num_atom_sum);
    let nconfigs = 0, natoms = 0;
    for (let i = 0; i < d.data_files.length; i++) {
      const g = d.group_names[i];
      let weGroup: number, wfGroup: number;
      if (d.we_map.has(g)) { weGroup = d.we_map.get(g)!; wfGroup = d.wf_map.get(g)!; } else { weGroup = d.fitting_weights[0]; wfGroup = d.fitting_weights[1]; }
      this.readExyz(d, texts[i], nconfigs, natoms, species, weGroup, wfGroup);
      nconfigs += d.num_config[i];
      natoms += d.num_atom_each_file[i];
    }
    const len = d.num_atom.length;
    let mn = d.num_atom[0], mx = d.num_atom[0];
    for (const a of d.num_atom) { if (a < mn) mn = a; if (a > mx) mx = a; }
    d.num_atom_min = mn; d.num_atom_max = mx;
    d.num_atom_cumsum = podCumsum(d.num_atom, len + 1);
    d.num_config_cumsum = podCumsum(d.num_config, d.data_files.length + 1);
    // convert all structures to the triclinic system
    for (let ci = 0; ci < len; ci++) {
      const natom = d.num_atom[ci];
      const ac = d.num_atom_cumsum[ci];
      const L = d.lattice;
      const o = 9 * ci;
      const Q = matrix33Inverse(L, o);
      triclinicLatticeConversion(L, o);
      // matrix33_multiplication(Qmat, lattice, Qmat, 3): column i of Q becomes lattice * Q[:, i]
      for (let i = 0; i < 3; i++) {
        const x1 = Q[3 * i], x2 = Q[1 + 3 * i], x3 = Q[2 + 3 * i];
        Q[3 * i] = L[o] * x1 + L[o + 3] * x2 + L[o + 6] * x3;
        Q[1 + 3 * i] = L[o + 1] * x1 + L[o + 4] * x2 + L[o + 7] * x3;
        Q[2 + 3 * i] = L[o + 2] * x1 + L[o + 5] * x2 + L[o + 8] * x3;
      }
      rotate(d.position, 3 * ac, Q, natom);
      rotate(d.force, 3 * ac, Q, natom);
    }
    this.log(`minimum number of atoms: ${d.num_atom_min}`);
    this.log(`maximum number of atoms: ${d.num_atom_max}`);
  }

  /** FitPOD::read_exyz_file. */
  private readExyz(d: DataStruct, text: string, c0: number, a0: number, species: string[], weGroup: number, wfGroup: number): void {
    let cfi = c0, nat = a0;
    for (const line of text.split('\n')) {
      const w = words(line);
      if (w.length === 0) continue;
      if (line.split('#')[0].includes('attice')) {
        let index = w.findIndex((s) => s.includes('attice'));
        const latStart = w[index].includes('=') ? index + 1 : index + 2;
        for (let k = 0; k < 9; k++) d.lattice[k + 9 * cfi] = numeric(w[latStart + k]);
        if (this.computeDescriptors === 0) {
          index = w.findIndex((s) => s.includes('nergy'));
          if (index < 0) index = w.length;
          if (index < w.length && w[index].includes('=')) d.energy[cfi] = numeric(w[index].slice(w[index].indexOf('=') + 1));
          else d.energy[cfi] = numeric(w[index + 2]);
          index = w.findIndex((s) => s.includes('tress'));
          if (index >= 0) {
            const st = w[index].includes('=') ? index + 1 : index + 2;
            for (let k = 0; k < 9; k++) d.stress[k + 9 * cfi] = numeric(w[st + k]);
          }
        }
        d.we[cfi] = weGroup;
        d.wf[cfi] = wfGroup;
        cfi++;
      } else if (w.length > 1) {
        for (let ii = 0; ii < species.length; ii++) if (species[ii] === w[0]) d.atomtype[nat] = ii + 1;
        if (this.computeDescriptors > 0) {
          for (let k = 0; k < 3; k++) d.position[k + 3 * nat] = numeric(w[1 + k]);
        } else {
          for (let k = 0; k < 6; k++) {
            if (k <= 2) d.position[k + 3 * nat] = numeric(w[1 + k]);
            else d.force[k - 3 + 3 * nat] = numeric(w[1 + k]);
          }
        }
        nat++;
      }
    }
  }

  /** FitPOD::podfullneighborlist: the periodic images (latticecoords) and the neighbours within rcut (podneighborlist). */
  private neighborList(d: DataStruct, ci: number, pbc: number[], rcut: number): number {
    const natom = d.num_atom[ci];
    const ac = d.num_atom_cumsum[ci];
    const L = d.lattice, o = 9 * ci;
    const x = d.position.subarray(3 * ac, 3 * (ac + natom));
    let m = 0, n = 0, p = 0;
    if (pbc[0] === 1) m = Math.ceil(rcut / L[o]);
    if (pbc[1] === 1) n = Math.ceil(rcut / L[o + 4]);
    if (pbc[2] === 1) p = Math.ceil(rcut / L[o + 8]);
    const ind = m + (2 * m + 1) * n + (2 * m + 1) * (2 * n + 1) * p;
    const nl = (2 * m + 1) * (2 * n + 1) * (2 * p + 1);
    const N = natom * nl;
    const y = new Float64Array(3 * N);
    y.set(x);
    let q = natom;
    for (let i = 0; i < 2 * p + 1; i++) {
      for (let j = 0; j < 2 * n + 1; j++) {
        for (let k = 0; k < 2 * m + 1; k++) {
          const ii = k + (2 * m + 1) * j + (2 * m + 1) * (2 * n + 1) * i;
          if (ii === ind) continue;
          const x0 = L[o] * (k - m) + L[o + 3] * (j - n) + L[o + 6] * (i - p);
          const x1 = L[o + 1] * (k - m) + L[o + 4] * (j - n) + L[o + 7] * (i - p);
          const x2 = L[o + 2] * (k - m) + L[o + 5] * (j - n) + L[o + 8] * (i - p);
          for (let jj = 0; jj < natom; jj++) {
            y[3 * q] = x0 + x[3 * jj];
            y[1 + 3 * q] = x1 + x[1 + 3 * jj];
            y[2 + 3 * q] = x2 + x[2 + 3 * jj];
            q++;
          }
        }
      }
    }
    const alist = new Int32Array(N);
    for (let i = 0; i < nl; i++) for (let j = 0; j < natom; j++) alist[j + natom * i] = j;
    const rcutsq = rcut * rcut;
    const jl: number[] = [];
    const num = new Int32Array(natom);
    for (let i = 0; i < natom; i++) {
      let inc = 0;
      for (let j = 0; j < N; j++) {
        const dx = y[3 * i] - y[3 * j], dy = y[3 * i + 1] - y[3 * j + 1], dz = y[3 * i + 2] - y[3 * j + 2];
        const rijsq = dx * dx + dy * dy + dz * dz;
        if (rijsq > SMALL && rijsq <= rcutsq) { inc++; jl.push(j); }
      }
      num[i] = inc;
    }
    const pairnumsum = new Int32Array(natom + 1);
    for (let j = 1; j <= natom; j++) pairnumsum[j] = num[j - 1] + pairnumsum[j - 1];
    this.nb = { y, alist, jlist: Int32Array.from(jl), pairnumsum };
    return jl.length;
  }

  /** FitPOD::local_descriptors_fastpod: gd, gdd and the base descriptors of configuration ci. */
  private localDescriptors(d: DataStruct, ci: number, pbc: number[], rcut: number): void {
    const natom = d.num_atom[ci];
    this.neighborList(d, ci, pbc, rcut);
    const nCoeffAll = this.model.nCoeffAll;
    if (this.gdd.length < 3 * natom * nCoeffAll) this.gdd = new Float64Array(3 * natom * nCoeffAll);
    if (this.bd.length < natom * this.model.Mdesc) this.bd = new Float64Array(natom * this.model.Mdesc);
    const ac = d.num_atom_cumsum[ci];
    this.model.descriptors(this.gd, this.gdd, this.bd, this.nb, d.atomtype.subarray(ac, ac + natom), natom);
  }

  /** FitPOD::least_squares_matrix. */
  private leastSquaresMatrix(d: DataStruct, ci: number): void {
    const natom = d.num_atom[ci];
    const nCoeffAll = this.model.nCoeffAll;
    const nforce = 3 * natom;
    let normconst = 1.0;
    if (d.normalizeenergy === 1) normconst = 1.0 / natom;
    const we = d.we[ci], wf = d.wf[ci];
    const we2 = (we * we) * (normconst * normconst);
    const wf2 = wf * wf;
    const energy = d.energy[ci];
    const ac = d.num_atom_cumsum[ci];
    const force = d.force.subarray(3 * ac, 3 * (ac + natom));
    const A = this.A, b = this.b, gd = this.gd, gdd = this.gdd;
    // podKron: A[idx] += we2 * gd[ia] * gd[ib], idx = ia*M2 + ib
    for (let idx = 0; idx < nCoeffAll * nCoeffAll; idx++) {
      const ib = idx % nCoeffAll;
      const ia = (idx - ib) / nCoeffAll;
      A[idx] += we2 * gd[ia] * gd[ib];
    }
    // DGEMM('T','N', n, n, nforce, wf2, gdd, nforce, gdd, nforce, 1, A, n), reference order
    for (let j = 0; j < nCoeffAll; j++) {
      for (let i = 0; i < nCoeffAll; i++) {
        let temp = 0.0;
        for (let l = 0; l < nforce; l++) temp += gdd[l + nforce * i] * gdd[l + nforce * j];
        A[i + nCoeffAll * j] = wf2 * temp + A[i + nCoeffAll * j];
      }
    }
    const wee = we2 * energy;
    for (let i = 0; i < nCoeffAll; i++) b[i] += wee * gd[i];
    // DGEMV('T', nforce, n, wf2, gdd, nforce, force, 1, 1, b, 1), reference order
    for (let j = 0; j < nCoeffAll; j++) {
      let temp = 0.0;
      for (let i = 0; i < nforce; i++) temp += gdd[i + nforce * j] * force[i];
      b[j] += wf2 * temp;
    }
  }

  /** FitPOD::least_squares_fit. */
  private leastSquaresFit(d: DataStruct, pbc: number[], rcut: number): void {
    this.log('**************** Begin of Least-Squares Fitting ****************');
    const nCoeffAll = this.model.nCoeffAll;
    for (let ci = 0; ci < d.num_atom.length; ci++) {
      if (ci % 100 === 0) this.log(`Configuration: # ${ci + 1}`);
      this.localDescriptors(d, ci, pbc, rcut);
      if (this.saveDescriptors > 0) {
        const natom = d.num_atom[ci];
        const vals: number[] = Array.from(this.gd.subarray(0, nCoeffAll));
        if (this.saveDescriptors === 2) for (const v of this.gdd.subarray(0, 3 * natom * nCoeffAll)) vals.push(v);
        this.sys.writeFile(`${d.data_path}/descriptors_config${ci + 1}.bin`, doublesToBinary(vals), false);
      }
      this.leastSquaresMatrix(d, ci);
    }
    const A = this.A, c = this.c;
    // symmetrize A
    for (let i = 0; i < nCoeffAll; i++) {
      for (let j = i; j < nCoeffAll; j++) {
        const a1 = A[i + nCoeffAll * j], a2 = A[j + nCoeffAll * i];
        A[i + nCoeffAll * j] = 0.5 * (a1 + a2);
        A[j + nCoeffAll * i] = 0.5 * (a1 + a2);
      }
    }
    const reg = d.fitting_weights[11];
    for (let i = 0; i < nCoeffAll; i++) {
      c[i] = this.b[i];
      A[i + nCoeffAll * i] = A[i + nCoeffAll * i] * (1.0 + reg);
      if (A[i + nCoeffAll * i] < reg) A[i + nCoeffAll * i] = reg;
    }
    dposv(A, c, nCoeffAll);
    this.model.mknewcoeff(c);
  }

  /** FitPOD::error_analysis and print_analysis. */
  private errorAnalysis(d: DataStruct, pbc: number[], rcut: number): void {
    const nfiles = d.data_files.length;
    const ncfg = d.num_atom.length;
    const m = 8;
    const out = new Float64Array(m * ncfg);
    const ssr = new Float64Array(ncfg);
    const errors = new Float64Array(4 * (nfiles + 1));
    const force = new Float64Array(3 * d.num_atom_max);
    this.log('**************** Begin of Error Calculation ****************');
    let ci = 0;
    for (let file = 0; file < nfiles; file++) {
      for (let ii = 0; ii < d.num_config[file]; ii++) {
        if (ci % 100 === 0) this.log(`Configuration: # ${ci + 1}`);
        const natom = d.num_atom[ci];
        const nforce = 3 * natom;
        const energy = this.energyforceConfig(force, d, ci, pbc, rcut);
        const dft = d.energy[ci];
        const ac = d.num_atom_cumsum[ci];
        const dftf = d.force.subarray(3 * ac, 3 * (ac + natom));
        out[m * ci] = ci + 1;
        out[1 + m * ci] = natom;
        out[2 + m * ci] = energy;
        out[3 + m * ci] = dft;
        out[4 + m * ci] = Math.abs(dft - energy) / natom;
        out[5 + m * ci] = norm(force, nforce);
        out[6 + m * ci] = norm(dftf, nforce);
        let sum = 0.0, s2 = 0.0;
        for (let j = 0; j < nforce; j++) {
          const diff = dftf[j] - force[j];
          sum += Math.abs(diff);
          s2 += diff * diff;
        }
        out[7 + m * ci] = sum / nforce;
        ssr[ci] = s2;
        ci++;
      }
    }
    ci = 0;
    let nc = 0, nf = 0;
    for (let file = 0; file < nfiles; file++) {
      let emae = 0.0, essr = 0.0, fmae = 0.0, fssr = 0.0;
      let nforceall = 0;
      let nconfigs = d.num_config[file];
      nc += nconfigs;
      for (let ii = 0; ii < nconfigs; ii++) {
        const nforce = 3 * d.num_atom[ci];
        emae += out[4 + m * ci];
        essr += out[4 + m * ci] * out[4 + m * ci];
        fmae += out[7 + m * ci] * nforce;
        fssr += ssr[ci];
        nforceall += nforce;
        ci++;
      }
      const q = file + 1;
      if (nconfigs === 0) nconfigs = 1;
      if (nforceall === 0) nforceall = 1;
      errors[4 * q] = emae / nconfigs;
      errors[1 + 4 * q] = Math.sqrt(essr / nconfigs);
      errors[2 + 4 * q] = fmae / nforceall;
      errors[3 + 4 * q] = Math.sqrt(fssr / nforceall);
      nf += nforceall;
      errors[0] += emae;
      errors[1] += essr;
      errors[2] += fmae;
      errors[3] += fssr;
    }
    if (nc === 0) nc = 1;
    if (nf === 0) nf = 1;
    errors[0] = errors[0] / nc;
    errors[1] = Math.sqrt(errors[1] / nc);
    errors[2] = errors[2] / nf;
    errors[3] = Math.sqrt(errors[3] / nf);
    this.log('**************** End of Error Calculation ****************');
    this.printAnalysis(d, out, errors);
  }

  /** FitPOD::print_analysis: the errors and analysis files (and the log table). */
  private printAnalysis(d: DataStruct, out: Float64Array, errors: Float64Array): void {
    const nfiles = d.data_files.length;
    let lm = 10;
    for (let i = 0; i < nfiles; i++) lm = Math.max(lm, d.filenames[i].length);
    lm += 2;
    const kind = d.training ? 'training' : 'test';
    const Kind = d.training ? 'Training' : 'Test';
    let fe = '';
    let fa = '';
    const head = `**************** Begin of Error Analysis for the ${Kind} Data Set ****************\n`;
    this.logBlock(head);
    fe += head;
    const sa = `${'-'.repeat(lm + 80)}\n`;
    const sb = ` ${center('File', lm)} | # configs |  # atoms  | MAE energy  | RMSE energy | MAE force  | RMSE force\n`;
    this.logBlock((sa + sb + sa));
    fe += sa + sb + sa;
    const m = 8;
    let ci = 0, nc = 0, nf = 0;
    const f6 = (v: number) => formatNumber(v, '%-10.6f');
    for (let file = 0; file < nfiles; file++) {
      fa += `# ${d.filenames[file]}\n`;
      fa += '  config   # atoms       volume        energy        DFT energy     energy error     force          DFT force       force error\n';
      let nforceall = 0;
      const nconfigs = d.num_config[file];
      nc += nconfigs;
      for (let ii = 0; ii < nconfigs; ii++) {
        fa += `${fmtDouble(out[m * ci]).padStart(6)}   ${fmtDouble(out[1 + m * ci]).padStart(8)}    `;
        fa += `${formatNumber(latticeVolume(d.lattice, 9 * ci), '%-15.10g')} `;
        for (let k = 2; k < m; k++) fa += `${formatNumber(out[k + m * ci], '%-15.10g')} `;
        fa += '\n';
        nforceall += 3 * d.num_atom[ci];
        ci++;
      }
      nf += nforceall;
      const q = file + 1;
      const s = `${d.filenames[file].padEnd(lm)} ${String(nconfigs).padStart(10)} ${String(nforceall / 3).padStart(11)}     ${f6(errors[4 * q])}    ${f6(errors[1 + 4 * q])}    ${f6(errors[2 + 4 * q])}    ${f6(errors[3 + 4 * q])}\n`;
      this.logBlock(s);
      fe += s;
    }
    this.logBlock(sa);
    fe += sa;
    const s = `${'All files'.padEnd(lm)} ${String(nc).padStart(10)} ${String(nf / 3).padStart(11)}     ${f6(errors[0])}    ${f6(errors[1])}    ${f6(errors[2])}    ${f6(errors[3])}\n`;
    this.logBlock((s + sa));
    fe += s + sa;
    const tail = `**************** End of Error Analysis for the ${Kind} Data Set ****************\n`;
    this.logBlock(tail);
    fe += tail;
    this.sys.writeFile(`${d.filenametag}_${kind}_errors.pod`, fe, false);
    this.sys.writeFile(`${d.filenametag}_${kind}_analysis.pod`, fa, false);
  }

  /** FitPOD::energyforce_calculation_fastpod. */
  private energyforceConfig(force: Float64Array, d: DataStruct, ci: number, pbc: number[], rcut: number): number {
    const natom = d.num_atom[ci];
    this.neighborList(d, ci, pbc, rcut);
    const ac = d.num_atom_cumsum[ci];
    return this.model.energyforce(force, this.nb, d.atomtype.subarray(ac, ac + natom), natom);
  }

  /** FitPOD::energyforce_calculation: energyforce_config<N>.bin with the energy and the forces. */
  private energyforceCalculation(d: DataStruct, pbc: number[], rcut: number): void {
    this.log('**************** Begin of Energy/Force Calculation ****************');
    const force = new Float64Array(3 * d.num_atom_max);
    for (let ci = 0; ci < d.num_atom.length; ci++) {
      if (ci % 100 === 0) this.log(`Configuration: # ${ci + 1}`);
      const natom = d.num_atom[ci];
      const e = this.energyforceConfig(force, d, ci, pbc, rcut);
      const vals = [e, ...force.subarray(0, 3 * natom)];
      this.sys.writeFile(`energyforce_config${ci + 1}.bin`, doublesToBinary(vals), false);
    }
    this.log('**************** End of Energy/Force Calculation ****************');
  }

  /** FitPOD::descriptors_calculation: basedescriptors_config<N>.bin and globaldescriptors_config<N>.bin. */
  private descriptorsCalculation(d: DataStruct, pbc: number[], rcut: number): void {
    this.log('**************** Begin Calculating Descriptors ****************');
    const nCoeffAll = this.model.nCoeffAll, Mdesc = this.model.Mdesc;
    for (let ci = 0; ci < d.num_atom.length; ci++) {
      if (ci % 100 === 0) this.log(`Configuration: # ${ci + 1}`);
      this.localDescriptors(d, ci, pbc, rcut);
      const natom = d.num_atom[ci];
      this.sys.writeFile(`${d.data_path}/basedescriptors_config${ci + 1}.bin`, doublesToBinary([natom, Mdesc, ...this.bd.subarray(0, natom * Mdesc)]), false);
      const g = [natom, nCoeffAll, ...this.gd.subarray(0, nCoeffAll)];
      if (this.computeDescriptors === 2) for (const v of this.gdd.subarray(0, 3 * natom * nCoeffAll)) g.push(v);
      this.sys.writeFile(`${d.data_path}/globaldescriptors_config${ci + 1}.bin`, doublesToBinary(g), false);
    }
    this.log('**************** End Calculating Descriptors ****************');
  }
}

/** fmt's {:^w}: centred, an odd remainder on the right. */
const center = (s: string, w: number): string => {
  if (s.length >= w) return s;
  const left = Math.floor((w - s.length) / 2);
  return ' '.repeat(left) + s + ' '.repeat(w - s.length - left);
};


const norm = (a: ArrayLike<number>, n: number): number => {
  let e = a[0] * a[0];
  for (let i = 1; i < n; i++) e += a[i] * a[i];
  return Math.sqrt(e);
};

/** linspace(start, end, num): rounded, evenly spaced integers (FitPOD::linspace). */
const linspace = (start: number, end: number, num: number): number[] => {
  const out: number[] = [];
  if (num === 0) return out;
  if (num === 1) { out.push(Math.round(start)); return out; }
  const delta = (end - start) / (num - 1);
  for (let i = 0; i < num - 1; i++) out.push(roundHalfAway(start + delta * i));
  out.push(roundHalfAway(end));
  return out;
};

/** FitPOD::shuffle: num of the integers start..end in random order (std::shuffle with std::random_device natively). */
const shuffle = (start: number, end: number, num: number): number[] => {
  const v: number[] = [];
  for (let i = start; i <= end; i++) v.push(i);
  for (let i = v.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const t = v[i]; v[i] = v[j]; v[j] = t;
  }
  return v.slice(0, num);
};

/** std::round: halves away from zero. */
const roundHalfAway = (v: number): number => (v < 0 ? -Math.round(-v) : Math.round(v));

/** FitPOD::matrix33_inverse of the lattice columns a1, a2, a3 at L[o..o+8]. */
const matrix33Inverse = (L: Float64Array, o: number): Float64Array => {
  const a11 = L[o], a21 = L[o + 1], a31 = L[o + 2];
  const a12 = L[o + 3], a22 = L[o + 4], a32 = L[o + 5];
  const a13 = L[o + 6], a23 = L[o + 7], a33 = L[o + 8];
  const detA = (a11 * a22 * a33 - a11 * a23 * a32 - a12 * a21 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a13 * a22 * a31);
  const inv = new Float64Array(9);
  inv[0] = (a22 * a33 - a23 * a32) / detA;
  inv[1] = (a23 * a31 - a21 * a33) / detA;
  inv[2] = (a21 * a32 - a22 * a31) / detA;
  inv[3] = (a13 * a32 - a12 * a33) / detA;
  inv[4] = (a11 * a33 - a13 * a31) / detA;
  inv[5] = (a12 * a31 - a11 * a32) / detA;
  inv[6] = (a12 * a23 - a13 * a22) / detA;
  inv[7] = (a13 * a21 - a11 * a23) / detA;
  inv[8] = (a11 * a22 - a12 * a21) / detA;
  return inv;
};

/** FitPOD::triclinic_lattice_conversion, in place on the lattice columns at L[o..o+8]. */
const triclinicLatticeConversion = (L: Float64Array, o: number): void => {
  const A = [L[o], L[o + 1], L[o + 2]], B = [L[o + 3], L[o + 4], L[o + 5]], C = [L[o + 6], L[o + 7], L[o + 8]];
  const Anorm = Math.sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
  const Bnorm = Math.sqrt(B[0] * B[0] + B[1] * B[1] + B[2] * B[2]);
  const Cnorm = Math.sqrt(C[0] * C[0] + C[1] * C[1] + C[2] * C[2]);
  const Ahat = [A[0] / Anorm, A[1] / Anorm, A[2] / Anorm];
  const ax = Anorm;
  const bx = B[0] * Ahat[0] + B[1] * Ahat[1] + B[2] * Ahat[2];
  const by = Math.sqrt(Bnorm * Bnorm - bx * bx);
  const cx = C[0] * Ahat[0] + C[1] * Ahat[1] + C[2] * Ahat[2];
  const cy = (B[0] * C[0] + B[1] * C[1] + B[2] * C[2] - bx * cx) / by;
  const cz = Math.sqrt(Cnorm * Cnorm - cx * cx - cy * cy);
  L[o] = ax; L[o + 1] = 0.0; L[o + 2] = 0.0;
  L[o + 3] = bx; L[o + 4] = by; L[o + 5] = 0.0;
  L[o + 6] = cx; L[o + 7] = cy; L[o + 8] = cz;
};

/** FitPOD::matrix33_multiplication(x, Q, x, natom) in place at offset o. */
const rotate = (x: Float64Array, o: number, Q: Float64Array, natom: number): void => {
  for (let i = 0; i < natom; i++) {
    const k = o + 3 * i;
    const x1 = x[k], x2 = x[k + 1], x3 = x[k + 2];
    x[k] = Q[0] * x1 + Q[3] * x2 + Q[6] * x3;
    x[k + 1] = Q[1] * x1 + Q[4] * x2 + Q[7] * x3;
    x[k + 2] = Q[2] * x1 + Q[5] * x2 + Q[8] * x3;
  }
};

/**
 * DPOSV('U', n, 1, A, n, b, n): the Cholesky factor U (A = U^T U, the upper triangle, column by column as the unblocked
 * reference DPOTF2 does) and the two triangular solves (DPOTRS). Native ignores the info code; so does this.
 */
const dposv = (A: Float64Array, b: Float64Array, n: number): void => {
  for (let j = 0; j < n; j++) {
    let ajj = A[j + n * j];
    for (let k = 0; k < j; k++) ajj -= A[k + n * j] * A[k + n * j];
    if (!(ajj > 0)) { A[j + n * j] = ajj; return; }
    ajj = Math.sqrt(ajj);
    A[j + n * j] = ajj;
    for (let i = j + 1; i < n; i++) {
      let t = A[j + n * i];
      for (let k = 0; k < j; k++) t -= A[k + n * j] * A[k + n * i];
      A[j + n * i] = t / ajj;
    }
  }
  // U^T y = b
  for (let i = 0; i < n; i++) {
    let t = b[i];
    for (let k = 0; k < i; k++) t -= A[k + n * i] * b[k];
    b[i] = t / A[i + n * i];
  }
  // U x = y
  for (let i = n - 1; i >= 0; i--) {
    let t = b[i];
    for (let k = i + 1; k < n; k++) t -= A[i + n * k] * b[k];
    b[i] = t / A[i + n * i];
  }
};

const fitpod: Handler = ({ sys }, a) => {
  new FitPOD(sys).run(a);
};

export const COMMANDS = { fitpod };
