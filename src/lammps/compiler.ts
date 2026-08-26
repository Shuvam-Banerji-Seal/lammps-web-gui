/**
 * LAMMPS build-system catalog — packages, presets, accelerator backends.
 * Sources [VERIFIED 2026-08-22, git 4Jul2026]:
 *  - lammps/lammps develop branch: cmake/presets/most.cmake (68 packages)
 *  - docs.lammps.org/Build_settings.html (FFT, sizes, gzip, curl, memalign)
 *  - docs.lammps.org Build chapter (accelerator packages, Windows notes)
 */

export interface LmpPackage {
  name: string;         // CMake flag suffix: PKG_<NAME>
  description: string;
  category: 'core-adjacent' | 'force-fields' | 'methods' | 'molecular' | 'ml' | 'mesoscale' | 'io' | 'accel';
  /** Requires external libraries / special toolchain. */
  heavy?: boolean;
}

export const LMP_PACKAGES: LmpPackage[] = [
  // Force fields & many-body
  { name: 'AMOEBA', description: 'AMOEBA/HIPPO polarizable force fields', category: 'force-fields' },
  { name: 'CLASS2', description: 'Class II force fields (COMPASS)', category: 'force-fields' },
  { name: 'COLLOID', description: 'Colloid interactions (coarse spheres)', category: 'force-fields' },
  { name: 'DIELECTRIC', description: 'Surface polarization / dielectrics', category: 'force-fields' },
  { name: 'DIPOLE', description: 'Point dipole interactions', category: 'force-fields' },
  { name: 'DRUDE', description: 'Drude oscillators (polarizable)', category: 'force-fields', heavy: true },
  { name: 'FEP', description: 'Free-energy perturbation styles', category: 'force-fields' },
  { name: 'GRANULAR', description: 'Granular (sand/sphere) contacts', category: 'force-fields' },
  { name: 'MANYBODY', description: 'Tersoff, SW, EIM, AIREBO, Stillinger-Weber…', category: 'force-fields' },
  { name: 'MEAM', description: 'Modified embedded atom method', category: 'force-fields' },
  { name: 'MOLECULE', description: 'Bonds/angles/dihedrals/impropers', category: 'molecular' },
  { name: 'QEQ', description: 'Charge equilibration (ReaxFF aid)', category: 'force-fields' },
  { name: 'REACTION', description: 'Reactive force fields + bond breaking', category: 'force-fields' },
  { name: 'REAXFF', description: 'ReaxFF reactive force field', category: 'force-fields' },
  { name: 'SPIN', description: 'Magnetic spin dynamics', category: 'force-fields' },

  // Methods
  { name: 'KSPACE', description: 'Long-range Coulomb (PPPM/Ewald/MSM)', category: 'methods' },
  { name: 'RIGID', description: 'Rigid body integration', category: 'methods' },
  { name: 'SHOCK', description: 'Shock dynamics (SPaSM)', category: 'methods' },
  { name: 'MC', description: 'Monte Carlo (fix gcmc etc.)', category: 'methods' },
  { name: 'PHONON', description: 'Phonon/DOS analysis (dyn mat, QM)', category: 'methods' },
  { name: 'REPLICA', description: 'Replica exchange, NEB, PRD, TAD', category: 'methods' },
  { name: 'UEF', description: 'Uniaxial extensional flow', category: 'methods' },
  { name: 'INTERLAYER', description: 'Layered materials potentials', category: 'force-fields' },

  // ML potentials
  { name: 'ML-SNAP', description: 'SNAP / qSNAP potentials', category: 'ml' },
  { name: 'ML-POD', description: 'Pytorch-free POD potentials', category: 'ml' },
  { name: 'ML-IAP', description: 'Implicit/analytic ML potentials (mliap)', category: 'ml' },
  { name: 'ML-UF3', description: 'UFL3 four-body ML potentials', category: 'ml' },

  // Mesoscale
  { name: 'DPD-BASIC', description: 'Dissipative particle dynamics (basic)', category: 'mesoscale' },
  { name: 'DPD-MESO', description: 'eDPD/mDPD/tDPD mesoscale', category: 'mesoscale' },
  { name: 'DPD-REACT', description: 'Reactive DPD', category: 'mesoscale' },
  { name: 'DPD-SMOOTH', description: 'Smoothed DPD', category: 'mesoscale' },
  { name: 'PERI', description: 'Peridynamics', category: 'mesoscale' },
  { name: 'RHEO', description: 'RHEO solid/fluid mesoscale', category: 'mesoscale' },
  { name: 'SPH', description: 'Smoothed particle hydrodynamics', category: 'mesoscale' },
  { name: 'MACHDYN', description: 'Smooth Mach dynamics (SMD)', category: 'mesoscale' },
  { name: 'BPM', description: 'Bonded particle models', category: 'mesoscale' },

  // Specialized
  { name: 'ASPHERE', description: 'Ellipsoid/line/tri aspherical particles', category: 'methods' },
  { name: 'BODY', description: 'Arbitrary body particles', category: 'mesoscale' },
  { name: 'BROWNIAN', description: 'Brownian pair styles', category: 'force-fields' },
  { name: 'BOCS', description: 'BOCS bottom-up coarse-graining', category: 'methods' },
  { name: 'CG-DNA', description: 'Coarse-grained DNA (oxDNA)', category: 'molecular' },
  { name: 'CG-SPICA', description: 'SPICA coarse-grained FF', category: 'molecular' },
  { name: 'CORESHELL', description: 'Core-shell particles', category: 'methods' },
  { name: 'COLVARS', description: 'Collective variables (enhanced sampling)', category: 'methods', heavy: true },
  { name: 'COMPRESS', description: 'Compressed dump/read via zlib', category: 'io' },
  { name: 'DIFFRACTION', description: 'X-ray/electron diffraction computes', category: 'methods' },
  { name: 'EFF', description: 'Electron force field', category: 'force-fields' },
  { name: 'ELECTRODE', description: 'Constant-potential electrodes', category: 'methods', heavy: true },
  { name: 'EXTRA-COMMAND', description: 'Extra commands (balance, quip…)', category: 'methods' },
  { name: 'EXTRA-COMPUTE', description: 'Extra compute styles', category: 'methods' },
  { name: 'EXTRA-DUMP', description: 'Extra dump styles', category: 'io' },
  { name: 'EXTRA-FIX', description: 'Extra fix styles', category: 'methods' },
  { name: 'EXTRA-MOLECULE', description: 'Extra molecule/bond/angle styles', category: 'molecular' },
  { name: 'EXTRA-PAIR', description: 'Extra pair styles', category: 'force-fields' },
  { name: 'GRANSURF', description: 'Granular surfaces', category: 'force-fields' },
  { name: 'GRAPHICS', description: 'Dump styles for rendering (image/vtk)', category: 'io' },
  { name: 'LEPTON', description: 'Lepton expression potentials', category: 'force-fields' },
  { name: 'MESONT', description: 'Mesoporous nanoparticles', category: 'mesoscale' },
  { name: 'MISC', description: 'Miscellaneous fixes/computes', category: 'methods' },
  { name: 'MOFFF', description: 'Metal-organic framework diffusion', category: 'methods' },
  { name: 'ORIENT', description: 'Orientational potentials (resquared…)', category: 'force-fields' },
  { name: 'PLUGIN', description: 'Load external plugins at runtime', category: 'io' },
  { name: 'SRD', description: 'Stochastic rotation dynamics', category: 'mesoscale' },
  { name: 'TALLY', description: 'Per-atom energy/stress tallying', category: 'methods' },
  { name: 'VORONOI', description: 'Voronoi volume computes', category: 'methods', heavy: true },
  { name: 'YAFF', description: 'YAFF force field styles', category: 'force-fields' },
  { name: 'OPENMP', description: 'OpenMP-accelerated styles (/omp)', category: 'accel' },
  { name: 'OPT', description: 'Optimized CPU styles (/opt)', category: 'accel' },
  { name: 'INTEL', description: 'Intel SIMD-accelerated styles', category: 'accel', heavy: true },
  { name: 'GPU', description: 'NVIDIA/AMD GPU package (lib/gpu)', category: 'accel', heavy: true },
  { name: 'KOKKOS', description: 'Kokkos C++ performance portability', category: 'accel', heavy: true },
];

export const PACKAGE_CATEGORIES = [
  'accel',
  'force-fields',
  'molecular',
  'methods',
  'ml',
  'mesoscale',
  'io',
] as const;

/** Accelerator / hardware backends. */
export interface Accelerator {
  id: 'cpu' | 'openmp' | 'opt' | 'intel' | 'gpu-cuda' | 'kokkos-cuda' | 'kokkos-hip' | 'kokkos-sycl';
  label: string;
  vendor: 'any' | 'nvidia' | 'amd' | 'intel';
  packages: string[];         // PKG_* flags
  extraFlags?: string[];      // additional -D flags
  notes?: string;
}

export const ACCELERATORS: Accelerator[] = [
  {
    id: 'cpu',
    label: 'CPU (serial / MPI only)',
    vendor: 'any',
    packages: [],
  },
  {
    id: 'openmp',
    label: 'CPU + OpenMP threads (/omp styles)',
    vendor: 'any',
    packages: ['OPENMP'],
  },
  {
    id: 'opt',
    label: 'CPU optimized styles (/opt)',
    vendor: 'any',
    packages: ['OPT'],
  },
  {
    id: 'intel',
    label: 'Intel CPU SIMD (INTEL package)',
    vendor: 'intel',
    packages: ['INTEL'],
    notes: 'Best with Intel compilers / MKL.',
  },
  {
    id: 'gpu-cuda',
    label: 'NVIDIA GPU — GPU package (CUDA)',
    vendor: 'nvidia',
    packages: ['GPU'],
    extraFlags: [],
    notes: 'Requires CUDA toolkit; pair styles get /gpu suffix at runtime. Set GPU_API=cuda + GPU_ARCH for your card (Build options).',
  },
  {
    id: 'kokkos-cuda',
    label: 'NVIDIA GPU — KOKKOS/CUDA (full GPU-resident)',
    vendor: 'nvidia',
    packages: ['KOKKOS'],
    extraFlags: [
      '-D Kokkos_ENABLE_CUDA=yes',
      '-D CMAKE_CXX_STANDARD=17',
      '-D Kokkos_ARCH_VOLTA70=yes',
    ],
    notes: 'Set Kokkos_ARCH to your GPU arch (VOLTA70/AMPERE80/…).',
  },
  {
    id: 'kokkos-hip',
    label: 'AMD GPU — KOKKOS/HIP',
    vendor: 'amd',
    packages: ['KOKKOS'],
    extraFlags: [
      '-D Kokkos_ENABLE_HIP=yes',
      '-D CMAKE_CXX_STANDARD=17',
      '-D Kokkos_ARCH_VEGA90A=yes',
    ],
    notes: 'Requires ROCm; set Kokkos_ARCH for your GPU (VEGA90A/MI200…).',
  },
  {
    id: 'kokkos-sycl',
    label: 'Intel GPU — KOKKOS/SYCL (oneAPI)',
    vendor: 'intel',
    packages: ['KOKKOS'],
    extraFlags: [
      '-D Kokkos_ENABLE_SYCL=yes',
      '-D CMAKE_CXX_STANDARD=17',
    ],
    notes: 'Requires Intel oneAPI DPC++ toolchain.',
  },
];

export interface BuildOption {
  key: string;               // -D key
  label: string;
  values: string[];          // allowed values ('' = boolean yes/no)
  default: string;
  help?: string;
}

export const BUILD_OPTIONS: BuildOption[] = [
  // --- basics (docs.lammps.org/Build_basics.html) ---
  { key: 'BUILD_OMP', label: 'OpenMP threading', values: ['yes', 'no'], default: 'yes', help: 'Core + OPENMP/INTEL/KOKKOS packages; set OMP_NUM_THREADS at runtime.' },
  { key: 'BUILD_SHARED_LIBS', label: 'Shared library (liblammps.so)', values: ['yes', 'no'], default: 'no' },
  { key: 'BUILD_TOOLS', label: 'Build auxiliary tools', values: ['yes', 'no'], default: 'yes' },
  { key: 'BUILD_LAMMPS_GUI', label: 'Build LAMMPS GUI', values: ['yes', 'no'], default: 'no' },
  { key: 'BUILD_WHAM', label: 'Build WHAM (with GUI)', values: ['yes', 'no'], default: 'yes' },
  { key: 'LAMMPS_INSTALL_RPATH', label: 'Embed runtime library path', values: ['no', 'yes'], default: 'no', help: 'Handy for installs outside system paths.' },
  // --- FFT (docs.lammps.org/Build_settings.html) ---
  { key: 'FFT', label: 'FFT library', values: ['FFTW3', 'MKL', 'NVPL', 'KISS'], default: 'FFTW3', help: 'Used by KSPACE (pppm). KISS is bundled.' },
  { key: 'FFT_KOKKOS', label: 'FFT library (Kokkos)', values: ['KISS', 'FFTW3', 'MKL', 'NVPL', 'CUFFT', 'HIPFFT', 'MKL_GPU'], default: 'KISS', help: 'Applies when Kokkos styles run; must match the back end (CUFFT for CUDA).' },
  { key: 'FFT_SINGLE', label: 'Single-precision FFT', values: ['yes', 'no'], default: 'no', help: 'Trades a little accuracy for less memory/communication.' },
  { key: 'FFT_PACK', label: 'FFT data packing', values: ['array', 'pointer', 'memcpy'], default: 'array' },
  { key: 'FFT_USE_HEFFTE', label: 'Use heFFTe FFT', values: ['yes', 'no'], default: 'no', help: 'Highly optimized MPI FFT communication layer.' },
  { key: 'FFT_HEFFTE_BACKEND', label: 'heFFTe back end', values: ['', 'FFTW', 'MKL'], default: '', help: "Empty = stock back end (testing only)." },
  // --- I/O & sizes ---
  { key: 'WITH_GZIP', label: 'gzip I/O support', values: ['yes', 'no'], default: 'yes' },
  { key: 'WITH_CURL', label: 'geturl download support', values: ['yes', 'no'], default: 'yes' },
  { key: 'WITH_JPEG', label: 'JPEG output (dump image)', values: ['yes', 'no'], default: 'yes', help: 'GRAPHICS package.' },
  { key: 'WITH_PNG', label: 'PNG output (dump image)', values: ['yes', 'no'], default: 'yes', help: 'GRAPHICS package.' },
  { key: 'WITH_FFMPEG', label: 'dump movie support', values: ['yes', 'no'], default: 'yes', help: 'GRAPHICS package; needs ffmpeg at runtime.' },
  { key: 'LAMMPS_SIZES', label: 'Integer sizes', values: ['smallbig', 'bigbig'], default: 'smallbig', help: 'bigbig for >2B atom IDs / image flags.' },
  { key: 'LAMMPS_MEMALIGN', label: 'Memory alignment', values: ['0', '8', '16', '32', '64'], default: '64' },
  { key: 'LAMMPS_EXCEPTIONS', label: 'C++ exception handling', values: ['yes', 'no'], default: 'no', help: 'For library use — errors throw instead of aborting.' },
  { key: 'LAMMPS_LONGLONG_TO_LONG', label: 'long long → long workaround', values: ['yes', 'no'], default: 'no' },
  { key: 'DOWNLOAD_POTENTIALS', label: 'Download large potentials', values: ['yes', 'off'], default: 'yes' },
  // --- accelerator tuning (docs.lammps.org/Build_extras.html) ---
  { key: 'GPU_API', label: 'GPU back end', values: ['opencl', 'cuda', 'hip'], default: 'opencl', help: 'GPU package only.' },
  { key: 'GPU_PREC', label: 'GPU precision', values: ['mixed', 'double', 'single'], default: 'mixed', help: 'GPU package only. mixed = most of the speed, forces in double.' },
  { key: 'GPU_ARCH', label: 'GPU architecture', values: ['', 'sm_75', 'sm_80', 'sm_86', 'sm_89', 'sm_90', 'gfx906', 'gfx1030', 'gfx1100', 'spirv'], default: '', help: 'GPU package only. Empty = multiarch (slower builds).' },
  { key: 'KOKKOS_PREC', label: 'Kokkos precision', values: ['double', 'mixed', 'single'], default: 'double', help: 'KOKKOS package only. mixed = FP64 accumulation, FP32 elsewhere.' },
  { key: 'Kokkos_ENABLE_DEBUG', label: 'Kokkos debug checks', values: ['no', 'yes'], default: 'no', help: 'KOKKOS package only. Big performance cost — development only.' },
  { key: 'Kokkos_ENABLE_CUDA_UVM', label: 'Kokkos CUDA UVM', values: ['no', 'yes'], default: 'no', help: 'KOKKOS package only. Lets RAM supplement GPU memory (slower).' },
  { key: 'Kokkos_ENABLE_OPENMP', label: 'Kokkos OpenMP host', values: ['no', 'yes'], default: 'no', help: 'KOKKOS package only. Requires BUILD_OMP=yes.' },
  { key: 'KOKKOS_LAYOUT', label: 'Kokkos array layout', values: ['legacy', 'default'], default: 'legacy', help: 'KOKKOS package only. default (LayoutLeft) may speed up some GPU models.' },
  // --- GPU package extras ---
  { key: 'GPU_DEBUG', label: 'GPU debug code', values: ['no', 'yes'], default: 'no', help: 'GPU package only. Developer debugging.' },
  { key: 'CUDA_MPS_SUPPORT', label: 'CUDA MPS support', values: ['no', 'yes'], default: 'no', help: 'GPU package only. For nvidia-cuda-mps daemon.' },
  { key: 'CUDA_BUILD_MULTIARCH', label: 'CUDA multiarch kernels', values: ['yes', 'no'], default: 'yes', help: 'GPU package only. Build for all supported GPU archs.' },
  { key: 'USE_STATIC_OPENCL_LOADER', label: 'Static OpenCL loader', values: ['yes', 'no'], default: 'yes', help: 'GPU package only. Downloads OpenCL ICD loader — no local headers needed.' },
  // --- package-specific downloads & options ---
  { key: 'LAMMPS_MACHINE', label: 'Executable suffix', values: ['', 'mpi', 'serial'], default: '', help: 'Names the binary lmp_<suffix>. Empty = plain lmp.' },
  { key: 'DOWNLOAD_KIM', label: 'Download OpenKIM API', values: ['no', 'yes'], default: 'no', help: 'KIM package only.' },
  { key: 'KIM_EXTRA_UNITTESTS', label: 'KIM extra unit tests', values: ['no', 'yes'], default: 'no', help: 'KIM package only. Requires internet + Python.' },
  { key: 'LEPTON_ENABLE_JIT', label: 'Lepton JIT compiler', values: ['yes', 'no'], default: 'yes', help: 'LEPTON package only. Auto-detected on x86.' },
  { key: 'DOWNLOAD_EIGEN3', label: 'Download Eigen3', values: ['no', 'yes'], default: 'no', help: 'MACHDYN package only.' },
  { key: 'MLIAP_ENABLE_PYTHON', label: 'ML-IAP Python support', values: ['no', 'yes'], default: 'no', help: 'ML-IAP package only. Requires Python 3.6+ and cython.' },
  { key: 'DOWNLOAD_VORO', label: 'Download Voro++', values: ['no', 'yes'], default: 'no', help: 'VORONOI package only.' },
  { key: 'COLVARS_LEPTON', label: 'Colvars Lepton support', values: ['yes', 'no'], default: 'yes', help: 'COLVARS package only.' },
  { key: 'COLVARS_DEBUG', label: 'Colvars debug messages', values: ['no', 'yes'], default: 'no', help: 'COLVARS package only. Verbose output.' },
  { key: 'Python_EXECUTABLE', label: 'Python executable path', values: ['', '/usr/bin/python3'], default: '', help: 'PYTHON package only. Empty = CMake auto-detect.' },
  { key: 'DOWNLOAD_QUIP', label: 'Download QUIP library', values: ['no', 'yes'], default: 'no', help: 'ML-QUIP package only. Non-commercial licence.' },
  { key: 'USE_INTERNAL_LINALG', label: 'Use internal LAPACK', values: ['no', 'yes'], default: 'no', help: 'ML-QUIP/ELECTRODE package only. Workaround for LAPACK link issues.' },
];

export interface NamedPreset {
  id: string;
  label: string;
  description: string;
  packages: string[];
}

/** Presets modeled on the repo's cmake/presets/*.cmake files. */
export const PRESETS: NamedPreset[] = [
  {
    id: 'minimal',
    label: 'Minimal (core only)',
    description: 'Core + MOLECULE + KSPACE — smallest useful build.',
    packages: ['MOLECULE', 'KSPACE'],
  },
  {
    id: 'most',
    label: 'Most packages (mirrors most.cmake)',
    description: 'Wide range incl. manybody, ML, mesoscale; no heavy externals.',
    packages: [
      'AMOEBA', 'ASPHERE', 'BOCS', 'BODY', 'BPM', 'BROWNIAN', 'CG-DNA', 'CG-SPICA',
      'CLASS2', 'COLLOID', 'COLVARS', 'COMPRESS', 'CORESHELL', 'DIELECTRIC',
      'DIFFRACTION', 'DIPOLE', 'DPD-BASIC', 'DPD-MESO', 'DPD-REACT', 'DPD-SMOOTH',
      'DRUDE', 'EFF', 'ELECTRODE', 'EXTRA-COMMAND', 'EXTRA-COMPUTE', 'EXTRA-DUMP',
      'EXTRA-FIX', 'EXTRA-MOLECULE', 'EXTRA-PAIR', 'FEP', 'GRAPHICS', 'GRANULAR',
      'GRANSURF', 'INTERLAYER', 'KSPACE', 'LEPTON', 'MACHDYN', 'MANYBODY', 'MC',
      'MEAM', 'MESONT', 'MISC', 'ML-IAP', 'ML-POD', 'ML-SNAP', 'ML-UF3', 'MOFFF',
      'MOLECULE', 'OPENMP', 'OPT', 'ORIENT', 'PERI', 'PHONON', 'PLUGIN', 'QEQ',
      'REACTION', 'REAXFF', 'REPLICA', 'RHEO', 'RIGID', 'SHOCK', 'SPH', 'SPIN',
      'SRD', 'TALLY', 'UEF', 'VORONOI', 'YAFF',
    ],
  },
  {
    id: 'bio',
    label: 'Biomolecular',
    description: 'Molecular dynamics for proteins / DNA: CHARMM/AMBER styles, Drude, FEP.',
    packages: ['MOLECULE', 'KSPACE', 'FEP', 'DRUDE', 'CG-DNA', 'EXTRA-MOLECULE', 'RIGID'],
  },
  {
    id: 'materials',
    label: 'Materials / metals',
    description: 'Metals & semiconductors: EAM, MEAM, manybody, KSPACE.',
    packages: ['MANYBODY', 'MEAM', 'KSPACE', 'EXTRA-PAIR'],
  },
  {
    id: 'reactive',
    label: 'Reactive chemistry',
    description: 'ReaxFF + QEQ + molecule topologies.',
    packages: ['REAXFF', 'QEQ', 'MOLECULE', 'KSPACE', 'EXTRA-FIX'],
  },
];

export type OsTarget = 'linux' | 'windows';

export interface CompilerOptions {
  os: OsTarget;
  presetId: string;                 // '' = manual selection
  manualPackages: string[];         // used when presetId === ''
  accelerator: Accelerator['id'];
  buildType: 'Release' | 'Debug' | 'RelWithDebInfo';
  withMpi: boolean;
  options: Record<string, string>;  // BUILD_OPTIONS overrides
  jobs: number;                     // parallel build -j
  repoUrl: string;
  branch: string;
}

export const DEFAULT_COMPILER_OPTIONS: CompilerOptions = {
  os: 'linux',
  presetId: 'most',
  manualPackages: [],
  accelerator: 'cpu',
  buildType: 'Release',
  withMpi: true,
  options: Object.fromEntries(BUILD_OPTIONS.map(o => [o.key, o.default])),
  jobs: 8,
  repoUrl: 'https://github.com/lammps/lammps.git',
  branch: 'develop',
};

export interface CompilerScript {
  /** One big shell/PowerShell script text. */
  text: string;
  /** cmake -D flag list (for display). */
  flags: string[];
  /**
   * Structured view of `flags` with human-readable descriptions and the
   * UI group each flag came from — powers the click-to-inspect chip list.
   */
  flagDetails: FlagDetail[];
  warnings: string[];
}

export type FlagGroup = 'package' | 'accelerator' | 'option' | 'build' | 'mpi';

export interface FlagDetail {
  flag: string;
  description: string;
  group: FlagGroup;
  /** Where in the UI this flag can be changed. */
  source: string;
}

const FLAG_DESCRIPTIONS: Record<string, { description: string; source: string }> = {
  '-D BUILD_MPI=no': {
    description: 'Compile without MPI support: produces a serial lmp binary that runs on one process.',
    source: 'MPI toggle',
  },
  '-D CUDPP_OPT=yes': {
    description: 'Build the bundled CUDPP library used by the GPU package for faster neighbor sorting on NVIDIA GPUs.',
    source: 'Accelerator backend',
  },
};

const kokkosArchHint = (flag: string): string =>
  `Selects the Kokkos GPU architecture to compile for (${flag.replace('-D Kokkos_ARCH_', '').replace('=yes', '')}). ` +
  'Set it to match your hardware, e.g. VOLTA70, AMPERE80, VEGA90A, MI300.';

const describeFlag = (
  flag: string,
  packages: Map<string, LmpPackage>,
  accelerators: Map<string, Accelerator>,
  buildOptions: Map<string, BuildOption>
): { description: string; group: FlagGroup; source: string } => {
  if (flag.startsWith('-D PKG_')) {
    const name = flag.slice('-D PKG_'.length).replace(/=.*$/, '');
    const pkg = packages.get(name);
    return {
      description: pkg
        ? `${pkg.description}${pkg.heavy ? ' Requires external libraries or a special toolchain.' : ''}`
        : 'Optional LAMMPS package compiled into the binary.',
      group: 'package',
      source: 'Package preset / manual selection',
    };
  }
  if (FLAG_DESCRIPTIONS[flag]) {
    const known = FLAG_DESCRIPTIONS[flag];
    const acc = accelerators.get('gpu-cuda');
    return {
      ...known,
      group: flag === '-D BUILD_MPI=no' ? 'mpi' : 'accelerator',
      ...(flag.startsWith('-D CUDPP') && acc?.notes ? {} : {}),
      source: known.source,
    };
  }
  if (flag.startsWith('-D Kokkos_ARCH_')) {
    return {
      description: kokkosArchHint(flag),
      group: 'accelerator',
      source: 'Accelerator backend',
    };
  }
  if (flag.startsWith('-D Kokkos_ENABLE_')) {
    const backend = flag.replace('-D Kokkos_ENABLE_', '').replace('=yes', '');
    return {
      description: `Enables the ${backend} backend of the bundled Kokkos runtime so pair styles run on that device.`,
      group: 'accelerator',
      source: 'Accelerator backend',
    };
  }
  if (flag === '-D CMAKE_CXX_STANDARD=17') {
    return {
      description: 'Pins the C++ standard to 17 — the minimum required by modern LAMMPS and its Kokkos dependency.',
      group: 'accelerator',
      source: 'Accelerator backend',
    };
  }
  if (flag.startsWith('-D CMAKE_BUILD_TYPE=')) {
    return {
      description:
        'Compiler optimization level: Release = fully optimized (-O3), RelWithDebInfo = optimized + debug symbols, Debug = unoptimized with assertions.',
      group: 'build',
      source: 'Build type selector',
    };
  }
  // Remaining -D <KEY>=<value> entries come from BUILD_OPTIONS.
  const key = flag.replace(/^-\s*/, '').replace(/^-\D\s*/, '').split('=')[0].trim();
  const optKey = flag.match(/-D ([A-Z0-9_]+)=/)?.[1] ?? '';
  const bo = buildOptions.get(optKey || key);
  if (bo) {
    return {
      description: `${bo.help ?? bo.label} (default: ${bo.default}).`,
      group: 'option',
      source: 'Build options',
    };
  }
  return {
    description: 'CMake configure flag for this LAMMPS build.',
    group: 'build',
    source: 'Generated configuration',
  };
};

const resolvePackages = (opts: CompilerOptions): string[] => {
  const preset = PRESETS.find(p => p.id === opts.presetId);
  const base = preset ? preset.packages : opts.manualPackages;
  const acc = ACCELERATORS.find(a => a.id === opts.accelerator);
  const set = new Set([...base, ...(acc?.packages ?? [])]);
  return Array.from(set).sort();
};

/**
 * Generate a full clone + configure + build script for the chosen target.
 * Linux → bash; Windows → PowerShell. Commands are printed for the user
 * to run locally — nothing is executed here.
 */
export const generateBuildScript = (opts: CompilerOptions): CompilerScript => {
  const warnings: string[] = [];
  const flags: string[] = [];
  const flagDetails: FlagDetail[] = [];

  const packages = resolvePackages(opts);
  const pkgMap = new Map(LMP_PACKAGES.map(p => [p.name, p]));
  const accMap = new Map(ACCELERATORS.map(a => [a.id, a]));
  const optMap = new Map(BUILD_OPTIONS.map(o => [o.key, o]));

  for (const pkg of packages) {
    const flag = `-D PKG_${pkg}=yes`;
    flags.push(flag);
    const d = describeFlag(flag, pkgMap, accMap, optMap);
    flagDetails.push({ flag, ...d });
  }

  const acc = ACCELERATORS.find(a => a.id === opts.accelerator);
  for (const f of acc?.extraFlags ?? []) {
    flags.push(f);
    const d = describeFlag(f, pkgMap, accMap, optMap);
    flagDetails.push({ flag: f, ...d });
  }
  if (acc?.notes) warnings.push(`${acc.label}: ${acc.notes}`);

  if (!opts.withMpi) {
    const flag = '-D BUILD_MPI=no';
    flags.push(flag);
    const d = describeFlag(flag, pkgMap, accMap, optMap);
    flagDetails.push({ flag, ...d });
    warnings.push('MPI disabled — serial build (fine for single-workstation runs).');
  }

  for (const bo of BUILD_OPTIONS) {
    const v = opts.options[bo.key] ?? bo.default;
    if (v !== bo.default) {
      const flag = `-D ${bo.key}=${v}`;
      flags.push(flag);
      const d = describeFlag(flag, pkgMap, accMap, optMap);
      flagDetails.push({ flag, ...d });
    }
  }
  flags.push(`-D CMAKE_BUILD_TYPE=${opts.buildType}`);
  flagDetails.push({
    flag: `-D CMAKE_BUILD_TYPE=${opts.buildType}`,
    ...describeFlag('-D CMAKE_BUILD_TYPE=x', pkgMap, accMap, optMap),
  });

  const cloneCmd = `git clone --depth 1 --branch ${opts.branch} ${opts.repoUrl} lammps`;
  const cmakeBase = ['cmake', '../cmake', ...flags].join(' \\\n    ');

  if (opts.os === 'linux') {
    const text = `#!/usr/bin/env bash
# ============================================================
# LAMMPS build script — generated by Molecule3D Workbench
# by Shuvam Banerji Seal
# Target: Linux · preset=${opts.presetId || 'manual'} · accelerator=${opts.accelerator}
# ============================================================
set -euo pipefail

# --- prerequisites (Debian/Ubuntu; adapt for your distro) ---
# sudo apt-get install -y build-essential cmake git mpi-default-dev \\
#   libfftw3-dev libopenmpi-dev

${cloneCmd}
cd lammps
mkdir -p build && cd build

# --- configure ---
${cmakeBase}

# --- build & install ---
cmake --build . --parallel ${opts.jobs}
sudo cmake --install .        # installs lmp + library (optional)

# --- sanity check ---
lmp -h | head -n 30
`;
    return { text, flags, flagDetails, warnings };
  }

  // Windows: PowerShell
  const text = `# ============================================================
# LAMMPS build script — generated by Molecule3D Workbench
# by Shuvam Banerji Seal
# Target: Windows · preset=${opts.presetId || 'manual'} · accelerator=${opts.accelerator}
# Run inside "x64 Native Tools Command Prompt for VS 2022" or with
# the Visual Studio generator available.
# ============================================================
$ErrorActionPreference = "Stop"

# --- prerequisites ---
# winget install Kitware.CMake Git.Git
# Install Visual Studio 2022 with "Desktop development with C++"
# Optional MPI: Microsoft MPI (msmpisetup.exe + msmpisdk.msi)

${cloneCmd}
cd lammps
if (-Not (Test-Path build)) { New-Item -ItemType Directory build | Out-Null }
cd build

# --- configure (Visual Studio generator) ---
${cmakeBase} -G "Visual Studio 17 2022" -A x64

# --- build ---
cmake --build . --config ${opts.buildType} --parallel ${opts.jobs}

# binary: .\\bin\\lmp.exe (copy DLLs next to it as needed)
.\\bin\\lmp.exe -h | Select-Object -First 30
`;
  return { text, flags, flagDetails, warnings };
};
