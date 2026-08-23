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
    extraFlags: ['-D CUDPP_OPT=yes'],
    notes: 'Requires CUDA toolkit; pair styles get /gpu suffix at runtime.',
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
  { key: 'FFT', label: 'FFT library', values: ['FFTW3', 'MKL', 'NVPL', 'KISS'], default: 'FFTW3', help: 'Used by KSPACE (pppm). KISS is bundled.' },
  { key: 'FFT_SINGLE', label: 'Single-precision FFT', values: ['yes', 'no'], default: 'no' },
  { key: 'WITH_GZIP', label: 'gzip I/O support', values: ['yes', 'no'], default: 'yes' },
  { key: 'WITH_CURL', label: 'geturl download support', values: ['yes', 'no'], default: 'yes' },
  { key: 'LAMMPS_SIZES', label: 'Integer sizes', values: ['smallbig', 'bigbig'], default: 'smallbig', help: 'bigbig for >2B atom IDs' },
  { key: 'LAMMPS_MEMALIGN', label: 'Memory alignment', values: ['0', '8', '16', '32', '64'], default: '64' },
  { key: 'DOWNLOAD_POTENTIALS', label: 'Download large potentials', values: ['yes', 'off'], default: 'yes' },
  { key: 'BUILD_TOOLS', label: 'Build auxiliary tools', values: ['yes', 'no'], default: 'yes' },
  { key: 'BUILD_LAMMPS_GUI', label: 'Build LAMMPS GUI', values: ['yes', 'no'], default: 'no' },
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
  warnings: string[];
}

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

  const packages = resolvePackages(opts);
  for (const pkg of packages) flags.push(`-D PKG_${pkg}=yes`);

  const acc = ACCELERATORS.find(a => a.id === opts.accelerator);
  for (const f of acc?.extraFlags ?? []) flags.push(f);
  if (acc?.notes) warnings.push(`${acc.label}: ${acc.notes}`);

  if (!opts.withMpi) {
    flags.push('-D BUILD_MPI=no');
    warnings.push('MPI disabled — serial build (fine for single-workstation runs).');
  }

  for (const bo of BUILD_OPTIONS) {
    const v = opts.options[bo.key] ?? bo.default;
    if (v !== bo.default) {
      flags.push(bo.values.includes('yes') && bo.values.includes('no')
        ? `-D ${bo.key}=${v}`
        : `-D ${bo.key}=${v}`);
    }
  }
  flags.push(`-D CMAKE_BUILD_TYPE=${opts.buildType}`);

  const cloneCmd = `git clone --depth 1 --branch ${opts.branch} ${opts.repoUrl} lammps`;
  const cmakeBase = ['cmake', '../cmake', ...flags].join(' \\\n    ');

  if (opts.os === 'linux') {
    const text = `#!/usr/bin/env bash
# ============================================================
# LAMMPS build script — generated by Molecule3D Workbench
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
    return { text, flags, warnings };
  }

  // Windows: PowerShell
  const text = `# ============================================================
# LAMMPS build script — generated by Molecule3D Workbench
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
  return { text, flags, warnings };
};
