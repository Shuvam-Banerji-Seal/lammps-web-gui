import { describe, it, expect } from 'vitest';
import {
  generateBuildScript,
  DEFAULT_COMPILER_OPTIONS,
  LMP_PACKAGES,
  PRESETS,
  ACCELERATORS,
  CompilerOptions,
} from '../src/lammps/compiler';

/**
 * The full ALL_PACKAGES list from lammps/lammps `cmake/presets/most.cmake`
 * (fetched 2026-09-22). Snapshotted so the catalog cannot silently drift out
 * of sync with upstream — a missing package means the Compiler Helper cannot
 * build an input that needs it.
 */
const MOST_CMAKE_PACKAGES = [
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
];

describe('LAMMPS compiler helper', () => {
  it('covers every package in most.cmake, plus the special-toolchain ones', () => {
    const names = new Set(LMP_PACKAGES.map(p => p.name));
    const missing = MOST_CMAKE_PACKAGES.filter(p => !names.has(p));
    expect(missing).toEqual([]);
    // most.cmake deliberately omits these: they need CUDA/HIP/oneAPI.
    for (const p of ['GPU', 'INTEL', 'KOKKOS']) expect(names.has(p)).toBe(true);
  });

  it('has no duplicate package entries', () => {
    const names = LMP_PACKAGES.map(p => p.name);
    expect(names.length).toBe(new Set(names).size);
  });

  it('every package description is specific enough to be useful', () => {
    for (const pkg of LMP_PACKAGES) {
      expect(pkg.description.length, pkg.name).toBeGreaterThan(20);
      // Descriptions come from the package's own docs "Contents" line, so they
      // must not just restate the package name.
      expect(pkg.description.toUpperCase(), pkg.name).not.toBe(pkg.name);
    }
  });

  it('carries the corrected package descriptions (regression)', () => {
    const byName = new Map(LMP_PACKAGES.map(p => [p.name, p.description]));
    // Each of these was materially WRONG before the 2026-09-22 docs pass.
    expect(byName.get('MESONT')).toMatch(/nanotube/i);
    expect(byName.get('MESONT')).not.toMatch(/mesoporous/i);
    expect(byName.get('SHOCK')).toMatch(/msst|nphug|shock/i);
    expect(byName.get('SHOCK')).not.toMatch(/SPaSM/i);
    expect(byName.get('ORIENT')).toMatch(/grain.boundary/i);
    expect(byName.get('ORIENT')).not.toMatch(/resquared/i);
    expect(byName.get('ML-UF3')).toMatch(/ultra-fast/i);
    expect(byName.get('ML-UF3')).not.toMatch(/UFL3|four-body/i);
    expect(byName.get('MOFFF')).toMatch(/MOF-FF/);
    expect(byName.get('MOFFF')).not.toMatch(/diffusion/i);
    expect(byName.get('PHONON')).toMatch(/dynamical matri/i);
    expect(byName.get('BOCS')).toMatch(/barostat/i);
    expect(byName.get('YAFF')).toMatch(/pair_style yaff/);
  });

  it('presets reference known packages only', () => {
    const known = new Set(LMP_PACKAGES.map(p => p.name));
    for (const preset of PRESETS) {
      for (const pkg of preset.packages) {
        expect(known.has(pkg), `${preset.id} references unknown ${pkg}`).toBe(true);
      }
    }
  });

  it('linux script: clone + cmake -D PKG flags + parallel build', () => {
    const opts: CompilerOptions = {
      ...DEFAULT_COMPILER_OPTIONS,
      os: 'linux',
      presetId: 'minimal',
      accelerator: 'openmp',
      jobs: 12,
    };
    const out = generateBuildScript(opts);
    expect(out.text).toContain('git clone --depth 1 --branch develop');
    expect(out.text).toContain('-D PKG_MOLECULE=yes');
    expect(out.text).toContain('-D PKG_KSPACE=yes');
    expect(out.text).toContain('-D PKG_OPENMP=yes');
    expect(out.text).toContain('cmake --build . --parallel 12');
    expect(out.text).toContain('#!/usr/bin/env bash');
    expect(out.flags.some(f => f.startsWith('-D PKG_'))).toBe(true);
  });

  it('windows script: PowerShell + VS generator + lmp.exe path', () => {
    const opts: CompilerOptions = {
      ...DEFAULT_COMPILER_OPTIONS,
      os: 'windows',
      presetId: 'materials',
      withMpi: false,
    };
    const out = generateBuildScript(opts);
    expect(out.text).toContain('$ErrorActionPreference');
    expect(out.text).toContain('-G "Visual Studio 17 2022" -A x64');
    expect(out.text).toContain('.\\bin\\lmp.exe');
    expect(out.flags).toContain('-D BUILD_MPI=no');
    expect(out.warnings.join(' ')).toMatch(/serial build/i);
  });

  it('kokkos-cuda adds backend flags and warning', () => {
    const out = generateBuildScript({
      ...DEFAULT_COMPILER_OPTIONS,
      presetId: '',
      manualPackages: [],
      accelerator: 'kokkos-cuda',
    });
    expect(out.flags).toContain('-D PKG_KOKKOS=yes');
    expect(out.flags).toContain('-D Kokkos_ENABLE_CUDA=yes');
    expect(out.warnings.join(' ')).toMatch(/Kokkos_ARCH/);
  });

  it('non-default build options are emitted as -D flags', () => {
    const out = generateBuildScript({
      ...DEFAULT_COMPILER_OPTIONS,
      presetId: '',
      manualPackages: [],
      options: { ...DEFAULT_COMPILER_OPTIONS.options, FFT: 'MKL', LAMMPS_SIZES: 'bigbig' },
    });
    expect(out.flags).toContain('-D FFT=MKL');
    expect(out.flags).toContain('-D LAMMPS_SIZES=bigbig');
  });

  it('accelerator packages merge into preset selection without duplicates', () => {
    const out = generateBuildScript({
      ...DEFAULT_COMPILER_OPTIONS,
      presetId: 'most', // includes OPENMP
      accelerator: 'openmp',
    });
    const openmpFlags = out.flags.filter(f => f === '-D PKG_OPENMP=yes');
    expect(openmpFlags).toHaveLength(1);
  });

  it('accelerator catalog covers nvidia/amd/intel paths', () => {
    const vendors = ACCELERATORS.map(a => a.vendor);
    expect(vendors).toContain('nvidia');
    expect(vendors).toContain('amd');
    expect(vendors).toContain('intel');
  });

  it('flagDetails mirror flags 1:1 with non-empty descriptions and valid groups', () => {
    const out = generateBuildScript({
      ...DEFAULT_COMPILER_OPTIONS,
      presetId: 'most',
      accelerator: 'kokkos-cuda',
      withMpi: false,
      options: { ...DEFAULT_COMPILER_OPTIONS.options, FFT: 'MKL' },
    });
    expect(out.flagDetails.map(d => d.flag)).toEqual(out.flags);
    const groups = ['package', 'accelerator', 'option', 'build', 'mpi'];
    for (const d of out.flagDetails) {
      expect(d.description.trim().length, d.flag).toBeGreaterThan(5);
      expect(groups, d.flag).toContain(d.group);
      expect(d.source.trim().length, d.flag).toBeGreaterThan(0);
    }
    const byFlag = new Map(out.flagDetails.map(d => [d.flag, d]));
    expect(byFlag.get('-D PKG_KSPACE=yes')?.description).toMatch(/long-range|Coulomb/i);
    expect(byFlag.get('-D Kokkos_ENABLE_CUDA=yes')?.group).toBe('accelerator');
    expect(byFlag.get('-D BUILD_MPI=no')?.group).toBe('mpi');
    expect(byFlag.get('-D FFT=MKL')?.group).toBe('option');
    expect(byFlag.get('-D CMAKE_BUILD_TYPE=Release')?.description).toMatch(/optimiz/i);
  });

  it('package descriptions surface from the catalog', () => {
    const out = generateBuildScript({ ...DEFAULT_COMPILER_OPTIONS, presetId: 'minimal' });
    const kspace = out.flagDetails.find(d => d.flag === '-D PKG_KSPACE=yes');
    expect(kspace?.description).toMatch(/PPPM|Ewald|long-range/i);
    expect(kspace?.source).toMatch(/preset/i);
  });
});
