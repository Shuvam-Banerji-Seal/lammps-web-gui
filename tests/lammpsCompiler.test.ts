import { describe, it, expect } from 'vitest';
import {
  generateBuildScript,
  DEFAULT_COMPILER_OPTIONS,
  LMP_PACKAGES,
  PRESETS,
  ACCELERATORS,
  CompilerOptions,
} from '../src/lammps/compiler';

describe('LAMMPS compiler helper', () => {
  it('package catalog contains the most.cmake set and accelerators', () => {
    const names = new Set(LMP_PACKAGES.map(p => p.name));
    for (const p of ['MOLECULE', 'KSPACE', 'REAXFF', 'KOKKOS', 'GPU', 'OPENMP']) {
      expect(names.has(p)).toBe(true);
    }
    expect(LMP_PACKAGES.length).toBeGreaterThanOrEqual(60);
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
