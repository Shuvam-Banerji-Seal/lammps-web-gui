import { describe, it, expect } from 'vitest';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { readFileSync, rmSync, writeFileSync } from 'node:fs';
import {
  generateBuildScript,
  DEFAULT_COMPILER_OPTIONS,
  BUILD_OPTIONS,
  ACCELERATORS,
  type CompilerOptions,
} from '../src/lammps/compiler';

/**
 * Domain checks for the generated build scripts, pinned against
 * docs.lammps.org/Build_cmake.html and /Build_extras.html (fetched 2026-10-07).
 * NOTATION: a single \ below means one literal backslash in the generated text.
 */

const base = (over: Partial<CompilerOptions>): CompilerOptions => ({
  ...DEFAULT_COMPILER_OPTIONS,
  ...over,
});

/** Continuation markers: bash ' \' (docs.lammps.org/Build_cmake.html example),
 *  PowerShell ' `' (PowerShell has no backslash continuation). */
const isContinuation = (l: string): boolean => l.endsWith(' \\') || l.endsWith(' `');

/** The multi-line cmake configure command: the first line starting with
 *  'cmake ' plus every following continuation line (inclusive of the final,
 *  unterminated line). */
const cmakeCommandLines = (text: string): string[] => {
  const lines = text.split('\n');
  const start = lines.findIndex(l => l.startsWith('cmake '));
  expect(start, 'configure command found').toBeGreaterThanOrEqual(0);
  const cmd: string[] = [];
  for (let i = start; i < lines.length; i++) {
    cmd.push(lines[i]);
    if (!isContinuation(lines[i])) break;
  }
  return cmd;
};

const checkLinuxShape = (text: string): void => {
  const cmd = cmakeCommandLines(text);
  expect(cmd.length).toBeGreaterThan(2);
  for (let i = 0; i < cmd.length - 1; i++) {
    expect(cmd[i].endsWith(' \\'), cmd[i]).toBe(true);
  }
  expect(cmd[cmd.length - 1].endsWith(' \\')).toBe(false);
};

const checkWindowsShape = (text: string): void => {
  // No PowerShell line may end with a backslash — PowerShell has no '\'
  // continuation, so such a line would be a syntax / command error.
  for (const l of text.split('\n')) {
    expect(l.endsWith('\\'), JSON.stringify(l)).toBe(false);
  }
  const cmd = cmakeCommandLines(text);
  expect(cmd.length).toBeGreaterThan(2);
  // Every line of the cmake command except the last ends with ' `'
  // (space + backtick, nothing after it); the last has no continuation.
  for (let i = 0; i < cmd.length - 1; i++) {
    expect(cmd[i].endsWith(' `'), cmd[i]).toBe(true);
  }
  expect(cmd[cmd.length - 1].endsWith(' `'), cmd[cmd.length - 1]).toBe(false);
  // The Visual Studio generator arguments are part of the same command,
  // exactly once (no duplicate append after the joined command).
  expect(cmd.join('\n').match(/-G "Visual Studio 17 2022"/g)).toHaveLength(1);
  expect(cmd.join('\n')).toContain('-G "Visual Studio 17 2022" -A x64');
};

const runBashN = (text: string, tag: string): void => {
  const file = join(tmpdir(), `lmp-build-${tag}-${process.pid}-${Date.now()}.sh`);
  writeFileSync(file, text);
  try {
    expect(() => execFileSync('bash', ['-n', file], { stdio: 'pipe' })).not.toThrow();
  } finally {
    rmSync(file, { force: true });
  }
};

describe('generated compiler scripts (docs verified 2026-10-07)', () => {
  it('(A) linux default script passes bash -n and uses backslash continuations', () => {
    const out = generateBuildScript(base({ os: 'linux' }));
    checkLinuxShape(out.text);
    runBashN(out.text, 'default');
  });

  it('(A) windows default script: backtick continuations, no trailing backslash', () => {
    const out = generateBuildScript(base({ os: 'windows' }));
    checkWindowsShape(out.text);
  });

  it('(B) windows drops CMAKE_BUILD_TYPE and uses .\<buildType>\lmp.exe', () => {
    const out = generateBuildScript(base({ os: 'windows' }));
    // Multi-config generator: build type is chosen at build time with --config
    // (docs.lammps.org/Build_cmake.html), so it must not be configured.
    expect(out.text).not.toContain('-D CMAKE_BUILD_TYPE=');
    expect(out.flags.some(f => f.startsWith('-D CMAKE_BUILD_TYPE='))).toBe(false);
    expect(out.flagDetails.some(d => d.flag.startsWith('-D CMAKE_BUILD_TYPE='))).toBe(false);
    expect(out.text).toContain('cmake --build . --config Release');
    expect(out.text).toContain(
      '# binary: .\\Release\\lmp.exe (multi-config generator: one folder per build type)'
    );
    expect(out.text).toContain('.\\Release\\lmp.exe -h | Select-Object -First 30');
    expect(out.text).not.toContain('.\\bin\\lmp.exe');
  });

  it('(B) windows buildType Debug → .\Debug\lmp.exe', () => {
    const out = generateBuildScript(base({ os: 'windows', buildType: 'Debug' }));
    expect(out.text).toContain(
      '# binary: .\\Debug\\lmp.exe (multi-config generator: one folder per build type)'
    );
    expect(out.text).toContain('.\\Debug\\lmp.exe -h | Select-Object -First 30');
    expect(out.text).not.toContain('-D CMAKE_BUILD_TYPE=');
    checkWindowsShape(out.text);
  });

  it('(B) linux keeps -D CMAKE_BUILD_TYPE (single-config Makefile generator)', () => {
    const out = generateBuildScript(base({ os: 'linux', buildType: 'RelWithDebInfo' }));
    expect(out.flags).toContain('-D CMAKE_BUILD_TYPE=RelWithDebInfo');
  });

  it('(C) linux tail: commented optional install + file-based sanity check', () => {
    const out = generateBuildScript(base({ os: 'linux' }));
    expect(out.text).toContain(
      '# cmake --install .   # optional: installs into ${HOME}/.local (no sudo needed)'
    );
    expect(out.text).not.toContain('sudo cmake --install');
    expect(out.text).toContain('./lmp -h > lmp-help.txt');
    expect(out.text).toContain('head -n 30 lmp-help.txt');
    // 'lmp -h | head' exits 141 (SIGPIPE) under pipefail — must not be emitted.
    expect(out.text).not.toMatch(/^lmp -h \| head/m);
  });

  it('(D) Kokkos_ENABLE_CUDA_UVM replaced by Kokkos_ENABLE_IMPL_CUDA_UNIFIED_MEMORY', () => {
    expect(BUILD_OPTIONS.find(o => o.key === 'Kokkos_ENABLE_CUDA_UVM')).toBeUndefined();
    expect(BUILD_OPTIONS.find(o => o.key === 'Kokkos_ENABLE_IMPL_CUDA_UNIFIED_MEMORY')).toEqual({
      key: 'Kokkos_ENABLE_IMPL_CUDA_UNIFIED_MEMORY',
      label: 'Kokkos CUDA unified memory',
      values: ['no', 'yes'],
      default: 'no',
      help: 'KOKKOS+CUDA only. GPU memory as CUDA managed memory; needs CUDA 12.2+.',
    });
    // generateBuildScript only iterates BUILD_OPTIONS, so a stale saved option
    // object that still carries the removed key must not emit any flag from it.
    const stale = generateBuildScript(base({
      os: 'linux',
      options: { ...DEFAULT_COMPILER_OPTIONS.options, Kokkos_ENABLE_CUDA_UVM: 'yes' },
    }));
    expect(stale.text).not.toContain('Kokkos_ENABLE_CUDA_UVM');
    expect(stale.flags).not.toContain('-D Kokkos_ENABLE_CUDA_UVM=yes');
    const fresh = generateBuildScript(base({
      os: 'linux',
      options: { ...DEFAULT_COMPILER_OPTIONS.options, Kokkos_ENABLE_IMPL_CUDA_UNIFIED_MEMORY: 'yes' },
    }));
    expect(fresh.flags).toContain('-D Kokkos_ENABLE_IMPL_CUDA_UNIFIED_MEMORY=yes');
  });

  it('(E) kokkos-hip uses AMD_GFX90A; no VEGA remains', () => {
    const hip = ACCELERATORS.find(a => a.id === 'kokkos-hip');
    expect(hip?.extraFlags).toContain('-D Kokkos_ARCH_AMD_GFX90A=yes');
    expect(hip?.notes).toContain('AMD_GFX90A = MI200, AMD_GFX942 = MI300');
    for (const os of ['linux', 'windows'] as const) {
      const out = generateBuildScript(
        base({ os, presetId: '', manualPackages: [], accelerator: 'kokkos-hip' })
      );
      expect(out.flags).toContain('-D Kokkos_ARCH_AMD_GFX90A=yes');
      expect(out.text).not.toContain('VEGA');
    }
    const out = generateBuildScript(
      base({ os: 'linux', presetId: '', manualPackages: [], accelerator: 'kokkos-hip' })
    );
    const detail = out.flagDetails.find(d => d.flag === '-D Kokkos_ARCH_AMD_GFX90A=yes');
    expect(detail?.description).toContain('VOLTA70, AMPERE80, HOPPER90, AMD_GFX90A, AMD_GFX942');
    // And no 'VEGA' anywhere in the module source (vitest jsdom has no
    // file: import.meta.url, so resolve from the repo root).
    const src = readFileSync(join(process.cwd(), 'src', 'lammps', 'compiler.ts'), 'utf8');
    expect(src).not.toContain('VEGA');
  });

  for (const acc of ['kokkos-cuda', 'kokkos-hip'] as const) {
    it(`(A) ${acc} linux script passes bash -n`, () => {
      const out = generateBuildScript(
        base({ os: 'linux', presetId: '', manualPackages: [], accelerator: acc })
      );
      checkLinuxShape(out.text);
      runBashN(out.text, acc);
    });

    it(`(A) ${acc} windows script: backtick continuations`, () => {
      const out = generateBuildScript(
        base({ os: 'windows', presetId: '', manualPackages: [], accelerator: acc })
      );
      checkWindowsShape(out.text);
    });
  }

  it('kokkos-cuda keeps its NVIDIA arch flag on both OSes', () => {
    for (const os of ['linux', 'windows'] as const) {
      const out = generateBuildScript(
        base({ os, presetId: '', manualPackages: [], accelerator: 'kokkos-cuda' })
      );
      expect(out.flags).toContain('-D Kokkos_ARCH_VOLTA70=yes');
      expect(out.flags).toContain('-D Kokkos_ENABLE_CUDA=yes');
    }
  });
});
