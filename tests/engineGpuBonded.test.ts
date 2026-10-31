import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { System } from '../src/engine/system';
import { pairSpecEval, type GpuPairSpec } from '../src/engine/gpu/pairSpec';
import { accelerator } from '../src/engine/run/accel';
import { buildAtomMap, type SpecialList } from '../src/engine/atoms';
import { CpuForceBackend } from '../src/engine/cpu/forces';
import type { EngineEvent, ForceBackend, ForceResult, SimState, ThermoRow } from '../src/engine/types';

/*
 * Molecular systems on the GPU path (run/accel.ts): the bonded terms run on the CPU and the GPU adds the pair term,
 * with the special_bonds weights of 1-2, 1-3 and 1-4 pairs applied in the kernel (webgpuForces.ts KERNEL_STYLES).
 * The backend here does what that kernel does in fp64 (every pair by the minimum image, the special entry of the
 * partner scaling the short-range and the Coulomb parts, a pair with both weights 0 skipped), so a run through the
 * accelerated loop must give the CPU engine's thermo. The WGSL kernel itself is checked in Chromium (SwiftShader).
 */

/** Chains of 8 atoms along x on a simple cubic lattice of 8 x 8 x 8 sites (spacing 1.125, box 9). */
const chainData = (charged: boolean): string => {
  const a = 1.125, m = 8;
  const atoms: string[] = [], bonds: string[] = [], angles: string[] = [], diheds: string[] = [];
  let id = 0, mol = 0;
  for (let k = 0; k < m; k++) {
    for (let j = 0; j < m; j++) {
      mol++;
      const first = id + 1;
      for (let i = 0; i < m; i++) {
        id++;
        // a small deterministic offset, so the forces are not those of a perfect lattice
        const dx = 0.05 * Math.sin(1.7 * id), dy = 0.05 * Math.cos(2.3 * id), dz = 0.05 * Math.sin(0.9 * id + 1);
        const type = 1 + (i % 2);
        const q = charged ? (i % 2 ? -0.4 : 0.4) : 0;
        atoms.push(charged
          ? `${id} ${mol} ${type} ${q} ${(i + 0.5) * a + dx} ${(j + 0.5) * a + dy} ${(k + 0.5) * a + dz}`
          : `${id} ${mol} ${type} ${(i + 0.5) * a + dx} ${(j + 0.5) * a + dy} ${(k + 0.5) * a + dz}`);
      }
      for (let i = 0; i < m - 1; i++) bonds.push(`${bonds.length + 1} 1 ${first + i} ${first + i + 1}`);
      for (let i = 0; i < m - 2; i++) angles.push(`${angles.length + 1} 1 ${first + i} ${first + i + 1} ${first + i + 2}`);
      for (let i = 0; i < m - 3; i++) diheds.push(`${diheds.length + 1} 1 ${first + i} ${first + i + 1} ${first + i + 2} ${first + i + 3}`);
    }
  }
  const L = m * a;
  return [
    'chains', '',
    `${atoms.length} atoms`, `${bonds.length} bonds`, `${angles.length} angles`, `${diheds.length} dihedrals`, '',
    '2 atom types', '1 bond types', '1 angle types', '1 dihedral types', '',
    `0 ${L} xlo xhi`, `0 ${L} ylo yhi`, `0 ${L} zlo zhi`, '',
    'Masses', '', '1 1.0', '2 1.5', '',
    `Atoms # ${charged ? 'full' : 'molecular'}`, '', ...atoms, '',
    'Bonds', '', ...bonds, '',
    'Angles', '', ...angles, '',
    'Dihedrals', '', ...diheds, '',
  ].join('\n');
};

/** What the KERNEL_STYLES kernel computes, in fp64 over every pair (minimum image). */
class KernelInFp64 implements ForceBackend {
  readonly kind = 'webgpu' as const;
  readonly label = 'WebGPU · fp64 stand-in';
  calls = 0;
  specialSeen = false;
  compute(): ForceResult { throw new Error('the stand-in runs computeSpec only'); }
  dispose(): void {}
  async computeSpec(st: SimState, spec: GpuPairSpec, special: SpecialList | null = null): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    this.calls++;
    const n = st.n;
    const L = [0, 1, 2].map((d) => st.box.hi[d] - st.box.lo[d]);
    const idx = buildAtomMap(st);
    // the order of each special partner of atom i (0: none)
    const orderOf = (i: number, j: number): number => {
      if (!spec.special || !special) return 0;
      for (let e = special.offset[i]; e < special.offset[i + 1]; e++) if (idx[special.partner[e]] === j) return special.order[e];
      return 0;
    };
    st.f.fill(0, 0, 3 * n);
    let evdwl = 0, ecoul = 0, virial = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = [0, 1, 2].map((k) => { const v = st.x[3 * i + k] - st.x[3 * j + k]; return v - L[k] * Math.round(v / L[k]); });
        const rsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        const o = orderOf(i, j);
        if (o > 0) this.specialSeen = true;
        const flj = o ? spec.special!.lj[o] : 1, fc = o ? spec.special!.coul[o] : 1;
        if (flj === 0 && fc === 0) continue;
        const qi = spec.charges ? st.q[i] : 0, qj = spec.charges ? st.q[j] : 0;
        const short = pairSpecEval(spec, st.type[i], st.type[j], rsq, 0, 0);
        const all = pairSpecEval(spec, st.type[i], st.type[j], rsq, qi, qj);
        const fpair = flj * short.fpair + fc * (all.fpair - short.fpair);
        evdwl += flj * short.evdwl;
        ecoul += fc * all.ecoul;
        virial += fpair * rsq;
        for (let k = 0; k < 3; k++) { st.f[3 * i + k] += fpair * d[k]; st.f[3 * j + k] -= fpair * d[k]; }
      }
    }
    return { pe: evdwl + ecoul, virial, evdwl, ecoul };
  }
}

const SETUP = (charged: boolean, pair: string) => `units lj
atom_style ${charged ? 'full' : 'molecular'}
read_data chains.data
${pair}
bond_style harmonic
bond_coeff 1 200.0 1.125
angle_style harmonic
angle_coeff 1 20.0 170.0
dihedral_style opls
dihedral_coeff 1 0.5 0.2 0.3 0.0
special_bonds lj 0.0 0.5 0.8 coul 0.0 0.0 0.6
velocity all create 1.0 4928 loop geom
fix 1 all nve
thermo_style custom step temp pe ke etotal evdwl ecoul ebond eangle edihed press
thermo 10
run 40
`;

const run = async (backend: ForceBackend, charged: boolean, pair: string): Promise<{ rows: ThermoRow[]; logs: string[]; sys: System }> => {
  const rows: ThermoRow[] = [], logs: string[] = [];
  const s = new Session({
    emit: (ev: EngineEvent) => {
      if (ev.kind === 'thermo') rows.push(ev.row);
      else if (ev.kind === 'log') logs.push(ev.text);
    },
    writeFile: () => {},
  }, backend);
  s.addFile('chains.data', chainData(charged));
  await s.execute(SETUP(charged, pair));
  return { rows, logs, sys: (s as unknown as { sys: System }).sys };
};

const CASES: Record<string, { charged: boolean; pair: string; style: number }> = {
  'lj/cut (spec style 10)': { charged: false, pair: 'pair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\npair_coeff 1 2 0.8 1.05 2.4\npair_modify shift yes', style: 10 },
  'lj/cut/coul/cut': { charged: true, pair: 'pair_style lj/cut/coul/cut 2.5 2.9\npair_coeff * * 1.0 1.0\npair_coeff 1 2 0.8 1.05', style: 1 },
};

describe('molecular systems on the GPU path: CPU bonded terms plus the GPU pair term with special weights', () => {
  for (const [name, c] of Object.entries(CASES)) {
    it(`${name}: the thermo of the CPU engine`, async () => {
      const cpu = await run(new CpuForceBackend(), c.charged, c.pair);
      const fake = new KernelInFp64();
      const gpu = await run(fake, c.charged, c.pair);
      // the run took the GPU path (no fallback line), the special list reached the kernel
      expect(gpu.logs.filter((l) => l.includes('general fp64 CPU engine'))).toEqual([]);
      expect(fake.calls).toBe(41);
      expect(fake.specialSeen).toBe(true);
      expect(gpu.rows.length).toBe(cpu.rows.length);
      expect(cpu.rows.length).toBe(5);
      for (let r = 0; r < cpu.rows.length; r++) {
        for (const k of ['temp', 'pe', 'ke', 'etotal', 'evdwl', 'ecoul', 'ebond', 'eangle', 'edihed', 'press']) {
          const a = cpu.rows[r][k], b = gpu.rows[r][k];
          expect(Math.abs(a - b), `row ${r} ${k}: ${a} vs ${b}`).toBeLessThan(1e-9 * Math.max(1, Math.abs(a)));
        }
      }
      // the bonded and the special-weighted pair terms both matter here
      expect(Math.abs(cpu.rows[0].ebond)).toBeGreaterThan(1e-3);
      expect(Math.abs(cpu.rows[0].edihed)).toBeGreaterThan(1e-3);
      if (c.charged) expect(Math.abs(cpu.rows[0].ecoul)).toBeGreaterThan(1e-3);
    });
  }

  it('the accelerator: a spec with the special weights, not resident; kspace and the lj/cut threads stay on the CPU', async () => {
    const gpu = { kind: 'webgpu', label: 'WebGPU · test', compute: () => ({ pe: 0, virial: 0 }), dispose() {}, canAdvance: () => true, canAdvanceSpec: () => true } as unknown as ForceBackend;
    const { sys } = await run(new CpuForceBackend(), false, CASES['lj/cut (spec style 10)'].pair);
    const a = accelerator(sys, gpu);
    expect(a.reason).toBeNull();
    expect(a.accel?.spec?.style).toBe(10);
    expect(a.accel?.bonded).toBe(true);
    expect(a.accel?.resident).toBe(false);
    expect(a.accel?.spec?.special).toEqual({ lj: [1, 0, 0.5, 0.8], coul: [1, 0, 0, 0.6] });
    // the lj/cut threads of cpu/parallel.ts have no special bonds
    const threads = { kind: 'cpu', label: 'CPU · threads', compute: () => ({ pe: 0, virial: 0 }), dispose() {} } as unknown as ForceBackend;
    expect(accelerator(sys, threads).reason).toBe('bonded terms or molecular topology');
    const charged = await run(new CpuForceBackend(), true, 'pair_style lj/cut/coul/long 2.5 2.9\npair_coeff * * 1.0 1.0\nkspace_style ewald 1e-4');
    expect(accelerator(charged.sys, gpu).reason).toMatch(/pair style lj\/cut\/coul\/long|kspace/);
  });
});
