import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { System } from '../src/engine/system';
import { buildAtomMap, type SpecialList } from '../src/engine/atoms';
import { tip4pGather, type GpuTip4pSpec, type Tip4pSites } from '../src/engine/gpu/tip4pSpec';
import { accelerator } from '../src/engine/run/accel';
import { CpuForceBackend } from '../src/engine/cpu/forces';
import type { EngineEvent, ForceBackend, ForceResult, SimState, ThermoRow } from '../src/engine/types';

/*
 * The TIP4P pair styles on the GPU path (gpu/tip4pSpec.ts, gpu/tip4pKernels.ts, webgpuForces.ts computeTip4p): a run
 * through the accelerated loop with a backend that does what the kernel and the site spreading do, in fp64
 * (tip4pGather), gives the CPU engine's forces and thermo, with the bonded terms and pppm/tip4p on the CPU beside it.
 * The WGSL itself is checked in Chromium (SwiftShader).
 */

/** n^3 rigid-geometry TIP4P waters on a cubic grid of spacing a, random orientations (as engineTip4pThreads.test.ts). */
const waterData = (n: number, a: number): string => {
  let seed = 12345;
  const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
  const r0 = 0.9572, half = (104.52 / 2) * Math.PI / 180;
  const atoms: string[] = [], bonds: string[] = [], angles: string[] = [];
  let id = 0, mol = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) {
    mol++;
    const o = [(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a];
    const th = Math.PI * rnd(), ph = 2 * Math.PI * rnd(), ps = 2 * Math.PI * rnd();
    const u = [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
    const t0 = Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const v = [u[1] * t0[2] - u[2] * t0[1], u[2] * t0[0] - u[0] * t0[2], u[0] * t0[1] - u[1] * t0[0]];
    const vn = Math.hypot(v[0], v[1], v[2]);
    const w = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]].map((c) => c / vn);
    const e = v.map((c, m) => (c / vn) * Math.cos(ps) + w[m] * Math.sin(ps));
    const h1 = o.map((c, m) => c + r0 * (Math.cos(half) * u[m] + Math.sin(half) * e[m]));
    const h2 = o.map((c, m) => c + r0 * (Math.cos(half) * u[m] - Math.sin(half) * e[m]));
    const oi = ++id;
    atoms.push(`${oi} ${mol} 1 -1.0484 ${o.join(' ')}`);
    atoms.push(`${++id} ${mol} 2 0.5242 ${h1.join(' ')}`);
    atoms.push(`${++id} ${mol} 2 0.5242 ${h2.join(' ')}`);
    bonds.push(`${bonds.length + 1} 1 ${oi} ${oi + 1}`, `${bonds.length + 2} 1 ${oi} ${oi + 2}`);
    angles.push(`${angles.length + 1} 1 ${oi + 1} ${oi} ${oi + 2}`);
  }
  const L = n * a;
  return ['water box', '', `${id} atoms`, `${bonds.length} bonds`, `${angles.length} angles`, '2 atom types', '1 bond types', '1 angle types', '',
    `0 ${L} xlo xhi`, `0 ${L} ylo yhi`, `0 ${L} zlo zhi`, '', 'Masses', '', '1 15.9994', '2 1.008', '',
    'Atoms # full', '', ...atoms, '', 'Bonds', '', ...bonds, '', 'Angles', '', ...angles, ''].join('\n');
};

/** A backend that does what computeTip4p does, in fp64. */
class Tip4pInFp64 implements ForceBackend {
  readonly kind = 'webgpu' as const;
  readonly label = 'WebGPU · fp64 stand-in';
  calls = 0;
  specialSeen = false;
  compute(): ForceResult { throw new Error('the stand-in runs computeTip4p only'); }
  dispose(): void {}
  async computeTip4p(st: SimState, spec: GpuTip4pSpec, special: SpecialList | null, sites: Tip4pSites): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    this.calls++;
    const L = [0, 1, 2].map((d) => st.box.hi[d] - st.box.lo[d]);
    const idx = buildAtomMap(st);
    const order = (i: number, j: number): number => {
      if (!spec.special || !special) return 0;
      for (let e = special.offset[i]; e < special.offset[i + 1]; e++) {
        if (idx[special.partner[e]] === j) { this.specialSeen = true; return special.order[e]; }
      }
      return 0;
    };
    const r = tip4pGather(spec, st.n, st.x, st.type, st.q, sites, order, L);
    st.f.set(r.f.subarray(0, 3 * st.n));
    return { pe: r.evdwl + r.ecoul, virial: r.virial, evdwl: r.evdwl, ecoul: r.ecoul };
  }
}

const SETUP = (pair: string) => `units real
atom_style full
read_data water.data
bond_style harmonic
bond_coeff 1 450.0 0.9572
angle_style harmonic
angle_coeff 1 55.0 104.52
${pair}
velocity all create 300.0 4928459 loop geom
`;

const RUN = `fix 1 all nve
timestep 0.5
thermo_style custom step temp pe ke etotal evdwl ecoul elong ebond eangle press
thermo 10
run 40
`;

const CASES: Record<string, string> = {
  'lj/cut/tip4p/long + pppm/tip4p': 'pair_style lj/cut/tip4p/long 1 2 1 1 0.1546 3.5\npair_coeff 1 1 0.1852 3.1589\npair_coeff 1 2 0.0 1.0\npair_coeff 2 2 0.0 1.0\npair_modify table 0\nkspace_style pppm/tip4p 1.0e-4',
  'lj/cut/tip4p/cut': 'pair_style lj/cut/tip4p/cut 1 2 1 1 0.1546 3.5\npair_coeff 1 1 0.1852 3.1589\npair_coeff 1 2 0.0 1.0\npair_coeff 2 2 0.0 1.0',
  'tip4p/long + pppm/tip4p': 'pair_style tip4p/long 1 2 1 1 0.1546 3.5\npair_coeff * *\npair_modify table 0\nkspace_style pppm/tip4p 1.0e-4',
  'tip4p/cut, special weights': 'pair_style tip4p/cut 1 2 1 1 0.1546 3.5\npair_coeff * *\nspecial_bonds coul 0.0 0.5 0.5',
};

const session = (backend?: ForceBackend) => {
  const rows: ThermoRow[] = [], logs: string[] = [];
  const s = new Session({
    emit: (ev: EngineEvent) => { if (ev.kind === 'thermo') rows.push(ev.row); else if (ev.kind === 'log') logs.push(ev.text); },
    writeFile: () => {},
  }, backend);
  s.addFile('water.data', waterData(3, 3.1));
  return { s, rows, logs, sys: () => (s as unknown as { sys: System }).sys };
};

describe('the TIP4P pair styles on the GPU path', () => {
  for (const [name, pair] of Object.entries(CASES)) {
    it(`${name}: the forces of the CPU engine at the setup`, async () => {
      const cpu = session(new CpuForceBackend());
      await cpu.s.execute(`${SETUP(pair)}fix 1 all nve\nrun 0\n`);
      const stand = new Tip4pInFp64();
      const gpu = session(stand);
      await gpu.s.execute(`${SETUP(pair)}fix 1 all nve\nrun 0\n`);
      expect(gpu.logs.filter((l) => l.includes('general fp64 CPU engine'))).toEqual([]);
      expect(stand.calls).toBe(1);
      expect(stand.specialSeen).toBe(true);
      const a = cpu.sys().state, b = gpu.sys().state;
      let worst = 0, scale = 0;
      for (let k = 0; k < 3 * a.n; k++) { worst = Math.max(worst, Math.abs(a.f[k] - b.f[k])); scale = Math.max(scale, Math.abs(a.f[k])); }
      expect(scale).toBeGreaterThan(1);
      expect(worst / scale).toBeLessThan(1e-10);
    });

    it(`${name}: a run through the accelerated loop gives the CPU engine's thermo`, async () => {
      const cpu = session(new CpuForceBackend());
      await cpu.s.execute(`${SETUP(pair)}${RUN}`);
      const stand = new Tip4pInFp64();
      const gpu = session(stand);
      await gpu.s.execute(`${SETUP(pair)}${RUN}`);
      expect(gpu.logs.filter((l) => l.includes('general fp64 CPU engine'))).toEqual([]);
      expect(stand.calls).toBe(41);
      expect(cpu.rows.length).toBe(5);
      for (let r = 0; r < cpu.rows.length; r++) {
        for (const k of ['temp', 'pe', 'ke', 'etotal', 'evdwl', 'ecoul', 'elong', 'ebond', 'eangle', 'press']) {
          const x = cpu.rows[r][k], y = gpu.rows[r][k];
          expect(Math.abs(x - y), `row ${r} ${k}: ${x} vs ${y}`).toBeLessThan(1e-9 * Math.max(1, Math.abs(x)));
        }
      }
      expect(Math.abs(cpu.rows[0].ecoul)).toBeGreaterThan(1);
    });
  }

  it('the accelerator: a TIP4P spec with kspace on the CPU, not resident; a short box stays on the CPU', async () => {
    const gpu = { kind: 'webgpu', label: 'WebGPU · test', compute: () => ({ pe: 0, virial: 0 }), dispose() {} } as unknown as ForceBackend;
    const { s, sys } = session();
    await s.execute(`${SETUP(CASES['lj/cut/tip4p/long + pppm/tip4p'])}fix 1 all nve\nrun 0\n`);
    const a = accelerator(sys(), gpu);
    expect(a.reason).toBeNull();
    expect(a.accel?.tip4p?.alpha).toBeCloseTo(0.1546 / (0.9572 * Math.cos((104.52 / 2) * Math.PI / 180)), 12);
    expect(a.accel?.kspace).toBe(true);
    expect(a.accel?.bonded).toBe(true);
    expect(a.accel?.resident).toBe(false);
    const big = session();
    await big.s.execute(`${SETUP(CASES['lj/cut/tip4p/long + pppm/tip4p'].replace('0.1546 3.5', '0.1546 4.5'))}fix 1 all nve\nrun 0\n`);
    expect(accelerator(big.sys(), gpu).reason).toBe('a box shorter than 2 pair cutoffs along a periodic edge');
  });
});
