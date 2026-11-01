import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { System } from '../src/engine/system';
import { pairSpecEval, type GpuPairSpec } from '../src/engine/gpu/pairSpec';
import { excludedPair, type GpuExclusions } from '../src/engine/gpu/exclusions';
import { accelerator } from '../src/engine/run/accel';
import { CpuForceBackend } from '../src/engine/cpu/forces';
import type { EngineEvent, ForceBackend, ForceResult, SimState, ThermoRow } from '../src/engine/types';

/*
 * neigh_modify exclude and include on the GPU path (gpu/exclusions.ts, the styles kernel): the type pairs are zero
 * cutoffs in the packed coefficients, the group and molecule rules and the include group are tested per pair in the
 * kernel. A backend that does what computeSpec does in fp64 (pairSpecEval on the packed coefficients, excludedPair)
 * gives the CPU engine's thermo through the accelerated loop, lj/cut (as the styles kernel's style 10) and
 * lj/cut/coul/long with ewald (an excluded pair has no kspace correction either).
 */

class SpecInFp64 implements ForceBackend {
  readonly kind = 'webgpu' as const;
  readonly label = 'WebGPU · fp64 stand-in';
  calls = 0;
  seen: GpuExclusions | null = null;
  compute(): ForceResult { throw new Error('the stand-in runs computeSpec only'); }
  dispose(): void {}
  async computeSpec(st: SimState, spec: GpuPairSpec, _special: unknown, ex: GpuExclusions | null = null): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    this.calls++;
    this.seen = ex;
    const n = st.n, L = [0, 1, 2].map((d) => st.box.hi[d] - st.box.lo[d]);
    st.f.fill(0, 0, 3 * n);
    let evdwl = 0, ecoul = 0, virial = 0;
    const vt = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (ex && excludedPair(ex, st.mask, st.molecule, i, j)) continue;
        const d = [0, 1, 2].map((k) => { const v = st.x[3 * i + k] - st.x[3 * j + k]; return v - L[k] * Math.round(v / L[k]); });
        const rsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        const r = pairSpecEval(spec, st.type[i], st.type[j], rsq, spec.charges ? st.q[i] : 0, spec.charges ? st.q[j] : 0);
        evdwl += r.evdwl; ecoul += r.ecoul; virial += r.fpair * rsq;
        vt[0] += r.fpair * d[0] * d[0]; vt[1] += r.fpair * d[1] * d[1]; vt[2] += r.fpair * d[2] * d[2];
        vt[3] += r.fpair * d[0] * d[1]; vt[4] += r.fpair * d[0] * d[2]; vt[5] += r.fpair * d[1] * d[2];
        for (let k = 0; k < 3; k++) { st.f[3 * i + k] += r.fpair * d[k]; st.f[3 * j + k] -= r.fpair * d[k]; }
      }
    }
    return { pe: evdwl + ecoul, virial, evdwl, ecoul, virialTensor: vt };
  }
}

const BOX = (charged: boolean) => `units lj
atom_style ${charged ? 'full' : 'molecular'}
lattice fcc 0.8442
region box block 0 5 0 5 0 5
create_box 2 box
create_atoms 1 box
mass * 1.0
set group all type/fraction 2 0.4 4321
${charged ? 'set type 1 charge 0.4\nset type 2 charge -0.6\n' : ''}region lo block INF INF INF INF INF 3.0 units box
region hi block INF INF INF INF 3.0 INF units box
group lower region lo
group upper region hi
variable m atom 1+floor(x/2.5)
set group all mol v_m
velocity all create 1.2 87287 loop geom
`;

const LJ = 'pair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\npair_coeff 1 2 0.8 1.1';
const COUL = 'pair_style lj/cut/coul/long 2.5\npair_coeff * * 1.0 1.0\npair_coeff 1 2 0.8 1.1\npair_modify table 0\nkspace_style ewald 1e-4';

const CASES: Record<string, { charged: boolean; pair: string; neigh: string }> = {
  'exclude type': { charged: false, pair: LJ, neigh: 'neigh_modify exclude type 1 2' },
  'exclude group': { charged: false, pair: LJ, neigh: 'neigh_modify exclude group lower upper' },
  'exclude molecule/intra': { charged: false, pair: LJ, neigh: 'neigh_modify exclude molecule/intra all' },
  'exclude molecule/inter and type': { charged: false, pair: LJ, neigh: 'neigh_modify exclude molecule/inter upper exclude type 2 2' },
  include: { charged: false, pair: LJ, neigh: 'atom_modify first lower\nneigh_modify include lower' },
  'lj/cut/coul/long with ewald, exclude type and group': { charged: true, pair: COUL, neigh: 'neigh_modify exclude type 1 1 exclude group lower upper' },
};

const RUN = `fix 1 all nve
timestep 0.004
thermo_style custom step temp pe evdwl ecoul elong press
thermo 10
run 30
`;

const session = (backend?: ForceBackend) => {
  const rows: ThermoRow[] = [], logs: string[] = [];
  const s = new Session({
    emit: (ev: EngineEvent) => { if (ev.kind === 'thermo') rows.push(ev.row); else if (ev.kind === 'log') logs.push(ev.text); },
    writeFile: () => {},
  }, backend);
  return { s, rows, logs, sys: () => (s as unknown as { sys: System }).sys };
};

describe('neigh_modify exclude and include on the GPU path', () => {
  for (const [name, c] of Object.entries(CASES)) {
    it(`${name}: the CPU engine's thermo`, async () => {
      const input = `${BOX(c.charged)}${c.pair}\n${c.neigh}\n${RUN}`;
      const cpu = session(new CpuForceBackend());
      await cpu.s.execute(input);
      const stand = new SpecInFp64();
      const gpu = session(stand);
      await gpu.s.execute(input);
      expect(gpu.logs.filter((l) => l.includes('general fp64 CPU engine'))).toEqual([]);
      expect(stand.calls).toBe(31);
      expect(stand.seen).not.toBeNull();
      expect(cpu.rows.length).toBe(4);
      for (let r = 0; r < cpu.rows.length; r++) {
        for (const k of ['temp', 'pe', 'evdwl', 'ecoul', 'elong', 'press']) {
          const a = cpu.rows[r][k], b = gpu.rows[r][k];
          expect(Math.abs(a - b), `row ${r} ${k}: ${a} vs ${b}`).toBeLessThan(1e-9 * Math.max(1, Math.abs(a)));
        }
      }
      // the exclusion matters: the same run without it has another energy
      const plain = session(new CpuForceBackend());
      await plain.s.execute(`${BOX(c.charged)}${c.pair}\n${RUN}`);
      expect(Math.abs(plain.rows[0].pe - cpu.rows[0].pe)).toBeGreaterThan(1e-3);
    });
  }

  it('the accelerator: the styles kernel, not resident; the lj/cut threads stay on the CPU', async () => {
    const run = async (lines: string) => {
      const { s, sys } = session();
      await s.execute(`${BOX(false)}${lines}\nfix 1 all nve\nrun 0\n`);
      return sys();
    };
    const gpu = { kind: 'webgpu', label: 'WebGPU · test', compute: () => ({ pe: 0, virial: 0 }), dispose() {}, canAdvance: () => true, canAdvanceSpec: () => true } as unknown as ForceBackend;
    const threads = { kind: 'cpu', label: 'CPU · threads', compute: () => ({ pe: 0, virial: 0 }), dispose() {} } as unknown as ForceBackend;
    const sys = await run(`${LJ}\nneigh_modify exclude group lower upper`);
    const a = accelerator(sys, gpu);
    expect(a.reason).toBeNull();
    expect(a.accel?.spec?.style).toBe(10);
    expect(a.accel?.exclude?.rules.length).toBe(1);
    expect(a.accel?.resident).toBe(false);
    expect(accelerator(sys, threads).reason).toBe('neigh_modify exclude/include');
    // (an exclude group of a dynamic group is refused at its neigh_modify, as native does: "Neigh_modify exclude group is
    // not compatible with dynamic groups"; the accelerator's own check is a guard behind it)
    const soft = await run('pair_style soft 1.2\npair_coeff * * 10.0\nneigh_modify exclude type 1 2');
    expect(accelerator(soft, gpu).reason).toBeNull();
  });
});
