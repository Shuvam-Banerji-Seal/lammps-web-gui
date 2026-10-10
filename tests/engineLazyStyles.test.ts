import { describe, expect, it } from 'vitest';
import { LAZY_GROUPS, type StyleKind } from '../src/engine/lazyStyles';
import { COMPUTE_STYLES, PAIR_STYLES, ensureStyles, styleNames } from '../src/engine/styles';
import { Session } from '../src/engine/interpreter';

// src/engine/lazyStyles.ts lists the style names of the registries the engine loads on first use; the lists
// are written by hand when a group is added, so they are checked against the modules here.
describe('lazily loaded style registries', () => {
  it('lists exactly the names each registry module registers', async () => {
    for (const g of LAZY_GROUPS) {
      const mod = await g.load();
      for (const [kind, reg] of Object.entries(mod)) {
        if (!/^[A-Z]+S$/.test(kind) || !reg || typeof reg !== 'object') continue;
        expect([...(g.names[kind as StyleKind] ?? [])].sort(), `${g.id} ${kind}`).toEqual(Object.keys(reg).sort());
      }
    }
  });

  it('keeps every lazy name out of the static registries and in the style lists', () => {
    const names = styleNames();
    for (const g of LAZY_GROUPS) {
      for (const n of g.names.PAIRS ?? []) expect(names.pair_style).toContain(n);
      for (const n of g.names.FIXES ?? []) expect(names.fix).toContain(n);
      for (const n of g.names.COMPUTES ?? []) expect(names.compute).toContain(n);
    }
    // nothing is loaded yet in this file: a lazy pair style is absent from the registry object itself
    expect('colloid' in PAIR_STYLES).toBe(false);
    expect('centro/atom' in COMPUTE_STYLES).toBe(false);
  });

  it('loads a group when a command names one of its styles, also through a variable', async () => {
    const errors: string[] = [];
    const s = new Session({ emit: (e: { kind: string; message?: string }) => { if (e.kind === 'error') errors.push(e.message ?? ''); }, writeFile: () => {} } as never);
    await s.execute(`units lj
lattice fcc 0.8442
region b block 0 3 0 3 0 3
create_box 1 b
create_atoms 1 box
mass 1 1.0
variable ps string colloid
pair_style \${ps} 2.5
pair_coeff 1 1 1.0 1.0 0.0 0.0 2.5
compute c all centro/atom fcc
compute m all reduce max c_c
thermo_style custom step pe c_m
run 0
`);
    expect(errors).toEqual([]);
    expect('colloid' in PAIR_STYLES).toBe(true);
    expect('centro/atom' in COMPUTE_STYLES).toBe(true);
  });

  it('ignores words that name no lazy style', async () => {
    await expect(ensureStyles(['run', '100', 'lj/cut'])).resolves.toBeUndefined();
  });
});
