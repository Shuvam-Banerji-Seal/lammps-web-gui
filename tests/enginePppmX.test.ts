import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * Unit tests of the pppm/stagger and pppm/rk styles (src/engine-gpl/pppmx). The G and grid values are the ones native
 * LAMMPS (b332f28) prints for the same inputs as the w237pk oracle cases ("G vector" and "grid" lines of the log).
 */

const SETUP = `units metal
atom_style charge
lattice fcc 5.64
region box block 0 3 0 3 0 3
create_box 2 box
create_atoms 1 box
lattice fcc 5.64 origin 0.5 0.0 0.0
create_atoms 2 box
mass 1 22.99
mass 2 35.45
set type 1 charge 1.0
set type 2 charge -1.0
pair_style lj/cut/coul/long 7.0 7.5
pair_coeff 1 1 0.0100 2.35
pair_coeff 2 2 0.0050 4.40
pair_coeff 1 2 0.0070 3.00
pair_modify table 0
`;

/** Runs a script on a fresh session and returns the log lines and the error messages. */
const run = async (script: string) => {
  const logs: string[] = [];
  const errors: string[] = [];
  const s = new Session({
    emit: (e: { kind: string; message?: string; text?: string }) => {
      if (e.kind === 'error') errors.push(e.message ?? '');
      if (e.kind === 'log' && e.text) logs.push(e.text);
    },
    writeFile: () => {},
  } as never);
  try {
    await s.execute(script);
  } catch (err) {
    errors.push(err instanceof Error ? err.message : String(err));
  }
  return { logs: logs.join('\n'), errors };
};

describe('kspace_style pppm/stagger', () => {
  it('chooses the native G-ewald and mesh on the automatic cubic crystal (w237pk_stagger_auto)', async () => {
    const r = await run(`${SETUP}kspace_style pppm/stagger 1.0e-4
run 0
`);
    expect(r.errors).toEqual([]);
    expect(r.logs).toMatch(/G vector \(1\/distance\) = 0\.3907577\n/);
    expect(r.logs).toMatch(/grid = 12 12 12\n/);
  }, 60_000);

  it('chooses the native G-ewald and mesh with order 4 at accuracy 1e-5 (w237pk_stagger_acc5_order4)', async () => {
    const r = await run(`${SETUP}kspace_style pppm/stagger 1.0e-5
kspace_modify order 4
run 0
`);
    expect(r.errors).toEqual([]);
    expect(r.logs).toMatch(/G vector \(1\/distance\) = 0\.41559309\n/);
    expect(r.logs).toMatch(/grid = 16 16 16\n/);
  }, 60_000);

  it('keeps the user G and mesh (w237pk_stagger_fixed)', async () => {
    const r = await run(`${SETUP}kspace_style pppm/stagger 1.0e-4
kspace_modify gewald 0.35 mesh 12 12 12 order 5
run 0
`);
    expect(r.errors).toEqual([]);
    expect(r.logs).toMatch(/G vector \(1\/distance\) = 0\.35\n/);
    expect(r.logs).toMatch(/grid = 12 12 12\n/);
  }, 60_000);

  it('chooses the native G-ewald and mesh for the slab geometry (w237pk_stagger_slab_auto)', async () => {
    const r = await run(`${SETUP.replace('units metal\n', 'units metal\nboundary p p f\n')}kspace_style pppm/stagger 1.0e-4
kspace_modify slab 3.0
run 0
`);
    expect(r.errors).toEqual([]);
    expect(r.logs).toMatch(/G vector \(1\/distance\) = 0\.36585425\n/);
    expect(r.logs).toMatch(/grid = 10 10 30\n/);
  }, 60_000);

  it('refuses kspace_modify diff ad, naming the option', async () => {
    const r = await run(`${SETUP}kspace_style pppm/stagger 1.0e-4
kspace_modify diff ad
run 0
`);
    expect(r.errors.join('\n')).toMatch(/diff ad.*pppm\/stagger/);
  }, 60_000);

  it('refuses per-atom energy (compute pe/atom), naming the style', async () => {
    const r = await run(`${SETUP}kspace_style pppm/stagger 1.0e-4
compute pe all pe/atom
compute s all reduce sum c_pe
thermo_style custom step c_s
run 0
`);
    expect(r.errors.join('\n')).toMatch(/pe\/atom.*pppm\/stagger/);
  }, 60_000);

  it('refuses a triclinic box', async () => {
    const r = await run(`${SETUP}change_box all triclinic\nchange_box all xy final 1.0
kspace_style pppm/stagger 1.0e-4
run 0
`);
    expect(r.errors.join('\n')).toContain('Cannot (yet) use kspace_style pppm/stagger with triclinic systems');
  }, 60_000);
});

describe('kspace_style pppm/rk', () => {
  it('is refused: it needs two partitions (-partition and run_style verlet/split/rk)', async () => {
    const r = await run(`${SETUP}kspace_style pppm/rk 1.0e-4
run 0
`);
    expect(r.errors.join('\n')).toContain('pppm/rk requires 2 partitions');
  }, 60_000);
});
