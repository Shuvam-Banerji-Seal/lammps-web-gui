import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * fix box/relax dilate all and partial as native LAMMPS (src/fix_box_relax.cpp remap at b332f28): with partial only the
 * atoms of the fix group are remapped when the box changes, the others keep their coordinates. Measured with native
 * LAMMPS (/home/roy/.local/bin/lmp, b332f28): an fcc LJ crystal with half the atoms in the fix group, minimized with
 * iso and aniso pressure targets; each row is the input and native's result (OK, then the printed values).
 */

const outcome = async (input: string): Promise<string> => {
  const vals: string[] = [];
  const s = new Session({ emit: (ev: { kind: string; text?: string }) => { if (ev.kind === 'log' && ev.text?.startsWith('VAL ')) vals.push(ev.text); }, writeFile: () => {} } as never);
  try { await s.execute(input); } catch (e) { return 'ERROR: ' + (e as Error).message.split('\n')[0].replace(/^line \d+: /, ''); }
  return vals.length ? `OK | ${vals.join(' ')}` : 'OK';
};

const ROWS: [string, string][] = [
  ["units lj\nlattice fcc 0.9\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\ndisplace_atoms all random 0.03 0.03 0.03 4711\nregion half block 0 1.5 INF INF INF INF\ngroup half region half\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\nfix 1 half box/relax aniso 0.5 vmax 0.001 dilate all\nthermo_style custom step pe press vol lx ly lz\nthermo_modify format float %.15g\nminimize 0.0 1.0e-8 200 2000\nprint \"VAL $(pe:%.12g) $(vol:%.12g) $(lx:%.12g) $(ly:%.12g)\"\n", "OK | VAL -8.0980198744 99.7685980225 4.63800582569 4.63800582569"],
  ["units lj\nlattice fcc 0.9\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\nfix 1 all box/relax iso 0.0 dilate some\n", "ERROR: Illegal fix box/relax dilate command: some unkown"],
  ["units lj\nlattice fcc 0.9\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\ndisplace_atoms all random 0.03 0.03 0.03 4711\nregion half block 0 1.5 INF INF INF INF\ngroup half region half\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\nfix 1 half box/relax aniso 0.5 vmax 0.001 dilate partial\nthermo_style custom step pe press vol lx ly lz\nthermo_modify format float %.15g\nminimize 0.0 1.0e-8 200 2000\nprint \"VAL $(pe:%.12g) $(vol:%.12g) $(lx:%.12g) $(ly:%.12g)\"\n", "OK | VAL -7.47665993746 109.935638232 4.79111136285 4.78938384835"],
  ["units lj\nlattice fcc 0.9\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\ndisplace_atoms all random 0.03 0.03 0.03 4711\nregion half block 0 1.5 INF INF INF INF\ngroup half region half\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\nfix 1 half box/relax iso 0.0 vmax 0.001 dilate partial\nthermo_style custom step pe press vol lx ly lz\nthermo_modify format float %.15g\nminimize 0.0 1.0e-8 200 2000\nprint \"VAL $(pe:%.12g) $(vol:%.12g) $(lx:%.12g) $(ly:%.12g)\"\n", "OK | VAL -7.47380631508 109.85983332 4.78938384835 4.78938384835"],
];

describe('fix box/relax dilate as native LAMMPS', () => {
  it.each(ROWS.map(([input, native], k) => [k + 1, input, native] as const))('input %i', async (_k, input, native) => {
    expect(await outcome(input)).toBe(native);
  });
});
