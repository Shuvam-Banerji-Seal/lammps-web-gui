import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * dump image bonds with transparency: an atom-coloured bond is drawn in two halves, each with its atom's opacity
 * (dump_modify atrans), the other bonds with their type's opacity (dump_modify btrans), and autobond halves with the atoms'
 * opacity; as native LAMMPS does (src/GRAPHICS/dump_image.cpp create_image at b332f28, draw_cylinder with aopacity or
 * bopacity). Measured with native LAMMPS: the three images of the input below (a periodic bond and two bonded chains)
 * have the pixel SHA-256 digests in NATIVE; the engine's images are pixel-identical.
 */

const INPUT = `units           lj
atom_style      bond
boundary        p p p
region          box block 0 10 0 10 0 10
create_box      2 box bond/types 2 extra/bond/per/atom 2
create_atoms    1 single 3.0 5.0 5.0
create_atoms    2 single 5.0 5.5 5.0
create_atoms    1 single 7.0 5.0 5.2
create_atoms    2 single 9.6 5.0 5.0
create_atoms    1 single 0.6 5.0 5.0
mass            * 1.0
bond_style      harmonic
bond_coeff      * 100.0 2.0
create_bonds    single/bond 1 1 2
create_bonds    single/bond 2 2 3
create_bonds    single/bond 1 4 5
pair_style      zero 2.5
pair_coeff      * *
dump            a all image 1 a.*.ppm type type size 240 240 zoom 2.6 view 80 10 bond atom 0.4 box no 0.02
dump_modify     a atrans 1 0.5
dump            b all image 1 b.*.ppm type type size 240 240 zoom 2.6 view 80 10 bond type 0.4 box no 0.02
dump_modify     b btrans 2 0.3 bcolor 1 red bcolor 2 blue atrans 2 0.6
dump            c all image 1 c.*.ppm type type size 240 240 zoom 2.6 view 80 10 autobond 2.5 0.3 box no 0.02
dump_modify     c atrans 1 0.4
run             0
`;

const NATIVE: Record<string, string> = {
  'a.0.ppm': '70b1ef57576a1b0ed70b411bf1c6a22862773fbef6f6c69a0ef5e18b02e1ece2',
  'b.0.ppm': 'e88cd484f185a2da0d2697a159fccec2488f89d82366a4f65b5185951f840da0',
  'c.0.ppm': '1698f67b613403b384a2c8c4452915e0f396684a7bf7ca1b9a71abd78adbff6c',
};

/** The pixel bytes of a P6 PPM (after the header and its comment lines). */
const pixels = (b: Buffer): Buffer => {
  let i = 0;
  let fields = 0;
  while (fields < 4) {
    while (b[i] === 0x20 || b[i] === 0x0a || b[i] === 0x0d || b[i] === 0x09) i++;
    if (b[i] === 0x23) { while (b[i] !== 0x0a) i++; continue; }
    while (!(b[i] === 0x20 || b[i] === 0x0a || b[i] === 0x0d || b[i] === 0x09)) i++;
    fields++;
  }
  return b.subarray(i + 1);
};

describe('dump image bonds with atrans and btrans', () => {
  it('renders the halves and the bond types with their opacity, pixel-identical to native', async () => {
    const files: Record<string, string> = {};
    const session = new Session({ emit: () => {}, writeFile: (n: string, t: string) => { files[n] = t; } });
    await session.execute(INPUT);
    for (const [name, digest] of Object.entries(NATIVE)) {
      const t = files[name];
      expect(t, name).toBeDefined();
      const buf = t.startsWith('data:') ? Buffer.from(t.slice(t.indexOf(',') + 1), 'base64') : Buffer.from(t, 'latin1');
      expect(createHash('sha256').update(pixels(buf)).digest('hex'), name).toBe(digest);
    }
  });

  it('refuses btrans without bond types and checks its value (native messages)', async () => {
    const run = async (text: string): Promise<string> => {
      const s = new Session({ emit: () => {}, writeFile: () => {} });
      try { await s.execute(text); return ''; } catch (e) { return String((e as Error).message); }
    };
    const base = INPUT.split('dump            a')[0];
    expect(await run(`${base}dump d all image 1 d.*.ppm type type bond atom 0.4\ndump_modify d btrans 1 1.5\n`)).toContain('Invalid btrans opacity in dump_modify command');
    expect(await run('units lj\natom_style atomic\nregion box block 0 4 0 4 0 4\ncreate_box 1 box\ndump d all image 1 d.*.ppm type type\ndump_modify d btrans 1 0.5\n')).toContain('Dump modify btrans not allowed with no bond types');
  });
});
