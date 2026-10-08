import { describe, expect, it } from 'vitest';
import { parseMoleculeFile, specialFromBonds, rotationMatrix, geometricCenter } from '../src/engine/molecule';
import { Session } from '../src/engine/interpreter';
import type { EngineError } from '../src/engine/types';

/*
 * Molecule templates (src/engine/molecule.ts). Placement, rotation, offsets,
 * molecule IDs and charges against native LAMMPS: oracle case molecule_create.
 */

const O = { toff: 0, boff: 0, aoff: 0, doff: 0, ioff: 0, scale: 1 };
const CHAIN = `# chain of four
4 atoms
3 bonds

Coords

1 0 0 0
2 1 0 0   # trailing comment
3 2 0 0
4 3 0 0

Types

1 1
2 1
3 2
4 2

Bonds

1 1 1 2
2 1 2 3
3 2 3 4
`;

describe('molecule templates', () => {
  it('reads header, sections, comments, offsets and scale', () => {
    const t = parseMoleculeFile('c', 'chain.txt', CHAIN, { ...O, toff: 1, boff: 2, scale: 2 });
    expect(t.natoms).toBe(4);
    expect(Array.from(t.type)).toEqual([2, 2, 3, 3]);
    expect(t.bonds).toEqual([[3, 1, 2], [3, 2, 3], [4, 3, 4]]);
    expect(t.x[9]).toBe(6);
    expect(geometricCenter(t)).toEqual([3, 0, 0]);
  });

  it('derives 1-2, 1-3, 1-4 neighbors from bonds and checks explicit special lists against them', () => {
    const t = parseMoleculeFile('c', 'chain.txt', CHAIN, O);
    expect(specialFromBonds(t)[0]).toEqual([[2], [3], [4]]);
    const good = CHAIN + '\nSpecial Bond Counts\n\n1 1 1 1\n2 2 1 0\n3 2 1 0\n4 1 1 1\n\nSpecial Bonds\n\n1 2 3 4\n2 1 3 4\n3 2 4 1\n4 3 2 1\n';
    expect(() => parseMoleculeFile('c', 'chain.txt', good, O)).not.toThrow();
    const bad = good.replace('1 2 3 4\n2 1', '1 2 4 3\n2 1');
    expect(() => parseMoleculeFile('c', 'chain.txt', bad, O)).toThrow(/differ from those its bonds imply/);
  });

  it('names what it cannot do', () => {
    expect(() => parseMoleculeFile('c', 'w.json', '{}', O)).toThrow(/JSON molecule files/);
    expect(() => parseMoleculeFile('c', 'm.txt', CHAIN + '\nMasses\n\n1 1\n2 1\n3 1\n4 1\n', O)).toThrow(/per-atom masses/);
    expect(() => parseMoleculeFile('c', 'm.txt', CHAIN.replace('Types', 'Typo'), O)).toThrow(/section 'Typo'/);
    expect(() => parseMoleculeFile('c', 'm.txt', '# x\n\nCoords\n\n1 0 0 0\n', O)).toThrow(/needs an "N atoms" header/);
    expect(() => parseMoleculeFile('c', 'm.txt', CHAIN.replace('4 2\n\nBonds', '\nBonds'), O)).toThrow(/bad line 'Bonds' in the Types section/);
  });

  it('rotation follows the right-hand rule', () => {
    const R = rotationMatrix(Math.PI / 2, 0, 0, 1);
    expect(R[1][0]).toBeCloseTo(1, 15); // x -> y
    expect(R[0][1]).toBeCloseTo(-1, 15);
  });

  it('create_atoms mol checks the atom style and the bond types', async () => {
    const run = async (style: string, extra: string) => {
      const s = new Session({ emit: () => {}, writeFile: () => {} });
      s.addFile('chain.txt', CHAIN);
      try {
        await s.execute(`units lj\natom_style ${style}\nregion b block 0 10 0 10 0 10\ncreate_box 2 b ${extra}\nmolecule c chain.txt\ncreate_atoms 0 single 5 5 5 mol c 1 rotate 0 0 0 1 units box`);
        return null;
      } catch (e) { return e as EngineError; }
    };
    expect((await run('atomic', ''))?.message).toMatch(/atom_style atomic cannot store/);
    expect((await run('bond', 'bond/types 1'))?.message).toMatch(/bond type 2 is outside 1..1/);
    expect(await run('bond', 'bond/types 2 extra/bond/per/atom 2')).toBeNull();
  });
});
