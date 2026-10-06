import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseXYZFile } from '../src/services/xyzParser';
import { parseDumpFile } from '../src/services/dumpParser';
import { parsePDBFile } from '../src/services/pdbParser';
import { inferBonds } from '../src/services/bondInference';
import { Atom } from '../src/types';

const readExample = (name: string): string =>
  readFileSync(join(process.cwd(), 'public', 'examples', name), 'utf8');

const mkAtom = (id: number, type: number, x: number, y: number, z: number): Atom =>
  ({ id, molId: 1, type, charge: 0, x, y, z });

describe('Domain: bundled demo files', () => {
  it('LJ-melt dump (108 Ar) infers zero bonds — noble gases do not bond', () => {
    const data = parseDumpFile(readExample('lj-melt.lammpstrj'));
    expect(data.atoms).toHaveLength(108);
    // Before the noble-gas guard this produced 1554 phantom Ar-Ar "bonds".
    console.log('[domain] lj-melt.lammpstrj bonds:', data.bonds.length);
    expect(data.bonds).toHaveLength(0);
  });

  it('benzene.pdb keeps its 12 bonds (6 C-C + 6 C-H)', () => {
    const data = parsePDBFile(readExample('benzene.pdb'));
    console.log('[domain] benzene.pdb bonds:', data.bonds.length);
    expect(data.bonds).toHaveLength(12);
  });
});

describe('extxyz Lattice', () => {
  // Hand computation (docs.lammps.org/Howto_triclinic.html, A along +x):
  //   lx = |A| = 5.43 ; xy = B·Â = 0 ; ly = sqrt(5.43^2 - 0) = 5.43
  //   xz = C·Â = 0 ; yz = (B·C - xy*xz)/ly = 0 ; lz = sqrt(5.43^2) = 5.43
  // → box 0..5.43 on every axis, no tilt factors.
  const CUBIC = `2
Lattice="5.43 0 0 0 5.43 0 0 0 5.43" Properties=species:S:1:pos:R:3 pbc="T T T"
Si 0.0 0.0 0.0
Si 2.715 2.715 2.715
`;

  it('cubic Lattice gives an orthogonal 0..5.43 box with no tilt keys', () => {
    const d = parseXYZFile(CUBIC);
    expect(d.box).toEqual({
      xlo: 0, xhi: 5.43,
      ylo: 0, yhi: 5.43,
      zlo: 0, zhi: 5.43,
    });
    expect(d.box?.xy).toBeUndefined();
    expect(d.box?.xz).toBeUndefined();
    expect(d.box?.yz).toBeUndefined();
  });

  it('aligned lattice leaves atom positions bit-identical', () => {
    const d = parseXYZFile(`1
Lattice="5.43 0 0 0 5.43 0 0 0 5.43"
Si 1.234567 -0.5 2.5
`);
    expect(d.atoms[0].x).toBe(1.234567);
    expect(d.atoms[0].y).toBe(-0.5);
    expect(d.atoms[0].z).toBe(2.5);
  });

  // Hand computation for A=(2.46,0,0), B=(-1.23,2.130422,0), C=(0,0,10):
  //   lx = 2.46 ; xy = B·(1,0,0) = -1.23
  //   ly = sqrt(1.23^2 + 2.130422^2 - 1.23^2) = 2.130422
  //   xz = 0 ; yz = (B·C - xy*xz)/ly = 0 ; lz = sqrt(10^2) = 10
  const HEX = `2
Lattice="2.46 0 0 -1.23 2.130422 0 0 0 10"
Si 0 0 0
Si 1.23 0.7101407 5.0
`;

  it('hexagonal Lattice: lx 2.46, xy -1.23, ly 2.130422, lz 10, only xy tilted', () => {
    const d = parseXYZFile(HEX);
    const b = d.box!;
    expect(b.xlo).toBe(0);
    expect(b.xhi).toBeCloseTo(2.46, 12);
    expect(b.xy).toBeCloseTo(-1.23, 12);
    expect(b.yhi).toBeCloseTo(2.130422, 5);
    expect(b.zhi).toBeCloseTo(10, 12);
    expect(b.xz).toBeUndefined();
    expect(b.yz).toBeUndefined();
  });

  // Hand computation for a cube rotated 45 deg about z:
  //   A=(3.84,3.84,0)  → |A| = 3.84·sqrt(2) = 5.43059…  (≈5.43 within 1e-3)
  //   Â = (0.70711, 0.70711, 0) ; B=(-3.84,3.84,0) → xy = B·Â = 0
  //   ly = sqrt(|B|^2 - 0) = 5.43059… ; xz = 0 ; yz = 0 ; lz = 5.43
  //   e2 = normalize(B - 0) = (-0.70711, 0.70711, 0) ; e3 = e1×e2 = (0,0,1)
  //   atom (3.84,3.84,0): r·e1 = 3.84·0.70711·2 = 5.43059…, r·e2 = 0, r·e3 = 0
  const ROTATED = `1
Lattice="3.84 3.84 0 -3.84 3.84 0 0 0 5.43"
Si 3.84 3.84 0.0
`;

  it('45-deg-rotated cubic: lx=ly=5.43 (1e-3), no tilt, atom maps to (5.43,0,0)', () => {
    const d = parseXYZFile(ROTATED);
    const b = d.box!;
    expect(Math.abs(b.xhi - 5.43)).toBeLessThan(1e-3);
    expect(Math.abs(b.yhi - 5.43)).toBeLessThan(1e-3);
    expect(b.zhi).toBeCloseTo(5.43, 12);
    expect(b.xy).toBeUndefined();
    expect(b.xz).toBeUndefined();
    expect(b.yz).toBeUndefined();
    expect(Math.abs(d.atoms[0].x - 5.43)).toBeLessThan(1e-3);
    expect(Math.abs(d.atoms[0].y)).toBeLessThan(1e-9);
    expect(Math.abs(d.atoms[0].z)).toBeLessThan(1e-9);
  });

  // (A×B)·C = (0,0,5.43^2)·(0,0,-5.43) = -160.1 < 0 → left-handed.
  it('throws on a left-handed lattice', () => {
    const LEFT = `1
Lattice="5.43 0 0 0 5.43 0 0 0 -5.43"
Si 0 0 0
`;
    expect(() => parseXYZFile(LEFT)).toThrow('left-handed lattice is not supported');
  });

  it('every frame keeps its own box; MoleculeData.box is frame 0', () => {
    const d = parseXYZFile(`2
Lattice="4 0 0 0 4 0 0 0 4"
Si 0 0 0
Si 2 2 2
2
Lattice="6 0 0 0 6 0 0 0 6" Properties=species:S:1:pos:R:3
Si 0 0 0
Si 3 3 3
`);
    expect(d.frames).toHaveLength(2);
    expect(d.frames![0].box?.xhi).toBe(4);
    expect(d.frames![1].box?.xhi).toBe(6);
    expect(d.box?.xhi).toBe(4);
  });

  it('3-number diagonal Lattice and Origin are honoured; unknown keys ignored', () => {
    const d = parseXYZFile(`1
note="whatever" step=7 Lattice="10 20 30" Origin="1 2 3" pbc="F T T"
C 5 5 5
`);
    const b = d.box!;
    expect(b).toEqual({
      xlo: 1, xhi: 11,
      ylo: 2, yhi: 22,
      zlo: 3, zhi: 33,
    });
    expect(b.xy).toBeUndefined();
  });
});

describe('extxyz Properties', () => {
  it('reads columns in declared order: pos:R:3 before species:S:1', () => {
    const d = parseXYZFile(`2
Lattice="4 0 0 0 4 0 0 0 4" Properties=pos:R:3:species:S:1
1.0 2.0 3.0 Si
2.0 3.0 4.0 O
`);
    expect(d.atoms[0]).toMatchObject({ x: 1, y: 2, z: 3 });
    expect(d.atoms[0].type).toBe(14); // Si
    expect(d.atoms[1]).toMatchObject({ x: 2, y: 3, z: 4 });
    expect(d.atoms[1].type).toBe(8); // O
    expect(d.atomTypes[14].element).toBe('Si');
  });

  it('Z:I:1 atomic-number column maps 14 -> Si', () => {
    const d = parseXYZFile(`2
Properties=Z:I:1:pos:R:3
14 0.0 0.0 0.0
8 1.0 1.0 1.0
`);
    expect(d.atoms[0].type).toBe(14);
    expect(d.atoms[1].type).toBe(8);
    expect(d.atomTypes[14].element).toBe('Si');
    expect(d.atomTypes[8].element).toBe('O');
  });

  it('missing Properties = plain xyz species:S:1:pos:R:3', () => {
    const d = parseXYZFile(`2
Lattice="4 0 0 0 4 0 0 0 4"
O 0 0 0
H 0.96 0 0
`);
    expect(d.atoms[0].type).toBe(8);
    expect(d.atoms[1].type).toBe(1);
    expect(d.atoms[1].x).toBeCloseTo(0.96);
  });

  it('throws a clear error when Properties has no pos column', () => {
    expect(() =>
      parseXYZFile(`1
Properties=species:S:1
Si
`)
    ).toThrow(/no position column/);
  });
});

describe('plain XYZ behaviour is unchanged', () => {
  it('a comment without "=" yields no box', () => {
    const d = parseXYZFile(`3
Water molecule
O 0.0 0.0 0.0
H 0.96 0.0 0.0
H -0.24 0.93 0.0
`);
    expect(d.box).toBeUndefined();
    expect(d.frames).toBeUndefined();
    expect(d.atoms[0].type).toBe(8);
  });
});

describe('bond inference: noble gases and overlaps', () => {
  it('never bonds Ar-Ar (type 18) at liquid densities', () => {
    // LJ melt nearest-neighbour spacing (~1.1 in reduced units).
    const ar = [mkAtom(1, 18, 0, 0, 0), mkAtom(2, 18, 1.1, 0, 0), mkAtom(3, 18, 0, 1.1, 0)];
    expect(inferBonds(ar)).toHaveLength(0);
  });

  it('never bonds He, Ne, Kr, Xe, Rn (types 2/10/36/54/86) nor Og (118)', () => {
    for (const z of [2, 10, 18, 36, 54, 86, 118]) {
      const pair = [mkAtom(1, z, 0, 0, 0), mkAtom(2, z, 0.5, 0, 0)];
      expect(inferBonds(pair)).toHaveLength(0);
    }
  });

  it('does not bond a noble gas to a bondable element either', () => {
    const mix = [mkAtom(1, 18, 0, 0, 0), mkAtom(2, 6, 1.2, 0, 0)]; // Ar-C at 1.2 A
    expect(inferBonds(mix)).toHaveLength(0);
  });

  it('still bonds ordinary elements (control)', () => {
    expect(inferBonds([mkAtom(1, 1, 0, 0, 0), mkAtom(2, 1, 0.74, 0, 0)])).toHaveLength(1);
    expect(inferBonds([mkAtom(1, 6, 0, 0, 0), mkAtom(2, 6, 1.54, 0, 0)])).toHaveLength(1);
  });

  it('skips overlap artefacts closer than 0.1*(ra+rb)', () => {
    // 0.1*(0.76+0.76) = 0.152 A — a duplicated C site at 0.10 A is no bond.
    expect(inferBonds([mkAtom(1, 6, 0, 0, 0), mkAtom(2, 6, 0.10, 0, 0)])).toHaveLength(0);
    // 0.1*(0.66+0.31) = 0.097 A — H glued onto O at 0.05 A is no bond.
    expect(inferBonds([mkAtom(1, 8, 0, 0, 0), mkAtom(2, 1, 0.05, 0, 0)])).toHaveLength(0);
    // Just past the overlap cutoff the covalent criterion applies again:
    // C-C at 1.54 A (> 0.152) bonds; 0.2 A (> 0.152, < 1.824) also bonds.
    expect(inferBonds([mkAtom(1, 6, 0, 0, 0), mkAtom(2, 6, 0.2, 0, 0)])).toHaveLength(1);
  });
});
