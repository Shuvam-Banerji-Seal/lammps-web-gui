import { describe, it, expect } from 'vitest';
import { parseDumpFile, boundBoxToTriclinic } from '../src/services/dumpParser';
import { detectFileFormat, detectFormatFromContent } from '../src/services/fileParser';

// Orthogonal, 2 frames, custom columns incl. element + q
const ORTHO = `ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0.0000000000000000e+00 1.0000000000000000e+01
0.0000000000000000e+00 1.0000000000000000e+01
0.0000000000000000e+00 1.0000000000000000e+01
ITEM: ATOMS id type element x y z q
1 1 O 1.0 2.0 3.0 -0.8
2 1 H 2.0 2.0 3.0 0.4
3 2 Na 9.0 9.0 9.0 1.0
ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0.0000000000000000e+00 1.0000000000000000e+01
0.0000000000000000e+00 1.0000000000000000e+01
0.0000000000000000e+00 1.0000000000000000e+01
ITEM: ATOMS id type element x y z q
1 1 O 1.5 2.5 3.5 -0.8
2 1 H 2.5 2.5 3.5 0.4
3 2 Na 8.5 8.5 8.5 1.0
`;

// Triclinic, style-atom default columns (id type xs ys zs — SCALED).
// tilts: xy=-2, xz=-0.5, yz=-1 ; true box lx=10 ly=8 lz=6, origin (0,0,0).
// Per Howto_triclinic: xlo_bound = 0 + MIN(0,-2,-.5,-2.5) = -2.5 ; xhi_bound = 10 + MAX(...)=10
// ylo_bound = 0 + MIN(0,-1) = -1 ; yhi_bound = 8 + MAX(0,-1) = 8 ; z 0..6
const TRICLINIC = `ITEM: TIMESTEP
500
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS xy xz yz pp pp pp
-2.5 10.0 -2.0
-1.0 8.0 -0.5
0.0 6.0 -1.0
ITEM: ATOMS id type xs ys zs
1 1 0.0 0.0 0.0
2 1 1.0 1.0 1.0
`;

describe('LAMMPS dump parser', () => {
  it('parses orthogonal multi-frame dumps with element/q columns', () => {
    const d = parseDumpFile(ORTHO);
    expect(d.atoms).toHaveLength(3);
    expect(d.frames).toHaveLength(2);
    expect(d.atoms[0]).toMatchObject({ id: 1, type: 8, charge: -0.8, x: 1, y: 2, z: 3 });
    expect(d.atoms[1].type).toBe(1); // H
    expect(d.atoms[2].type).toBe(11); // Na
    expect(d.box).toEqual({ xlo: 0, xhi: 10, ylo: 0, yhi: 10, zlo: 0, zhi: 10 });
    // frame 2 swapped in playback keeps topology metadata stable
    expect(d.frames![1].atoms[0]).toMatchObject({ x: 1.5, y: 2.5, z: 3.5 });
    expect(d.frames![0].comment).toContain('timestep 0');
    expect(d.frames![1].comment).toContain('timestep 100');
    // min/max span ALL frames
    expect(d.max.x).toBeCloseTo(9);
    expect(d.min.x).toBeCloseTo(1);
  });

  it('resolves elements from the element column, else type-as-atomic-number', () => {
    const d = parseDumpFile(ORTHO);
    expect(d.atomTypes[8].element).toBe('O');
    expect(d.atomTypes[11].element).toBe('Na');
    const noElem = parseDumpFile(`ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
1 6 5 5 5
`);
    expect(noElem.atomTypes[6].element).toBe('C');
  });

  it('converts triclinic bounding box to true box (verified Howto_triclinic inverse)', () => {
    const box = boundBoxToTriclinic(
      [-2.5, 10], [-1, 8], [0, 6], -2.0, -0.5, -1.0,
    );
    expect(box.xlo).toBeCloseTo(0);
    expect(box.xhi).toBeCloseTo(10);
    expect(box.ylo).toBeCloseTo(0);
    expect(box.yhi).toBeCloseTo(8);
    expect(box.zlo).toBeCloseTo(0);
    expect(box.zhi).toBeCloseTo(6);
    expect(box.xy).toBeCloseTo(-2);
    expect(box.xz).toBeCloseTo(-0.5);
    expect(box.yz).toBeCloseTo(-1);
  });

  it('de-scales fractional coords in the triclinic vector basis', () => {
    const d = parseDumpFile(TRICLINIC);
    // atom 2 at (1,1,1): x = 0 + 10 - 2 - 0.5 = 7.5 ; y = 0 + 8 - 1 = 7 ; z = 6
    const a2 = d.atoms[1];
    expect(a2.x).toBeCloseTo(7.5);
    expect(a2.y).toBeCloseTo(7);
    expect(a2.z).toBeCloseTo(6);
    // atom 1 at origin corner
    expect(d.atoms[0]).toMatchObject({ x: 0, y: 0, z: 0 });
    expect(d.box!.xy).toBeCloseTo(-2);
  });

  it('prefers x/y/z over xu/yu/zu over xs/ys/zs and errors without any', () => {
    const unwrapped = parseDumpFile(`ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type xu yu zu
1 1 12.0 3.0 4.0
`);
    expect(unwrapped.atoms[0]).toMatchObject({ x: 12, y: 3, z: 4 });

    expect(() => parseDumpFile(`ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type vx vy vz
1 1 0 0 0
`)).toThrow(/coord/i);
  });

  it('throws on empty input and atom-count mismatch in the first frame', () => {
    expect(() => parseDumpFile('')).toThrow(/dump/i);
    expect(() => parseDumpFile(`ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
1 1 0 0 0
`)).toThrow(/expected 3 atoms/);
  });

  it('format detection: extension and content sniffing', () => {
    expect(detectFileFormat('run.lammpstrj')).toBe('lammpsdump');
    expect(detectFileFormat('snap.dump')).toBe('lammpsdump');
    expect(detectFormatFromContent(ORTHO)).toBe('lammpsdump');
  });
});
