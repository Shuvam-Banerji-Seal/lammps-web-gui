import { describe, expect, it } from 'vitest';
import { VoroCellNeighbor } from '../src/engine-gpl/voro/cell';
import { VoroContainer } from '../src/engine-gpl/voro/container';
import { WL, WL_HGRIDCU, WL_SEQ_LENGTH } from '../src/engine-gpl/voro/worklist';
import { nativeNeighborRows } from '../src/engine-gpl/voro/nativeNeighbors';

/*
 * Unit checks of the voro++ 0.4.6 port (src/engine-gpl/voro/): the worklist table, a single cut of the cell, the
 * neighbour ids of a lattice cell, the native row order of a periodic simple cubic box (the 162 rows of the native
 * fixture tests/oracle/w199vo_sc), and the refusal of layouts with more than one ghost swap per side.
 */

describe('voro++ worklist table', () => {
  it('has 64 worklists of 64 entries, each starting with the count of the entries it lists', () => {
    expect(WL.length).toBe(WL_HGRIDCU * WL_SEQ_LENGTH);
    for (let w = 0; w < WL_HGRIDCU; w++) {
      const count = WL[w * WL_SEQ_LENGTH];
      expect(count).toBeGreaterThan(0);
      expect(count).toBeLessThan(WL_SEQ_LENGTH);
    }
  });
});

describe('voronoicell_neighbor', () => {
  it('cut of the unit box by one plane gives the plane id as a face and keeps the wall ids', () => {
    const c = new VoroCellNeighbor();
    c.init(-1, 1, -1, 1, -1, 1);
    // the plane x = 0.6 (points at distance 0.6 along +x from the cell centre): a new face with id 7
    expect(c.nplane(0.6, 0, 0, 0.36, 7)).toBe(true);
    const nb = c.neighbors();
    expect(nb.length).toBe(6);
    expect(nb).toContain(7);
    // the other five faces are the box walls (negative ids of voronoicell_neighbor::init)
    expect(nb.filter((v) => v < 0).length).toBe(5);
  });

  it('a plane that does not cut the box leaves the six wall faces', () => {
    const c = new VoroCellNeighbor();
    c.init(-1, 1, -1, 1, -1, 1);
    expect(c.nplane(5, 0, 0, 25, 3)).toBe(true);
    expect(c.neighbors().length).toBe(6);
    expect(c.faceAreas().length).toBe(6);
  });
});

describe('native row order (compute voronoi/atom neighbors yes)', () => {
  it('a periodic 3x3x3 simple cubic box gives 162 rows with unit areas, six per owner', () => {
    const x = new Float64Array(27 * 3);
    let k = 0;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let l = 0; l < 3; l++) {
      x[3 * k] = i; x[3 * k + 1] = j; x[3 * k + 2] = l; k++;
    }
    const rows = nativeNeighborRows({
      dim: 3, lo: [0, 0, 0], hi: [3, 3, 3], periodic: [true, true, true], cutghost: 1.8, x,
      owner: new Uint8Array(27).fill(1), pool: new Uint8Array(27).fill(1), faceThreshold: 0,
    });
    expect(rows).not.toBeNull();
    expect(rows!.length).toBe(162);
    const perOwner = new Map<number, number>();
    for (const r of rows!) {
      expect(Math.abs(r.area - 1)).toBeLessThan(1e-12);
      perOwner.set(r.owner, (perOwner.get(r.owner) ?? 0) + 1);
    }
    expect([...perOwner.values()].every((n) => n === 6)).toBe(true);
    // the neighbours of owner 0 are its six nearest periodic images (distance 1 in the 3x3x3 box)
    const owner0 = rows!.filter((r) => r.owner === 0);
    expect(owner0.length).toBe(6);
    for (const r of owner0) {
      let d2 = 0;
      for (let d = 0; d < 3; d++) {
        let dd = Math.abs(x[3 * r.neighbor + d] - x[d]);
        dd = Math.min(dd, 3 - dd);
        d2 += dd * dd;
      }
      expect(Math.abs(Math.sqrt(d2) - 1)).toBeLessThan(1e-12);
    }
  });

  it('returns null when the ghost cutoff reaches a box length (several swaps per side are not reproduced)', () => {
    const x = new Float64Array(8 * 3);
    for (let k = 0; k < 8; k++) { x[3 * k] = k & 1; x[3 * k + 1] = (k >> 1) & 1; x[3 * k + 2] = (k >> 2) & 1; }
    const rows = nativeNeighborRows({
      dim: 3, lo: [0, 0, 0], hi: [1, 1, 1], periodic: [true, true, true], cutghost: 1.5, x,
      owner: new Uint8Array(8).fill(1), pool: new Uint8Array(8).fill(1), faceThreshold: 0,
    });
    expect(rows).toBeNull();
  });

  it('a container holds the particles it was given and the block search finds the cell of one particle', () => {
    const con = new VoroContainer(0, 2, 0, 2, 0, 2, 2, 2, 2);
    con.put(0, 0.5, 0.5, 0.5);
    con.put(1, 1.5, 0.5, 0.5);
    expect(con.co[0]).toBe(1);
    const c = new VoroCellNeighbor();
    expect(con.computeCell(c, 0, 0, 0, 0, 0)).toBe(true);
    expect(c.neighbors()).toContain(1);
  });
});
