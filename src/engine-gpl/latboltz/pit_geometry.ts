/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/LATBOLTZ/fix_lb_fluid.cpp at commit b332f28 (initializeGlobalGeometry, addslit, addpit and the
 * geometry part of initializeGeometry, serial). Contributing authors: Frances Mackay, Santtu Ollila, Tyson Whitehead,
 * Colin Denniston (UWO). Copyright (2003) Sandia Corporation, GNU General Public License. This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_lb_fluid.html: "npits npits h_p l_p l_pp l_e: setup to use pit geometry ... wp w_p: y-width of
 * slit regions (defaults to full width) ... sw: y-sidewalls (in xz plane) on/off".
 *
 * The lattice is stored flat as native's wholelattice[Nbx][Nby][Nbz]: index (i * Nby + j) * Nbz + k. Native writes
 * the node HP+HS+1 (= Nbz) of addslit one step past the k range; that write lands on the next (j) row exactly as in
 * the C array, and writes past the end of the array are dropped (the typed-array behaviour), which native's own
 * layout makes harmless.
 */

/** Site types of the lattice (fix_lb_fluid.cpp): 0 bulk fluid, 1 boundary (bounce-back), 2 outside the domain, 3 flat wall. */
export interface PitSpec {
  npits: number;
  /** h_p: z-height of the pit regions (floor to the bottom of the slit). */
  hp: number;
  /** l_p: x-length of a pit. */
  lp: number;
  /** l_pp: x-length of the slit between consecutive pits. */
  lpp: number;
  /** l_e: x-length of the slit at each end. */
  le: number;
  /** w_p: y-width of the slit regions. */
  wp: number;
  /** sw: side walls on (1) or off (0). */
  sw: number;
}

export interface LatticeGeometry {
  Nbx: number;
  Nby: number;
  Nbz: number;
  /** Site type and orientation per whole-lattice node, flat (see the header). */
  type: Int8Array;
  ori: Int8Array;
}

export const wholeIndex = (g: { Nby: number; Nbz: number }, i: number, j: number, k: number): number =>
  (i * g.Nby + j) * g.Nbz + k;

/** A new lattice of the given size, every node outside the domain (type 2, orientation 0). */
export const newLattice = (Nbx: number, Nby: number, Nbz: number): LatticeGeometry => {
  const type = new Int8Array(Nbx * Nby * Nbz).fill(2);
  return { Nbx, Nby, Nbz, type, ori: new Int8Array(Nbx * Nby * Nbz) };
};

/**
 * initializeGlobalGeometry() without pits (npits = -1): flat walls at k = 0 and k = Nbz - 1 when z is not periodic,
 * every other node bulk fluid.
 */
export const flatGeometry = (Nbx: number, Nby: number, Nbz: number, walls: boolean): LatticeGeometry => {
  const g = newLattice(Nbx, Nby, Nbz);
  for (let i = 0; i < Nbx; i++)
    for (let j = 0; j < Nby; j++) {
      const b = wholeIndex(g, i, j, 0), t = wholeIndex(g, i, j, Nbz - 1);
      if (walls) {
        g.type[b] = 3;
        g.ori[b] = 3;
        g.type[t] = 3;
        g.ori[t] = 16;
      } else {
        g.type[b] = 0;
        g.ori[b] = 0;
        g.type[t] = 0;
        g.ori[t] = 0;
      }
      for (let k = 1; k < Nbz - 1; k++) {
        g.type[wholeIndex(g, i, j, k)] = 0;
        g.ori[wholeIndex(g, i, j, k)] = 0;
      }
    }
  return g;
};

/** addslit(): a slit (the channel between pits) of length LE from x0; returns the new x0. */
export const addslit = (g: LatticeGeometry, x0: number, HS: number, HP: number, LE: number, SW: number): number => {
  const { Nbx, Nby } = g;
  const xend = x0 + LE < Nbx ? LE : Nbx - x0;
  const put = (i: number, j: number, k: number, type: number, ori: number) => {
    const n = wholeIndex(g, i, j, k);
    g.type[n] = type;
    g.ori[n] = ori;
  };
  for (let ii = 0; ii < xend; ii++)
    for (let jj = 0; jj < Nby; jj++)
      for (let kk = 0; kk < HS + HP + 1; kk++) put(x0 + ii, jj, kk, 2, 0);

  for (let ii = 0; ii < xend; ii++) {
    for (let jj = 0; jj < Nby; jj++) {
      for (let kk = HP; kk < HS + HP + 1; kk++) {
        let type: number, ori: number;
        if (kk === HP) {
          // bottom of slit & bottom edges
          type = 1;
          ori = 3;
          if (SW === 1 && jj === 0) ori = 6;
          else if (SW === 1 && jj === Nby - 1) ori = 19;
        } else if (kk === HP + HS) {
          // slit ceiling & top edges
          type = 1;
          ori = 16;
          if (SW === 1 && jj === 0) ori = 27;
          else if (SW === 1 && jj === Nby - 1) ori = 28;
        } else {
          // bulk nodes & side walls
          type = 0;
          ori = 0;
          if (SW === 1 && jj === 0) {
            ori = 2;
            type = 1;
          } else if (SW === 1 && jj === Nby - 1) {
            ori = 15;
            type = 1;
          }
        }
        put(x0 + ii, jj, kk, type, ori);
      }
      // native writes node HP+HS+1 of this column (one past the k range; see the header)
      put(x0 + ii, jj, HP + HS + 1, 2, 0);
    }
  }
  return x0 + xend;
};

/**
 * addpit(): a pit of length LP and width WP from x0, with side walls when SW = 1. Returns the new x0.
 *
 * Native keeps one `temp` site for the whole routine and several branches set only the type or only the orientation,
 * so a node keeps the value of the previous node in that case. `tType` and `tOri` reproduce that.
 */
export const addpit = (g: LatticeGeometry, x0: number, HS: number, HP: number, WP: number, LP: number, SW: number): number => {
  const { Nbx, Nby, Nbz } = g;
  let p1 = Math.trunc((Nby - WP) / 2);
  let p2 = p1 + WP - 1;
  if (SW === 1) {
    p1 = 0;
    p2 = Nby - 1;
  }
  if (SW === 0) {
    if (p1 <= 0) p1 = -1;
    if (p2 >= Nby - 1) p2 = Nby;
  }
  const xend = x0 + LP < Nbx ? x0 + LP : Nbx - 1;
  const ZTOP = Nbz - 1;

  let tType = 2;
  let tOri = 0;
  const store = (ii: number, jj: number, kk: number) => {
    const n = wholeIndex(g, ii, jj, kk);
    g.type[n] = tType;
    g.ori[n] = tOri;
  };

  // initialize the whole pit block to "not in fluid"
  for (let ii = x0; ii <= xend; ii++)
    for (let jj = 0; jj < Nby; jj++)
      for (let kk = 0; kk < HS + HP + 1; kk++) store(ii, jj, kk);

  if (SW === 0) {
    for (let ii = x0; ii <= xend; ii++) {
      for (let jj = 0; jj < Nby; jj++) {
        for (let kk = 0; kk < HS + HP + 1; kk++) {
          if ((jj < p1 || jj > p2) && kk < HP) {
            tOri = 0;
            tType = 2;
          }
          // top plate
          else if (kk === HP + HS && HP < ZTOP) {
            tOri = 16;
            tType = 1;
          }
          // slit regions: bottom plate
          else if ((jj < p1 || jj > p2) && kk === HP) {
            tOri = 3;
            tType = 1;
          }
          // bottom of pit, ok for troughs, too
          else if (ii > x0 && ii < LP + x0 && jj > p1 && jj < p2 && kk === 0 && HP < ZTOP) {
            tOri = 3;
            tType = 1;
          }
          // top corners at the level of slit floor WHEN THERE ARE NO SIDE WALLS
          else if (ii === x0 && jj === p1 && p1 > 0 && kk === HP) {
            if (x0 > 0 && LP < Nbx) tOri = 10;
            else tOri = 7;
            tType = 1;
          } else if (ii === x0 + LP && jj === p1 && p1 > 0 && kk === HP) {
            if (x0 + LP < Nbx - 1 && LP < Nbx) tOri = 23;
            else tOri = 7;
            tType = 1;
          } else if (ii === x0 && jj === p2 && p2 < Nby && kk === HP) {
            if (x0 > 0 && LP < Nbx) tOri = 11;
            else tOri = 20;
            tType = 1;
          } else if (ii === x0 + LP && jj === p2 && p2 < Nby && kk === HP) {
            if (x0 + LP < Nbx - 1 && LP < Nbx) tOri = 24;
            else tOri = 20;
            tType = 1;
          }
          // UPSTREAM corners at the level of pit floor
          else if (ii === x0 && jj === p1 && p1 > 0 && kk === 0) {
            if (x0 > 0 && LP < Nbx && HP < ZTOP) tOri = 25;
            else if (x0 > 0 && LP < Nbx && HP >= ZTOP) tOri = 22;
            else tOri = 6;
            tType = 1;
          }
          // DOWNSTREAM corners at the level of pit floor
          else if (ii === x0 + LP && jj === p1 && p1 > 0 && kk === 0) {
            if (x0 + LP < Nbx - 1 && LP < Nbx && HP < ZTOP) tOri = 12;
            else if (x0 + LP < Nbx - 1 && LP < Nbx && HP >= ZTOP) tOri = 9;
            else tOri = 6;
            tType = 1;
          }
          // UPSTREAM corners at the level of pit floor
          else if (ii === x0 && jj === p2 && p2 < Nby && kk === 0) {
            if (x0 > 0 && LP < Nbx && HP < ZTOP) tOri = 26;
            else if (x0 > 0 && LP < Nbx && HP >= ZTOP) tOri = 21;
            else tOri = 19;
            tType = 1;
          }
          // DOWNSTREAM corners at the level of pit floor
          else if (ii === x0 + LP && jj === p2 && p2 < Nby && kk === 0) {
            if (x0 + LP < Nbx - 1 && LP < Nbx && HP < ZTOP) tOri = 13;
            else if (x0 + LP < Nbx - 1 && LP < Nbx && HP >= ZTOP) tOri = 8;
            else tOri = 19;
            tType = 1;
          }
          // side walls + edges
          else if (ii === x0 && jj > p1 && jj < p2) {
            if (x0 > 0 && HP > ZTOP && LP < Nbx && kk < HP) {
              tOri = 1;
              tType = 1;
            } // wall of VERTICAL SHUTE with normal (+1,0,0)
            else if (x0 > 0 && kk < HP && LP < Nbx && kk > 0) {
              tOri = 1;
              tType = 1;
            } // UPSTREAM WALL with normal (+1,0,0)
            else if (kk === 0) {
              // bottom edge along y
              if (x0 > 0 && LP < Nbx) {
                tOri = 5;
                tType = 1;
              } else {
                tOri = 3;
                tType = 1;
              }
            } else if (kk === HP) {
              // bottom edge along y
              if (x0 > 0 && LP < Nbx) {
                tOri = 4;
                tType = 1;
              } else {
                tOri = 0;
                tType = 0;
              }
            } else {
              tOri = 0;
              tType = 0;
            }
          } else if (ii === x0 + LP && jj > p1 && jj < p2) {
            if (x0 + LP < Nbx - 1 && LP < Nbx && kk < HP && HP > ZTOP) {
              tOri = 14;
              tType = 1;
            } else if (x0 + LP < Nbx - 1 && LP < Nbx && kk < HP && kk > 0) {
              tOri = 14;
              tType = 1;
            } else if (kk === 0) {
              // bottom edge along y
              if (x0 + LP < Nbx - 1 && LP < Nbx) {
                tOri = 18;
                tType = 1;
              } else {
                tOri = 3;
                tType = 1;
              }
            } else if (kk === HP) {
              if (LP < Nbx) {
                tOri = 17;
                tType = 1;
              } else {
                tOri = 0;
                tType = 0;
              }
            } else {
              tOri = 0;
              tType = 0;
            }
          } else if (ii > x0 && ii < x0 + LP && jj === p1 && p1 > 0 && kk < HP) {
            if (HP >= ZTOP) {
              tOri = 2;
              tType = 1;
            } else if (kk > 0 && HP < ZTOP) {
              tOri = 2;
              tType = 1;
            } else if (kk === 0) {
              // edge along x axis
              tOri = 6;
              tType = 1;
            } else {
              tOri = 0;
              tType = 0;
            }
          } else if (ii > x0 && ii < x0 + LP && jj === p2 && p2 < Nby && kk < HP) {
            if (HP >= ZTOP) {
              tOri = 15;
              tType = 1;
            } else if (kk > 0 && HP < ZTOP) {
              tOri = 15;
              tType = 1;
            } else if (kk === 0) {
              // edge along x axis
              tOri = 19;
              tType = 1;
            } else {
              tOri = 0;
              tType = 0;
            }
          }
          // top edges along x axis. If sidewalls => they become part of flat wall with (0,+/-1,0) surface normals
          else if (ii > x0 && ii < x0 + LP && jj === p1 && kk === HP) {
            tOri = 7;
            tType = 1;
          } else if (ii > x0 && ii < x0 + LP && jj === p2 && kk === HP) {
            tOri = 20;
            tType = 1;
          }
          // edges along z axis
          else if (ii === x0 && jj === p1 && kk < HP) {
            if (x0 > 0 && LP < Nbx && HP >= ZTOP) {
              tOri = 22;
              tType = 1;
            } else if (x0 > 0 && LP < Nbx && kk > 0 && HP < ZTOP) {
              tOri = 22;
              tType = 1;
            } else {
              tOri = 2;
              tType = 1;
            }
          } else if (ii === x0 && jj === p2 && kk < HP) {
            if (x0 > 0 && LP < Nbx && HP >= ZTOP) {
              tOri = 21;
              tType = 1;
            } else if (x0 > 0 && LP < Nbx && kk > 0 && HP < ZTOP) {
              tOri = 21;
              tType = 1;
            } else {
              tOri = 15;
              tType = 1;
            }
          } else if (ii === x0 + LP && jj === p1 && kk < HP) {
            if (x0 + LP < Nbx - 1 && LP < Nbx && HP >= ZTOP) {
              tOri = 9;
              tType = 1;
            } else if (x0 + LP < Nbx - 1 && LP < Nbx && kk > 0 && HP < ZTOP) {
              tOri = 9;
              tType = 1;
            } else {
              tOri = 2;
              tType = 1;
            }
          } else if (ii === x0 + LP && jj === p2 && kk < HP) {
            if (x0 + LP < Nbx - 1 && LP < Nbx && HP >= ZTOP) {
              tOri = 8;
              tType = 1;
            } else if (x0 + LP < Nbx - 1 && LP < Nbx && kk > 0 && HP < ZTOP) {
              tOri = 8;
              tType = 1;
            } else {
              tOri = 15;
              tType = 1;
            }
          } else {
            tOri = 0;
            tType = 0;
          }
          store(ii, jj, kk);
        }
      }
    }
  } else if (SW === 1) {
    // PIT REGION IN PRESENCE OF SIDE WALLS IN SYSTEM
    for (let ii = x0; ii <= xend; ii++) {
      for (let jj = 0; jj < Nby; jj++) {
        for (let kk = 0; kk < HS + HP + 1; kk++) {
          // ceiling & edges
          if (kk === HP + HS && HP < ZTOP) {
            if (jj === p1) {
              tOri = 27;
              tType = 1;
            } else if (jj === p2) {
              tOri = 28;
              tType = 1;
            } else {
              tOri = 16;
              tType = 1;
            }
          } else if (kk > HP && kk < HP + HS) {
            if (jj === p1) {
              tOri = 2;
              tType = 1;
            } else if (jj === p2) {
              tOri = 15;
              tType = 1;
            } else {
              tOri = 0;
              tType = 0;
            }
          }
          // bottom of pit, ok for troughs, too
          else if (ii > x0 && ii < LP + x0 && jj > p1 && jj < p2 && kk === 0 && HP < ZTOP) {
            tOri = 3;
            tType = 1;
          }
          // top corners at the level of slit floor WHEN THERE ARE SIDE WALLS
          else if (ii === x0 && jj === p1 && kk === HP) {
            if (x0 > 0 && LP < Nbx) tOri = 29;
            else tOri = 2;
            tType = 1;
          } else if (ii === x0 + LP && jj === p1 && kk === HP) {
            if (x0 + LP - 1 < Nbx - 1 && LP < Nbx) {
              tOri = 31;
              tType = 1;
            }
          } else if (ii === x0 && jj === p2 && kk === HP) {
            if (x0 > 0 && LP < Nbx) tOri = 30;
            else tOri = 15;
            tType = 1;
          } else if (ii === x0 + LP && jj === p2 && kk === HP) {
            if (x0 + LP - 1 < Nbx - 1 && LP < Nbx) {
              tOri = 32;
              tType = 1;
            }
          }
          // UPSTREAM corners at the bottom of the pit
          else if (ii === x0 && jj === p1 && kk === 0) {
            if (x0 > 0 && LP < Nbx) tOri = 25;
            else tOri = 6;
            tType = 1;
          }
          // DOWNSTREAM corners at the bottom of the pit
          else if (ii === x0 + LP && jj === p1 && kk === 0) {
            if (x0 + LP - 1 < Nbx - 1 && LP < Nbx) tOri = 12;
            tType = 1;
          }
          // UPSTREAM corners at the bottom of the pit
          else if (ii === x0 && jj === p2 && kk === 0) {
            if (x0 > 0 && LP < Nbx) tOri = 26;
            else tOri = 19;
            tType = 1;
          }
          // DOWNSTREAM corners at the bottom of the pit
          else if (ii === x0 + LP && jj === p2 && kk === 0) {
            if (x0 + LP - 1 < Nbx - 1 && LP < Nbx) tOri = 13;
            tType = 1;
          }
          // UPSTREAM and DOWNSTREAM pit walls + edges
          else if (ii === x0 && jj > p1 && jj < p2) {
            if (x0 > 0 && kk < HP && LP < Nbx && kk > 0) {
              tOri = 1;
              tType = 1;
            } // UPSTREAM WALL with normal (+1,0,0)
            else if (kk === 0) {
              // bottom edge along y
              if (x0 > 0 && LP < Nbx) {
                tOri = 5;
                tType = 1;
              } else {
                tOri = 3;
                tType = 1;
              }
            } else if (kk === HP) {
              // bottom edge along y
              if (x0 > 0 && LP < Nbx) {
                tOri = 4;
                tType = 1;
              } else {
                tOri = 0;
                tType = 0;
              }
            } else {
              tOri = 0;
              tType = 0;
            }
          } else if (ii === x0 + LP && jj > p1 && jj < p2) {
            if (x0 + LP < Nbx - 1 && LP < Nbx && kk < HP && kk > 0) {
              tOri = 14;
              tType = 1;
            } else if (kk === 0) {
              // bottom edge along y
              if (x0 + LP < Nbx - 1 && LP < Nbx) {
                tOri = 18;
                tType = 1;
              } else {
                tOri = 3;
                tType = 1;
              }
            } else if (kk === HP) {
              if (LP < Nbx) {
                tOri = 17;
                tType = 1;
              } else {
                tOri = 0;
                tType = 0;
              }
            } else {
              tOri = 0;
              tType = 0;
            }
          }
          // SIDE WALLS INSIDE PIT
          else if (ii > x0 && ii < x0 + LP && jj === p1) {
            if (kk > 0 && kk < HP + HS) {
              tOri = 2;
              tType = 1;
            } // wall, surface normal (0,+1,0)
            else if (kk === 0) {
              // edge along x axis
              tOri = 6;
              tType = 1;
            } else {
              tOri = 0;
              tType = 0;
            }
          } else if (ii > x0 && ii < x0 + LP && jj === p2) {
            if (kk > 0 && kk < HP + HS) {
              tOri = 15;
              tType = 1;
            } // wall, surface normal (0,-1,0)
            else if (kk === 0) {
              // edge along x axis
              tOri = 19;
              tType = 1;
            } else {
              tOri = 0;
              tType = 0;
            }
          }
          // edges along z axis
          else if (ii === x0 && jj === p1 && kk < HP) {
            if (x0 > 0 && LP < Nbx && kk > 0) {
              tOri = 22;
              tType = 1;
            } else {
              tOri = 2;
              tType = 1;
            }
          } else if (ii === x0 && jj === p2 && kk < HP) {
            if (x0 > 0 && LP < Nbx && kk > 0) {
              tOri = 21;
              tType = 1;
            } else {
              tOri = 15;
              tType = 1;
            }
          } else if (ii === x0 + LP && jj === p1 && kk < HP) {
            if (x0 + LP < Nbx - 1 && LP < Nbx && kk > 0) {
              tOri = 9;
              tType = 1;
            }
          } else if (ii === x0 + LP && jj === p2 && kk < HP) {
            if (x0 + LP < Nbx - 1 && LP < Nbx && kk > 0) {
              tOri = 8;
              tType = 1;
            }
          } else {
            tOri = 0;
            tType = 0;
          }
          store(ii, jj, kk);
        }
      }
    }
  }
  return xend + 1;
};

/**
 * initializeGlobalGeometry() with pits (npits >= 0): the slits and pits are laid out along x block by block.
 * `Nbz` already includes the extra point of the z walls.
 */
export const pitGeometry = (Nbx: number, Nby: number, Nbz: number, HS: number, pit: PitSpec): LatticeGeometry => {
  const g = newLattice(Nbx, Nby, Nbz);
  let ix = 0;
  if (pit.npits === 0) ix = addslit(g, ix, HS, pit.hp, Nbx - 1, pit.sw);
  else ix = addslit(g, ix, HS, pit.hp, pit.le, pit.sw);
  for (let nn = 0; nn < pit.npits; nn++) {
    ix = addpit(g, ix, HS, pit.hp, pit.wp, pit.lp, pit.sw);
    if (nn < pit.npits - 1) ix = addslit(g, ix, HS, pit.hp, pit.lpp - 1, pit.sw);
  }
  if (pit.npits > 0 && ix < Nbx - 1) ix = addslit(g, ix, HS, pit.hp, pit.le - 1, pit.sw);
  return g;
};
