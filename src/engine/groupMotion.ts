/*
 * Group motion about the centre of mass (Group::xcm, angmom, inertia and omega in src/group.cpp at b332f28): the
 * velocity zero angular, fix momentum angular and the variable omega() function use it. The solve is the lazily
 * loaded helper of the command word (src/engine-gpl/group/group_omega.ts, through gplHook()), so a caller that runs
 * before the word is loaded gets a StyleError naming it.
 */
import { gplHook } from './gplHooks';
import { massOf } from './atoms';
import type { System } from './system';

export interface GroupMotion {
  masstotal: number;
  xcm: number[];
  omega: number[];
}

/** The lazily loaded helper stored under a command word (lazyStyles.ts group gpl_group_omega). */
export interface GroupMotionHook {
  motion(pos: number[][], vel: number[][], mass: number[]): GroupMotion;
}

/**
 * The motion of the members (atom indices in ascending order): positions unwrapped with the image flags, the
 * velocities and the masses. word is the command word whose helper computes it: velocity, momentum or omega.
 */
export const groupMotionOf = (sys: System, members: number[], word: string): GroupMotion => {
  const s = sys.state;
  const pos: number[][] = [];
  const vel: number[][] = [];
  const mass: number[] = [];
  for (const i of members) {
    const u = [0, 0, 0];
    sys.geom.unwrap(s.x, s.image, i, u);
    pos.push(u);
    vel.push([s.v[3 * i], s.v[3 * i + 1], s.v[3 * i + 2]]);
    mass.push(massOf(s, i));
  }
  return gplHook<GroupMotionHook>(word).motion(pos, vel, mass);
};
