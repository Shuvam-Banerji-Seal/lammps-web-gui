/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the registrations of src/REPLICA/temper.h (Command temper), src/REPLICA/temper_npt.h (temper/npt),
 * src/REPLICA/temper_grem.h (temper/grem), src/REPLICA/prd.h (Command prd), src/REPLICA/tad.h (Command tad),
 * src/REPLICA/fix_alchemy.h (FixStyle(alchemy,FixAlchemy)) and src/REPLICA/compute_pressure_alchemy.h
 * (ComputeStyle(pressure/alchemy,ComputePressureAlchemy)) at commit b332f28; the commands and classes are in
 * ../replica/ (see each file's header).
 *
 * Loaded through src/engine/lazyStyles.ts (group gpl_replica); the session runs the tempering, prd and tad commands
 * once, over the replicas (src/engine/session.ts REPLICA_DRIVEN).
 */
import type { Handler } from '../../engine/commands/args';
import type { ComputeFactory, FixFactory } from '../../engine/styles';
import { COMMANDS as TEMPER_COMMANDS } from '../replica/temper';
import { runPrd } from '../replica/prd';
import { runTad } from '../replica/tad';
import { FixAlchemy } from '../replica/fix_alchemy';
import { ComputePressureAlchemy } from '../replica/compute_pressure_alchemy';

export const COMMANDS: Record<string, Handler> = {
  ...TEMPER_COMMANDS,
  prd: (ctx, a) => runPrd(ctx, a),
  tad: (ctx, a) => runTad(ctx, a),
};

export const FIXES: Record<string, FixFactory> = {
  alchemy: (sys, id, group, args) => new FixAlchemy(sys, id, group, args),
};

export const COMPUTES: Record<string, ComputeFactory> = {
  'pressure/alchemy': (sys, id, group, args) => new ComputePressureAlchemy(sys, id, group, args),
};
