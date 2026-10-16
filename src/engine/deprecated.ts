import { StyleError } from './force/types';

/*
 * Commands and styles that LAMMPS has removed or renamed — docs.lammps.org/Commands_removed.html: "This page lists
 * LAMMPS commands and packages that have been removed from the distribution and provides suggestions for alternatives
 * or replacements.  LAMMPS has special dummy styles implemented, that will stop LAMMPS and print a suitable error
 * message in most cases". The native binary still knows these
 * names (lmp -h lists them) and stops with a note on what replaced them, so the engine gives the same answer instead
 * of a generic "not supported" (as native LAMMPS does, src/*_deprecated.cpp and src/deprecated.cpp at b332f28: a
 * note, then "<Kind> style <name> is no longer available."). The renamed reset commands run under their new name.
 *
 * Measured with native LAMMPS: pair_style reax/c stops with Pair style reax/c is no longer available after the note
 * Pair style 'reax/c' has been renamed to 'reaxff'; reset_atom_ids prints WARNING: 'reset_atom_ids' has been renamed
 * to 'reset_atoms id'. Please update your input. and runs reset_atoms id.
 */

/** The style kinds with removed names, keyed by the command that names them. */
export type RemovedKind = 'pair_style' | 'fix' | 'compute' | 'dump' | 'min_style' | 'command';

const LABEL: Record<RemovedKind, string> = {
  pair_style: 'Pair style', fix: 'Fix style', compute: 'Compute style', dump: 'Dump style',
  min_style: 'Minimize style', command: 'The',
};

const AVE_SPATIAL =
  "Fix styles 'ave/spatial' and 'ave/spatial/sphere' have been replaced by the more general fix ave/chunk and " +
  'compute chunk/atom commands. All ave/spatial and ave/spatial/sphere functionality is available in these new ' +
  'commands. These ave/spatial keywords & options are part of fix ave/chunk: Nevery, Nrepeat, Nfreq, input values, ' +
  'norm, ave, file, overwrite, title123. These ave/spatial keywords & options for binning are part of compute ' +
  'chunk/atom: dim, origin, delta, region, bound, discard, units.';
const MPIIO = 'The MPIIO package has been removed from LAMMPS.';
const MESSAGE = 'The MESSAGE package has been replaced by the MDI package.';

/** name -> the note native prints before stopping ('' when it prints none). */
const REMOVED: Record<RemovedKind, Record<string, string>> = {
  pair_style: {
    reax: "Pair style 'reax' has been removed from LAMMPS after the 12 December 2018 version.",
    'reax/c': "Pair style 'reax/c' has been renamed to 'reaxff'.",
    'meam/c': "Pair style 'meam/c' has been renamed to 'meam'.",
    'mesont/tpm': "Pair style 'mesont/tpm' has been removed from LAMMPS. Please use pair style 'mesocnt' instead.",
    'awpmd/cut': 'The AWPMD package has been removed from LAMMPS.',
    agni: "Pair style 'agni' has been removed from LAMMPS.",
    'agni/omp': "Pair style 'agni' has been removed from LAMMPS.",
    rann: "Pair style 'rann' and the ML-RANN package have been removed from LAMMPS.",
  },
  fix: {
    atc: 'The ATC package has been removed from LAMMPS.',
    'ave/spatial': AVE_SPATIAL,
    'ave/spatial/sphere': AVE_SPATIAL,
    'lb/pc': "Fix style 'lb/pc' has been removed from the LATBOLTZ package; 'fix nve' can be used in its place.",
    'lb/rigid/pc/sphere': "Fix style 'lb/rigid/pc/sphere' has been removed from the LATBOLTZ package; 'fix rigid' can be used in its place.",
    'nve/awpmd': 'The AWPMD package has been removed from LAMMPS.',
    poems: 'The POEMS package has been removed from LAMMPS.',
    'reax/c/bonds': "Fix style 'reax/c/bonds' has been renamed to 'reaxff/bonds'.",
    'reax/c/species': "Fix style 'reax/c/species' has been renamed to 'reaxff/species'.",
  },
  compute: {
    mesont: "Compute style 'mesont' and the associated pair style have been removed. Please use pair style 'mesocnt' instead.",
  },
  dump: { 'atom/mpiio': MPIIO, 'cfg/mpiio': MPIIO, 'custom/mpiio': MPIIO, 'xyz/mpiio': MPIIO },
  min_style: { 'fire/old': "Minimize style 'fire/old' has been removed from LAMMPS. Please use 'min_style fire' instead." },
  command: { message: MESSAGE, server: MESSAGE },
};

// native error->all ends with utils::errorurl(38)
const MORE = 'For more information see https://docs.lammps.org/err0038';

/** The error for a removed name of this kind, or null when the name was not removed. */
export const removedStyleError = (kind: RemovedKind, name: string): StyleError | null => {
  if (!Object.prototype.hasOwnProperty.call(REMOVED[kind], name)) return null;
  const note = REMOVED[kind][name];
  const what = kind === 'command' ? `The ${name} command is no longer available.`
    : kind === 'min_style' && note ? '' : `${LABEL[kind]} ${name} is no longer available.`;
  return new StyleError([note, what, MORE].filter(Boolean).join(' '));
};

/**
 * Commands that LAMMPS renamed: Commands_removed.html, "The reset_ids, reset_atom_ids, and reset_mol_ids commands
 * have been folded into the reset_atoms command.  If present, LAMMPS will replace the commands accordingly and print a
 * warning." The kim_ commands are replaced the same way (src/deprecated.cpp at b332f28: kim with the rest of the name
 * as its first word).
 */
export const renamedCommand = (cmd: string, args: string[]): { cmd: string; args: string[]; warning: string } | null => {
  let to: string[] | null = null;
  if (cmd === 'reset_ids' || cmd === 'reset_atom_ids') to = ['reset_atoms', 'id'];
  else if (cmd === 'reset_mol_ids') to = ['reset_atoms', 'mol'];
  else if (['kim_init', 'kim_interactions', 'kim_param', 'kim_property', 'kim_query'].includes(cmd)) to = ['kim', cmd.slice(4)];
  if (!to) return null;
  return {
    cmd: to[0], args: [...to.slice(1), ...args],
    warning: `'${cmd}' has been renamed to '${to.join(' ')}'. Please update your input.`,
  };
};
