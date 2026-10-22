import { StyleError } from './force/types';
import { gplHook } from './gplHooks';
import { ensureStyles } from './styles';

/*
 * Style names the engine does not have. Native LAMMPS stops on a name no package registers with "Unrecognized {kind}
 * style '{name}'" (utils::check_packages_for_style, src/utils.cpp at b332f28; the minimize and integrate styles with
 * "Unknown minimize style {name}" and "Unknown integrate style {name}", src/update.cpp). For a style some package has
 * but the engine does not, the engine refuses it by name with its own message.
 */

export type NativeStyleKind =
  'pair' | 'bond' | 'angle' | 'dihedral' | 'improper' | 'kspace' | 'fix' | 'compute' | 'dump' | 'region' | 'atom' | 'minimize' | 'integrate';

interface NativeStyles { known(kind: NativeStyleKind, name: string): boolean }

/** Loads the native style names (a lazy module of src/engine-gpl, loaded once). */
export const loadNativeStyles = (): Promise<void> => ensureStyles(['native_styles']);

/** Whether some LAMMPS package has the style; true when the names are not loaded (the engine's refusal then applies). */
export const isNativeStyle = (kind: NativeStyleKind, name: string): boolean => {
  let h: NativeStyles;
  try {
    h = gplHook<NativeStyles>('native_styles');
  } catch {
    return true;
  }
  return h.known(kind, name);
};

/**
 * Native's message for a style no LAMMPS package has, else the engine's refusal of a style it lacks. Native's line is
 * followed by the engine's list of the kind's styles from the refusal ("(supported: ...)", which the notebook folds
 * away and searches for a near name).
 */
export const unknownStyleError = (kind: NativeStyleKind, name: string, refusal: () => StyleError): StyleError => {
  const why = refusal();
  if (isNativeStyle(kind, name)) return why;
  const head = kind === 'minimize' || kind === 'integrate' ? `Unknown ${kind} style ${name}` : `Unrecognized ${kind} style '${name}'`;
  const list = /supported: ([^)]*?)\.?$/.exec(why.message)?.[1];
  return new StyleError(list ? `${head}\n(supported: ${list})` : head);
};
