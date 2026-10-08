import type { FixFactory } from '../styles';
import type { Bonded, Pair } from '../force/types';

/** Lepton-expression styles (wave 16); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {};
export const BONDS: Record<string, () => Bonded> = {};
export const ANGLES: Record<string, () => Bonded> = {};
export const DIHEDRALS: Record<string, () => Bonded> = {};
export const FIXES: Record<string, FixFactory> = {};
