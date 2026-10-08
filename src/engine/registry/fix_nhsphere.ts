import type { FixFactory } from '../styles';
import { FixNHSphere } from '../fix/nh_sphere';

/** fix nvt/sphere, npt/sphere, nph/sphere (Haiku wave); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  'nvt/sphere': (s, i, g, a) => new FixNHSphere(s, i, g, a, 'nvt'),
  'npt/sphere': (s, i, g, a) => new FixNHSphere(s, i, g, a, 'npt'),
  'nph/sphere': (s, i, g, a) => new FixNHSphere(s, i, g, a, 'nph'),
};
