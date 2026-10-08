import type { ComputeFactory, FixFactory } from '../styles';
import { FixNVEAsphere } from '../fix/nve_asphere';
import { ComputeERotateAsphere, ComputeTempAsphere } from '../compute/asphere';
import { FixNHAsphere } from '../fix/nh_asphere';

/** aspherical (ellipsoid) integrators and computes (wave 19); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  'nve/asphere': (s, i, g, a) => new FixNVEAsphere(s, i, g, a),
  'nve/asphere/noforce': (s, i, g, a) => new FixNVEAsphere(s, i, g, a, true),
  'nvt/asphere': (s, i, g, a) => new FixNHAsphere(s, i, g, a, 'nvt'),
  'npt/asphere': (s, i, g, a) => new FixNHAsphere(s, i, g, a, 'npt'),
  'nph/asphere': (s, i, g, a) => new FixNHAsphere(s, i, g, a, 'nph'),
};
export const COMPUTES: Record<string, ComputeFactory> = {
  'erotate/asphere': (s, i, g, a) => new ComputeERotateAsphere(s, i, g, a),
  'temp/asphere': (s, i, g, a) => new ComputeTempAsphere(s, i, g, a),
};
