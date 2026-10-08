import type { Bonded } from '../force/types';
import {
  ImproperCvff, ImproperUmbrella, ImproperCossq, ImproperFourier,
  ImproperDistance, ImproperZero,
} from '../force/improper/styles';

/** Improper styles added by the wave-1 rerun; merged into styles.ts. Style name -> factory. */
export const IMPROPERS: Record<string, () => Bonded> = {
  cvff: () => new ImproperCvff(),
  umbrella: () => new ImproperUmbrella(),
  cossq: () => new ImproperCossq(),
  fourier: () => new ImproperFourier(),
  distance: () => new ImproperDistance(),
  zero: () => new ImproperZero(),
};
