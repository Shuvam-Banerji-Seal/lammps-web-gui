import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): pair_modify shift yes, tail yes and mix arithmetic before the first run 0 of each pair style, pe, press and fnorm compared. */
paritySuite('pair-modify');
