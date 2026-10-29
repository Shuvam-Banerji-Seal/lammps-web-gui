import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): compute pair/local eng force reduced to sums before the first run 0 of each pair style (the style's single()), pe and the two sums compared. */
paritySuite('pair-local');
