import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): the last word of each fix and compute style line dropped. */
paritySuite('fix-compute-truncated');
