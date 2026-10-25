import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): the last number of each fix and compute style line replaced by the word bogus. */
paritySuite('fix-compute-last-word');
