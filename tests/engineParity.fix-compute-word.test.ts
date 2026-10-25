import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): the first number of each fix and compute style (from word 4) replaced by the word bogus. */
paritySuite('fix-compute-word');
