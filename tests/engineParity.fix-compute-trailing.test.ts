import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): an extra word after each fix and compute style line. */
paritySuite('fix-compute-trailing');
