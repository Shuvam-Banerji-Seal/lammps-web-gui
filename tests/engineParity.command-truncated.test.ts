import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): the last word of each other command dropped. */
paritySuite('command-truncated');
