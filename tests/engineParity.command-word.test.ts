import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): the first number of each other command (from word 1) replaced by the word bogus. */
paritySuite('command-word');
