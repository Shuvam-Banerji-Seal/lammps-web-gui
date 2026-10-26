import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): an extra word after each other command (one line per command and style). */
paritySuite('command-trailing');
