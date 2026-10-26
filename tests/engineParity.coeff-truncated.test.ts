import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): the last word of each coefficient line dropped. */
paritySuite('coeff-truncated');
