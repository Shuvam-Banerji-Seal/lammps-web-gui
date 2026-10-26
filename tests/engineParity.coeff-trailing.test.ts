import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): an extra word after each coefficient line (one line per style). */
paritySuite('coeff-trailing');
