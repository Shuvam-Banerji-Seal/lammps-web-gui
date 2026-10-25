import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): the first number after the type words of each coefficient line replaced by the word bogus. */
paritySuite('coeff-word');
