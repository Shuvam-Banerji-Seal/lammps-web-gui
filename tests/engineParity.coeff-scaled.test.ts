import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): the first number after the type words of each coefficient line multiplied by 1.1, pe and press compared after run 0. */
paritySuite('coeff-scaled');
