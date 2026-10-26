import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): the atom type of each coefficient, mass and create_atoms line replaced by 99. */
paritySuite('type');
