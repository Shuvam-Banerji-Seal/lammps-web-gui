import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): the group ID of each fix, compute, dump, velocity, displace_atoms, delete_atoms group and set group line replaced by nogroup. */
paritySuite('group-id');
