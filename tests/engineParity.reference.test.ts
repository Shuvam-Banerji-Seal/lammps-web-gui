import { paritySuite } from './parityHarness';

/* Parity with native LAMMPS (tests/parityHarness.ts): the first c_ID, f_ID or v_name reference of each line replaced by c_nosuch, f_nosuch or v_nosuch. */
paritySuite('reference');
