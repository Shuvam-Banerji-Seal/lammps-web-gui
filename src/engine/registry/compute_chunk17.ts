import type { ComputeFactory } from '../styles';
import {
  ComputeAngmomChunk, ComputeChunkSpreadAtom, ComputeDipole, ComputeDipoleChunk, ComputeGyrationChunk,
  ComputeGyrationShape, ComputeGyrationShapeChunk, ComputeInertiaChunk, ComputeMomentum, ComputeOmegaChunk,
  ComputePropertyChunk, ComputeReduceChunk,
} from '../compute/chunk17';
import { StyleError } from '../force/types';

/** chunk and shape computes (wave 17, GLM worker); merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'angmom/chunk': (sys, id, group, args) => new ComputeAngmomChunk(sys, id, group, args),
  'omega/chunk': (sys, id, group, args) => new ComputeOmegaChunk(sys, id, group, args),
  'inertia/chunk': (sys, id, group, args) => new ComputeInertiaChunk(sys, id, group, args),
  'gyration/chunk': (sys, id, group, args) => new ComputeGyrationChunk(sys, id, group, args),
  'gyration/shape': (sys, id, group, args) => new ComputeGyrationShape(sys, id, group, args),
  'gyration/shape/chunk': (sys, id, group, args) => new ComputeGyrationShapeChunk(sys, id, group, args),
  momentum: (sys, id, group, args) => new ComputeMomentum(sys, id, group, args),
  dipole: (sys, id, group, args) => new ComputeDipole(sys, id, group, args),
  'dipole/chunk': (sys, id, group, args) => new ComputeDipoleChunk(sys, id, group, args),
  'dipole/tip4p': () => { throw new StyleError('compute dipole/tip4p is not supported by the engine yet'); },
  'dipole/tip4p/chunk': () => { throw new StyleError('compute dipole/tip4p/chunk is not supported by the engine yet'); },
  'reduce/chunk': (sys, id, group, args) => new ComputeReduceChunk(sys, id, group, args),
  'chunk/spread/atom': (sys, id, group, args) => new ComputeChunkSpreadAtom(sys, id, group, args),
  'property/chunk': (sys, id, group, args) => new ComputePropertyChunk(sys, id, group, args),
};
