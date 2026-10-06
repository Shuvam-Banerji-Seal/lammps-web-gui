import type { FixSpec } from './types';
import { FixEnforce2d, FixNve, type Fix } from './integrate';

/*
 * Fix construction from parsed specs. The interpreter parses every
 * supported fix's arguments (docs.lammps.org/fix_<style>.html) into a
 * FixSpec; this module turns a spec into a Fix object.
 */

export class FixNotImplementedError extends Error {}

export const makeFix = (spec: FixSpec): Fix => {
  switch (spec.style) {
    case 'nve':
      return new FixNve(spec.id);
    case 'enforce2d':
      return new FixEnforce2d(spec.id);
    default:
      throw new FixNotImplementedError(`fix ${spec.style} is not available in this build of the notebook engine yet`);
  }
};
