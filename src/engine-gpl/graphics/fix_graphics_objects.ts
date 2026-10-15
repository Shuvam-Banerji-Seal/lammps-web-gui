/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/fix_graphics_objects.cpp and fix_graphics_objects.h (class FixGraphicsObjects, fix
 * graphics/objects: "fix ID group-ID graphics/objects N type args ..."). Copyright (2003) Sandia Corporation,
 * distributed under the GNU General Public License (the LAMMPS file headers). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Documented behaviour (docs.lammps.org/fix_graphics_objects.html): the objects are sphere, cylinder, arrow and
 * cone, each given by a colour type and geometry; "The radius ... is given" (the C++ doubles the argument into a
 * diameter, fix_graphics_objects.cpp: "sphere.diameter *= 2.0"). Objects are drawn by dump image through the
 * "fix ID type|element|const flag1 flag2" keyword. The progress bar object (progbar, progbar.ts) is drawn as native
 * cylinders by the GPL renderer (draw.ts).
 *
 * Difference: the browser engine evaluates the objects when the image is written (the same step as the dump,
 * when N divides the dump frequency, as in the examples), not at every N-th end of step.
 */
import type { System } from '../../engine/system';
import { Fix } from '../../engine/fix/fix';
import type { FixFactory } from '../../engine/styles';
import { StyleError } from '../../engine/force/types';
import type { ImageObject, Vec3 } from '../../engine/output/image_objects';
import { evalNumOrVar, parseInteger, parseNumber, parseNumOrVar, type NumOrVar } from './common';
import { parseProgbar, progbarObjects, type ProgbarSpec } from './progbar';
import { drawGraphicsObject } from './draw';

type Item =
  | { kind: 'progbar'; spec: ProgbarSpec; progress: NumOrVar }
  | { kind: 'sphere'; type: number; pos: NumOrVar[]; radius: NumOrVar }
  | { kind: 'cylinder'; type: number; p1: NumOrVar[]; p2: NumOrVar[]; radius: NumOrVar }
  | { kind: 'arrow'; type: number; bot: NumOrVar[]; tip: NumOrVar[]; radius: NumOrVar; ratio: number }
  | { kind: 'cone'; type: number; bot: NumOrVar[]; top: NumOrVar[]; r1: NumOrVar; r2: NumOrVar };

const vec = (sys: System, v: NumOrVar[]): Vec3 => [evalNumOrVar(sys, v[0]), evalNumOrVar(sys, v[1]), evalNumOrVar(sys, v[2])];

export class FixGraphicsObjects extends Fix {
  readonly style = 'graphics/objects';
  private readonly items: Item[] = [];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError(`usage: fix ${id} group-ID graphics/objects N type args ...`);
    this.nevery = parseInteger(args[0], `fix ${id} graphics/objects N`);
    if (this.nevery <= 0) throw new StyleError(`fix ${id} graphics/objects: N must be > 0`);
    const need = (k: number, name: string) => {
      if (args.length < k) throw new StyleError(`fix ${id} graphics/objects ${name}: missing arguments`);
    };
    const num = (w: string | undefined, what: string) => parseNumOrVar(w, `fix ${id} graphics/objects ${what}`);
    let i = 1;
    while (i < args.length) {
      const kw = args[i];
      if (kw === 'sphere') {
        need(i + 6, 'sphere');
        this.items.push({ kind: 'sphere', type: parseInteger(args[i + 1], 'sphere type'), pos: [num(args[i + 2], 'sphere x'), num(args[i + 3], 'sphere y'), num(args[i + 4], 'sphere z')], radius: num(args[i + 5], 'sphere radius') });
        i += 6;
      } else if (kw === 'cylinder') {
        need(i + 9, 'cylinder');
        this.items.push({ kind: 'cylinder', type: parseInteger(args[i + 1], 'cylinder type'), p1: [num(args[i + 2], 'cylinder x1'), num(args[i + 3], 'cylinder y1'), num(args[i + 4], 'cylinder z1')], p2: [num(args[i + 5], 'cylinder x2'), num(args[i + 6], 'cylinder y2'), num(args[i + 7], 'cylinder z2')], radius: num(args[i + 8], 'cylinder radius') });
        i += 9;
      } else if (kw === 'arrow') {
        need(i + 10, 'arrow');
        const ratio = Number(args[i + 9]);
        if (!(ratio >= 0.1 && ratio <= 0.5)) throw new StyleError(`fix ${id} graphics/objects arrow: tip ratio must be between 0.1 and 0.5`);
        this.items.push({ kind: 'arrow', type: parseInteger(args[i + 1], 'arrow type'), bot: [num(args[i + 2], 'arrow x1'), num(args[i + 3], 'arrow y1'), num(args[i + 4], 'arrow z1')], tip: [num(args[i + 5], 'arrow x2'), num(args[i + 6], 'arrow y2'), num(args[i + 7], 'arrow z2')], radius: num(args[i + 8], 'arrow radius'), ratio });
        i += 10;
      } else if (kw === 'cone') {
        need(i + 11, 'cone');
        const sides = parseInteger(args[i + 10], 'cone sides');
        if (sides < 0 || sides > 7) throw new StyleError(`fix ${id} graphics/objects cone: sides must be between 0 and 7`);
        // fix_graphics_objects.cpp draws a round cone when sides is 0; polygonal cones are not ported.
        if (sides !== 0) throw new StyleError(`fix ${id} graphics/objects cone: polygonal cones (sides ${sides}) are not supported by the browser engine; use sides 0`);
        this.items.push({ kind: 'cone', type: parseInteger(args[i + 1], 'cone type'), bot: [num(args[i + 2], 'cone x1'), num(args[i + 3], 'cone y1'), num(args[i + 4], 'cone z1')], top: [num(args[i + 5], 'cone x2'), num(args[i + 6], 'cone y2'), num(args[i + 7], 'cone z2')], r1: num(args[i + 8], 'cone bottom radius'), r2: num(args[i + 9], 'cone top radius') });
        i += 11;
      } else if (kw === 'progbar') {
        // fix_graphics_objects.cpp: progbar type1 type2 dim x y z length radius progress tics (11 words with the keyword)
        need(i + 11, 'progbar');
        const prd = this.sys.state.box;
        const prdLen: [number, number, number] = [prd.hi[0] - prd.lo[0], prd.hi[1] - prd.lo[1], prd.hi[2] - prd.lo[2]];
        const spec = parseProgbar(
          parseInteger(args[i + 1], 'progbar type1'),
          parseInteger(args[i + 2], 'progbar type2'),
          args[i + 3],
          [parseNumber(args[i + 4], 'progbar x'), parseNumber(args[i + 5], 'progbar y'), parseNumber(args[i + 6], 'progbar z')],
          parseNumber(args[i + 7], 'progbar length'),
          parseNumber(args[i + 8], 'progbar radius'),
          parseInteger(args[i + 10], 'progbar tics'),
          prdLen,
        );
        this.items.push({ kind: 'progbar', spec, progress: num(args[i + 9], 'progbar progress') });
        i += 11;
      } else {
        throw new StyleError(`fix ${id} graphics/objects: unknown object '${kw}' (sphere, cylinder, arrow or cone)`);
      }
    }
    if (this.items.length === 0) throw new StyleError(`fix ${id} graphics/objects needs at least one object`);
  }

  /** The objects for dump image (fix_graphics_objects.cpp end_of_step and image()). */
  imageObjects(): ImageObject[] {
    const sys = this.sys;
    const out: ImageObject[] = [];
    for (const it of this.items) {
      if (it.kind === 'progbar') {
        // the progress bar objects are native cylinders: drawn by the GPL renderer (dump_image.cpp CYLINDER)
        const progress = evalNumOrVar(sys, it.progress);
        for (const c of progbarObjects(it.spec, progress)) out.push({ kind: 'draw', draw: (f, st) => drawGraphicsObject(f, st, c) });
        continue;
      }
      out.push(this.imageItem(it));
    }
    return out;
  }

  private imageItem(it: Exclude<Item, { kind: 'progbar' }>): ImageObject {
    const sys = this.sys;
    if (it.kind === 'sphere') return { kind: 'sphere', type: it.type, center: vec(sys, it.pos), diameter: 2 * evalNumOrVar(sys, it.radius) };
    if (it.kind === 'cylinder') return { kind: 'cylinder', type: it.type, p1: vec(sys, it.p1), p2: vec(sys, it.p2), diameter: 2 * evalNumOrVar(sys, it.radius) };
    if (it.kind === 'cone') return { kind: 'cone', type: it.type, p1: vec(sys, it.bot), p2: vec(sys, it.top), d1: 2 * evalNumOrVar(sys, it.r1), d2: 2 * evalNumOrVar(sys, it.r2) };
    // an arrow runs from bot to tip: centre, unit direction and length as the image object takes them
    const b = vec(sys, it.bot), t = vec(sys, it.tip);
    // fix_graphics_objects.cpp (ARROW, the imgparms of end_of_step): the unit direction and the length |tip - bot|
    const raw: Vec3 = [t[0] - b[0], t[1] - b[1], t[2] - b[2]];
    const length = Math.hypot(raw[0], raw[1], raw[2]);
    const inv = length > 0 ? 1.0 / length : 0;
    const dir: Vec3 = [inv * raw[0], inv * raw[1], inv * raw[2]];
    const center: Vec3 = [0.5 * (t[0] + b[0]), 0.5 * (t[1] + b[1]), 0.5 * (t[2] + b[2])];
    return { kind: 'arrow', type: it.type, center, dir, length, diameter: 2 * evalNumOrVar(sys, it.radius), ratio: it.ratio };
  }
}

export const makeFixGraphicsObjects: FixFactory = (sys, id, group, args) => new FixGraphicsObjects(sys, id, group, args);
