/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/dump_cfg.cpp and dump_cfg.h (class
 * DumpCFG, dump style cfg; contributing author Liang Wan (Chinese Academy of Sciences); copyright Sandia Corporation): the
 * argument checks and the auxiliary names of the constructor, write_header and write_lines; with the peridynamic scale
 * factor of src/PERI/fix_peri_neigh.cpp init; and src/UEF/dump_cfg_uef.cpp (class DumpCFGUef, dump style cfg/uef; contributing
 * author David Nicholson (MIT)): its init_style and write_header. This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/dump.html (verbatim): "Style cfg has the same command syntax as style custom and writes extended CFG
 * format files, as used by the AtomEye visualization package. Since the extended CFG format uses a single snapshot of the
 * system per file, a wildcard "*" must be included in the filename"; "The list of atom attributes for style cfg must begin
 * with either "mass type xs ys zs" or "mass type xsu ysu zsu""; "When unwrapped coordinates xsu, ysu, and zsu are requested,
 * the nominal AtomEye periodic cell dimensions are expanded by a large factor UNWRAPEXPAND = 10.0".
 *
 * The core dump (src/engine/output/dump.ts) parses the attributes as for dump custom, selects and orders the atoms and
 * compiles the column formats; this helper checks the cfg attributes and writes the snapshot text.
 */
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { formatNumber } from '../../engine/cformat';
import { pow } from '../libm/pow';
import type { DumpCfgFactory, DumpCfgSnapshot } from '../../engine/output/dump';

/** dump_cfg.cpp: static constexpr double UNWRAPEXPAND = 10.0. */
const UNWRAPEXPAND = 10.0;

/**
 * dump_cfg.cpp constructor: auxname[i] for the attributes after the first five, 'X_ID[m]' (X = c, f, v, d, i) written as
 * 'X_ID_m' (ArgInfo with COMPUTE|FIX|VARIABLE|DNAME|INAME: a one-index reference, also the d2_/i2_ forms).
 */
const auxName = (arg: string): string => {
  const m = /^([cfvdi])2?_([^[]*)\[(-?\d+)\]$/.exec(arg);
  if (!m) return arg;
  return `${m[1]}_${m[2]}_${Number(m[3])}`;
};

/** The peridynamic scale factor (fix_peri_neigh.cpp init: 1.44 / pow(vave, 1/3), vave the mean vfrac; else 1). */
const periScale = (sys: System): number => {
  const s = sys.state;
  if (!s.vfrac) return 1.0;
  // fix PERI_NEIGH is created by the peri pair styles' init_style; without one atom->pdscale stays 1.0 (atom.cpp)
  const pair = sys.ff.pair as { name?: string; subs?: { name: string }[] } | null;
  const isPeri = (n?: string) => !!n && n.startsWith('peri/');
  if (!pair || !(isPeri(pair.name) || (pair.subs ?? []).some((x) => isPeri(x.name)))) return 1.0;
  let vone = 0.0;
  for (let i = 0; i < s.n; i++) vone += s.vfrac[i];
  let vave = vone;
  if (s.n) vave /= s.n;
  return vave > 0.0 ? 1.44 / pow(vave, 1.0 / 3.0) : 1.0;
};

/** What dump cfg/uef reads of its fix nvt/uef or npt/uef (engine-gpl/uef/fix_nh_uef.ts FixNHUef get_box, get_rot). */
interface UefFrame {
  getBox(out: number[][]): void;
  getRot(out: number[][]): void;
}

/** dump_cfg_uef.cpp init_style: the first fix whose style matches "^n[vp]t/uef". */
const findUefFix = (sys: System): UefFrame | null =>
  (sys.fixes.find((f) => /^n[vp]t\/uef/.test(f.style)) as unknown as UefFrame | undefined) ?? null;

export const createDumpCfg: DumpCfgFactory = (sys, earg) => makeDumpCfg(sys, earg, false);
export const createDumpCfgUef: DumpCfgFactory = (sys, earg) => makeDumpCfg(sys, earg, true);

const makeDumpCfg = (sys: System, earg: string[], uef: boolean) => {
  const nfield = earg.length;
  if (nfield < 5 || earg[0] !== 'mass' || earg[1] !== 'type' || (earg[2] !== 'xs' && earg[2] !== 'xsu')
    || (earg[3] !== 'ys' && earg[3] !== 'ysu') || (earg[4] !== 'zs' && earg[4] !== 'zsu')) {
    throw new StyleError("Dump cfg arguments must start with 'mass type xs ys zs' or 'mass type xsu ysu zsu'");
  }
  let unwrapflag: 0 | 1;
  if (earg[2] === 'xs') {
    if (earg[3] === 'ysu' || earg[4] === 'zsu') throw new StyleError('Dump cfg arguments can not mix xs|ys|zs with xsu|ysu|zsu');
    unwrapflag = 0;
  } else {
    if (earg[3] === 'ys' || earg[4] === 'zs') throw new StyleError('Dump cfg arguments can not mix xs|ys|zs with xsu|ysu|zsu');
    unwrapflag = 1;
  }
  const auxname = earg.slice(5).map(auxName);
  // atom->pdscale: set once, at the first init of fix PERI_NEIGH; here at the first snapshot
  let pdscale: number | null = null;
  // dump cfg/uef: the fix found at init
  let fixUef: UefFrame | null = null;

  const g = (v: number) => formatNumber(v, '%g');
  const writeHeader = (n: number, colnames: Map<number, string>): string => {
    let scale = 1.0;
    if (sys.state.vfrac) scale = (pdscale ??= periScale(sys));
    else if (unwrapflag === 1) scale = UNWRAPEXPAND;
    const geom = sys.geom;
    const [xy, xz, yz] = sys.state.box.tilt;
    let header = `Number of particles = ${n}\n`;
    header += `A = ${g(scale)} Angstrom (basic length-scale)\n`;
    if (uef) {
      // dump_cfg_uef.cpp write_header: the box in the flow frame. rot goes from the lab frame to the upper triangular
      // frame and its transpose takes the simulation box to the flow frame; in box[][] columns are cell edges, in H0 rows
      const box = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], rot = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      const fix = fixUef ?? findUefFix(sys);
      if (!fix) throw new StyleError("Can't use dump cfg/uef without defining a fix nvt/npt/uef");
      fix.getBox(box);
      fix.getRot(rot);
      for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) { const t = rot[i][j]; rot[i][j] = rot[j][i]; rot[j][i] = t; }
      // UEF_utils::mul_m2(rot, box): box = rot box
      const t = box.map((row) => [...row]);
      for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) box[k][j] = rot[k][0] * t[0][j] + rot[k][1] * t[1][j] + rot[k][2] * t[2][j];
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) header += `H0(${r + 1},${c + 1}) = ${g(box[c][r])} A\n`;
    } else {
      header += `H0(1,1) = ${g(geom.lx)} A\n`;
      header += 'H0(1,2) = 0 A\n';
      header += 'H0(1,3) = 0 A\n';
      header += `H0(2,1) = ${g(xy)} A\n`;
      header += `H0(2,2) = ${g(geom.ly)} A\n`;
      header += 'H0(2,3) = 0 A\n';
      header += `H0(3,1) = ${g(xz)} A\n`;
      header += `H0(3,2) = ${g(yz)} A\n`;
      header += `H0(3,3) = ${g(geom.lz)} A\n`;
    }
    header += '.NO_VELOCITY.\n';
    header += `entry_count = ${nfield - 2}\n`;
    for (let i = 5; i < nfield; i++) header += `auxiliary[${i - 5}] = ${colnames.get(i) ?? auxname[i - 5]}\n`;
    return header;
  };

  return {
    init(): void {
      if (!uef) return;
      // dump_cfg_uef.cpp init_style: after DumpCFG's checks, a fix nvt/uef or npt/uef must exist
      fixUef = findUefFix(sys);
      if (!fixUef) throw new StyleError("Can't use dump cfg/uef without defining a fix nvt/npt/uef");
    },
    text(snap: DumpCfgSnapshot): string {
      const s = sys.state;
      const out: string[] = [];
      if (snap.header) out.push(writeHeader(snap.rows.length, snap.colnames));
      const typename = (i: number) => (snap.element.length ? snap.element[s.type[i] - 1] : 'C');
      const { values, formatters } = snap;
      // write_lines: mass with "%f \n", the type name with "%s \n", then the other columns with their formats
      // (vformat: a blank after each but the last), unwrapped coordinates mapped back into the expanded cell
      for (let r = 0; r < snap.rows.length; r++) {
        const i = snap.rows[r];
        let line = `${formatNumber(values[0][i], '%f')} \n${typename(i)} \n`;
        for (let j = 2; j < nfield; j++) {
          const v = values[j][i];
          const text = earg[j] === 'element' ? typename(i) : formatters[j](unwrapflag === 1 && j <= 4 ? (v - 0.5) / UNWRAPEXPAND + 0.5 : v);
          line += j < nfield - 1 ? `${text} ` : text;
        }
        out.push(`${line}\n`);
      }
      return out.join('');
    },
  };
};
