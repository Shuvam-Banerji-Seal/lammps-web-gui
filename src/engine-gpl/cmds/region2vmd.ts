/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/GRAPHICS/region2vmd.cpp and region2vmd.h (Axel Kohlmeyer, Temple U). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * region2vmd: writes a VMD Tcl script that draws regions with VMD graphics primitives (docs.lammps.org/region2vmd.html).
 * Translated: the keywords color, material, command and region; the block, cylinder, ellipsoid, prism and sphere
 * styles with their open faces; the move displacement of block, cylinder and ellipsoid (native applies none to sphere
 * and prism). Refused with a StyleError naming the region: rotating regions (native skips them with "Cannot (yet)
 * handle rotating region"), cone (native draws it; not ported here), and the other styles native skips (plane,
 * union, intersect). Nothing is written when the command fails.
 */
import type { Handler } from '../../engine/commands/args';
import { StyleError } from '../../engine/force/types';
import { formatNumber } from '../../engine/cformat';
import { BlockRegion, ConeRegion, EllipsoidRegion, PrismRegion, SphereRegion, type Region } from '../../engine/region';

/** region2vmd.cpp: constexpr int RESOLUTION = 20; constexpr double DELTA = 1.0e-5. */
const RESOLUTION = 20;
const DELTA = 1.0e-5;

/** region2vmd.cpp vmdcolors. */
const VMD_COLORS = new Set([
  'blue', 'red', 'gray', 'orange', 'yellow', 'tan', 'silver', 'green', 'white',
  'pink', 'cyan', 'purple', 'lime', 'mauve', 'ochre', 'iceblue', 'black', 'yellow2',
  'yellow3', 'green2', 'green3', 'cyan2', 'cyan3', 'blue2', 'blue3', 'violet', 'violet2',
  'magenta', 'magenta2', 'red2', 'red3', 'orange2', 'orange3',
]);

/** region2vmd.cpp vmdmaterials. */
const VMD_MATERIALS = new Set([
  'Opaque', 'Transparent', 'BrushedMetal', 'Diffuse', 'Ghost', 'Glass1',
  'Glass2', 'Glass3', 'Glossy', 'HardPlastic', 'MetallicPastel', 'Steel',
  'Translucent', 'Edgy', 'EdgyShiny', 'EdgyGlass', 'Goodsell', 'AOShiny',
  'AOChalky', 'AOEdgy', 'BlownGlass', 'GlassBubble', 'RTChrome',
]);

/**
 * The "{}" conversion of a double in fmt (the formatter of utils::print, src/fmt/format-inl.h and format.h at
 * b332f28): the shortest round-trip digits, in fixed notation when the decimal exponent is in [-4, 16) and in
 * d.ddde+XX (at least two exponent digits) otherwise. "1.0" prints as "1" and 1e20 as "1e+20".
 */
const fmtDefault = (x: number): string => {
  if (Number.isNaN(x)) return 'nan';
  if (!Number.isFinite(x)) return x > 0 ? 'inf' : '-inf';
  if (x === 0) return Object.is(x, -0) ? '-0' : '0';
  const [mant, expText] = Math.abs(x).toExponential().split('e');
  const digits = mant.replace('.', '');
  const e = Number(expText);
  let body: string;
  if (e < -4 || e >= 16) {
    const ae = Math.abs(e);
    const m = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    body = `${m}e${e < 0 ? '-' : '+'}${ae < 10 ? '0' : ''}${ae}`;
  } else if (e >= 0) {
    const intLen = e + 1;
    body = digits.length <= intLen
      ? digits + '0'.repeat(intLen - digits.length)
      : `${digits.slice(0, intLen)}.${digits.slice(intLen)}`;
  } else {
    body = `0.${'0'.repeat(-e - 1)}${digits}`;
  }
  return x < 0 ? `-${body}` : body;
};

/** The "{:.15}" conversion of region2vmd.cpp (lid + DELTA): C "%.15g". */
const g15 = (x: number): string => formatNumber(x, '%.15g');

/** Open-face flags of a region: native open_faces[k] is set when the region command had "open k+1". */
const openFlags = (ids: number[], n: number): boolean[] => Array.from({ length: n }, (_, k) => ids.includes(k + 1));

/**
 * Prism open faces as native stores them. RegPrism::shape_update (src/region_prism.cpp at b332f28, run once by the
 * constructor) remaps the user numbering: "open_faces[0] = temp[4]; open_faces[1] = temp[5]; open_faces[2] = temp[2];
 * open_faces[3] = temp[3]; open_faces[4] = temp[0]; open_faces[5] = temp[1];".
 */
const PRISM_OPEN_PERM = [4, 5, 2, 3, 0, 1];

/** region2vmd.cpp: the draw ellipsoid Tcl function, written once before the first ellipsoid region. */
const DRAW_ELLIPSOID_FUNCTION =
    "\n# VMD script code to emulate ellipsoids with trinorm graphics objects\n" +
    "proc vmd_draw_ellipsoid {mol level {center {0.0 0.0 0.0}} {radius {1.0 1.0 1.0}}} {\n" +
    "   set orient {1.0 0.0 0.0 0.0}\n" +
    "   set gid {}\n\n" +
    "   # vertices of an octahedron inscribed\n" +
    "   # in a sphere with radius 1.0.\n" +
    "   set vec1 {-1.0  0.0  0.0}\n" +
    "   set vec2 { 1.0  0.0  0.0}\n" +
    "   set vec3 { 0.0 -1.0  0.0}\n" +
    "   set vec4 { 0.0  1.0  0.0}\n" +
    "   set vec5 { 0.0  0.0 -1.0}\n" +
    "   set vec6 { 0.0  0.0  1.0}\n\n" +
    "   # build list of triangles representing\n" +
    "   # the octahedron. the points of the \n" +
    "   # triangles have to be given clockwise from\n" +
    "   # looking at the surface from the outside.\n" +
    "   set trilist [list \\\n" +
    "                [list $vec1 $vec4 $vec5] \\\n" +
    "                [list $vec5 $vec4 $vec2] \\\n" +
    "                [list $vec2 $vec4 $vec6] \\\n" +
    "                [list $vec6 $vec4 $vec1] \\\n" +
    "                [list $vec5 $vec3 $vec1] \\\n" +
    "                [list $vec2 $vec3 $vec5] \\\n" +
    "                [list $vec6 $vec3 $vec2] \\\n" +
    "                [list $vec1 $vec3 $vec6] ]\n\n" +
    "   # refinement iterations to approximate a sphere:\n" +
    "   # each triangle is split into 4 subtriangles\n" +
    "   #        p2\n" +
    "   #        /\\\n" +
    "   #       /  \\\n" +
    "   #    pb/____\\pc\n" +
    "   #     /\\    /\\\n" +
    "   #    /  \\  /  \\\n" +
    "   #   /____\\/____\\\n" +
    "   # p1     pa    p3\n\n" +
    "   # we construct vectors to the three midpoints and rescale\n" +
    "   # them to unit length. then build new triangles enumerating\n" +
    "   # the points in clockwise order again.\n" +
    "   for {set i 0} {$i < $level} {incr i} {\n" +
    "       set newlist {}\n" +
    "       foreach tri $trilist {\n" +
    "           foreach {p1 p2 p3} $tri {}\n" +
    "           set pa [vecnorm [vecadd $p1 $p3]]\n" +
    "           set pb [vecnorm [vecadd $p1 $p2]]\n" +
    "           set pc [vecnorm [vecadd $p2 $p3]]\n" +
    "           lappend newlist [list $p1 $pb $pa]\n" +
    "           lappend newlist [list $pb $p2 $pc]\n" +
    "           lappend newlist [list $pa $pb $pc]\n" +
    "           lappend newlist [list $pa $pc $p3]\n" +
    "       }\n" +
    "       set trilist $newlist\n" +
    "   }\n\n" +
    "   # compute vertex scaling factor to deform a sphere to an ellipsoid\n" +
    "   proc radscale {radius vector} {\n" +
    "       foreach {a b c} $radius {}\n" +
    "       foreach {x y z} $vector {}\n" +
    "       return [expr {sqrt(1.0/($x/$a*$x/$a+$y/$b*$y/$b+$z/$c*$z/$c))}]\n" +
    "   }\n\n" +
    "   # convert quaternion to rotation matrix\n" +
    "   proc quattorot {quat} {\n" +
    "       foreach {w x y z} $quat {}\n" +
    "       set norm [expr {$w*$w + $x*$x + $y*$y +$z*$z}]\n" +
    "       set sc 0.0\n" +
    "       if {$norm > 0.0} { set s [expr {2.0/$norm}] }\n" +
    "       set X [expr {$x*$s}]\n" +
    "       set Y [expr {$y*$s}]\n" +
    "       set Z [expr {$z*$s}]\n" +
    "       return [list \\\n" +
    "           [list [expr {1.0-($y*$y*$s + $z*$z*$s)}] \\\n" +
    "                 [expr {$x*$y*$s - $w*$z*$s}] \\\n" +
    "                 [expr {$x*$z*$s + $w*$y*$s}] ] \\\n" +
    "           [list [expr {$x*$y*$s + $w*$z*$s}] \\\n" +
    "                 [expr {1.0-($x*$x*$s + $z*$z*$s)}] \\\n" +
    "                 [expr {$y*$z*$s - $w*$x*$s}] ] \\\n" +
    "           [list [expr {$x*$z*$s - $w*$y*$s}] \\\n" +
    "                 [expr {$y*$z*$s + $w*$x*$s}] \\\n" +
    "                 [expr {1.0-($x*$x*$s + $y*$y*$s)}] ] ]\n" +
    "   }\n\n" +
    "   # apply rotation matrix to vector\n" +
    "   proc rotvec {mat vec} {\n" +
    "       set new {}\n" +
    "       foreach c $mat {\n" +
    "           lappend new [vecdot $c $vec]\n" +
    "       }\n" +
    "       return $new\n" +
    "   }\n\n" +
    "   foreach tri $trilist {\n" +
    "       foreach {vec1 vec2 vec3} $tri {}\n" +
    "       # rescale to desired radius\n" +
    "       set rad1 [radscale $radius $vec1]\n" +
    "       set rad2 [radscale $radius $vec2]\n" +
    "       set rad3 [radscale $radius $vec3]\n" +
    "       # get and apply rotation matrix\n" +
    "       set mat [quattorot $orient]\n" +
    "       set vec1 [rotvec $mat $vec1]\n" +
    "       set vec2 [rotvec $mat $vec2]\n" +
    "       set vec3 [rotvec $mat $vec3]\n" +
    "       # deform sphereoid and translate\n" +
    "       set pos1 [vecadd [vecscale $rad1 $vec1] $center]\n" +
    "       set pos2 [vecadd [vecscale $rad2 $vec2] $center]\n" +
    "       set pos3 [vecadd [vecscale $rad3 $vec3] $center]\n" +
    "       # since the original vectors to the vertices are those of\n" +
    "       # a unit sphere, we can use them directly as surface normals.\n" +
    "       lappend gid [graphics $mol trinorm $pos1 $pos2 $pos3 $vec1 $vec2 $vec3]\n" +
    "   }\n" +
    "   return $gid\n" +
    "}\n\n";

/**
 * Region2VMD::write_region (region2vmd.cpp at b332f28): the VMD graphics lines of one region. Native applies the
 * displacement dx, dy, dz of a moving region to block, cylinder and ellipsoid, and writes sphere and prism without it.
 */
const regionText = (reg: Region): string => {
  if (reg.rotate) throw new StyleError(`Cannot (yet) handle rotating region ${reg.id}`);
  // native prematch(): dx, dy, dz are the move displacement of the region (zero when it has no move keyword)
  const [dx, dy, dz] = reg.transformState().d;
  const F = fmtDefault;

  if (reg instanceof BlockRegion) {
    // native: a block is 12 triangles; a triangle is commented out when its side is open
    const [xlo, xhi, ylo, yhi, zlo, zhi] = reg.bounds();
    const s = [xlo + dx, xhi + dx, ylo + dy, yhi + dy, zlo + dz, zhi + dz].map(F);
    const o = openFlags(reg.openFaces, 6).map((v) => (v ? '# ' : ''));
    const V = (i: number, j: number, k: number) => `{${s[i]} ${s[j]} ${s[k]}}`;
    const T = [
      [V(0, 2, 4), V(0, 2, 5), V(0, 3, 4)], [V(0, 3, 4), V(0, 3, 5), V(0, 2, 5)],
      [V(0, 2, 4), V(0, 2, 5), V(1, 2, 4)], [V(1, 2, 4), V(1, 2, 5), V(0, 2, 5)],
      [V(1, 2, 4), V(1, 2, 5), V(1, 3, 4)], [V(1, 3, 4), V(1, 3, 5), V(1, 2, 5)],
      [V(0, 2, 4), V(1, 2, 4), V(1, 3, 4)], [V(0, 2, 4), V(0, 3, 4), V(1, 3, 4)],
      [V(0, 2, 5), V(1, 2, 5), V(1, 3, 5)], [V(0, 2, 5), V(0, 3, 5), V(1, 3, 5)],
      [V(0, 3, 4), V(0, 3, 5), V(1, 3, 5)], [V(0, 3, 4), V(1, 3, 4), V(1, 3, 5)],
    ];
    // the open-face prefix of each triangle: {6}..{11} are open_faces[0..5]
    const face = [0, 0, 2, 2, 1, 1, 4, 4, 5, 5, 3, 3];
    return T.map((t, k) => `${o[face[k]]}draw triangle ${t[0]} ${t[1]} ${t[2]}\n`).join('');
  }

  if (reg instanceof ConeRegion && reg.style === 'cylinder') {
    const geo = reg.geometry();
    const ax = geo.axis;
    const open = openFlags(reg.openFaces, 3);
    for (const k of reg.openFaces) {
      if (k > 3) throw new StyleError(`Illegal region cylinder open face: ${k}`);
    }
    const r = geo.radlo; // native cyl->radius: a cylinder has equal radii
    // native: the centre coordinates of the two axes perpendicular to the cylinder axis, displaced as native does
    const d1 = ax === 0 ? 1 : 0;
    const d2 = ax === 2 ? 1 : 2;
    const dd = [dx, dy, dz];
    const c1 = geo.c1 + dd[d1];
    const c2 = geo.c2 + dd[d2];
    // the two end points along the axis, in the coordinate order of the axis
    const pair = (p0: string, p1: string, c1s: string, c2s: string) => (
      ax === 0 ? `{${p0} ${c1s} ${c2s}} {${p1} ${c1s} ${c2s}}`
        : ax === 1 ? `{${c1s} ${p0} ${c2s}} {${c1s} ${p1} ${c2s}}`
          : `{${c1s} ${c2s} ${p0}} {${c1s} ${c2s} ${p1}}`
    );
    const filled = !open[0] && !open[1] && !open[2] ? 'filled yes' : 'filled no';
    let t = '';
    // the cylinder uses a single primitive, drawn only when the lateral face is closed
    if (!open[2]) {
      t += `draw cylinder ${pair(F(geo.lo + dd[ax]), F(geo.hi + dd[ax]), F(c1), F(c2))} radius ${F(r)} resolution ${RESOLUTION} ${filled}\n`;
    }
    // lids: drawn as thin filled cylinders when the cylinder is not filled
    if (filled === 'filled no' && !open[0]) {
      const lid = geo.lo + dd[ax];
      t += `draw cylinder ${pair(F(lid), g15(lid + DELTA), F(c1), F(c2))} radius ${F(r)} resolution ${RESOLUTION} filled yes\n`;
    }
    if (filled === 'filled no' && !open[1]) {
      const lid = geo.hi + dd[ax];
      t += `draw cylinder ${pair(F(lid), g15(lid + DELTA), F(c1), F(c2))} radius ${F(r)} resolution ${RESOLUTION} filled yes\n`;
    }
    return t;
  }

  if (reg instanceof EllipsoidRegion) {
    // native: a custom VMD function (DRAW_ELLIPSOID_FUNCTION) emulates the ellipsoid with trinorm primitives
    const [xc, yc, zc] = reg.center();
    const [a, b, c] = reg.axes();
    return `draw ellipsoid 3 {${F(xc + dx)} ${F(yc + dy)} ${F(zc + dz)}} {${F(a)} ${F(b)} ${F(c)}}\n`;
  }

  if (reg instanceof PrismRegion) {
    // native: a prism is 12 triangles; {14}..{19} are the open-face prefixes after the native remap
    const [xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz] = reg.values();
    const s = [
      xlo, xhi, ylo, yhi, zlo, zhi,
      xlo + xy, xhi + xy, xhi + xz, ylo + yz, xlo + xz, xhi + xy + xz, yhi + yz, xlo + xy + xz,
    ].map(F);
    const user = openFlags(reg.openFaces, 6);
    const o = PRISM_OPEN_PERM.map((k) => (user[k] ? '# ' : ''));
    const V = (i: number, j: number, k: number) => `{${s[i]} ${s[j]} ${s[k]}}`;
    const T = [
      [V(0, 2, 4), V(1, 2, 4), V(7, 3, 4)], [V(0, 2, 4), V(6, 3, 4), V(7, 3, 4)],
      [V(0, 2, 4), V(1, 2, 4), V(8, 9, 5)], [V(0, 2, 4), V(10, 9, 5), V(8, 9, 5)],
      [V(1, 2, 4), V(8, 9, 5), V(7, 3, 4)], [V(11, 12, 5), V(8, 9, 5), V(7, 3, 4)],
      [V(0, 2, 4), V(6, 3, 4), V(13, 12, 5)], [V(0, 2, 4), V(10, 9, 5), V(13, 12, 5)],
      [V(10, 9, 5), V(8, 9, 5), V(11, 12, 5)], [V(10, 9, 5), V(13, 12, 5), V(11, 12, 5)],
      [V(6, 3, 4), V(7, 3, 4), V(11, 12, 5)], [V(6, 3, 4), V(13, 12, 5), V(11, 12, 5)],
    ];
    // prefixes {14}, {14}, {16}, {16}, {19}, {19}, {18}, {18}, {15}, {15}, {17}, {17}
    const face = [0, 0, 2, 2, 5, 5, 4, 4, 1, 1, 3, 3];
    return T.map((t, k) => `${o[face[k]]}draw triangle ${t[0]} ${t[1]} ${t[2]}\n`).join('');
  }

  if (reg instanceof SphereRegion) {
    // native: one sphere primitive; the displacement is not applied (region2vmd.cpp, sphere branch)
    const [xc, yc, zc] = reg.center();
    return `draw sphere {${F(xc)} ${F(yc)} ${F(zc)}} radius ${F(reg.radius())} resolution ${RESOLUTION}\n`;
  }

  if (reg.style === 'cone') {
    throw new StyleError(`region2vmd cannot translate region ${reg.id} of style cone: this browser engine does not port the cone graphics`);
  }
  throw new StyleError(`Cannot (yet) translate region ${reg.id} of style ${reg.style} to VMD graphics`);
};

/**
 * region2vmd command (docs.lammps.org/region2vmd.html): "region2vmd file keyword values ...", the keywords color,
 * material, command and region (repeatable). Native writes the file as it goes; this port builds the script and
 * writes it at the end, so a failing command leaves no file.
 */
export const region2vmd: Handler = ({ sys }, a) => {
  if (a.length < 3) throw new StyleError('Illegal region2vmd command: missing argument(s)');
  const file = a[0];
  const out: string[] = ['# save old top molecule index\nset oldtop [molinfo top]\n'];
  sys.log(`Writing region visualizations to VMD Tcl script file ${file}:`);

  let color = 'silver';
  let material = 'Transparent';
  let defEllipsoidFunc = false;

  const n = a.length;
  let i = 1;
  while (i < n) {
    if (i + 2 > n) throw new StyleError('Illegal region2vmd command: missing argument(s)');
    const kw = a[i];
    i++;
    const val = a[i];
    if (kw === 'color') {
      if (!VMD_COLORS.has(val)) throw new StyleError(`Color ${val} is not a known VMD color`);
      color = val;
    } else if (kw === 'material') {
      if (!VMD_MATERIALS.has(val)) throw new StyleError(`Material ${val} is not a known VMD material`);
      material = val;
    } else if (kw === 'command') {
      out.push(`\n# custom command\n${val}\n`);
    } else if (kw === 'region') {
      const reg = sys.regions.get(val);
      if (!reg) throw new StyleError(`Region ${val} does not exist`);
      if (!defEllipsoidFunc && reg.style === 'ellipsoid') {
        out.push(DRAW_ELLIPSOID_FUNCTION);
        defEllipsoidFunc = true;
      }
      sys.log(` writing region ${reg.id} ...`);
      out.push(`\n# region ${reg.id} of style ${reg.style}\n`);
      out.push('# create new empty VMD molecule to store the graphics primitives\nset gfxmol [mol new]\nmol top $gfxmol\n');
      out.push(`mol rename $gfxmol {LAMMPS region ${reg.id}}\n`);
      out.push('# set color and material\n');
      out.push(`graphics $gfxmol color ${color}\n`);
      out.push(`graphics $gfxmol material ${material}\n`);
      out.push(regionText(reg));
      sys.log(' done');
    } else {
      throw new StyleError(`Unknown region2vmd keyword ${kw}`);
    }
    i++;
  }

  // reset views and restore previous top molecule
  out.push('after idle {if {$oldtop >= 0} {mol top $oldtop}; display resetview}\n');
  sys.writeFile(file, out.join(''), false);
};
