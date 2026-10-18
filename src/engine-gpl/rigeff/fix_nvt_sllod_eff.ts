/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EFF/fix_nvt_sllod_eff.cpp and
 * fix_nvt_sllod_eff.h (class FixNVTSllodEff, fix nvt/sllod/eff): the SLLOD thermostat of an electron force field
 * system, a Nose-Hoover chain (FixNHEff) on the thermal velocities with the streaming velocity of the box deformation
 * (fix deform, remap v) removed and added back. Keywords psllod, peculiar, kick and integrator as in the source
 * ("reversible" is the default integrator, "legacy" the older half-step scaling). Contributing author (from the
 * source): Andres Jaramillo-Botero (Caltech). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the
 * GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The Nose-Hoover base (fix_nh.cpp via FixNHBase) and the EFF electron terms (FixNHEff) are the engine's own ports
 * (src/engine-gpl/eff/fix_nh.ts, fix_nh_eff.ts), reused here by import. The temperature is the wrapped compute
 * temp/deform/eff (ComputeTempDeformEff), or temp/eff with peculiar yes.
 */
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, ELECTRON_RFORCE, ELECTRON_RVEL, ELECTRON_RADIUS, ELECTRON_SPIN } from '../../engine/atoms';
import { ownCompute, removeCompute } from '../../engine/fix/util';
import { FixNHEff } from '../eff/fix_nh_eff';
import { ComputeTempDeformEff, boxH, deformRates, multiplyShapeShape } from './compute_temp_deform_eff';
import { exp as libmExp } from '../libm/exp';

/** logical keyword value: yes / no (utils::logical). */
const logical = (w: string | undefined, what: string): boolean => {
  if (w === 'yes') return true;
  if (w === 'no') return false;
  throw new StyleError(`Expected yes or no for ${what}, got '${w ?? ''}'`);
};

const OWN_KEYWORDS = new Set(['psllod', 'peculiar', 'kick', 'integrator']);

export class FixNVTSllodEff extends FixNHEff {
  private readonly ownTempId: string;
  private psllod = false;
  private peculiar = false;
  private legacy = false;
  private kick = false;
  private nondeformbias = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    // the sllod keywords are read here, the base fix_nh keywords go to the Nose-Hoover base; default tchain is 1 (mtchain_default_flag)
    const base: string[] = [];
    const own: string[] = [];
    let userKick = false, hasTchain = false;
    for (let k = 0; k < args.length;) {
      const w = args[k];
      if (OWN_KEYWORDS.has(w)) {
        if (k + 1 >= args.length) throw new StyleError(`Illegal fix nvt/sllod/eff command: missing argument for ${w}`);
        own.push(w, args[k + 1]);
        if (w === 'kick') userKick = true;
        k += 2;
      } else {
        if (w === 'tchain') hasTchain = true;
        base.push(w);
        k++;
      }
    }
    if (!hasTchain) base.push('tchain', '1');
    const tempId = `${id}_temp`;
    super(sys, id, group, base, tempId, 'nvt/sllod/eff');
    this.ownTempId = tempId;
    if (!this.tstat) throw new StyleError('Temperature control must be used with fix nvt/sllod/eff');
    if (this.pstat) throw new StyleError('Pressure control can not be used with fix nvt/sllod/eff');
    this.psllod = false;
    this.peculiar = false;
    this.legacy = false;
    for (let k = 0; k < own.length; k += 2) {
      const w = own[k], v = own[k + 1];
      if (w === 'psllod') this.psllod = logical(v, 'psllod');
      else if (w === 'peculiar') this.peculiar = logical(v, `fix nvt/sllod/eff peculiar`);
      else if (w === 'kick') this.kick = logical(v, 'kick');
      else if (w === 'integrator') {
        if (v === 'reversible') this.legacy = false;
        else if (v === 'legacy') this.legacy = true;
        else throw new StyleError(`Unknown fix nvt/sllod/eff integrator argument: ${v}`);
      }
    }
    if (!userKick) this.kick = !this.peculiar;
    if (this.legacy && this.peculiar) throw new StyleError('fix nvt/sllod/eff legacy integrator is not compatible with peculiar=yes');
    ownCompute(sys, tempId, group, this.peculiar ? 'temp/eff' : 'temp/deform/eff', []);
  }

  override destroy(): void {
    removeCompute(this.sys, this.ownTempId);
  }

  /** the wrapped temperature compute as the deform compute, when it is one (dynamic_cast in the source) */
  private deformTemp(): ComputeTempDeformEff | null {
    const t = this.tempCompute();
    return t instanceof ComputeTempDeformEff ? t : null;
  }

  /** fix_nvt_sllod_eff.cpp init */
  override init(): void {
    super.init();
    const t = this.tempCompute();
    if (!this.peculiar && !t.hasBias()) throw new StyleError('Temperature for fix nvt/sllod/eff does not have a bias');
    this.nondeformbias = false;
    if (!/^temp\/deform\/eff/.test(t.style)) {
      if (this.legacy) {
        this.nondeformbias = true;
        if (this.kick) throw new StyleError('fix nvt/sllod/eff with peculiar=no and kick=yes requires temperature bias to be calculated by compute temp/deform/eff');
      } else if (!this.peculiar) {
        throw new StyleError('Fix nvt/sllod/eff used with lab-frame velocity and non-deform temperature bias. For non-deform biases, either set peculiar = yes or pass an explicit temp/deform/eff with an extra bias');
      }
    }
    const deforms = this.sys.fixes.filter((f) => f.style === 'deform');
    if (deforms.length === 0) throw new StyleError('Using fix nvt/sllod/eff with no fix deform defined');
    for (const f of deforms) {
      const df = f as unknown as { remapFlag(): string; init?(): void; style: string };
      const remap = df.remapFlag();
      if ((this.peculiar && remap !== 'none') || (!this.peculiar && remap !== 'v')) {
        throw new StyleError(`Using fix nvt/sllod/eff with inconsistent fix ${df.style} remap option`);
      }
      if (this.kick) {
        if (!this.peculiar) {
          df.init?.();
          const dt = this.deformTemp();
          if (dt) dt.applyDeformBiasAll(0.0);
          this.kick = false;
        }
      }
      break;
    }
  }

  /** fix_nvt_sllod_eff.cpp nve_x: the SLLOD position and velocity update (reversible, the default) or the base drift (legacy). */
  protected override nveX(): void {
    if (this.legacy) { super.nveX(); return; }
    const s = this.sys.state;
    const { x, v, mask } = s;
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    const eradius = s.custom.get(ELECTRON_RADIUS)!.data;
    const ervel = s.custom.get(ELECTRON_RVEL)!.data;
    const erforce = s.custom.get(ELECTRON_RFORCE)!.data;
    const { hInv } = boxH(this.sys);
    const gu = multiplyShapeShape(deformRates(this.sys).hRate, hInv);
    const dthalf = this.dthalf, dtv = this.dtv;
    const xfac = [libmExp(gu[0] * dthalf), libmExp(gu[1] * dthalf), libmExp(gu[2] * dthalf)];
    const vfac = [libmExp(-gu[0] * dthalf), libmExp(-gu[1] * dthalf), libmExp(-gu[2] * dthalf)];
    const dt = this.deformTemp();
    if (!this.peculiar && dt) dt.removeDeformBiasAll();
    const box = s.box;
    const xmid = [0, 1, 2].map((d) => (box.hi[d] + box.lo[d]) / 2.0);
    const xlo = box.lo;
    const ylo2 = xmid[1] + (xlo[1] - xmid[1]) * xfac[1] * xfac[1];
    const zlo2 = xmid[2] + (xlo[2] - xmid[2]) * xfac[2] * xfac[2];
    const mefactor = this.sys.dimension / 4.0;
    const g0 = gu[0], g1 = gu[1], g2 = gu[2], g3 = gu[3], g4 = gu[4], g5 = gu[5];
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const dtfm = this.dtf / massOf(s, i);
      if (this.psllod) {
        v[3 * i + 2] -= dthalf * g2 * g2 * x[3 * i + 2];
        v[3 * i + 1] -= dthalf * g3 * v[3 * i + 2] + dthalf * g1 * g1 * x[3 * i + 1];
        v[3 * i] -= dthalf * (g5 * v[3 * i + 1] + g4 * v[3 * i + 2]) + dthalf * g0 * g0 * x[3 * i];
      } else {
        v[3 * i + 1] -= dthalf * g3 * v[3 * i + 2];
        v[3 * i] -= dthalf * (g5 * v[3 * i + 1] + g4 * v[3 * i + 2]);
      }
      v[3 * i] *= vfac[0];
      v[3 * i + 1] *= vfac[1];
      v[3 * i + 2] *= vfac[2];
      x[3 * i + 1] += dthalf * g3 * (x[3 * i + 2] - xlo[2]);
      x[3 * i] += dthalf * (g5 * (x[3 * i + 1] - xlo[1]) + g4 * (x[3 * i + 2] - xlo[2]));
      x[3 * i] = xmid[0] + (x[3 * i] - xmid[0]) * xfac[0];
      x[3 * i + 1] = xmid[1] + (x[3 * i + 1] - xmid[1]) * xfac[1];
      x[3 * i + 2] = xmid[2] + (x[3 * i + 2] - xmid[2]) * xfac[2];
      x[3 * i] += dtv * v[3 * i];
      x[3 * i + 1] += dtv * v[3 * i + 1];
      x[3 * i + 2] += dtv * v[3 * i + 2];
      if (Math.abs(spin[i]) === 1) {
        ervel[i] += dtfm * erforce[i] / mefactor - dtv * Math.sqrt(g0 * g0 + g1 * g1 + g2 * g2);
        eradius[i] += dtv * ervel[i];
      }
      x[3 * i] = xmid[0] + (x[3 * i] - xmid[0]) * xfac[0];
      x[3 * i + 1] = xmid[1] + (x[3 * i + 1] - xmid[1]) * xfac[1];
      x[3 * i + 2] = xmid[2] + (x[3 * i + 2] - xmid[2]) * xfac[2];
      x[3 * i] += dthalf * (g5 * (x[3 * i + 1] - ylo2) + g4 * (x[3 * i + 2] - zlo2));
      x[3 * i + 1] += dthalf * g3 * (x[3 * i + 2] - zlo2);
      v[3 * i] *= vfac[0];
      v[3 * i + 1] *= vfac[1];
      v[3 * i + 2] *= vfac[2];
      if (this.psllod) {
        v[3 * i] -= dthalf * (g5 * v[3 * i + 1] + g4 * v[3 * i + 2]) + dthalf * g0 * g0 * x[3 * i];
        v[3 * i + 1] -= dthalf * g3 * v[3 * i + 2] + dthalf * g1 * g1 * x[3 * i + 1];
        v[3 * i + 2] -= dthalf * g2 * g2 * x[3 * i + 2];
      } else {
        v[3 * i] -= dthalf * (g5 * v[3 * i + 1] + g4 * v[3 * i + 2]);
        v[3 * i + 1] -= dthalf * g3 * v[3 * i + 2];
      }
    }
    if (!this.peculiar && dt) dt.applyDeformBiasAll(dtv);
  }

  /**
   * fix_nvt_sllod_eff.cpp nh_v_temp: the legacy half-step scaling of the thermal velocities and the streaming term;
   * the reversible integrator uses the Nose-Hoover base and the electron scaling of FixNHEff.
   */
  protected override nhVTemp(): void {
    if (!this.legacy) {
      // fix_nh.cpp nh_v_temp with which == BIAS (the temperature has a bias): the bias is removed from each atom,
      // the thermal velocity is scaled and the bias added back. The electron radial velocities follow (fix_nh_eff.cpp).
      const t = this.tempCompute();
      const s = this.sys.state;
      const { v, mask } = s;
      const ervel = s.custom.get(ELECTRON_RVEL)!.data;
      const spin = s.custom.get(ELECTRON_SPIN)!.data;
      const biased = t.hasBias();
      const dt = this.deformTemp();
      if (biased && !dt) throw new StyleError('fix nvt/sllod/eff: the temperature bias is not a temp/deform/eff bias');
      const vi = [0, 0, 0];
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & this.groupBit)) continue;
        if (biased) {
          vi[0] = v[3 * i]; vi[1] = v[3 * i + 1]; vi[2] = v[3 * i + 2];
          dt!.removeDeformBiasVec(i, vi);
          vi[0] *= this.factorEta; vi[1] *= this.factorEta; vi[2] *= this.factorEta;
          dt!.restoreDeformBiasVec(i, vi);
          v[3 * i] = vi[0]; v[3 * i + 1] = vi[1]; v[3 * i + 2] = vi[2];
        } else {
          v[3 * i] *= this.factorEta;
          v[3 * i + 1] *= this.factorEta;
          v[3 * i + 2] *= this.factorEta;
        }
        if (Math.abs(spin[i]) === 1) ervel[i] *= this.factorEta;
      }
      return;
    }
    const t = this.tempCompute();
    if (this.nondeformbias) { t.invalidate(); t.scalarValue(); }
    const s = this.sys.state;
    const { v, mask } = s;
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    const ervel = s.custom.get(ELECTRON_RVEL)!.data;
    const { hInv } = boxH(this.sys);
    const h2 = multiplyShapeShape(deformRates(this.sys).hRate, hInv);
    const dt = this.deformTemp();
    const removeBias = (i: number, vi: number[]) => { if (dt) dt.removeDeformBiasVec(i, vi); };
    const restoreBias = (i: number, vi: number[]) => { if (dt) dt.restoreDeformBiasVec(i, vi); };
    const vi = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      vi[0] = v[3 * i]; vi[1] = v[3 * i + 1]; vi[2] = v[3 * i + 2];
      if (!this.psllod) removeBias(i, vi);
      const vdelu0 = h2[0] * vi[0] + h2[5] * vi[1] + h2[4] * vi[2];
      const vdelu1 = h2[1] * vi[1] + h2[3] * vi[2];
      const vdelu2 = h2[2] * vi[2];
      if (this.psllod) removeBias(i, vi);
      vi[0] = vi[0] * this.factorEta - this.dthalf * vdelu0;
      vi[1] = vi[1] * this.factorEta - this.dthalf * vdelu1;
      vi[2] = vi[2] * this.factorEta - this.dthalf * vdelu2;
      restoreBias(i, vi);
      v[3 * i] = vi[0]; v[3 * i + 1] = vi[1]; v[3 * i + 2] = vi[2];
      if (Math.abs(spin[i]) === 1) {
        ervel[i] = ervel[i] * this.factorEta - this.dthalf * Math.sqrt(vdelu0 * vdelu0 + vdelu1 * vdelu1 + vdelu2 * vdelu2);
      }
    }
  }
}
