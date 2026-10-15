/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/SMTBQ/pair_smtbq.cpp (PairSMTBQ::read_file, the potential-file reader) and src/SMTBQ/pair_smtbq.h,
 * at commit b332f28; the line reader is the TextFileReader of src/text_file_reader.cpp (comments after '#' are
 * dropped, blank lines are skipped) and the tokenizer separators are the SMTBQ_SEPARATORS of the C++ source
 * ("' \t\n\r": quote marks separate words). Authors credited in the C++ file headers: N. Salles, O. Politano,
 * E. Amzallag, R. Tetot (SMTBQ) and E. Maras (Aalto University). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The file is read as a sequence of lines, one value line after another, as the C++ reader does. Two details of
 * that reader are kept on purpose: the interaction loop reads one more line than the file has interactions
 * (the header of the cutoff section is taken as the label of that failed read, and the line after it is then the
 * cutoff line), and the "Verbose" block reads the "parameters for mode" line as a skipped line.
 */

/** One element (the params array of PairSMTBQ). */
export interface SmtbqParam {
  nom: string;
  sto: number;
  n0: number;
  ne: number;
  chi: number;
  dj: number;
  R: number;
  dzeta: number;
  qform: number;
  masse: number;
}

/** One interaction (the intparams array of PairSMTBQ); index 0 is unused. */
export interface SmtbqInter {
  typepot: string;
  mode: string;
  intsm: number;
  a: number;
  p: number;
  ksi: number;
  q: number;
  r0: number;
  dc1: number;
  dc2: number;
  abuck: number;
  rhobuck: number;
  aOO: number;
  bOO: number;
  r1OO: number;
  r2OO: number;
}

/** Everything read_file sets. */
export interface SmtbqFile {
  ntypes: number;
  params: SmtbqParam[];
  /** intype[i * ntypes + j]: interaction number of the element pair (0 = none). */
  intype: Int32Array;
  maxintparam: number;
  maxintsm: number;
  intparams: SmtbqInter[];
  coordOxBB: number;
  coordOxBulk: number;
  coordOxSurf: number;
  ROxBB: number;
  ROxSurf: number;
  cutmax: number;
  rmin: number;
  dr: number;
  Qstep: number;
  loopmax: number;
  precision: number;
  r1Coord: number;
  r2Coord: number;
  QInitMode: string;
  QOxInit: number;
  QEqMode: string;
  zlim1QEq: number;
  zlim2QEq: number;
  Bavard: string;
  writeenerg: string;
  Nevery: number;
  writepot: string;
  Neverypot: number;
  ncov: number;
}

const fail = (what: string): Error => new Error(`Error parsing SMTBQ parameter file: ${what}`);

/** TextFileReader + ValueTokenizer: the non-blank lines of the file, comments removed, as word lists. */
class SmtbqLines {
  private readonly lines: string[][] = [];
  private pos = 0;

  constructor(text: string) {
    for (const raw of text.split('\n')) {
      const hash = raw.indexOf('#');
      const body = hash >= 0 ? raw.slice(0, hash) : raw;
      const words = body.split(/['\t\n\r ]+/).filter((w) => w.length > 0);
      if (words.length > 0) this.lines.push(words);
    }
  }

  /** next_line: the words of the next line. */
  next(): string[] {
    const line = this.lines[this.pos++];
    if (line === undefined) throw fail('unexpected end of file');
    return line;
  }

  /** skip_line */
  skip(): void { this.next(); }
}

const num = (w: string | undefined): number => {
  const v = w === undefined ? Number.NaN : Number(w);
  if (!Number.isFinite(v)) throw fail(`expected a number, found '${w ?? ''}'`);
  return v;
};

const int = (w: string | undefined): number => {
  const v = num(w);
  if (!Number.isInteger(v)) throw fail(`expected an integer, found '${w ?? ''}'`);
  return v;
};

/** Reads an SMTBQ potential file (pair_smtbq.cpp read_file). */
export function readSmtbqFile(text: string): SmtbqFile {
  const R = new SmtbqLines(text);
  let line = R.next();
  const ntypes = int(line[1]);
  const intype = new Int32Array(ntypes * ntypes);
  let m = 0;
  for (let i = 0; i < ntypes; i++) {
    for (let j = 0; j < ntypes; j++) {
      if (j < i) intype[i * ntypes + j] = intype[j * ntypes + i];
      else { intype[i * ntypes + j] = 0; m = m + 1; }
    }
  }
  const maxintparam = m;
  const params: SmtbqParam[] = [];
  let coordOxBB = 0, coordOxBulk = 0, coordOxSurf = 0, ROxBB = 0, ROxSurf = 0;

  // one block per element: header, name and stoichiometry, charges, QEq parameters, (O only) coordination, shared states
  for (let i = 0; i < ntypes; i++) {
    R.skip();
    line = R.next();
    const p: SmtbqParam = {
      nom: line[1], sto: num(line[2]), n0: 0, ne: 0, chi: 0, dj: 0, R: 0, dzeta: 0, qform: 0, masse: 0,
    };
    line = R.next();
    p.qform = num(line[1]);
    p.masse = num(line[2]);
    line = R.next();
    p.ne = num(line[1]);
    p.chi = num(line[2]);
    p.dj = num(line[3]);
    if (p.nom !== 'O') p.R = num(line[4]);
    if (p.nom === 'O') {
      line = R.next();
      coordOxBB = num(line[1]);
      coordOxBulk = num(line[2]);
      coordOxSurf = num(line[3]);
      ROxBB = num(line[4]);
      p.R = num(line[5]);
      ROxSurf = num(line[6]);
    }
    line = R.next();
    p.n0 = num(line[1]);
    p.dzeta = (2.0 * p.ne + 1.0) / (4.0 * p.R);
    params.push(p);
  }

  // interactions: one more read than the file has (see the header comment)
  const intparams: SmtbqInter[] = [];
  for (let k = 0; k <= maxintparam; k++) {
    intparams.push({
      typepot: '', mode: '', intsm: 0, a: 0, p: 0, ksi: 0, q: 0, r0: 0, dc1: 0, dc2: 0, abuck: 0, rhobuck: 0,
      aOO: 0, bOO: 0, r1OO: 0, r2OO: 0,
    });
  }
  let maxintsm = 0;
  let cutLine: string[] = [];
  let test = 0;
  let mm = 0;
  for (let k = 0; k < maxintparam; k++) {
    mm += 1;
    R.skip();  // label line of the interaction
    const words = R.next();
    test = 0;
    if (words.length < 3) test = 1;
    let i = 0, j = 0;
    for (i = 0; i < ntypes && test === 0; i++) {
      if (words[1] === params[i].nom) break;
      if (i === ntypes - 1) test = 1;
    }
    for (j = 0; j < ntypes && test === 0; j++) {
      if (words[2] === params[j].nom) break;
      if (j === ntypes - 1) test = 1;
    }
    if (test === 1) {
      // C++: the failed read leaves the cutoff line in the reader's buffer: its words are the cutoff line
      cutLine = words;
      break;
    }
    intype[i * ntypes + j] = mm;
    intype[j * ntypes + i] = intype[i * ntypes + j];
    const it = intparams[mm];
    it.typepot = words[3];
    it.intsm = 0;
    if (it.typepot !== 'second_moment' && it.typepot !== 'buck' && it.typepot !== 'buckPlusAttr') {
      throw fail("Potential must be 'second_moment' or 'buckingham'");
    }
    if (it.typepot === 'second_moment') {
      maxintsm += 1;
      it.mode = words[4];
      it.intsm = maxintsm;
      if (it.mode !== 'oxide' && it.mode !== 'metal') {
        throw fail("Mode of second moment interaction must be 'oxide' or 'metal'");
      }
      line = R.next();
      it.a = num(line[1]);
      it.p = num(line[2]);
      it.ksi = num(line[3]);
      it.q = num(line[4]);
      line = R.next();
      it.dc1 = num(line[1]);
      it.dc2 = num(line[2]);
      it.r0 = num(line[3]);
    } else if (it.typepot === 'buck') {
      line = R.next();
      it.abuck = num(line[1]);
      it.rhobuck = num(line[2]);
    } else {
      line = R.next();
      it.abuck = num(line[1]);
      it.rhobuck = num(line[2]);
      line = R.next();
      it.aOO = num(line[1]);
      it.bOO = num(line[2]);
      it.r1OO = num(line[3]);
      it.r2OO = num(line[4]);
    }
  }

  // tables parameters: the Rcoul line (after the header when every interaction was read)
  if (test === 0) {
    R.skip();
    cutLine = R.next();
  }
  const cutmax = num(cutLine[1]);

  const tabLine = R.next();
  const rmin = num(tabLine[1]);
  const dr = num(tabLine[2]);

  R.skip();
  const qLine = R.next();
  const Qstep = int(qLine[1]);

  const loopLine = R.next();
  const loopmax = int(loopLine[1]);
  const precision = num(loopLine[2]);

  R.skip();
  const coordLine = R.next();
  const r1Coord = num(coordLine[1]);
  const r2Coord = num(coordLine[2]);

  R.skip();
  const initLine = R.next();
  const QInitMode = initLine[1];
  const QOxInit = QInitMode === 'true' ? num(initLine[2]) : 0.0;

  R.skip();
  const qeqLine = R.next();
  const QEqMode = qeqLine[1];
  let zlim1QEq = 0, zlim2QEq = 0;
  if (QEqMode === 'BulkFromSlab') {
    zlim1QEq = num(qeqLine[2]);
    zlim2QEq = num(qeqLine[3]);
  } else if (QEqMode === 'Surface') {
    zlim1QEq = num(qeqLine[2]);
  } else if (QEqMode !== 'QEqAll' && QEqMode !== 'QEqAllParallel') {
    throw fail(`The QEq Mode is not known (${QEqMode}). QEq mode should be QEqAll, QEqAllParallel, Surface or BulkFromSlab`);
  }

  R.skip();
  const bavardLine = R.next();
  const Bavard = bavardLine[1];

  const enLine = R.next();
  const writeenerg = enLine[1];
  const Nevery = writeenerg === 'true' ? num(enLine[2]) : 0.0;

  const potLine = R.next();
  const writepot = potLine[1];
  const Neverypot = writepot === 'true' ? num(potLine[2]) : 0.0;

  // ncov: the shared-state count the charge-dependent covalent terms use (C++: the loop reads params[1] for every i)
  let ncov = params[0].sto * params[0].n0;
  for (let i = 1; i < ntypes; i++) {
    const v = params[1].sto * params[1].n0;
    ncov = v < ncov ? v : ncov;
  }

  return {
    ntypes, params, intype, maxintparam, maxintsm, intparams, coordOxBB, coordOxBulk, coordOxSurf, ROxBB,
    ROxSurf, cutmax, rmin, dr, Qstep, loopmax, precision, r1Coord, r2Coord, QInitMode, QOxInit, QEqMode,
    zlim1QEq, zlim2QEq, Bavard, writeenerg, Nevery, writepot, Neverypot, ncov,
  };
}
