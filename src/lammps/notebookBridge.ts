/**
 * Script hand-over between the MD Notebook and the Script Builder.
 *
 * Notebook -> Builder: the cells become one input script (cellsToScript), which the Builder imports as a new tab
 * with parseScript, the same importer its Import button uses. Builder -> Notebook: the generated script text is
 * offered to the notebook as cells (the notebook already accepts it via its incoming banner).
 */

/** One input script from notebook cells: the non-empty cells in order, separated by a blank line. */
export const cellsToScript = (cells: readonly string[]): string =>
  cells.map((c) => c.replace(/\s+$/, '')).filter((c) => c.trim() !== '').join('\n\n') + '\n';

/** A script handed from the notebook to the Script Builder. */
export interface BuilderIncoming {
  text: string;
  /** Title of the new Builder tab. */
  name: string;
}
