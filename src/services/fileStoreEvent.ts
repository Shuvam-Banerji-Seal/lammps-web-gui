/** The window event the file store dispatches after every successful put, remove or clear (both backends). Kept apart from
 * fileStore.ts so a view can follow the store without loading it.
 */
export const FILESTORE_EVENT = 'lammps-filestore-change';
