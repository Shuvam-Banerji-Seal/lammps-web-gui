import type { ImageDumpFactory } from '../output/dump';
import { ImageDump } from '../output/dump_image';

/**
 * The dump image writer (output/dump_image.ts), loaded when a command names image (dump ... image, write_dump ... image;
 * lazyStyles.ts), so its renderer stays out of the engine worker's first chunk.
 */
export const HOOKS: Record<string, { createDump: ImageDumpFactory }> = {
  image: { createDump: (sys, id, group, file, cols, multifileOverride, every) => new ImageDump(sys, id, group, file, cols, multifileOverride, every) },
};
