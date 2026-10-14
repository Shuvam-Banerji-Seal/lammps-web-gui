import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['tests/**/*.test.ts', 'tests/**/*.test.tsx'],
    // engine tests run real MD steps; on a loaded or shared host a few PPPM and spin tests took just over the
    // 5 s default, so the default per-test limit is 30 s (the oracle cases set their own 120 s limit)
    testTimeout: 30000,
  }
});
