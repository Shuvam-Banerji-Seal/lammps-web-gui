/*
 * Cross-origin isolation for static hosting that cannot set response headers (GitHub Pages).
 * SharedArrayBuffer, which the multi-threaded simulation engine uses to share atom data between
 * workers, needs a cross-origin-isolated page: the document must be served with
 * Cross-Origin-Opener-Policy: same-origin and Cross-Origin-Embedder-Policy (here credentialless,
 * so cross-origin resources such as web fonts still load without CORP headers). This service
 * worker adds both headers to every response it serves. src/coi.ts registers it and reloads the
 * page once so the document itself is served through it. Browsers without COEP credentialless
 * stay non-isolated and the engine falls back to message passing.
 */
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  let request = event.request;
  // requests that only read the cache across origins cannot be re-fetched here
  if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') return;
  // COEP credentialless: no-cors requests to other origins go without credentials
  if (request.mode === 'no-cors' && new URL(request.url).origin !== self.location.origin) {
    request = new Request(request, { credentials: 'omit' });
  }
  event.respondWith(
    fetch(request).then((response) => {
      if (response.status === 0) return response; // opaque: headers cannot be changed
      const headers = new Headers(response.headers);
      headers.set('Cross-Origin-Embedder-Policy', 'credentialless');
      headers.set('Cross-Origin-Opener-Policy', 'same-origin');
      return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
    }),
  );
});
