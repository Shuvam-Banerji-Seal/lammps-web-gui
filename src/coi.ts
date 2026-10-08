/**
 * Registers public/coi-sw.js, which serves the app with the COOP/COEP headers that make it
 * cross-origin isolated (needed for SharedArrayBuffer, i.e. shared-memory simulation threads),
 * and reloads once so the page is served through it. GitHub Pages cannot send those headers
 * itself. Nothing happens when the page is already isolated (a host that sends the headers), in
 * development, without service workers, or after one reload attempt (a browser that does not
 * support COEP credentialless stays non-isolated and the engine uses message passing).
 */
export const ensureCrossOriginIsolation = (): void => {
  if (typeof window === 'undefined' || !import.meta.env.PROD) return;
  if (window.crossOriginIsolated || !window.isSecureContext || !('serviceWorker' in navigator)) return;
  const base = import.meta.env.BASE_URL;
  const flag = 'coi-reload-attempted';
  let attempted = false;
  try { attempted = sessionStorage.getItem(flag) === '1'; } catch { attempted = true; }
  const reloadOnce = () => {
    if (attempted) return;
    try { sessionStorage.setItem(flag, '1'); } catch { return; }
    window.location.reload();
  };
  navigator.serviceWorker.register(`${base}coi-sw.js`, { scope: base }).then((reg) => {
    // already controlled but still not isolated: the browser lacks COEP credentialless
    if (navigator.serviceWorker.controller) return;
    if (reg.active) { reloadOnce(); return; }
    const sw = reg.installing ?? reg.waiting;
    sw?.addEventListener('statechange', () => { if (sw.state === 'activated') reloadOnce(); });
  }, (err: unknown) => {
    console.warn('cross-origin isolation service worker not registered; shared-memory threads are unavailable', err);
  });
};
