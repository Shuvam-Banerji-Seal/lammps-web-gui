# Deployment

## Pipeline

```
push to main ──► ci.yml (typecheck · tests · npm audit · build)
             └─► deploy.yml (typecheck · tests · audit · build ──► GitHub Pages)
```

Both workflows gate on `npm audit --audit-level=high`: a high/critical
dependency advisory blocks merge and deploy until patched.

Dependabot opens grouped minor/patch npm updates weekly (major bumps like
three.js are left for deliberate review) plus weekly Actions updates.

## GitHub Pages configuration

- Source: **GitHub Actions** (`actions/deploy-pages@v4`)
- Base path: `/lammps-web-gui/` — set in `vite.config.ts` when
  `NODE_ENV=production`; all asset URLs use `import.meta.env.BASE_URL`

## Publishing the wiki (one-time)

GitHub requires the wiki to be initialized once through the web UI:

1. Open the repo → **Wiki** tab → *Create the first page*, save any content.
2. Then publish the versioned pages:

```bash
git clone https://github.com/Shuvam-Banerji-Seal/lammps-web-gui.wiki.git wiki-repo
cp wiki/*.md wiki-repo/
cd wiki-repo && git add . && git commit -m "docs: sync wiki" && git push
```

Afterwards keep editing `wiki/*.md` in-repo and re-run the copy+push.

## Search indexing

The site ships robots.txt, sitemap.xml, canonical URL, Open Graph/Twitter
cards and JSON-LD structured data. To (re-)request indexing after a major
release: [Google Search Console](https://search.google.com/search-console) →
URL inspection → *Request indexing*. Bing uses *IndexNow*/Webmaster Tools.
