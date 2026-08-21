# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| latest `main` | ✅ |
| older tags/commits | ❌ |

The deployed site always tracks `main` via GitHub Actions.

## Reporting a vulnerability

**Do not open a public issue for security problems.**

Use GitHub's *Report a vulnerability* button under
**Security → Advisories → New draft security advisory** on this repository.
This keeps the report private while allowing coordinated disclosure.

You can expect:

- Acknowledgement within **72 hours**
- A fix or mitigation plan within **14 days** for high-severity issues
- Credit in the release notes (unless you prefer anonymity)

## Scope notes

This project is a fully client-side static site:

- No accounts, no cookies, no analytics, no telemetry.
- User files are parsed **locally in the browser**; nothing is uploaded.
- The only runtime network requests are static assets from this repo's own
  GitHub Pages origin (no third-party CDNs).

Of particular interest for reports: XSS via crafted file content rendered into
the DOM (e.g., through error messages or labels), dependency vulnerabilities
(`npm audit`), and workflow injection in GitHub Actions.
