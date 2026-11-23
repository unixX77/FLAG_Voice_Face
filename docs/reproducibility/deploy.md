# Publishing the repository and the website (requires the owner's approval)

Published on 2026-11-23 as `unixX77/FLAG_Voice_Face`. Steps used:

1. Choose a licence (`licensing.md`), add `LICENSE`, and fill in your name in `CITATION.cff` (currently only the GitHub alias).
2. Create an empty GitHub repository, `unixX77/FLAG_Voice_Face` (public).
3. From this folder:
   ```bash
   git init -b main && git add . && git commit -m "Public release of the FLAG 2027 research project"
   git remote add origin https://github.com/unixX77/FLAG_Voice_Face.git && git push -u origin main
   ```
4. In the repository settings, go to **Pages → Build and deployment → Source: GitHub Actions**. The `Deploy site to GitHub Pages` workflow builds `site/` with base path `/<repository name>/` and publishes it to `https://unixx77.github.io/FLAG_Voice_Face/`.
5. `CI` runs the privacy scan, the tests and the site checks on every push.

If the repository gets a different name, nothing needs to change: the workflows read `github.event.repository.name`. The source links inside the site use `REPO_URL` in `site/src/data/registry.ts`; update it if the owner or name differs.
