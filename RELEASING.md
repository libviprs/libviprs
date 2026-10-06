# Releasing libviprs

This is the sequence I ran for 0.5.1, which is the 0.5.0 sequence with the traps written down. Everything up to the tag is reversible. The crates.io upload is not (you can only yank), so it goes last.

## 1. Open the issue

Every PR needs a closing keyword, and the cut is a PR. Open "Release libviprs X.Y.Z", assign it to spdrman, and say what the cut carries.

## 2. Cut on a branch off `main`

Name it `cut/X.Y.Z` (a branch called `release/...` can't exist, because `release` is already a branch). In one commit, `release: cut X.Y.Z`:

- `Cargo.toml`: bump `version`. `Cargo.lock` is not tracked, so there is nothing to edit there.
- `CHANGELOG.md`: leave a bare `## [Unreleased]`, put the notes under `## [X.Y.Z] — YYYY-MM-DD` directly below it, and add the `[X.Y.Z]: https://github.com/libviprs/libviprs/releases/tag/vX.Y.Z` reference above the previous one.
- Say in the body why a pre-commit bypass was needed, if one was.

Then run these before pushing, because a cut is the one document shape the normal PR run never sees:

- `cargo fmt --check`
- `cargo test --test changelog_preamble --test changelog_release_claims --test pmtiles_release_readiness`

A patch release has no `### Breaking` section, so `release_notes` in `tests/changelog_preamble.rs` walks down to the newest section that has one. If you cut something that changes how that file finds its notes, expect it to go red here first.

## 3. Check the other repo's goldens

`libviprs-tests` compares encoder output with committed files. The Radiance header carries `SOFTWARE=libviprs <version>`, and that cell masks the version (libviprs-tests#264). If a new cell pins the version string, fix it in libviprs-tests and merge it before pushing the cut, because the pre-push hook runs the tests suite against the core you are pushing.

## 4. Push and merge the cut PR

- `git push -u origin cut/X.Y.Z` (the pre-push hook runs the full tests suite, 10 to 40 minutes, so run it in the background).
- Open the PR with `Closes #N` in the body and merge it. CI has been paused since September, so that is `gh pr merge --admin --merge` until it is back on.
- `main` now has the merge commit. Tag that commit.

## 5. Dry-run the publish

Before anything irreversible:

```
gh workflow run publish.yml --ref main -f dry_run=true
```

It packages and verifies with `cargo publish --dry-run --locked`. Wait for it to go green.

## 6. Tag, `release`, GitHub release

```
git tag -a vX.Y.Z <merge commit> -F tagmsg.txt     # annotated, with a short summary
git push origin vX.Y.Z
git worktree add ../rel -B rel origin/release
cd ../rel && git merge -m 'Merge main for X.Y.Z' vX.Y.Z && git push origin HEAD:release
gh release create vX.Y.Z --verify-tag --title 'libviprs X.Y.Z' --notes-file notes.md
```

`notes.md` is the new section of `CHANGELOG.md`, and `release` is unprotected, so the merge pushes directly. Pushing a tag also runs the pre-push hook.

## 7. Publish to crates.io

```
gh workflow run publish.yml --ref vX.Y.Z -f dry_run=false
```

`publish.yml` is manual-only on purpose. It skips a version that is already on crates.io, refuses to upload unless the published pdfium contract holds, and needs the `CARGO_REGISTRY_TOKEN` repository secret. Check `https://crates.io/api/v1/crates/libviprs` afterwards.

## Traps

- **The pre-commit hook runs the amd64 image under Rosetta on an Apple Silicon Mac** and can sit for 40 minutes. Run fmt and the guards above in an `--platform linux/arm64` container and commit with `--no-verify`, and say so in the PR body. Anything that must be x64 runs on MARS or the NAS (see `CLAUDE.md`).
- **A worktree's `.git` file points outside the container mount**, so a test that shells out to `git` fails with "not a git repository". Mount the main checkout's `.git` at the same path too.
- **Tags come from `git tag`**, and `changelog_release_claims` reads them, so fetch tags before running it.
