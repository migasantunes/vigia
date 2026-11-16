# scripts

One-off entry points that rebuild everything derived from the raw data: downloading a
dataset into `data/`, building its manifest, and writing the resized frame cache. Every
script is a thin wrapper over a `vigia` CLI subcommand (for example `vigia manifest
figlib`), so the logic lives in the package and is tested there; nothing in `scripts/` is
imported by the package.
