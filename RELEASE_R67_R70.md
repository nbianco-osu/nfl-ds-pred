# R67-R70: reliable delivery and visible evidence

- R67: A 12,511-byte manifest replaces the 33 MB all-dataset startup dependency.
  A session pins one revision; dataset bytes are SHA-256 checked. Failed individual
  files never switch to another revision. A failed cloud manifest may select the
  complete bundled revision, visibly labeled as potentially older.
- R68: Release Evidence shows the snapshot cutoff, reconciliation, missing-field
  counts, prospective fixed-review policy, measured latency, rollback rehearsal,
  pruning and retrospective scoring-event comparisons. Stale research is explicit.
  No browser approval or activation endpoint exists.
- R69: Compact content-addressed datasets and static hosting replace the giant
  embedded Worker. Existing datasets and historical archives remain unchanged.
  Panels request independent files; existing research panels may still preload
  their datasets. This is not a claim that all views are lazy-loaded.
- R70: Python and browser-loader unit tests cover stable identities, corruption,
  traversal, concurrent readers, fallback and missing datasets. CI and publication
  validate artifact hashes before deployment. No live endpoint availability SLA
  or automated browser screenshot coverage is implied.

The existing private scheduled writer generates and publishes subsequent manifests
and referenced objects. Operational timestamp-only changes remain excluded from
publication decisions. Old immutable objects are retained; storage compaction is
future work. The legacy live_snapshot.json remains for compatibility with older
site versions; it is no longer a startup dependency for this version.

Validation: 167 Python tests and four JavaScript loader tests passed. Production
forecasts, model versions and completed results were not retrained or rewritten.
