#!/usr/bin/env node
/**
 * Delete WPT snapshot files that should no longer exist, for the named backends.
 *
 * Usage:
 *   node scripts/prune_wpt_snapshots.mjs <backend> [<backend> ...]
 *
 * With the "PASS snapshots + expected-failures .txt" model, a backend test may
 * only have a snapshot when it PASSES. This script removes a snapshot when:
 *
 *   1. its test no longer exists in the upstream WPT corpus (orphaned), or
 *   2. its test is listed in `{backend}_expected_failures.txt` (it is a known
 *      failure, so it must not have a snapshot).
 *
 * Snapshot filenames follow the harness pattern:
 *   run_wpt_conformance__{backend}_{sanitize_test_id(case.name)}.snap
 *
 * Only the backends named on the command line are touched; other backends
 * (e.g. `trtx` when there is no GPU runner in CI) are left untouched.
 *
 * Note: this mirrors `sanitize_test_id` from tests/wpt_conformance/wpt_js_loader.rs
 * (non `[A-Za-z0-9_-]` characters become `_`). WPT test names are ASCII, so the
 * ASCII regex is equivalent.
 */
import { execFileSync } from 'node:child_process';
import { existsSync, readdirSync, readFileSync, unlinkSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');
const snapshotDir = path.join(repoRoot, 'tests', 'snapshots');
const dumpScript = path.join(repoRoot, 'scripts', 'wpt_bridge', 'dump_corpus.mjs');

const backends = process.argv.slice(2);
if (backends.length === 0) {
  console.error('Usage: node scripts/prune_wpt_snapshots.mjs <backend> [<backend> ...]');
  process.exit(2);
}

function sanitizeTestId(name) {
  return name.replace(/[^A-Za-z0-9_-]/g, '_');
}

// All sanitized test names present in the upstream corpus (regardless of backend).
function corpusNames() {
  const stdout = execFileSync('node', [dumpScript], {
    cwd: repoRoot,
    encoding: 'utf8',
    // The corpus JSON is >1 MB and grows with upstream WPT.
    maxBuffer: 100 * 1024 * 1024,
  });
  const corpus = JSON.parse(stdout);
  return new Set((corpus.cases ?? []).map((c) => sanitizeTestId(c.name)));
}

// Sanitized test names listed as known failures for a backend (from its .txt).
function expectedFailureNames(backend) {
  const file = path.join(
    repoRoot,
    'tests',
    'wpt_conformance',
    `${backend}_expected_failures.txt`
  );
  if (!existsSync(file)) {
    return new Set();
  }
  const names = new Set();
  for (const line of readFileSync(file, 'utf8').split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) {
      continue;
    }
    const sanitized = trimmed.split('::').pop();
    if (sanitized) {
      names.add(sanitized);
    }
  }
  return names;
}

const inCorpus = corpusNames();
let removed = 0;

for (const backend of backends) {
  const failing = expectedFailureNames(backend);
  const prefix = `run_wpt_conformance__${backend}_`;
  if (!existsSync(snapshotDir)) {
    continue;
  }
  for (const file of readdirSync(snapshotDir)) {
    if (!file.startsWith(prefix) || !file.endsWith('.snap')) {
      continue;
    }
    // run_wpt_conformance__{backend}_{sanitized}.snap -> {sanitized}
    const sanitized = file.slice(prefix.length, -'.snap'.length);
    const orphaned = !inCorpus.has(sanitized);
    const isKnownFailure = failing.has(sanitized);
    if (orphaned || isKnownFailure) {
      unlinkSync(path.join(snapshotDir, file));
      removed += 1;
      console.log(
        `removed snapshot (${orphaned ? 'orphaned' : 'expected failure'}): ${file}`
      );
    }
  }
}

console.log(`pruned ${removed} snapshot(s) for backends: ${backends.join(', ')}`);
