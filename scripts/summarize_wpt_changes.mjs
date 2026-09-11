#!/usr/bin/env node
/**
 * Generate a markdown PR body summarizing WPT snapshot + expected-failures changes.
 *
 * Run in the CI aggregate job AFTER backend patches have been applied to the
 * working tree. Uses `git diff HEAD` so no prior staging is required.
 *
 * Model: PASS snapshots + {backend}_expected_failures.txt. Correlates the two to
 * report "healed" (fail->pass) and "regressed" (pass->fail) transitions.
 */
import { execFileSync } from 'node:child_process';
import path from 'node:path';

const SNAP_DIR = 'tests/snapshots';
const CONFORMANCE_DIR = 'tests/wpt_conformance';

function git(args) {
  return execFileSync('git', args, {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
  }).trim();
}

// run_wpt_conformance__{backend}_{sanitized}.snap -> {sanitized}
function snapName(file) {
  const m = path.basename(file).match(/^run_wpt_conformance__([a-z0-9]+)_(.+)\.snap$/);
  return m ? { backend: m[1], name: m[2] } : null;
}

// {backend}::{operation}::{sanitized} -> {sanitized}
function txtName(line) {
  return line.split('::').pop();
}

// Parse a unified diff into added/removed sanitized names.
function txtDiff(file) {
  const diff = git(['diff', '--cached', 'HEAD', '--', file]);
  const added = new Set();
  const removed = new Set();
  for (const line of diff.split('\n')) {
    if (line.startsWith('+++') || line.startsWith('---')) continue;
    if (line.startsWith('+')) {
      const name = txtName(line.slice(1));
      if (name) added.add(name);
    } else if (line.startsWith('-')) {
      const name = txtName(line.slice(1));
      if (name) removed.add(name);
    }
  }
  return { added, removed };
}

// Stage snapshot/expected-failure changes so `git diff --cached` also sees
// newly-added (untracked) snapshots produced by `git apply`.
git(['add', '-A', '--', SNAP_DIR, CONFORMANCE_DIR]);

const nameStatus = git([
  'diff',
  '--cached',
  'HEAD',
  '--name-status',
  '--',
  SNAP_DIR,
  CONFORMANCE_DIR,
]);
const entries = nameStatus ? nameStatus.split('\n').filter(Boolean) : [];

// Per-backend aggregation keyed by backend name.
const stats = new Map();
const txtAdded = new Map();
const txtRemoved = new Map();
const newSnaps = new Map();
const removedSnaps = new Map();

function backendStat(backend) {
  if (!stats.has(backend)) {
    stats.set(backend, { added: 0, removed: 0, modified: 0 });
  }
  return stats.get(backend);
}
function setName(map, backend, name) {
  if (!map.has(backend)) map.set(backend, new Set());
  map.get(backend).add(name);
}

for (const line of entries) {
  const [status, file] = line.split('\t');
  if (!file) continue;

  if (file.endsWith('.snap')) {
    const parsed = snapName(file);
    if (!parsed) continue;
    const s = backendStat(parsed.backend);
    if (status === 'A') {
      s.added += 1;
      setName(newSnaps, parsed.backend, parsed.name);
    } else if (status === 'D') {
      s.removed += 1;
      setName(removedSnaps, parsed.backend, parsed.name);
    } else if (status === 'M') {
      s.modified += 1;
    }
  } else if (file.includes('_expected_failures.txt')) {
    const backend = file.split('/').pop().replace(/_expected_failures\.txt$/, '');
    const { added, removed } = txtDiff(file);
    if (added.size > 0) txtAdded.set(backend, added);
    if (removed.size > 0) txtRemoved.set(backend, removed);
  }
}

const out = [];
out.push('# WPT snapshot sync');
out.push('');
out.push('Updates WPT PASS snapshots and expected-failure lists to match the latest upstream WPT corpus.');
out.push('');

const allBackends = new Set([
  ...stats.keys(),
  ...txtAdded.keys(),
  ...txtRemoved.keys(),
]);

if (allBackends.size > 0) {
  out.push('## Summary');
  out.push('');
  out.push('| Backend | Snaps + | Snaps - | Snaps ~ | `.txt` + | `.txt` - |');
  out.push('|---------|---------|---------|---------|----------|----------|');
  for (const backend of [...allBackends].sort()) {
    const s = stats.get(backend) ?? { added: 0, removed: 0, modified: 0 };
    const a = txtAdded.get(backend)?.size ?? 0;
    const r = txtRemoved.get(backend)?.size ?? 0;
    out.push(`| ${backend} | ${s.added} | ${s.removed} | ${s.modified} | ${a} | ${r} |`);
  }
  out.push('');
}

function listTransitions(title, backendSets) {
  const items = [];
  for (const backend of [...backendSets.keys()].sort()) {
    for (const name of [...backendSets.get(backend)].sort()) {
      items.push(`- \`${backend}\` ${name}`);
    }
  }
  if (items.length === 0) return;
  out.push(`## ${title}`);
  out.push('');
  out.push(...items);
  out.push('');
}

// healed: new PASS snap whose test was removed from .txt (fail -> pass)
const healed = new Map();
for (const [backend, names] of newSnaps) {
  const removed = txtRemoved.get(backend);
  if (!removed) continue;
  for (const name of names) {
    if (removed.has(name)) setName(healed, backend, name);
  }
}
listTransitions('Healed (fail -> pass)', healed);

// regressed: removed PASS snap whose test was added to .txt (pass -> fail)
const regressed = new Map();
for (const [backend, names] of removedSnaps) {
  const added = txtAdded.get(backend);
  if (!added) continue;
  for (const name of names) {
    if (added.has(name)) setName(regressed, backend, name);
  }
}
listTransitions('Regressed (pass -> fail)', regressed);

const diffStat = git(['diff', '--cached', 'HEAD', '--stat', '--', SNAP_DIR, CONFORMANCE_DIR]);
if (diffStat) {
  out.push('## Diff stat');
  out.push('');
  out.push('```');
  out.push(diffStat);
  out.push('```');
}

console.log(out.join('\n'));
