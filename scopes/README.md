# Scope packs

A scope pack declares one landscape (segments, layers, vocabulary, sources, rubric, attributes,
profiles, views). See [ARCHITECTURE.md §3](../ARCHITECTURE.md#3-scope-pack).

- `_example/` is the only pack tracked in this public repository. It is illustrative and exists to
  prove the engine stays generic.
- Every other `scopes/<name>/` is a **private checkout** of its own repository and is gitignored
  here. Never copy its content, names or findings into this repository, its commits, PRs or
  handoffs (decision D-003).

## Using a private scope

To write a new pack from an empty directory, see [docs/new-scope.md](../docs/new-scope.md).

```sh
git clone <private-scope-repo-url> scopes/<name>
landscape validate --scope scopes/<name>
```

Lane worktrees do not contain gitignored folders. A lane that needs a private scope links it from
the main checkout:

```sh
ln -s "$MAIN_CHECKOUT/scopes/<name>" "<lane worktree>/scopes/<name>"
```

Scope-specific planning (its backlog, research and decisions) lives in the scope repository and
follows its own README. Engine work that a scope needs is filed in this repo's `todo.md` in
domain-neutral words.
