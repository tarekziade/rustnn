# Documentation

This directory contains the documentation for the WebNN Python API.

## Structure

```
docs/
├── index.md              # Home page
├── getting-started.md    # Installation and first steps
├── api-reference.md      # Complete API documentation
├── examples.md           # Code examples and tutorials
├── advanced.md           # Advanced usage patterns
├── architecture.md       # System architecture and design
├── development.md        # Development guide
├── implementation-status.md # Complete implementation status & testing strategy
└── requirements.txt      # Python dependencies for building docs
```

## Building Documentation Locally

### Prerequisites

```bash
pip install -r requirements.txt
```

### Serve Documentation

To preview documentation with live reload:

```bash
make docs-serve
```

Then open http://127.0.0.1:8000 in your browser.

### Build Documentation

To build the static site:

```bash
make docs-build
```

The built site will be in the `site/` directory.

### Strict Build

To build with strict mode (fails on warnings, used in CI):

```bash
make ci-docs
```

## Writing Documentation

### Markdown Features

The documentation uses [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) with these features:

#### Code Blocks

```python
import webnn
import numpy as np

ml = webnn.ML()
context = ml.create_context(accelerated=False)
```

#### Admonitions

```markdown
!!! note
    This is a note.

!!! warning
    This is a warning.

!!! tip
    This is a tip.
```

#### Tabs

```markdown
=== "Python"
    ```python
    print("Hello")
    ```

=== "Output"
    ```
    Hello
    ```
```

### Links

- Internal links: `[Getting Started](getting-started.md)`
- Anchors: `[API Reference](api-reference.md#class-ml)`
- External links: `[W3C WebNN](https://www.w3.org/TR/webnn/)`

## Documentation Deployment

Documentation is automatically built and deployed to GitHub Pages when changes are pushed to the `main` branch.

See [`.github/workflows/`](../.github/workflows/) for CI configuration.

## Style Guide

- Use present tense ("returns" not "will return")
- Use clear, simple language
- Include code examples for all features
- Add type hints to Python code examples
- Keep paragraphs short and focused
- Use headers to organize content hierarchically
- Show actual execution with `compute()`, not just graph building
- Use `make` commands instead of raw `cargo`/`maturin` commands

## Contributing

When adding new features to the Python API:

1. Document the feature in the appropriate section
2. Add code examples showing actual execution
3. Update the API reference
4. Test the documentation build locally: `make docs-serve`
5. Submit a pull request

The documentation will be automatically checked for build errors in the PR.
