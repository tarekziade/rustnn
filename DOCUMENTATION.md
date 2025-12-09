# Documentation Overview

Complete documentation setup for the WebNN Python API.

##  Documentation Site

The documentation is built with **MkDocs** using the **Material** theme and includes:

### Pages

1. **[Home](docs/index.md)** (`docs/index.md`)
   - Project overview
   - Key features
   - Quick example
   - Installation instructions

2. **[Getting Started](docs/getting-started.md)** (`docs/getting-started.md`)
   - Detailed installation guide
   - Your first graph tutorial
   - Step-by-step walkthrough
   - Common issues and troubleshooting

3. **[API Reference](docs/api-reference.md)** (`docs/api-reference.md`)
   - Complete API documentation
   - All classes and methods
   - Parameter descriptions
   - Code examples for each feature
   - Error handling guide

4. **[Examples](docs/examples.md)** (`docs/examples.md`)
   - Basic examples (addition, ReLU, etc.)
   - Intermediate examples (linear layers, multi-layer networks)
   - Advanced examples (multiple outputs, mixed precision)
   - Complete application example
   - Error handling patterns

5. **[Advanced Topics](docs/advanced.md)** (`docs/advanced.md`)
   - Performance optimization
   - Integration with other libraries
   - Graph introspection
   - Custom graph patterns
   - Testing strategies
   - Platform-specific features
   - Best practices

## [DEPLOY] Building Locally

### Install Dependencies

```bash
pip install -r docs/requirements.txt
```

### Serve with Live Reload

```bash
mkdocs serve
```

Open http://127.0.0.1:8000 in your browser.

### Build Static Site

```bash
mkdocs build
```

The built site will be in `site/`.

### Strict Mode (CI)

```bash
mkdocs build --strict
```

## [BOT] GitHub Actions

### Automatic Deployment

Documentation is automatically built and deployed to GitHub Pages on every push to `main`:

**Workflow**: `.github/workflows/docs.yml`

**What it does**:
1. [OK] Checks out the repository
2. [OK] Installs Python and dependencies
3. [OK] Builds documentation with `mkdocs build --strict`
4. [OK] Uploads build artifact
5. [OK] Deploys to GitHub Pages (main branch only)

**Triggers**:
- Push to `main` (with docs changes)
- Pull requests (build only, no deploy)
- Manual workflow dispatch

### PR Documentation Check

Documentation quality is checked on all PRs:

**Workflow**: `.github/workflows/docs-pr.yml`

**What it does**:
1. [OK] Builds documentation
2. [OK] Checks for broken links
3. [OK] Comments on PR with status

## [STYLE] Features

### Material for MkDocs Theme

-  Dark/light mode toggle
- [SEARCH] Full-text search
-  Mobile-responsive design
- [TARGET] Navigation tabs
-  Expandable sections
-  "Back to top" button
-  Code copy buttons

### Markdown Extensions

- Syntax highlighting for code blocks
- Admonitions (notes, warnings, tips)
- Tabbed content
- Tables
- Table of contents with permalinks
- In-HTML markdown

### Code Examples

All code examples include:
- Syntax highlighting
- Copy button
- Line numbers (where appropriate)
- Type hints

##  File Structure

```
rustnn/
 docs/                       # Documentation source
    index.md               # Home page
    getting-started.md     # Getting started guide
    api-reference.md       # API documentation
    examples.md            # Code examples
    advanced.md            # Advanced topics
    requirements.txt       # Python deps for building
    README.md              # Docs development guide

 mkdocs.yml                 # MkDocs configuration

 .github/workflows/
    docs.yml              # Build & deploy workflow
    docs-pr.yml           # PR check workflow
    README.md             # Workflows documentation

 site/                      # Built documentation (gitignored)
```

##  Configuration

### `mkdocs.yml`

Key configuration:
- Site metadata (name, description, URL)
- Material theme with dark/light mode
- Navigation structure
- Plugins (search, mkdocstrings)
- Markdown extensions
- Social links

### `docs/requirements.txt`

Required packages:
- `mkdocs` - Static site generator
- `mkdocs-material` - Material theme
- `mkdocstrings[python]` - API documentation from docstrings
- `pymdown-extensions` - Additional markdown features

##  Deployment

### GitHub Pages Setup

1. Go to repository **Settings** → **Pages**
2. Under "Build and deployment":
   - Source: **GitHub Actions**
3. Push to `main` branch
4. Documentation will be available at:
   ```
   https://your-org.github.io/rustnn/
   ```

### First Deployment

After the first successful workflow run:
1. Go to the Actions tab
2. Click on the "Build and Deploy Documentation" workflow
3. Click on the latest run
4. Find the deployment URL in the "deploy" job

##  Writing Documentation

### Adding a New Page

1. Create a new `.md` file in `docs/`
2. Add it to the `nav` section in `mkdocs.yml`
3. Test locally with `mkdocs serve`
4. Commit and push

### Code Examples

Use fenced code blocks with language:

````markdown
```python
import webnn

ml = webnn.ML()
```
````

### Admonitions

```markdown
!!! note
    This is an informational note.

!!! warning
    This is a warning message.

!!! tip
    This is a helpful tip.
```

### Internal Links

```markdown
[Getting Started](getting-started.md)
[API Reference - ML Class](api-reference.md#class-ml)
```

## [OK] Quality Checks

### Local Checks

Before committing:

```bash
# Build with strict mode
mkdocs build --strict

# Check for warnings
mkdocs build 2>&1 | grep -i warning
```

### CI Checks

On every PR:
- [OK] Documentation builds successfully
- [OK] No broken internal links
- [OK] No markdown syntax errors
- [OK] No MkDocs configuration errors

##  Links

- **Documentation Source**: `docs/`
- **Built Site**: `site/` (local) or GitHub Pages (deployed)
- **MkDocs**: https://www.mkdocs.org/
- **Material Theme**: https://squidfunk.github.io/mkdocs-material/
- **GitHub Actions Workflows**: `.github/workflows/`

## [TARGET] Next Steps

1. **Customize**: Update `your-org` in URLs to your GitHub organization
2. **Deploy**: Push to main and enable GitHub Pages
3. **Extend**: Add more examples and tutorials as needed
4. **Monitor**: Check GitHub Actions for build status

---

[OK] **Documentation is ready to use and deploy!**
