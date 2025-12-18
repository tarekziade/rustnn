# Makefile for rendering rustnn slides (updated: solarized, smaller size & CSS)

# Config
SLIDE_MD := rustnn_slides.md
OUT_HTML := index.html
OUT_PANDOC := rustnn_slides.html
MARPOUT := rustnn_slides_marp.html
REVEALJS_URL := https://unpkg.com/reveal.js@4.5.0
PANDOC := pandoc
MARP := marp
PY := python3
PORT := 8000
MERMAID_HEADER := mermaid.html

.PHONY: help slides slides-selfcontained marp serve open preview clean

help:
	@echo "Makefile targets:"
	@echo "  make slides              -> Render slides via pandoc + Reveal.js (fast, uses CDN)"
	@echo "  make slides-selfcontained -> Render slides as one self-contained HTML file (larger)"
	@echo "  make marp                -> Render slides using Marp (if you prefer)"
	@echo "  make serve               -> Serve the current directory on http://localhost:$(PORT)/"
	@echo "  make open                -> Open the generated HTML in the default browser (if available)"
	@echo "  make preview             -> Build slides and open them in browser (convenience)"
	@echo "  make clean               -> Remove generated HTML files"

# Default: revealjs via pandoc, uses CDN for reveal.js (fast)
# Uses solarized theme, explicit presentation size and a custom CSS override.
slides: $(SLIDE_MD)
	@command -v $(PANDOC) >/dev/null 2>&1 || { echo >&2 "pandoc not found. Install pandoc or use 'make marp'"; exit 1; }
	$(PANDOC) $(SLIDE_MD) -t revealjs -s \
	  -V revealjs-url=$(REVEALJS_URL) \
	  -V theme=solarized \
	  -V width=1280 \
	  -V height=720 \
	  -V margin=0.06 \
		--include-in-header=$(MERMAID_HEADER) \
	  --css reveal-overrides.css \
	  --slide-level=2 \
	  -o $(OUT_PANDOC)
	@echo "Generated: $(OUT_PANDOC)"

# Create a self-contained HTML (may be large). Useful for sharing.
slides-selfcontained: $(SLIDE_MD)
	@command -v $(PANDOC) >/dev/null 2>&1 || { echo >&2 "pandoc not found. Install pandoc to use this target."; exit 1; }
	$(PANDOC) $(SLIDE_MD) -t revealjs -s \
	  -V revealjs-url=$(REVEALJS_URL) \
	  -V theme=solarized \
	  -V width=1280 \
	  -V height=720 \
	  -V margin=0.06 \
		--include-in-header=$(MERMAID_HEADER) \
	  --css reveal-overrides.css \
	  --slide-level=2 \
	  --self-contained \
	  -o $(OUT_HTML)
	@echo "Generated (self-contained): $(OUT_HTML)"

# Marp alternative
marp: $(SLIDE_MD)
	@command -v $(MARP) >/dev/null 2>&1 || { echo >&2 "marp-cli not found. Install with: npm install -g @marp-team/marp-cli"; exit 1; }
	$(MARP) $(SLIDE_MD) -o $(MARPOUT)
	@echo "Generated (marp): $(MARPOUT)"

# Simple static server to preview slides
serve:
	@command -v $(PY) >/dev/null 2>&1 || { echo >&2 "python not found."; exit 1; }
	@echo "Serving . on http://localhost:$(PORT)/"
	$(PY) -m http.server $(PORT)

# Try to open generated slides in default browser (macOS/Linux)
open:
	@if [ -f $(OUT_PANDOC) ]; then \
		if command -v xdg-open >/dev/null 2>&1; then \
			xdg-open $(OUT_PANDOC) || true; \
		elif command -v open >/dev/null 2>&1; then \
			open $(OUT_PANDOC) || true; \
		else \
			echo "No open/xdg-open command found — please open $(OUT_PANDOC) manually."; \
		fi \
	else \
		echo "No generated slide file $(OUT_PANDOC) found. Run 'make slides' first."; \
	fi

# Convenience target: build slides and open them (uses file:// open)
preview: slides open

clean:
	rm -f $(OUT_PANDOC) $(OUT_HTML) $(MARPOUT)
	@echo "Cleaned generated files"
