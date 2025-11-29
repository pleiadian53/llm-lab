# LLM-Lab Scripts

Utility scripts for the LLM-Lab project.

## Available Scripts

### `md_to_pdf.py` - Markdown to PDF Converter (Primary)

**Uses Pandoc** - the industry-standard document converter.

Converts markdown documents with LaTeX math notation to professional PDFs using Pandoc.

**Advantages**:
- Robust handling of edge cases
- Excellent math support
- Automatic reference handling
- Well-tested and maintained
- Works with or without pypandoc

**Quick Start**:
```bash
# Convert to PDF
python3 scripts/md_to_pdf.py docs/llm/llm_tech_history.md

# With title and author
python3 scripts/md_to_pdf.py docs/llm/llm_tech_history.md \
    --title "LLM Evolution" \
    --author "Your Name"

# Generate LaTeX only
python3 scripts/md_to_pdf.py docs/llm/llm_tech_history.md --latex-only
```

**Requirements**:
- Pandoc (already installed)
- XeLaTeX (already installed)
- Optional: `pip install pypandoc` for Python API

---

### `md_to_pdf_v0.py` - Custom Markdown to PDF Converter (Legacy)

**Custom implementation** - no external dependencies beyond XeLaTeX.

Converts markdown documents with LaTeX math notation to professional PDFs.

**Note**: This is the original custom implementation. Use `md_to_pdf.py` (Pandoc-based) for production work.

**Features**:
- Converts markdown with inline math (`$...$`) and display math (`$$...$$`)
- Generates properly formatted LaTeX documents
- Compiles to PDF using XeLaTeX
- Supports headers, lists, code blocks, blockquotes, and links
- Automatic table of contents generation
- Professional document layout

**Quick Start**:
```bash
# Convert a markdown file to PDF
python scripts/md_to_pdf.py docs/llm/llm_tech_history.md

# With custom title and author
python scripts/md_to_pdf.py docs/llm/llm_tech_history.md \
    --title "LLM Evolution" \
    --author "Your Name"

# Generate LaTeX only (for debugging)
python scripts/md_to_pdf.py docs/llm/llm_tech_history.md --latex-only
```

**Requirements**:
- Python 3.8+
- XeLaTeX (see [LATEX_SETUP.md](../docs/LATEX_SETUP.md) for installation)

**See Also**:
- [LaTeX Setup Guide](../docs/LATEX_SETUP.md) - Complete installation and troubleshooting guide

## Installation

### 1. Install LaTeX

**macOS (BasicTeX - Recommended)**:
```bash
curl -L -o /tmp/BasicTeX.pkg https://mirror.ctan.org/systems/mac/mactex/BasicTeX.pkg
sudo installer -pkg /tmp/BasicTeX.pkg -target /
echo 'export PATH="/usr/local/texlive/2025basic/bin/universal-darwin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get update
sudo apt-get install texlive-xetex texlive-latex-extra
```

### 2. Verify Installation

```bash
xelatex --version
```

You should see output like:
```
XeTeX 3.141592653-2.6-0.999996 (TeX Live 2025)
```

## Usage Examples

### Convert Single File

```bash
python scripts/md_to_pdf.py docs/llm/llm_tech_history.md
```

Output: `docs/llm/llm_tech_history.pdf`

### Specify Output Path

```bash
python scripts/md_to_pdf.py docs/llm/llm_tech_history.md -o output/history.pdf
```

### Batch Convert All Markdown Files

```bash
for file in docs/llm/*.md; do
    python scripts/md_to_pdf.py "$file"
done
```

### Debug LaTeX Generation

```bash
# Generate .tex file without compiling
python scripts/md_to_pdf.py docs/llm/llm_tech_history.md --latex-only

# Inspect the generated LaTeX
cat docs/llm/llm_tech_history.tex

# Manually compile to see detailed errors
cd docs/llm
xelatex llm_tech_history.tex
```

## Troubleshooting

### "xelatex: command not found"

LaTeX is not installed or not in PATH. See [LATEX_SETUP.md](../docs/LATEX_SETUP.md).

### Compilation Errors

1. Generate LaTeX only: `--latex-only`
2. Inspect the `.tex` file
3. Check for unescaped special characters: `_`, `%`, `&`, `#`
4. Verify math delimiters are balanced: `$...$` and `$$...$$`

### Math Not Rendering

Common issues:
- Missing `$` delimiters around math
- Unescaped underscores outside math mode
- Special LaTeX characters not escaped

## Contributing

When adding new scripts:

1. Add a docstring explaining the script's purpose
2. Include usage examples in the docstring
3. Update this README with the new script
4. Add appropriate error handling
5. Make the script executable: `chmod +x scripts/your_script.py`

## License

Same as the parent project.
