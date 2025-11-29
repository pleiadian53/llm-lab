# Markdown to PDF Conversion Workflow

This guide explains how to convert markdown documents (especially technical documents with LaTeX math) to professional PDFs.

## Quick Start

```bash
# Generate LaTeX file (for review)
python3 scripts/md_to_pdf.py docs/llm/your_document.md --latex-only

# Generate PDF (requires XeLaTeX)
python3 scripts/md_to_pdf.py docs/llm/your_document.md
```

## Example: Converting LLM Tech History

```bash
# Generate LaTeX
python3 scripts/md_to_pdf.py docs/llm/llm_tech_evolution/llm_tech_history.md --latex-only

# Generate PDF
python3 scripts/md_to_pdf.py docs/llm/llm_tech_evolution/llm_tech_history.md
```

## Prerequisites

You need XeLaTeX installed. See [../setup/latex-setup.md](../setup/latex-setup.md) for installation instructions.

**Quick install (macOS)**:
```bash
curl -L -o /tmp/BasicTeX.pkg https://mirror.ctan.org/systems/mac/mactex/BasicTeX.pkg
sudo installer -pkg /tmp/BasicTeX.pkg -target /
echo 'export PATH="/usr/local/texlive/2025basic/bin/universal-darwin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

## Workflow

### 1. Review the Markdown

The markdown file uses standard LaTeX math notation:
- Inline math: `$x^2$`
- Display math: `$$...$$` (on separate lines)

### 2. Generate LaTeX

```bash
python3 scripts/md_to_pdf.py docs/llm/llm_tech_history.md --latex-only
```

This creates `docs/llm/llm_tech_history.tex` which you can review.

### 3. Check for Issues

Common issues to look for in the `.tex` file:
- Unescaped special characters: `_`, `%`, `&`, `#`
- Mismatched math delimiters
- Broken references

### 4. Generate PDF

```bash
python3 scripts/md_to_pdf.py docs/llm/llm_tech_history.md
```

This creates `docs/llm/llm_tech_history.pdf`.

### 5. Review PDF

```bash
open docs/llm/llm_tech_history.pdf  # macOS
# xdg-open docs/llm/llm_tech_history.pdf  # Linux
```

## Customization

### Add Title and Author

```bash
python3 scripts/md_to_pdf.py docs/llm/llm_tech_history.md \
    --title "Evolution of LLM Architectures" \
    --author "Your Name"
```

### Custom Output Path

```bash
python3 scripts/md_to_pdf.py docs/llm/llm_tech_history.md \
    -o output/llm_evolution.pdf
```

## Troubleshooting

### Math Symbols Not Rendering

**Issue**: Some math symbols appear as text.

**Solution**: Ensure they're wrapped in `$...$` for inline or `$$...$$` for display math.

### Compilation Errors

**Issue**: PDF generation fails.

**Solution**:
1. Generate LaTeX only: `--latex-only`
2. Review the `.tex` file
3. Look for error patterns:
   - Unescaped `_` outside math mode
   - Unescaped `%`, `&`, `#`
   - Mismatched braces `{}`

### Reference Links Not Working

**Issue**: `[text](url)` links don't work in PDF.

**Solution**: The converter transforms these to `\href{url}{text}` automatically. If they don't work, check the LaTeX output.

## Known Limitations

1. **Tables**: Not yet supported (markdown tables need custom handling)
2. **Images**: Not yet supported (would need path resolution)
3. **Nested lists**: Limited support
4. **Complex code blocks**: May need manual adjustment

## Next Steps

After generating the PDF, you might want to:

1. **Add to version control**:
   ```bash
   git add docs/llm/llm_tech_history.pdf
   git commit -m "Add PDF version of LLM tech history"
   ```

2. **Share the PDF**: The PDF is self-contained and can be shared directly

3. **Update the markdown**: If you find issues in the PDF, fix them in the markdown and regenerate

## Alternative: Pandoc

If you prefer using Pandoc:

```bash
# Install Pandoc
brew install pandoc  # macOS

# Convert
pandoc docs/llm/llm_tech_history.md \
    -o docs/llm/llm_tech_history.pdf \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    --toc \
    --number-sections
```

The custom script provides better control over formatting specific to this project.
