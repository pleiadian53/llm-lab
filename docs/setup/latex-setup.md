# LaTeX Setup Guide for LLM-Lab

This guide explains how to set up LaTeX for converting markdown documents with mathematical equations to professional PDFs.

## Why LaTeX?

The LLM research documents in this lab often contain complex mathematical equations. LaTeX is the gold standard for typesetting mathematics and ensures publication-quality rendering.

## Quick Start

### macOS - BasicTeX (Recommended, 116MB)

```bash
# Download and install BasicTeX
curl -L -o /tmp/BasicTeX.pkg https://mirror.ctan.org/systems/mac/mactex/BasicTeX.pkg
sudo installer -pkg /tmp/BasicTeX.pkg -target /

# Add to PATH
echo 'export PATH="/usr/local/texlive/2025basic/bin/universal-darwin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Verify installation
xelatex --version
```

### macOS - Full MacTeX (Alternative, 7GB)

```bash
# Using Homebrew
brew install --cask mactex

# Add to PATH
echo 'export PATH="/Library/TeX/texbin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Verify
xelatex --version
```

### Linux (Ubuntu/Debian)

```bash
# Install TeX Live
sudo apt-get update
sudo apt-get install texlive-xetex texlive-latex-extra

# Verify
xelatex --version
```

### Linux (Fedora/RHEL)

```bash
# Install TeX Live
sudo dnf install texlive-xetex texlive-collection-latexextra

# Verify
xelatex --version
```

## Verification

Test that everything works:

```bash
# Create test document
cat > /tmp/test.tex << 'EOF'
\documentclass{article}
\usepackage{amsmath}
\begin{document}
Test equation: $E = mc^2$

Display equation:
\[
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
\]
\end{document}
EOF

# Compile
cd /tmp
xelatex test.tex

# Check result
ls -lh test.pdf
open test.pdf  # macOS
# xdg-open test.pdf  # Linux
```

If `test.pdf` exists and displays correctly, you're ready!

## Using the Conversion Script

### Basic Usage

```bash
# Convert markdown to PDF
python scripts/md_to_pdf.py docs/llm/llm_tech_history.md

# Output: docs/llm/llm_tech_history.pdf
```

### Advanced Options

```bash
# Specify output path
python scripts/md_to_pdf.py docs/llm/llm_tech_history.md -o output/llm_history.pdf

# Add title and author
python scripts/md_to_pdf.py docs/llm/llm_tech_history.md \
    --title "Evolution of LLM Architectures" \
    --author "Your Name"

# Generate LaTeX only (for debugging)
python scripts/md_to_pdf.py docs/llm/llm_tech_history.md --latex-only
# Output: docs/llm/llm_tech_history.tex
```

### Batch Conversion

```bash
# Convert all markdown files in docs/llm/
for file in docs/llm/*.md; do
    python scripts/md_to_pdf.py "$file"
done
```

## Supported Markdown Features

The converter supports:

- ✅ **Headers**: `#`, `##`, `###`, etc.
- ✅ **Inline math**: `$x^2$`
- ✅ **Display math**: `$$...$$`
- ✅ **Bold**: `**text**`
- ✅ **Italic**: `*text*`
- ✅ **Code**: `` `code` ``
- ✅ **Code blocks**: ` ```...``` `
- ✅ **Lists**: `*` or `-` for bullets, `1.` for numbered
- ✅ **Blockquotes**: `> text`
- ✅ **Links**: `[text](url)`
- ✅ **Horizontal rules**: `---`

## Troubleshooting

### "xelatex: command not found"

**Problem**: LaTeX is not in your PATH.

**Solution (macOS)**:
```bash
# Find XeLaTeX
find /usr/local/texlive -name xelatex 2>/dev/null

# Add to PATH (adjust path based on find result)
export PATH="/usr/local/texlive/2025basic/bin/universal-darwin:$PATH"

# Make permanent
echo 'export PATH="/usr/local/texlive/2025basic/bin/universal-darwin:$PATH"' >> ~/.zshrc
```

**Solution (Linux)**:
```bash
# Check if installed
which xelatex

# If not found, reinstall
sudo apt-get install --reinstall texlive-xetex
```

### "! LaTeX Error: File `amsmath.sty' not found"

**Problem**: Missing LaTeX packages.

**Solution (BasicTeX)**:
```bash
# Update package manager
sudo tlmgr update --self

# Install missing packages
sudo tlmgr install amsmath amssymb amsfonts geometry fancyhdr enumitem xcolor
```

**Solution (Linux)**:
```bash
# Install extra packages
sudo apt-get install texlive-latex-extra
```

### Compilation Fails with Math Errors

**Problem**: Math symbols not rendering correctly.

**Common Issues**:
1. **Unescaped underscores**: Use `\_` outside math mode
2. **Special characters**: `%`, `&`, `#` need escaping: `\%`, `\&`, `\#`
3. **Mismatched delimiters**: Check all `$...$` and `$$...$$` are closed

**Debug**:
```bash
# Generate LaTeX file to inspect
python scripts/md_to_pdf.py your_file.md --latex-only

# Check the .tex file
cat your_file.tex

# Try compiling manually to see full error
cd /tmp
cp your_file.tex .
xelatex your_file.tex
# Read the error messages carefully
```

### PDF Not Generated

**Problem**: Compilation completes but no PDF.

**Solution**:
```bash
# Check for errors in log
cat document.log | grep "Error"

# Try compiling twice (for references)
xelatex document.tex
xelatex document.tex
```

## Performance

- **BasicTeX installation**: ~2 minutes
- **First compilation**: ~5-10 seconds (package loading)
- **Subsequent compilations**: ~2-3 seconds
- **Disk space**: 116MB (BasicTeX) to 7GB (Full MacTeX)

## Best Practices

1. **Use BasicTeX** unless you need specialized packages
2. **Keep TeX Live updated**: `sudo tlmgr update --all`
3. **Test with simple documents** first
4. **Check logs** if compilation fails
5. **Use `--latex-only`** flag for debugging

## Common Workflows

### Review and Fix Markdown

```bash
# 1. Generate LaTeX to check conversion
python scripts/md_to_pdf.py docs/llm/llm_tech_history.md --latex-only

# 2. Review the .tex file
cat docs/llm/llm_tech_history.tex

# 3. Fix any issues in the markdown
vim docs/llm/llm_tech_history.md

# 4. Generate final PDF
python scripts/md_to_pdf.py docs/llm/llm_tech_history.md
```

### Batch Processing

```bash
# Convert all markdown files in a directory
find docs/llm -name "*.md" -exec python scripts/md_to_pdf.py {} \;

# Or with a loop for better control
for file in docs/llm/*.md; do
    echo "Processing: $file"
    python scripts/md_to_pdf.py "$file" || echo "Failed: $file"
done
```

## Package Requirements

The conversion script automatically includes these LaTeX packages:

- `amsmath` - Advanced math typesetting
- `amssymb` - Math symbols  
- `amsfonts` - Math fonts
- `graphicx` - Graphics support
- `hyperref` - Hyperlinks and PDF metadata
- `geometry` - Page layout
- `fancyhdr` - Headers and footers
- `enumitem` - Enhanced lists
- `xcolor` - Color support

## Alternative: Pandoc

If you prefer using Pandoc:

```bash
# Install Pandoc
brew install pandoc  # macOS
# sudo apt-get install pandoc  # Linux

# Convert with Pandoc
pandoc docs/llm/llm_tech_history.md \
    -o output/llm_history.pdf \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    --toc
```

**Note**: The custom script provides better control over LaTeX formatting specific to this project.

## Resources

- [TeX Live Documentation](https://www.tug.org/texlive/)
- [LaTeX Project](https://www.latex-project.org/)
- [BasicTeX](https://www.tug.org/mactex/morepackages.html)
- [XeLaTeX](http://www.xelatex.org/)
- [Comprehensive LaTeX Symbol List](http://tug.ctan.org/info/symbols/comprehensive/symbols-a4.pdf)

## Support

If you encounter issues:

1. Check this guide's Troubleshooting section
2. Verify LaTeX installation: `xelatex --version`
3. Test with the verification example
4. Generate LaTeX only with `--latex-only` to debug
5. Check the `.log` file for detailed errors

---

**Quick Reference**:
```bash
# Install (macOS)
curl -L -o /tmp/BasicTeX.pkg https://mirror.ctan.org/systems/mac/mactex/BasicTeX.pkg && sudo installer -pkg /tmp/BasicTeX.pkg -target /

# Convert
python scripts/md_to_pdf.py docs/llm/llm_tech_history.md

# Debug
python scripts/md_to_pdf.py docs/llm/llm_tech_history.md --latex-only
```
