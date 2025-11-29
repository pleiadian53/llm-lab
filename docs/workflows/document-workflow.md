# Document Workflow: From Raw Text to Professional PDF

This guide describes the workflow for converting raw text, ill-formatted documents, or markdown files into professional PDFs with proper LaTeX math rendering.

## Overview

The workflow has three main stages:

1. **Text → Markdown**: Convert raw/ill-formatted text to proper markdown syntax
2. **Markdown → LaTeX**: Convert markdown to LaTeX (preserving equations)
3. **LaTeX → PDF**: Compile LaTeX to professional PDF

## Tools Available

### Option 1: Pandoc-based (Recommended)

**Script**: `scripts/md_to_pdf.py`

**Advantages**:
- Industry-standard tool (Pandoc)
- Robust handling of edge cases
- Excellent math support
- Automatic reference handling
- Well-tested and maintained

**Requirements**:
- Pandoc (already installed)
- XeLaTeX (already installed)
- Optional: `pip install pypandoc`

### Option 2: Custom Converter (Legacy)

**Script**: `scripts/md_to_pdf_v0.py`

**Advantages**:
- No external dependencies beyond XeLaTeX
- Full control over conversion logic
- Good for learning/customization

**Limitations**:
- Manual handling of edge cases
- Less robust than Pandoc

## Recommended Workflow

### Stage 1: Raw Text → Markdown

When you have raw or ill-formatted text:

1. **Show me the document** (paste content or provide file path)
2. **I'll help convert it to proper markdown**, ensuring:
   - Correct header levels (`#`, `##`, `###`)
   - Proper list formatting (`*`, `-`, `1.`)
   - Code blocks with ` ``` `
   - **Math equations** properly wrapped:
     - Inline: `$x^2$`
     - Display: `$$...$$` (on separate lines)
   - Links formatted as `[text](url)`
   - Blockquotes with `>`

3. **Save the cleaned markdown** to your docs directory

### Stage 2: Markdown → PDF

Once you have clean markdown:

```bash
# Using Pandoc (recommended)
python3 scripts/md_to_pdf.py docs/your_file.md \
    --title "Your Title" \
    --author "Your Name"

# Or using custom converter
python3 scripts/md_to_pdf.py docs/your_file.md \
    --title "Your Title" \
    --author "Your Name"
```

### Stage 3: Review and Iterate

1. **Open the PDF** and check:
   - Math equations render correctly
   - Headers are properly formatted
   - Lists and code blocks look good
   - References/links work

2. **If issues found**:
   - Generate LaTeX to debug: `--latex-only`
   - Review the `.tex` file
   - Fix issues in the markdown
   - Regenerate PDF

## Example Workflow

### Example 1: Converting Research Notes

```bash
# You provide raw notes (e.g., from ChatGPT, papers, etc.)
# I help format them as markdown with proper syntax

# Save to file
vim docs/llm/new_research_notes.md

# Convert to PDF
python3 scripts/md_to_pdf.py docs/llm/new_research_notes.md \
    --title "Research Notes on Attention Mechanisms" \
    --author "LLM Lab"

# Result: docs/llm/new_research_notes.pdf
```

### Example 2: Fixing Ill-Formatted Document

```bash
# You show me a document with issues:
# - Math not wrapped in $...$
# - Headers inconsistent
# - Lists broken

# I help you fix it, then:
python3 scripts/md_to_pdf.py docs/fixed_document.md
```

### Example 3: Batch Processing

```bash
# Convert all markdown files in a directory
for file in docs/llm/*.md; do
    echo "Processing: $file"
    python3 scripts/md_to_pdf.py "$file"
done
```

## Math Equation Guidelines

### Inline Math

Use `$...$` for inline equations:

```markdown
The loss function is $\mathcal{L} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$.
```

### Display Math

Use `$$...$$` on separate lines for display equations:

```markdown
The attention mechanism is defined as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

where $Q$, $K$, and $V$ are the query, key, and value matrices.
```

### Common Patterns

**Matrices**:
```markdown
$$
A = \begin{bmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{bmatrix}
$$
```

**Aligned equations**:
```markdown
$$
\begin{aligned}
f(x) &= x^2 + 2x + 1 \\
     &= (x + 1)^2
\end{aligned}
$$
```

**Cases**:
```markdown
$$
f(x) = \begin{cases}
x^2 & \text{if } x \geq 0 \\
-x^2 & \text{if } x < 0
\end{cases}
$$
```

## Troubleshooting

### Math Not Rendering

**Problem**: Equations appear as plain text.

**Solution**: Ensure math is wrapped in `$...$` or `$$...$$`:
```markdown
Wrong: The equation x^2 + y^2 = r^2
Right: The equation $x^2 + y^2 = r^2$
```

### Special Characters Breaking

**Problem**: Characters like `&`, `%`, `#` cause errors.

**Solution**: 
- Inside math mode: No escaping needed
- Outside math mode: Pandoc handles automatically
- If issues persist, escape with backslash: `\&`, `\%`, `\#`

### PDF Generation Fails

**Problem**: LaTeX compilation errors.

**Solution**:
```bash
# Generate LaTeX first to debug
python3 scripts/md_to_pdf.py your_file.md --latex-only

# Review the .tex file
cat your_file.tex

# Check for common issues:
# - Unmatched braces {}
# - Unmatched math delimiters $...$
# - Special characters outside math mode
```

### Reference Links Not Working

**Problem**: `[text](url)` links don't work.

**Solution**: Pandoc handles these automatically. If using custom converter, check URL escaping.

## Best Practices

1. **Always use proper markdown syntax** from the start
2. **Wrap all math in delimiters** (`$...$` or `$$...$$`)
3. **Test with small documents first** before batch processing
4. **Keep LaTeX files** (`.tex`) for debugging
5. **Use `--latex-only` flag** when debugging conversion issues
6. **Prefer Pandoc** (`md_to_pdf.py`) for production use
7. **Version control** both `.md` and `.pdf` files

## Quick Reference

### Convert to PDF
```bash
python3 scripts/md_to_pdf.py input.md
```

### With Custom Title/Author
```bash
python3 scripts/md_to_pdf.py input.md \
    --title "My Document" \
    --author "Author Name"
```

### Generate LaTeX Only
```bash
python3 scripts/md_to_pdf.py input.md --latex-only
```

### Specify Output Path
```bash
python3 scripts/md_to_pdf.py input.md -o output/document.pdf
```

## Integration with AI Workflow

When working with me (Cascade):

1. **Share raw content**: Paste text, provide file path, or describe document
2. **I'll format as markdown**: Proper syntax, equations wrapped correctly
3. **You save the markdown**: Copy to file or I can create it
4. **Run conversion script**: Use `md_to_pdf.py`
5. **Review and iterate**: If issues, I'll help debug

This workflow ensures high-quality, professional documents with minimal manual formatting.

## See Also

- [LATEX_SETUP.md](LATEX_SETUP.md) - LaTeX installation guide
- [scripts/README.md](../scripts/README.md) - Scripts documentation
- [Pandoc Manual](https://pandoc.org/MANUAL.html) - Full Pandoc documentation
- [LaTeX Math Symbols](http://tug.ctan.org/info/symbols/comprehensive/symbols-a4.pdf) - Comprehensive symbol list
