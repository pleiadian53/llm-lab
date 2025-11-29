#!/usr/bin/env python3
"""
Markdown to LaTeX/PDF Conversion Utility

Converts markdown documents with LaTeX math notation to properly formatted
LaTeX documents and compiles them to PDF using XeLaTeX.

Usage:
    python scripts/md_to_pdf.py docs/llm/llm_tech_history.md
    python scripts/md_to_pdf.py docs/llm/llm_tech_history.md --output output/llm_history.pdf
    python scripts/md_to_pdf.py docs/llm/llm_tech_history.md --latex-only
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple


class MarkdownToLatexConverter:
    """Converts Markdown with LaTeX math to pure LaTeX document."""
    
    def __init__(self):
        self.latex_preamble = r"""\documentclass[11pt]{article}

% Core packages (included in BasicTeX)
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{geometry}

% Page layout
\geometry{
    a4paper,
    left=1in,
    right=1in,
    top=1in,
    bottom=1in
}

% Hyperref setup
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    urlcolor=blue,
    citecolor=blue
}

% Simple header/footer
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\rhead{\thepage}
\lhead{\leftmark}
\renewcommand{\headrulewidth}{0.4pt}

"""
    
    def convert(self, markdown_content: str, title: str = "", author: str = "") -> str:
        """Convert markdown content to LaTeX."""
        
        # Start with preamble
        latex = self.latex_preamble
        
        # Add title and author if provided
        if title:
            latex += f"\\title{{{title}}}\n"
        if author:
            latex += f"\\author{{{author}}}\n"
        latex += "\\date{\\today}\n\n"
        
        # Begin document
        latex += "\\begin{document}\n\n"
        
        if title:
            latex += "\\maketitle\n\\tableofcontents\n\\newpage\n\n"
        
        # Convert markdown body
        latex += self._convert_body(markdown_content)
        
        # End document
        latex += "\n\\end{document}\n"
        
        return latex
    
    def _convert_body(self, content: str) -> str:
        """Convert markdown body to LaTeX."""
        
        lines = content.split('\n')
        latex_lines = []
        in_code_block = False
        in_blockquote = False
        in_list = False
        list_type = None  # 'itemize' or 'enumerate'
        code_lang = ""
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Handle code blocks
            if line.strip().startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    code_lang = line.strip()[3:].strip()
                    latex_lines.append("\\begin{verbatim}")
                else:
                    in_code_block = False
                    latex_lines.append("\\end{verbatim}")
                i += 1
                continue
            
            if in_code_block:
                latex_lines.append(line)
                i += 1
                continue
            
            # Handle display math ($$...$$)
            if line.strip() == '$$':
                # Start of multi-line display math
                math_lines = []
                i += 1
                while i < len(lines) and lines[i].strip() != '$$':
                    math_lines.append(lines[i])
                    i += 1
                # Add the math block
                latex_lines.append("\\[")
                latex_lines.extend(math_lines)
                latex_lines.append("\\]")
                i += 1  # Skip the closing $$
                continue
            
            # Handle single-line display math (rare case: $$...$$ on one line)
            if line.strip().startswith('$$') and line.strip().endswith('$$') and len(line.strip()) > 4:
                math_content = line.strip()[2:-2].strip()
                latex_lines.append(f"\\[")
                latex_lines.append(math_content)
                latex_lines.append("\\]")
                i += 1
                continue
            
            # Handle headers
            if line.startswith('#'):
                level, text = self._parse_header(line)
                if level == 1:
                    latex_lines.append(f"\\section{{{text}}}")
                elif level == 2:
                    latex_lines.append(f"\\subsection{{{text}}}")
                elif level == 3:
                    latex_lines.append(f"\\subsubsection{{{text}}}")
                else:
                    latex_lines.append(f"\\paragraph{{{text}}}")
                i += 1
                continue
            
            # Handle blockquotes
            if line.strip().startswith('>'):
                if not in_blockquote:
                    latex_lines.append("\\begin{quote}")
                    in_blockquote = True
                text = line.strip()[1:].strip()
                # Remove **bold** markers from blockquote
                text = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', text)
                # Handle inline math
                text = self._convert_inline_math(text)
                latex_lines.append(text)
            else:
                if in_blockquote:
                    latex_lines.append("\\end{quote}")
                    in_blockquote = False
            
            # Handle lists
            if line.strip().startswith('* ') or line.strip().startswith('- '):
                if not in_list:
                    latex_lines.append("\\begin{itemize}")
                    in_list = True
                    list_type = 'itemize'
                text = line.strip()[2:].strip()
                text = self._convert_inline_formatting(text)
                latex_lines.append(f"\\item {text}")
            elif line.strip().startswith(tuple(f"{i}. " for i in range(1, 10))):
                if not in_list:
                    latex_lines.append("\\begin{enumerate}")
                    in_list = True
                    list_type = 'enumerate'
                text = re.sub(r'^\d+\.\s+', '', line.strip())
                text = self._convert_inline_formatting(text)
                latex_lines.append(f"\\item {text}")
            else:
                if in_list:
                    latex_lines.append(f"\\end{{{list_type}}}")
                    in_list = False
                    list_type = None
            
            # Handle horizontal rules
            if line.strip() == '---':
                latex_lines.append("\\hrulefill")
                i += 1
                continue
            
            # Handle reference-style link definitions [1]: url "title"
            # These should be converted to footnote-style or just skipped
            if re.match(r'^\[\d+\]:\s+https?://', line):
                # Extract the reference
                match = re.match(r'^\[(\d+)\]:\s+(\S+)(?:\s+"([^"]+)")?', line)
                if match:
                    ref_num, url, title = match.groups()
                    # Escape underscores in URL for LaTeX
                    url_escaped = url.replace('_', '\\_')
                    if title:
                        latex_lines.append(f"\\noindent[{ref_num}] \\href{{{url_escaped}}}{{{title}}}")
                    else:
                        latex_lines.append(f"\\noindent[{ref_num}] \\url{{{url_escaped}}}")
                i += 1
                continue
            
            # Handle inline links [text](url)
            line = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\\href{\2}{\1}', line)
            
            # Handle regular paragraphs
            if not line.strip().startswith(('*', '-', '>', '#', '$$')) and line.strip():
                if not in_blockquote and not in_list:
                    text = self._convert_inline_formatting(line)
                    latex_lines.append(text)
            elif not line.strip():
                if not in_blockquote and not in_list:
                    latex_lines.append("")
            
            i += 1
        
        # Close any open environments
        if in_blockquote:
            latex_lines.append("\\end{quote}")
        if in_list and list_type:
            latex_lines.append(f"\\end{{{list_type}}}")
        
        return '\n'.join(latex_lines)
    
    def _parse_header(self, line: str) -> Tuple[int, str]:
        """Parse markdown header and return level and text."""
        match = re.match(r'^(#+)\s+(.+)$', line)
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            # Apply inline formatting (includes escaping special chars)
            text = self._convert_inline_formatting(text)
            return level, text
        return 0, line
    
    def _convert_inline_formatting(self, text: str) -> str:
        """Convert inline markdown formatting to LaTeX."""
        # First, protect inline math by temporarily replacing it
        math_placeholders = []
        def save_math(match):
            math_placeholders.append(match.group(0))
            return f"<<<MATH{len(math_placeholders)-1}>>>"
        
        # Save inline math
        text = re.sub(r'\$[^$]+\$', save_math, text)
        
        # Escape special LaTeX characters (now safe, math is protected)
        text = text.replace('&', '\\&')
        text = text.replace('%', '\\%')
        text = text.replace('#', '\\#')
        
        # Handle bold **text**
        text = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', text)
        
        # Handle italic *text*
        text = re.sub(r'\*([^*]+)\*', r'\\textit{\1}', text)
        
        # Handle inline code `code`
        text = re.sub(r'`([^`]+)`', r'\\texttt{\1}', text)
        
        # Restore inline math
        for i, math in enumerate(math_placeholders):
            text = text.replace(f"<<<MATH{i}>>>", math)
        
        return text
    
    def _convert_inline_math(self, text: str) -> str:
        """Convert inline math $...$ (keep as is, it's already LaTeX)."""
        # Inline math is already in correct format for LaTeX
        return text


class PDFCompiler:
    """Compiles LaTeX documents to PDF using XeLaTeX."""
    
    @staticmethod
    def find_xelatex() -> Optional[Path]:
        """Find XeLaTeX executable in common locations."""
        # Try PATH first
        try:
            result = subprocess.run(['which', 'xelatex'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except:
            pass
        
        # Try common installation paths
        common_paths = [
            Path("/usr/local/texlive"),
            Path("/Library/TeX/texbin/xelatex"),
            Path("/usr/bin/xelatex"),
        ]
        
        for base_path in common_paths:
            if base_path.exists():
                if base_path.is_file():
                    return base_path
                # Search for xelatex in subdirectories
                for xelatex_path in base_path.rglob('xelatex'):
                    if xelatex_path.is_file():
                        return xelatex_path
        
        return None
    
    @staticmethod
    def compile_latex(latex_content: str, output_pdf: Path, 
                     title: str = "Document") -> Tuple[bool, Optional[str]]:
        """
        Compile LaTeX content to PDF.
        
        Args:
            latex_content: LaTeX source code
            output_pdf: Path where PDF should be saved
            title: Document title for temp files
            
        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        xelatex = PDFCompiler.find_xelatex()
        if not xelatex:
            return False, "XeLaTeX not found. Please install TeX Live or BasicTeX."
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            tex_file = tmpdir_path / "document.tex"
            
            # Write LaTeX content
            tex_file.write_text(latex_content, encoding='utf-8')
            
            # Compile (run twice for references)
            for pass_num in [1, 2]:
                try:
                    result = subprocess.run(
                        [str(xelatex), '-interaction=nonstopmode', 
                         '-output-directory', str(tmpdir_path), 
                         str(tex_file)],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        cwd=tmpdir_path
                    )
                    
                    if result.returncode != 0 and pass_num == 2:
                        # Extract error from log
                        log_file = tmpdir_path / "document.log"
                        if log_file.exists():
                            log_content = log_file.read_text()
                            error = PDFCompiler._extract_error(log_content)
                            return False, f"LaTeX compilation failed:\n{error}"
                        return False, f"LaTeX compilation failed:\n{result.stderr}"
                
                except subprocess.TimeoutExpired:
                    return False, "LaTeX compilation timed out (60s limit)"
                except Exception as e:
                    return False, f"Compilation error: {str(e)}"
            
            # Copy PDF to output location
            pdf_file = tmpdir_path / "document.pdf"
            if pdf_file.exists():
                output_pdf.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(pdf_file, output_pdf)
                return True, None
            else:
                return False, "PDF was not generated (no error reported)"
    
    @staticmethod
    def _extract_error(log_content: str) -> str:
        """Extract meaningful error from LaTeX log."""
        lines = log_content.split('\n')
        errors = []
        
        for i, line in enumerate(lines):
            if line.startswith('!'):
                # Found an error
                errors.append(line)
                # Get next few lines for context
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip():
                        errors.append(lines[j])
        
        if errors:
            return '\n'.join(errors[:10])  # First 10 lines
        return "Unknown error (check log file)"


def main():
    parser = argparse.ArgumentParser(
        description='Convert Markdown with LaTeX math to PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to PDF
  python scripts/md_to_pdf.py docs/llm/llm_tech_history.md
  
  # Specify output path
  python scripts/md_to_pdf.py docs/llm/llm_tech_history.md -o output/history.pdf
  
  # Generate LaTeX only (no PDF compilation)
  python scripts/md_to_pdf.py docs/llm/llm_tech_history.md --latex-only
  
  # Custom title and author
  python scripts/md_to_pdf.py docs/llm/llm_tech_history.md --title "LLM History" --author "Your Name"
        """
    )
    
    parser.add_argument('input', type=Path, help='Input markdown file')
    parser.add_argument('-o', '--output', type=Path, 
                       help='Output PDF path (default: same name as input with .pdf extension)')
    parser.add_argument('--latex-only', action='store_true',
                       help='Generate LaTeX file only, do not compile to PDF')
    parser.add_argument('--title', type=str, default="",
                       help='Document title')
    parser.add_argument('--author', type=str, default="",
                       help='Document author')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.input.with_suffix('.pdf')
    
    # Read markdown
    print(f"📖 Reading markdown: {args.input}")
    markdown_content = args.input.read_text(encoding='utf-8')
    
    # Convert to LaTeX
    print("🔄 Converting to LaTeX...")
    converter = MarkdownToLatexConverter()
    latex_content = converter.convert(
        markdown_content, 
        title=args.title,
        author=args.author
    )
    
    # Save LaTeX if requested
    if args.latex_only:
        latex_path = args.input.with_suffix('.tex')
        latex_path.write_text(latex_content, encoding='utf-8')
        print(f"✅ LaTeX saved: {latex_path}")
        return
    
    # Compile to PDF
    print("📄 Compiling to PDF...")
    success, error = PDFCompiler.compile_latex(
        latex_content,
        output_path,
        title=args.title or args.input.stem
    )
    
    if success:
        print(f"✅ PDF generated: {output_path}")
        print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    else:
        print(f"❌ PDF generation failed:", file=sys.stderr)
        print(f"   {error}", file=sys.stderr)
        
        # Save LaTeX for debugging
        latex_path = args.input.with_suffix('.tex')
        latex_path.write_text(latex_content, encoding='utf-8')
        print(f"   LaTeX saved for debugging: {latex_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
