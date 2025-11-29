# LLM Documentation Naming Convention

This guide explains the naming convention for technical documents in the `docs/llm/` directory.

## Philosophy

**Use descriptive, topic-specific names instead of generic names like `summary.md`.**

### Why?

- ✅ Self-documenting - file name tells you what's inside
- ✅ Scalable - easy to add multiple documents per topic
- ✅ Discoverable - clear what each file covers 6 months later
- ✅ Professional - follows academic/technical writing conventions

## Naming Pattern

```
topic-specific-name.md
topic-specific-name.pdf
topic-specific-name.tex
```

**Format:**
- All lowercase
- Hyphens for spaces
- Descriptive and specific
- No abbreviations unless standard (e.g., `rlhf`, `dpo`)

## Examples

### ✅ Good Names

```
docs/llm/training_and_evaluation/
├── rlhf-methodology.md              # Clear: RLHF training methodology
├── dpo-vs-rlhf.md                   # Clear: Comparison of DPO and RLHF
├── reward-modeling.md               # Clear: Reward model training
└── ppo-deep-dive.md                 # Clear: Deep dive into PPO

docs/llm/memory/
├── how-memory-works.md              # Clear: Memory mechanisms overview
├── attention-mechanisms.md          # Clear: Attention variants
└── kv-cache-optimization.md         # Clear: KV cache techniques

docs/llm/architectures/
├── transformer-evolution.md         # Clear: Transformer history
├── state-space-models.md            # Clear: SSM overview
└── hybrid-architectures.md          # Clear: Hybrid models
```

### ❌ Bad Names

```
docs/llm/training_and_evaluation/
├── summary.md                       # Too generic
├── notes.md                         # Too vague
├── doc1.md                          # Meaningless
└── training.md                      # Too broad

docs/llm/memory/
├── memory.md                        # Too generic
├── stuff.md                         # Useless
└── temp.md                          # Temporary?
```

## Directory Structure

Each topic directory should contain:

```
topic_name/
├── README.md                        # Index of all documents in this topic
├── primary-document.md              # Main comprehensive document
├── primary-document.pdf
├── primary-document.tex
├── specific-subtopic-1.md           # Additional focused documents
├── specific-subtopic-2.md
└── figures/                         # Optional: images, diagrams
    ├── diagram-1.png
    └── diagram-2.png
```

## Current Structure

### Existing Documents

```
docs/llm/
├── README.md                                    # Index of all LLM content
├── llm_tech_evolution/
│   ├── llm_tech_history.md                     # ✅ Good: Descriptive
│   ├── llm_tech_history.pdf
│   └── llm_tech_history.tex
├── memory/
│   ├── how_memory_works                        # Notes/outline
│   ├── how_memory_works.md                     # ✅ Good: Descriptive
│   ├── how_memory_works.pdf
│   └── how_memory_works.tex
└── training_and_evaluation/
    ├── rlhf-methodology.md                     # ✅ Good: Specific
    ├── rlhf-methodology.pdf
    ├── rlhf-methodology.tex
    └── summary.md                              # ❌ Deprecated: Too generic
```

## Migration Strategy

When you have a generic name like `summary.md`:

### Option 1: Rename (Recommended)

```bash
# Rename to descriptive name
mv summary.md rlhf-methodology.md

# Regenerate PDF with new name
python3 scripts/md_to_pdf.py rlhf-methodology.md

# Update references in README
```

### Option 2: Keep as Notes

```bash
# Keep original as notes/outline
mv summary.md notes.md

# Create new properly-named document
# notes.md becomes your working draft
```

## Adding New Documents

### Step 1: Choose a Descriptive Name

Ask yourself:
- What specific topic does this cover?
- How would I search for this 6 months from now?
- Is this name unique within this directory?

### Step 2: Create the Document

```bash
# Create markdown file
vim docs/llm/topic_name/descriptive-name.md

# Write content following the template
```

### Step 3: Generate PDF

```bash
# Generate PDF
python3 scripts/md_to_pdf.py docs/llm/topic_name/descriptive-name.md \
    --title "Full Document Title" \
    --author "LLM Lab"
```

### Step 4: Update README

Add entry to `docs/llm/README.md` and `docs/llm/topic_name/README.md` (if it exists).

## Document Templates

### Comprehensive Document Template

For main topic documents like `rlhf-methodology.md`:

```markdown
# Document Title

Brief introduction explaining what this document covers.

**Key insight**: One-sentence summary of the main point.

---

## 1. Introduction

### 1.1 Background
### 1.2 Motivation

## 2. Core Concepts

### 2.1 Concept 1
### 2.2 Concept 2

## 3. Technical Details

### 3.1 Mathematical Formulation
### 3.2 Implementation

## 4. Practical Implications

### 4.1 For Practitioners
### 4.2 For Researchers

## 5. Comparison with Alternatives

## 6. Future Directions

## 7. Conclusion

## References

---

**Last updated**: Month Year
```

### Focused Document Template

For specific subtopics like `ppo-deep-dive.md`:

```markdown
# Specific Topic Title

Brief introduction.

---

## 1. Overview

## 2. Deep Dive

### 2.1 Detail 1
### 2.2 Detail 2

## 3. Examples

## 4. Best Practices

## References

---

**Last updated**: Month Year
```

## README Files

Each topic directory should have a `README.md`:

```markdown
# Topic Name

Brief description of this topic area.

## Documents

### [Primary Document Name](primary-document.md)

**Description**: What this document covers.

**Topics:**
- Topic 1
- Topic 2

**Files:**
- [`primary-document.md`](primary-document.md)
- [`primary-document.pdf`](primary-document.pdf)

### [Subtopic Document](subtopic-document.md)

**Description**: Focused coverage of specific aspect.

**Files:**
- [`subtopic-document.md`](subtopic-document.md)
- [`subtopic-document.pdf`](subtopic-document.pdf)

## Future Topics

Planned additions:
- [ ] Topic A
- [ ] Topic B

---

**Last updated**: Month Year
```

## Best Practices

### 1. Be Specific

```
❌ training.md
✅ rlhf-methodology.md

❌ attention.md
✅ attention-mechanisms-comparison.md

❌ models.md
✅ transformer-variants-survey.md
```

### 2. Use Standard Abbreviations

Common abbreviations are OK:
- `rlhf` - Reinforcement Learning from Human Feedback
- `dpo` - Direct Preference Optimization
- `ppo` - Proximal Policy Optimization
- `llm` - Large Language Model
- `ssm` - State Space Model

### 3. Indicate Document Type

For special document types:
- `topic-comparison.md` - Comparison of approaches
- `topic-tutorial.md` - Step-by-step guide
- `topic-survey.md` - Literature survey
- `topic-deep-dive.md` - Detailed analysis

### 4. Version Control

If you need versions:
```
topic-name.md              # Current version
topic-name-v1.md           # Archived version 1
topic-name-v2.md           # Archived version 2
```

Or use git tags/branches instead of file versions.

## Quick Reference

### Checklist for New Documents

- [ ] Name is descriptive and specific
- [ ] Name uses lowercase with hyphens
- [ ] Name is unique in the directory
- [ ] Document follows template structure
- [ ] PDF generated with proper title
- [ ] README updated with new document
- [ ] Links work correctly

### Common Patterns

| Topic | Naming Pattern | Example |
|-------|----------------|---------|
| Methodology | `method-name-methodology.md` | `rlhf-methodology.md` |
| Comparison | `topic1-vs-topic2.md` | `dpo-vs-rlhf.md` |
| Survey | `topic-survey.md` | `attention-mechanisms-survey.md` |
| Tutorial | `topic-tutorial.md` | `fine-tuning-tutorial.md` |
| Deep Dive | `topic-deep-dive.md` | `ppo-deep-dive.md` |
| Overview | `topic-overview.md` | `llm-architectures-overview.md` |

---

## Questions?

- See [`docs/workflows/document-workflow.md`](../workflows/document-workflow.md) for document creation process
- See [`docs/workflows/markdown-to-pdf-workflow.md`](../workflows/markdown-to-pdf-workflow.md) for PDF generation
- See [`docs/llm/README.md`](README.md) for current document index

---

**Last updated**: November 2025
