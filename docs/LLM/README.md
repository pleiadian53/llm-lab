# LLM Research Documentation

This directory contains technical research notes and documentation on large language models, their architectures, and underlying mechanisms.

## 📁 Contents

### [LLM Tech Evolution](llm_tech_evolution/)

**Evolution of LLM Architectures: From Transformers to Hybrid Models**

Comprehensive technical document tracing the evolution of LLM architectures from the original Transformer (2017) through modern hybrid systems (2024-2025).

**Topics covered:**
- Original Transformer architecture and attention mechanism
- Efficient/long-context Transformers (Longformer, BigBird, Performer)
- State Space Models (S4, structured state spaces)
- Selective SSMs (Mamba, Mamba-2)
- Retention and hybrid models (RetNet, Hyena, RWKV)
- Modern hybrids (Jamba, StripedHyena-2)

**Files:**
- [`llm_tech_history.md`](llm_tech_evolution/llm_tech_history.md) - Source markdown
- [`llm_tech_history.pdf`](llm_tech_evolution/llm_tech_history.pdf) - PDF version
- [`llm_tech_history.tex`](llm_tech_evolution/llm_tech_history.tex) - LaTeX source

---

### [Memory Mechanisms](memory/)

**Memory Mechanisms in Neural Sequence Models: From RNNs to State-of-the-Art Architectures**

Deep dive into how different neural architectures represent, store, update, and propagate memory across sequences.

**Topics covered:**
- RNNs and the vanishing gradient problem
- LSTM/GRU gated memory mechanisms
- CNNs and receptive fields as implicit memory
- Transformers: Attention as content-addressable memory
- KV cache and memory optimization techniques
- State Space Models (S4): Continuous-time memory
- Selective SSMs (Mamba): Input-dependent memory
- Memory-augmented Transformers (2024-2025)
- Retrieval-Augmented Generation (RAG)
- Hybrid memory architectures

**Files:**
- [`how_memory_works`](memory/how_memory_works) - Raw notes/outline
- [`how_memory_works.md`](memory/how_memory_works.md) - Source markdown
- [`how_memory_works.pdf`](memory/how_memory_works.pdf) - PDF version
- [`how_memory_works.tex`](memory/how_memory_works.tex) - LaTeX source

---

### [Training and Evaluation](training_and_evaluation/)

**RLHF/RLAIF Training and Evaluation Methodology**

Comprehensive explanation of how reinforcement learning from human feedback (RLHF) differs from traditional supervised learning in training and evaluation.

**Topics covered:**
- Why RLHF doesn't use traditional validation splits
- Reward model training and evaluation
- Off-policy evaluation in RL
- Hyperparameter tuning without validation sets
- Avoiding reward hacking and overfitting
- Comparison: SFT vs RM training vs RL training
- The complete RLHF pipeline with mathematical formulations
- Alternative approaches (DPO, GRPO, Constitutional AI)

**Files:**
- [`rlhf-methodology.md`](training_and_evaluation/rlhf-methodology.md) - Complete RLHF methodology
- [`rlhf-methodology.pdf`](training_and_evaluation/rlhf-methodology.pdf) - PDF version
- [`summary.md`](training_and_evaluation/summary.md) - Original notes (deprecated)

---

## 🔄 Generating PDFs

All markdown documents can be converted to professional PDFs using the conversion script:

```bash
# From llm-lab root directory
python3 scripts/md_to_pdf.py docs/llm/path/to/document.md \
    --title "Document Title" \
    --author "Author Name"
```

See the [Markdown to PDF Workflow](../workflows/markdown-to-pdf-workflow.md) for detailed instructions.

---

## 📝 Document Format

All technical documents follow this structure:

1. **Mathematical rigor**: Equations properly formatted with LaTeX
2. **Conceptual clarity**: Intuitive explanations alongside formal definitions
3. **Historical context**: Evolution and motivation for each approach
4. **Comparative analysis**: Trade-offs and performance characteristics
5. **Practical guidance**: Implementation tips and use cases
6. **References**: Citations to original papers and resources

---

## 🎯 Future Topics

Planned additions to this directory:

- **Attention mechanisms**: Deep dive into various attention variants
- **Tokenization**: BPE, WordPiece, SentencePiece, and modern approaches
- **Training techniques**: Pre-training, fine-tuning, RLHF, DPO
- **Scaling laws**: Chinchilla scaling, compute-optimal training
- **Inference optimization**: Quantization, pruning, distillation
- **Multi-modal models**: Vision-language models, unified architectures

---

## 📚 Related Resources

- [Document Workflow Guide](../workflows/document-workflow.md) - How to create and format technical documents
- [LaTeX Setup](../setup/latex-setup.md) - Install LaTeX for PDF generation
- [Quick Start](../quick-start.md) - Get started with llm-lab

---

**Last updated**: November 2025
