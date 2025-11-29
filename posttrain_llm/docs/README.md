# Post-Training Documentation

Documentation for the post-training (fine-tuning, alignment, evaluation) components of llm-lab.

## Available Guides

### [Transformers Tutorial](transformers-tutorial.md)

Comprehensive guide to using HuggingFace Transformers library for loading and using pretrained models.

**Topics covered:**
- Model loading from HuggingFace Hub and local paths
- Tokenization (encoding, decoding, padding, truncation)
- Text generation with various sampling strategies
- Quantization (4-bit and 8-bit) for memory efficiency
- Device management (CUDA, MPS, CPU)
- Best practices and common patterns
- Troubleshooting common issues

**Key sections:**
- Core concepts and model hub
- Advanced loading options and caching
- Generation parameters (temperature, top-k, top-p)
- Memory-efficient quantization
- Batch processing and streaming
- Implementation examples from ServeLLM

This tutorial is the foundation for understanding how `serve_llm.py` works and how to use the Transformers library effectively.

## Related Documentation

### In posttrain_llm/M1/

- **[README_DEVICE_SUPPORT.md](../M1/README_DEVICE_SUPPORT.md)** - Quick start guide
- **[DEVICE_USAGE_GUIDE.md](../M1/DEVICE_USAGE_GUIDE.md)** - Detailed usage examples
- **[CHANGELOG.md](../M1/CHANGELOG.md)** - Module changelog and features

### In docs/

- **[LLM Documentation](../../docs/LLM/)** - Technical notes on LLM architectures
- **[Setup Guides](../../docs/setup/)** - Environment and installation guides

## Quick Links

### For Beginners

1. Start with [Transformers Tutorial](transformers-tutorial.md) - Core concepts
2. Read [DEVICE_USAGE_GUIDE.md](../M1/DEVICE_USAGE_GUIDE.md) - Device setup
3. Try [M1 Notebook](../M1/M1_G1_Inspecting_Finetuned_vs_Base_Model.ipynb) - Hands-on practice

### For Developers
1. [Transformers Tutorial](transformers-tutorial.md) - API reference
2. [serve_llm.py](../M1/utils/serve_llm.py) - Implementation
3. [utils.py](../M1/utils/utils.py) - Helper functions

## Contributing

When adding new documentation:

1. Place tutorials and guides in `posttrain_llm/docs/`
2. Place notebook-specific docs in `posttrain_llm/M1/`
3. Update this README with links
4. Follow markdown best practices
5. Include code examples and use cases

## Documentation Structure

```
posttrain_llm/
├── docs/
│   ├── README.md                    ← You are here
│   └── transformers-tutorial.md     ← Transformers guide
├── M1/
│   ├── M1_G1_Inspecting_Finetuned_vs_Base_Model.ipynb
│   ├── utils/
│   │   ├── serve_llm.py            ← Implementation
│   │   └── utils.py                ← Helpers
│   ├── README_DEVICE_SUPPORT.md    ← Quick start
│   ├── DEVICE_USAGE_GUIDE.md       ← Detailed guide
│   └── CHANGELOG.md                ← Features & changelog
└── ...
```

---

**Last Updated**: November 2025  
**Maintained by**: llm-lab project
