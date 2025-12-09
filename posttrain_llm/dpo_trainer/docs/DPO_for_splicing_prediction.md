# DPO for Adaptive Splice Site Prediction: Meta-SpliceAI Use Cases

**Author**: AI Assistant  
**Date**: December 8, 2025  
**Related**: `DPO_for_computational_biology.md`, `DPO_explainer.md`

---

## Table of Contents

1. [Introduction](#introduction)
2. [Why DPO for Splicing?](#why-dpo-for-splicing)
3. [Meta-SpliceAI Architecture Overview](#meta-spliceai-architecture-overview)
4. [Use Case 1: Variant-Induced Splicing Changes](#use-case-1-variant-induced-splicing-changes)
5. [Use Case 2: Disease-State Splicing Alterations](#use-case-2-disease-state-splicing-alterations)
6. [Use Case 3: Treatment-Induced Splicing Effects](#use-case-3-treatment-induced-splicing-effects)
7. [Use Case 4: Stress-Response Splicing](#use-case-4-stress-response-splicing)
8. [Use Case 5: Tissue-Specific Splicing Adaptation](#use-case-5-tissue-specific-splicing-adaptation)
9. [Implementation Strategies](#implementation-strategies)
10. [Modality Considerations](#modality-considerations)
11. [Evaluation Framework](#evaluation-framework)
12. [Future Directions](#future-directions)

---

## Introduction

**Alternative splicing** is a critical regulatory mechanism where a single gene produces multiple mRNA isoforms through differential inclusion/exclusion of exons. This process is highly dynamic and responsive to:

- **Genetic variants** (SNPs, indels, structural variants)
- **Disease states** (cancer, neurodegeneration, autoimmune disorders)
- **Therapeutic interventions** (drugs, antisense oligonucleotides)
- **Environmental stress** (hypoxia, heat shock, oxidative stress)
- **Tissue/cell type** (brain vs. liver vs. immune cells)

**Meta-SpliceAI** is a meta-learning framework that improves upon base splice site prediction models (e.g., SpliceAI, OpenSpliceAI) by learning from their errors. This document explores how **Direct Preference Optimization (DPO)** can enhance Meta-SpliceAI's ability to predict **adaptive splicing changes** induced by external factors.

---

## Why DPO for Splicing?

### Current Limitations of Base Models

| Model | Limitation |
|-------|------------|
| **SpliceAI** | Trained on constitutive splicing; poor at variant effects |
| **OpenSpliceAI** | Static predictions; doesn't model context-dependent changes |
| **Pangolin** | Limited to specific variant types |
| **MMSplice** | Requires pre-computed features; not end-to-end |

### DPO Advantages for Splicing

1. **Preference-based learning**: Learn to prefer correct splice site predictions over base model errors
2. **Context-aware**: Incorporate external factors (variants, disease, treatment) into preferences
3. **Sample efficient**: Work with limited experimental validation data
4. **Interpretable**: Explain why a splice site is preferred in a given context
5. **Multi-model fusion**: Learn optimal weighting of SpliceAI, OpenSpliceAI, and other predictors

---

## Meta-SpliceAI Architecture Overview

Meta-SpliceAI provides **two complementary approaches** for meta-learning:

### 1. Tabular Approach (Legacy)
XGBoost-based with engineered k-mer features—fast but limited scalability for adaptive splicing.

### 2. Multimodal Meta-Layer ⭐ (Current Focus)
Deep learning fusion of DNA sequences + base model scores—more powerful and scalable for context-dependent splicing.

```
┌─────────────────────────────────────────────────────┐
│  Input: Genomic Context                             │
│  - DNA sequence (501 nt window)                     │
│  - Variant information (if applicable)              │
│  - External factors (disease, treatment, etc.)      │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Base Layer                                         │
│  - OpenSpliceAI / SpliceAI predictions              │
│  - Per-nucleotide splice scores                     │
│  - Model-agnostic via genomic_resources             │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Multimodal Meta-Layer (Current)                    │
│  ┌─────────────────────┬─────────────────────────┐ │
│  │ DNA Sequence (501nt)│ Score Features (43)     │ │
│  │        ↓            │         ↓               │ │
│  │  CNNEncoder/        │   ScoreEncoder          │ │
│  │  HyenaDNA           │   (MLP)                 │ │
│  │        ↓            │         ↓               │ │
│  │   [batch, 256]      │   [batch, 256]          │ │
│  └─────────────────────┴─────────────────────────┘ │
│                        ↓                            │
│              Fusion (concat/attention)              │
│                        ↓                            │
│              Classifier → [donor, acceptor, neither]│
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  DPO Enhancement Layer (PROPOSED)                   │
│  - Preference learning over multimodal predictions  │
│  - Context-dependent refinement (variants, disease) │
│  - End-to-end sequence understanding                │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Output: Context-Aware Predictions + Explanations   │
└─────────────────────────────────────────────────────┘
```

**Key Advantages of Multimodal Approach**:
- ✅ **End-to-end learning**: No manual k-mer feature engineering
- ✅ **Long-range context**: CNN/HyenaDNA captures variable splice motifs
- ✅ **Scalability**: Fixed-size embeddings regardless of gene length
- ✅ **Alternative splicing**: Learns complex patterns directly from sequence
- ✅ **GPU acceleration**: MPS (M1 Mac) and CUDA support

---

## Use Case 1: Variant-Induced Splicing Changes

### Problem Statement

Genetic variants can:
- **Create cryptic splice sites** (new AG/GT dinucleotides)
- **Disrupt canonical splice sites** (mutations in consensus sequences)
- **Alter splicing regulatory elements** (ESE/ESS motifs)

Base models often fail to predict these effects accurately.

### DPO Solution

#### Preference Pair Construction

```python
{
    "input": {
        "sequence_ref": "ATGCAG|GTAAGT...",  # Reference allele
        "sequence_alt": "ATGCAG|ATAAGT...",  # Alternate allele (GT→AT)
        "variant": "chr1:12345:G>A",
        "openspliceai_ref": 0.95,
        "openspliceai_alt": 0.12,
        "spliceai_delta": -0.83
    },
    "chosen": "Variant disrupts canonical donor site. Predicted to cause exon skipping. "
              "Validated by RNA-seq: PSI change = -0.78 (p<0.001).",
    "rejected": "Variant has minimal effect on splicing. Donor site remains functional."
}
```

#### Training Data Sources

**Chosen (correct predictions)**:
- ClinVar variants with splicing evidence + RNA-seq validation
- MANE transcripts with experimentally validated variant effects
- Published studies with functional validation (minigene assays)

**Rejected (incorrect predictions)**:
- Base model predictions that contradict experimental data
- Variants predicted to affect splicing but RNA-seq shows no change
- False positives from base models (high score but no functional impact)

### Expected Outcomes

1. **Improved variant effect prediction**: Better accuracy for splice-altering variants
2. **Reduced false positives**: Distinguish true splice-disrupting variants from benign changes
3. **Clinical utility**: Aid in variant interpretation for genetic diagnostics

### Example: BRCA1 Splice Variant

```python
# Real-world example
variant = "NM_007294.3:c.5075-1G>A"  # Canonical splice acceptor

# Base model (SpliceAI)
base_prediction = {
    "acceptor_loss": 0.65,  # Moderate confidence
    "interpretation": "Likely affects splicing"
}

# DPO-enhanced Meta-SpliceAI
dpo_prediction = {
    "acceptor_loss": 0.98,  # High confidence
    "interpretation": "Pathogenic. Disrupts canonical acceptor site, leading to "
                      "exon 17 skipping. Results in frameshift and loss of BRCT domain. "
                      "ClinVar: Pathogenic (4-star). RNA-seq evidence: PSI=0.02 (vs. 0.95 in WT)."
}
```

---

## Use Case 2: Disease-State Splicing Alterations

### Problem Statement

Disease states alter splicing through:
- **Dysregulated splicing factors** (e.g., SRSF1 overexpression in cancer)
- **Epigenetic changes** (altered chromatin accessibility)
- **RNA-binding protein mutations** (e.g., TDP-43 in ALS)

Base models trained on healthy tissue cannot predict these changes.

### DPO Solution (Multimodal Approach)

#### Preference Pair Construction

**Using multimodal meta-layer** (DNA sequence + base scores + context):

```python
{
    "input": {
        "sequence": "ATGCAG|GTAAGT...",  # 501 nt window around exon 10
        "scores": [0.50, 0.50, ...],  # Base model scores (43 features)
        "context": {
            "gene": "MAPT",
            "disease_id": "alzheimers",  # Encoded as integer ID
            "splicing_factors": {"SRSF2": 1.5, "PTBP1": 0.6},  # Fold-change
            "tissue_id": "brain_cortex"
        }
    },
    "chosen": {
        "label": "donor",  # Exon 10 included (4R tau)
        "psi": 0.72,
        "logits": [0.8, 0.1, 0.1]  # [donor, acceptor, neither]
    },
    "rejected": {
        "label": "neither",  # Exon 10 skipped (base model prediction - wrong)
        "psi": 0.50,
        "logits": [0.5, 0.0, 0.5]  # Base model output
    }
}
```

**Key insight**: DPO operates on **logits/probabilities**, not English text. The model learns:
- **Chosen**: High donor probability (0.8) in Alzheimer's context
- **Rejected**: Balanced probability (0.5) ignoring disease state

#### Training Data Sources

**Chosen**:
- Disease-specific RNA-seq datasets (TCGA for cancer, AMP-AD for neurodegeneration)
- Differential splicing analysis (rMATS, MAJIQ) with statistical validation
- Published disease-splicing studies with functional validation

**Rejected**:
- Base model predictions that assume healthy tissue context
- Predictions that ignore disease-specific splicing factor changes

### Example: Cancer-Specific Splicing

```python
# BCL-X splicing in cancer
{
    "input": {
        "gene": "BCL2L1",
        "exon": "exon 2",
        "condition": "Glioblastoma",
        "base_prediction": {"exon2_inclusion": 0.30}  # Healthy brain
    },
    "chosen": "In glioblastoma, BCL-X exon 2 inclusion increases to PSI=0.85, "
              "producing anti-apoptotic BCL-XL isoform. Driven by SRSF1 overexpression. "
              "Correlates with chemotherapy resistance (HR=2.3, p<0.01).",
    "rejected": "BCL-X splicing remains unchanged (PSI=0.30, producing pro-apoptotic BCL-XS)."
}
```

---

## Use Case 3: Treatment-Induced Splicing Effects

### Problem Statement

Therapeutic interventions can alter splicing:
- **Antisense oligonucleotides (ASOs)**: Target specific splice sites
- **Small molecule drugs**: Modulate splicing factors (e.g., splicing modulators)
- **Chemotherapy**: Stress-induced splicing changes

Predicting these effects is critical for drug development and precision medicine.

### DPO Solution (Multimodal Approach)

#### Preference Pair Construction

**Using multimodal meta-layer with treatment context**:

```python
{
    "input": {
        "sequence": "ATGCAG|GTAAGT...",  # 501 nt window around SMN2 exon 7
        "scores": [0.15, 0.85, ...],  # Base model scores (without treatment)
        "context": {
            "gene": "SMN2",
            "treatment_id": "nusinersen",  # Encoded as integer
            "mechanism": "aso_iss_n1_block",  # ASO targeting ISS-N1 silencer
            "tissue_id": "motor_neuron"
        }
    },
    "chosen": {
        "label": "donor",  # Exon 7 included (functional SMN protein)
        "psi": 0.65,
        "logits": [0.75, 0.05, 0.20]  # Post-treatment
    },
    "rejected": {
        "label": "neither",  # Exon 7 excluded (base model - no treatment)
        "psi": 0.15,
        "logits": [0.15, 0.0, 0.85]  # Pre-treatment
    }
}
```

**Key insight**: Model learns treatment effect by comparing:
- **Chosen**: Post-treatment logits (exon included)
- **Rejected**: Pre-treatment logits (exon excluded)

#### Training Data Sources

**Chosen**:
- Clinical trial RNA-seq data (before/after treatment)
- Published ASO studies with splicing validation
- Drug-induced splicing databases

**Rejected**:
- Pre-treatment predictions
- Predictions that ignore drug mechanism of action

### Example: Splicing Modulator Drug

```python
# Risdiplam (SMN2 splicing modulator)
{
    "input": {
        "gene": "SMN2",
        "drug": "Risdiplam",
        "mechanism": "Stabilizes U1 snRNP binding to 5' splice site",
        "base_prediction": {"exon7_inclusion": 0.15}
    },
    "chosen": "Risdiplam increases SMN2 exon 7 inclusion to PSI=0.55 by stabilizing "
              "U1 snRNP at the weak 5' splice site. Oral bioavailability enables "
              "systemic treatment. Phase 3 trial: 41% achieved motor milestones.",
    "rejected": "No change in SMN2 exon 7 splicing."
}
```

---

## Use Case 4: Stress-Response Splicing

### Problem Statement

Environmental stress triggers splicing changes:
- **Hypoxia**: Alters splicing of angiogenesis genes (VEGF, HIF1A)
- **Heat shock**: Induces splicing of stress response genes (HSPs)
- **Oxidative stress**: Affects mitochondrial gene splicing

These are adaptive responses that base models cannot predict.

### DPO Solution

#### Preference Pair Construction

```python
{
    "input": {
        "sequence": "ATGCAG|GTAAGT...",
        "gene": "VEGFA",
        "stress": "Hypoxia (1% O2, 24h)",
        "base_prediction": {"exon8_inclusion": 0.50}  # Normoxia
    },
    "chosen": "Under hypoxia, VEGFA exon 8 inclusion decreases to PSI=0.20, producing "
              "anti-angiogenic VEGF165b isoform. Regulated by SRSF1 phosphorylation. "
              "Validated in endothelial cells (n=3 replicates, p<0.001).",
    "rejected": "VEGFA exon 8 inclusion remains at 50% (balanced pro/anti-angiogenic isoforms)."
}
```

#### Training Data Sources

**Chosen**:
- Stress-response RNA-seq datasets (GEO, ArrayExpress)
- Time-course splicing analysis under stress conditions
- Published stress-splicing studies

**Rejected**:
- Baseline (no stress) predictions
- Predictions that ignore stress-induced signaling pathways

---

## Use Case 5: Tissue-Specific Splicing Adaptation

### Problem Statement

Splice site usage varies dramatically across tissues:
- **Brain**: High expression of neuronal splicing factors (NOVA, RBFOX)
- **Muscle**: Muscle-specific splicing factors (MBNL, CELF)
- **Immune cells**: Activation-dependent splicing changes

Base models trained on mixed tissue data miss these nuances.

### DPO Solution

#### Preference Pair Construction

```python
{
    "input": {
        "sequence": "ATGCAG|GTAAGT...",
        "gene": "NRXN1",
        "tissue": "Brain (cortex)",
        "base_prediction": {"exon20_inclusion": 0.50}  # Average across tissues
    },
    "chosen": "In brain cortex, NRXN1 exon 20 (SS4) inclusion is PSI=0.85 due to "
              "NOVA2 binding at downstream YCAY clusters. This produces synapse-specific "
              "neurexin isoform critical for synaptic plasticity. GTEx validation: "
              "brain-specific (PSI=0.85 vs. 0.12 in other tissues).",
    "rejected": "NRXN1 exon 20 inclusion is 50% across all tissues."
}
```

#### Training Data Sources

**Chosen**:
- GTEx tissue-specific RNA-seq (54 tissues)
- ENCODE tissue-specific splicing maps
- Brain-specific datasets (BrainSpan, PsychENCODE)

**Rejected**:
- Tissue-agnostic predictions
- Predictions that ignore tissue-specific splicing factor expression

---

## Implementation Strategies

### Strategy 1: DPO on Multimodal Meta-Layer ⭐ (Recommended)

**Approach**: Apply DPO to Meta-SpliceAI's multimodal architecture—DNA sequences + base model scores.

This is the **most elegant and scalable** approach for adaptive splice site prediction.

#### Architecture

```python
from meta_spliceai.splice_engine.meta_layer import (
    MetaSpliceModel, MetaLayerConfig, ArtifactLoader
)
import torch
import torch.nn as nn

class DPO_MetaSpliceModel(nn.Module):
    """
    DPO-enhanced multimodal splice predictor
    Combines DNA sequence understanding + base model scores
    """
    def __init__(self, base_meta_model):
        super().__init__()
        # Use existing Meta-SpliceAI multimodal architecture
        self.meta_model = base_meta_model
        
        # Freeze or fine-tune based on strategy
        # Option A: Freeze sequence encoder, fine-tune fusion layer
        # Option B: Fine-tune entire model with DPO
    
    def forward(self, sequence, scores, context=None):
        """
        Args:
            sequence: DNA sequence (501 nt)
            scores: Base model scores (43 features)
            context: Optional context (variant, disease state, etc.)
        
        Returns:
            logits: [batch, 3] (donor, acceptor, neither)
        """
        # Multimodal fusion
        logits = self.meta_model(sequence, scores)
        
        # Optional: Incorporate context for adaptive splicing
        if context is not None:
            logits = self.context_adapter(logits, context)
        
        return logits
```

#### Preference Pair Construction

```python
def build_multimodal_dpo_dataset(base_model='openspliceai'):
    """
    Generate preference pairs from Meta-SpliceAI artifacts
    Uses existing multimodal data pipeline
    """
    from meta_spliceai.splice_engine.meta_layer import ArtifactLoader, MetaLayerConfig
    
    config = MetaLayerConfig(base_model=base_model)
    loader = ArtifactLoader(config)
    
    # Load analysis sequences (DNA + scores + labels)
    df = loader.load_analysis_sequences(chromosomes=['21', '22'])
    
    pairs = []
    
    for idx, row in df.iterrows():
        sequence = row['sequence']  # 501 nt
        scores = row['score_features']  # 43 features
        true_label = row['label']  # Ground truth from MANE
        base_prediction = row['base_prediction']
        
        # Identify base model errors
        is_error = (true_label != base_prediction)
        
        if not is_error:
            continue  # Skip correct predictions
        
        # Construct preference pair
        if true_label == 'donor' and base_prediction == 'neither':
            # False negative: base model missed a donor site
            chosen = {
                "sequence": sequence,
                "scores": scores,
                "label": "donor",
                "explanation": f"True donor site at position {row['position']}. "
                               f"Base model scored {row['donor_score']:.2f} but "
                               f"validated in MANE transcripts."
            }
            rejected = {
                "sequence": sequence,
                "scores": scores,
                "label": "neither",
                "explanation": f"Not a splice site (base model prediction)."
            }
        
        elif true_label == 'neither' and base_prediction in ['donor', 'acceptor']:
            # False positive: base model hallucinated a splice site
            chosen = {
                "sequence": sequence,
                "scores": scores,
                "label": "neither",
                "explanation": f"False positive. High base score ({row['donor_score']:.2f}) "
                               f"but not present in MANE annotations."
            }
            rejected = {
                "sequence": sequence,
                "scores": scores,
                "label": base_prediction,
                "explanation": f"Predicted {base_prediction} site (incorrect)."
            }
        
        pairs.append({"chosen": chosen, "rejected": rejected})
    
    return pairs
```

#### DPO Training

```python
from trl import DPOTrainer, DPOConfig
from torch.utils.data import Dataset

class MultimodalSplicingDataset(Dataset):
    """PyTorch Dataset for DPO training on multimodal splice data"""
    def __init__(self, preference_pairs):
        self.pairs = preference_pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            "chosen_sequence": pair["chosen"]["sequence"],
            "chosen_scores": pair["chosen"]["scores"],
            "chosen_label": pair["chosen"]["label"],
            "rejected_sequence": pair["rejected"]["sequence"],
            "rejected_scores": pair["rejected"]["scores"],
            "rejected_label": pair["rejected"]["label"],
        }

# Load pre-trained Meta-SpliceAI model
base_model = MetaSpliceModel.load_from_checkpoint('models/meta_splice_best.pt')
dpo_model = DPO_MetaSpliceModel(base_model)

# Build preference dataset
preference_pairs = build_multimodal_dpo_dataset(base_model='openspliceai')
dataset = MultimodalSplicingDataset(preference_pairs)

# DPO training configuration
config = DPOConfig(
    beta=0.1,  # Conservative—preserve learned splice patterns
    learning_rate=1e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
)

# Train with DPO
trainer = DPOTrainer(
    model=dpo_model,
    ref_model=None,  # Use implicit reference
    train_dataset=dataset,
    args=config,
)

result = trainer.train()
```

**Advantages**:
- ✅ **End-to-end learning**: Leverages existing multimodal architecture
- ✅ **No feature engineering**: CNN/HyenaDNA learns from raw sequence
- ✅ **Scalable**: Fixed-size embeddings, GPU-accelerated
- ✅ **Context-aware**: Can incorporate variants, disease states
- ✅ **Seamless integration**: Uses Meta-SpliceAI's data pipeline

**Why This Works**:
1. **Multimodal fusion** already combines sequence + scores
2. **DPO refines** the fusion layer to prefer correct predictions
3. **No modality mismatch**: Both inputs are numeric (sequence embeddings + score features)
4. **Preserves pre-training**: DPO fine-tunes without catastrophic forgetting

---

### Strategy 2: Hybrid Architecture (Sequence Encoder + Text Decoder)

**Approach**: Use SpliceBERT for sequence understanding, add text decoder for explanations.

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM

class SpliceBERT_to_Text(nn.Module):
    """
    Multimodal model: DNA sequences → Text explanations
    """
    def __init__(self):
        super().__init__()
        
        # Frozen sequence encoder
        self.sequence_encoder = AutoModel.from_pretrained("chenkenbio/SpliceBERT")
        for param in self.sequence_encoder.parameters():
            param.requires_grad = False
        
        # Projection layer (SpliceBERT dim → GPT2 dim)
        self.projection = nn.Linear(768, 768)
        
        # Text decoder
        self.text_decoder = AutoModelForCausalLM.from_pretrained("gpt2")
    
    def forward(self, sequence_tokens, text_tokens):
        # Encode DNA sequence
        seq_outputs = self.sequence_encoder(sequence_tokens)
        seq_embedding = seq_outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Project to text space
        projected = self.projection(seq_embedding)
        
        # Prepend to text decoder inputs
        text_embeds = self.text_decoder.transformer.wte(text_tokens)
        combined_embeds = torch.cat([projected.unsqueeze(1), text_embeds], dim=1)
        
        # Generate text
        outputs = self.text_decoder(inputs_embeds=combined_embeds)
        return outputs

# Build preference dataset (DNA → English)
def build_multimodal_dpo_dataset():
    pairs = []
    
    for site in candidate_splice_sites:
        sequence = site['sequence']
        is_true = site['in_mane_annotation']
        variant = site.get('variant', None)
        
        # Generate text explanations (rule-based)
        if is_true:
            chosen_text = generate_true_site_explanation(site)
            rejected_text = generate_false_positive_explanation(site)
        else:
            chosen_text = generate_false_positive_explanation(site)
            rejected_text = generate_true_site_explanation(site)
        
        pairs.append({
            "sequence": sequence,
            "chosen": chosen_text,
            "rejected": rejected_text,
        })
    
    return Dataset.from_list(pairs)

# Train with DPO
model = SpliceBERT_to_Text()
trainer = DPOTrainer(model=model, train_dataset=build_multimodal_dpo_dataset(), ...)
trainer.train()
```

**Advantages**:
- ✅ Natural language explanations
- ✅ Leverages pre-trained SpliceBERT
- ✅ Interpretable for clinicians

**Limitations**:
- ❌ Complex architecture
- ❌ Requires training projection layer
- ❌ More compute-intensive

---

### Strategy 3: Two-Stage Pipeline (Separate Models)

**Approach**: Use Meta-SpliceAI (XGBoost) for classification, separate LLM for explanations.

```python
# Stage 1: Meta-SpliceAI classification (existing)
from meta_spliceai.splice_engine.meta_models import MetaModel

meta_model = MetaModel.load("openspliceai_meta_xgboost.pkl")
prediction = meta_model.predict(sequence, base_scores)

# Stage 2: LLM explanation generation (DPO-trained)
from transformers import AutoModelForCausalLM, AutoTokenizer

llm = AutoModelForCausalLM.from_pretrained("gpt2-dpo-splicing")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Convert features to text prompt
prompt = f"""
Sequence: {sequence}
Position: chr{chr}:{pos}
OpenSpliceAI: {base_scores['openspliceai']:.2f}
SpliceAI: {base_scores['spliceai']:.2f}
Meta-Model Prediction: {prediction['score']:.2f}
In MANE: {prediction['in_mane']}
Conservation: {features['phylop']:.2f}

Explain this splice site prediction:
"""

# Generate explanation
inputs = tokenizer(prompt, return_tensors="pt")
outputs = llm.generate(**inputs, max_length=200)
explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(explanation)
# Output: "This is a true donor site (GT dinucleotide) with strong consensus motif..."
```

**Advantages**:
- ✅ Clean separation of concerns
- ✅ Can use existing Meta-SpliceAI models
- ✅ LLM only needs to learn explanations, not classification

**Limitations**:
- ❌ Two models to maintain
- ❌ LLM doesn't directly see sequence data

---

## Modality Considerations

### Multimodal Meta-Layer: No Tokenization Mismatch! ✅

**Key Insight**: The multimodal meta-layer architecture **avoids the DNA-text tokenization problem** entirely.

| Component | Input Type | Representation | Compatibility |
|-----------|------------|----------------|---------------|
| **Sequence Encoder** | DNA (501 nt) | Numeric embedding `[batch, 256]` | ✅ Numeric |
| **Score Encoder** | Base scores (43) | Numeric embedding `[batch, 256]` | ✅ Numeric |
| **Fusion Layer** | Concatenated embeddings | `[batch, 512]` | ✅ Numeric |
| **DPO Training** | Logits `[batch, 3]` | Probability distributions | ✅ Numeric |

**Why This Works**:
1. **DNA sequences** are encoded to numeric embeddings (CNN/HyenaDNA)
2. **Base model scores** are already numeric
3. **DPO operates on logits**, not tokens
4. **No text generation** required for classification task

### Comparison: Tabular vs. Multimodal for DPO

| Aspect | Tabular (XGBoost) | Multimodal Meta-Layer ⭐ |
|--------|-------------------|--------------------------|
| **Feature engineering** | Manual k-mer extraction | End-to-end from raw sequence |
| **Scalability** | Feature matrix grows with gene length | Fixed-size embeddings |
| **Context modeling** | Limited to fixed windows | CNN/HyenaDNA captures long-range patterns |
| **DPO compatibility** | Requires feature → text mapping | Direct logit-based DPO |
| **Adaptive splicing** | Hard to incorporate context | Easy to add context inputs |
| **GPU acceleration** | CPU-bound | GPU-accelerated (MPS/CUDA) |

**Recommendation**: Use **multimodal meta-layer** for DPO-based adaptive splicing prediction.

---

## Evaluation Framework

### Metrics for DPO-Enhanced Meta-SpliceAI

#### 1. Classification Metrics

- **Precision/Recall/F1**: Standard classification metrics
- **AUPRC**: Area under precision-recall curve (better for imbalanced data)
- **AUROC**: Area under ROC curve

#### 2. Variant Effect Prediction

- **Concordance with ClinVar**: % agreement with expert-curated variants
- **RNA-seq validation**: Correlation with experimental PSI changes
- **Minigene assay agreement**: % correct predictions vs. functional assays

#### 3. Context-Specific Metrics

- **Disease-state accuracy**: Correct prediction of disease-altered splicing
- **Treatment effect prediction**: Accuracy for drug/ASO-induced changes
- **Tissue-specificity**: Correlation with GTEx tissue-specific PSI

#### 4. Explanation Quality (for text-generating models)

- **Factual accuracy**: % of statements supported by evidence
- **Citation correctness**: % of cited papers/databases that exist
- **Clinical utility**: Expert rating of explanation usefulness (1-5 scale)

### Benchmark Datasets

| Dataset | Size | Use Case | Ground Truth |
|---------|------|----------|--------------|
| **ClinVar splice variants** | ~50k | Variant effects | Expert curation |
| **GTEx tissue-specific** | ~10M events | Tissue adaptation | RNA-seq |
| **TCGA cancer splicing** | ~1M events | Disease states | RNA-seq |
| **ASO clinical trials** | ~1k events | Treatment effects | Clinical data |
| **MANE transcripts** | ~20k genes | Canonical splicing | Expert curation |

---

## Context-Dependent Adaptive Splicing with DPO

### Key Innovation: Context Adapter Layer

Extend the multimodal meta-layer to incorporate external context for adaptive splicing:

```python
class ContextAdaptiveMetaSplice(nn.Module):
    """
    DPO-enhanced model with context adaptation
    Handles variants, disease states, treatments, stress, tissue
    """
    def __init__(self, base_meta_model, context_dim=64):
        super().__init__()
        self.meta_model = base_meta_model  # Multimodal meta-layer
        
        # Context encoders for different factors
        self.variant_encoder = nn.Embedding(num_variants, context_dim)
        self.disease_encoder = nn.Embedding(num_diseases, context_dim)
        self.treatment_encoder = nn.Embedding(num_treatments, context_dim)
        self.tissue_encoder = nn.Embedding(num_tissues, context_dim)
        
        # Context fusion
        self.context_fusion = nn.Linear(context_dim * 4, 256)
        
        # Adaptive layer
        self.adaptive_layer = nn.Linear(512 + 256, 3)  # Meta output + context
    
    def forward(self, sequence, scores, context):
        """
        Args:
            sequence: DNA sequence (501 nt)
            scores: Base model scores (43 features)
            context: Dict with keys: variant_id, disease_id, treatment_id, tissue_id
        
        Returns:
            logits: [batch, 3] adapted to context
        """
        # Base multimodal prediction
        meta_output = self.meta_model(sequence, scores)  # [batch, 512]
        
        # Encode context
        variant_emb = self.variant_encoder(context['variant_id'])
        disease_emb = self.disease_encoder(context['disease_id'])
        treatment_emb = self.treatment_encoder(context['treatment_id'])
        tissue_emb = self.tissue_encoder(context['tissue_id'])
        
        # Fuse context
        context_emb = torch.cat([variant_emb, disease_emb, treatment_emb, tissue_emb], dim=-1)
        context_emb = self.context_fusion(context_emb)  # [batch, 256]
        
        # Adaptive prediction
        combined = torch.cat([meta_output, context_emb], dim=-1)
        logits = self.adaptive_layer(combined)
        
        return logits
```

### DPO Training for Adaptive Splicing

```python
def build_context_aware_dpo_dataset():
    """
    Generate preference pairs with context information
    """
    pairs = []
    
    # Example: Variant-induced splicing changes
    for variant in clinvar_variants:
        sequence_ref = variant['ref_sequence']
        sequence_alt = variant['alt_sequence']
        scores_ref = get_base_scores(sequence_ref)
        scores_alt = get_base_scores(sequence_alt)
        
        # Ground truth from RNA-seq
        psi_ref = variant['psi_ref']  # 0.95 (exon included)
        psi_alt = variant['psi_alt']  # 0.12 (exon skipped)
        
        if abs(psi_ref - psi_alt) > 0.5:  # Significant change
            chosen = {
                "sequence": sequence_alt,
                "scores": scores_alt,
                "context": {
                    "variant_id": variant['id'],
                    "disease_id": variant['disease'],
                    "treatment_id": 0,  # No treatment
                    "tissue_id": variant['tissue']
                },
                "label": "acceptor_loss",  # Exon skipped
                "psi": psi_alt
            }
            rejected = {
                "sequence": sequence_alt,
                "scores": scores_alt,
                "context": {
                    "variant_id": 0,  # No variant
                    "disease_id": 0,
                    "treatment_id": 0,
                    "tissue_id": variant['tissue']
                },
                "label": "normal",  # Base model prediction (wrong)
                "psi": psi_ref
            }
            pairs.append({"chosen": chosen, "rejected": rejected})
    
    return pairs
```

### Expected Outcomes

| Context | Base Model | DPO-Enhanced Model |
|---------|------------|-------------------|
| **Variant: BRCA1 c.5075-1G>A** | Moderate confidence (0.65) | High confidence (0.98) + "Disrupts canonical acceptor" |
| **Disease: Glioblastoma** | Ignores disease state | Predicts BCL-X isoform switch (PSI 0.30 → 0.85) |
| **Treatment: Nusinersen** | No treatment awareness | Predicts SMN2 exon 7 inclusion (PSI 0.15 → 0.65) |
| **Stress: Hypoxia** | Static prediction | Predicts VEGFA isoform change (PSI 0.50 → 0.20) |
| **Tissue: Brain** | Average across tissues | Brain-specific NRXN1 exon 20 (PSI 0.85 vs. 0.12) |

---

## Future Directions

### 1. Multi-Task DPO with Multimodal Architecture

Train a single model for multiple splicing tasks using the multimodal meta-layer:

```python
# Unified preference dataset
{
    "task": "variant_effect",
    "sequence": "ATGCAG|GTAAGT...",
    "scores": [0.85, 0.12, ...],
    "context": {"variant_id": 123, ...},
    "chosen_label": "donor_loss",
    "rejected_label": "normal"
},
{
    "task": "tissue_specific",
    "sequence": "ATGCAG|GTAAGT...",
    "scores": [0.50, 0.50, ...],
    "context": {"tissue_id": "brain", ...},
    "chosen_label": "donor",
    "rejected_label": "neither"
}
```

### 2. Active Learning with DPO

Iteratively improve model with expert feedback on uncertain predictions:

```
1. Model makes uncertain predictions (e.g., 0.4 < score < 0.6)
2. Expert validates a subset (chosen vs. rejected)
3. Fine-tune with DPO on new preferences
4. Repeat until performance plateaus
```

**Advantage**: Focus expert effort on hard cases where model is uncertain.

### 3. Causal Splicing Models

Learn causal relationships, not just correlations:

```python
{
    "input": {
        "sequence": "...",
        "variant": "c.123A>G disrupts ESE motif"
    },
    "chosen": "Causes exon skipping via loss of SRSF1 binding (causal)",
    "rejected": "Associated with exon skipping (correlation only)"
}
```

### 4. Integration with RNA Structure

Incorporate RNA secondary structure predictions:

```python
class StructureAwareMetaSplice(nn.Module):
    def __init__(self):
        self.sequence_encoder = CNNEncoder()
        self.structure_encoder = StructureEncoder()  # RNAfold predictions
        self.score_encoder = ScoreEncoder()
        self.fusion = MultimodalFusion()
```

### 5. Foundation Model Integration

Combine with genomic foundation models (Nucleotide Transformer, HyenaDNA):

```python
# Use pre-trained foundation model as sequence encoder
from transformers import AutoModel

foundation_model = AutoModel.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m")
meta_model = MetaSpliceModel(sequence_encoder=foundation_model)

# Fine-tune with DPO on splicing-specific preferences
dpo_trainer.train(meta_model, splicing_preferences)
```

---

## Conclusion

DPO offers a powerful framework for enhancing Meta-SpliceAI's **multimodal meta-layer** to predict **adaptive splicing changes** induced by variants, disease states, treatments, stress, and tissue context.

### Key Advantages of DPO + Multimodal Meta-Layer

1. **Preference-based learning**: Learn from base model errors without explicit reward functions
2. **End-to-end sequence understanding**: CNN/HyenaDNA captures splice motifs directly
3. **Context-awareness**: Easily incorporate external factors (variants, disease, treatment)
4. **Scalability**: Fixed-size embeddings, GPU-accelerated training
5. **No modality mismatch**: Operates on numeric embeddings, not incompatible tokenizers
6. **Sample efficiency**: Work with limited experimental validation data

### Why Multimodal > Tabular for Adaptive Splicing

| Requirement | Tabular (XGBoost) | Multimodal Meta-Layer ⭐ |
|-------------|-------------------|--------------------------|
| **Feature engineering** | Manual k-mer extraction | End-to-end from raw sequence |
| **Long-range context** | Fixed window (limited) | CNN/HyenaDNA (flexible) |
| **Context adaptation** | Hard to incorporate | Natural via context embeddings |
| **DPO compatibility** | Requires text mapping | Direct logit-based DPO |
| **Scalability** | Feature explosion | Fixed embeddings |
| **GPU acceleration** | CPU-bound | MPS/CUDA support |

### Recommended Implementation Path

1. **Phase 1**: Train multimodal meta-layer on MANE annotations (baseline)
2. **Phase 2**: Apply DPO to correct base model errors (false positives/negatives)
3. **Phase 3**: Add context adapter for variant-induced splicing (ClinVar validation)
4. **Phase 4**: Extend to disease-state, treatment, stress, tissue-specific splicing
5. **Phase 5**: Multi-task DPO for unified adaptive splicing model

### Integration with Existing Meta-SpliceAI

DPO enhances the **multimodal meta-layer** architecture:

```
Base Models (SpliceAI, OpenSpliceAI)
         ↓
Multimodal Meta-Layer (Current)
├── DNA Sequence (501nt) → CNNEncoder/HyenaDNA → [256]
└── Score Features (43)  → ScoreEncoder (MLP)  → [256]
         ↓
    Fusion Layer → [512]
         ↓
DPO Enhancement (Proposed) ← Preference learning
         ↓
Context Adapter (Optional) ← Variants, disease, treatment, etc.
         ↓
Final Prediction: (donor, acceptor, neither) + Confidence
```

**This approach is:**
- ✅ **More elegant** than tabular feature engineering
- ✅ **More scalable** for whole-genome analysis
- ✅ **More powerful** for adaptive splicing prediction
- ✅ **Production-ready** with Meta-SpliceAI's existing infrastructure

### Next Steps

1. **Implement `DPO_MetaSpliceModel`** wrapper around existing multimodal meta-layer
2. **Generate preference pairs** from Meta-SpliceAI error analysis (false positives/negatives)
3. **Train with DPO** on chromosomes 21-22 (validation set)
4. **Evaluate** on ClinVar splice variants and GTEx tissue-specific splicing
5. **Deploy** for adaptive splice site prediction in clinical/research settings

---

## Implementation Readiness and Practical Considerations

### Current State: Foundation Models Are Not DPO-Ready

**Critical Reality Check**: Most state-of-the-art DNA/RNA sequence-based foundation models are **not designed for DPO** and require substantial additional work to make them DPO-compatible.

#### Foundation Model Landscape

| Model | Pre-training Task | Output Type | DPO-Ready? | Work Required |
|-------|-------------------|-------------|------------|---------------|
| **SpliceBERT** | Masked language modeling | Embeddings (768-dim) | ❌ No | Add classification head + DPO implementation |
| **Nucleotide Transformer** | Masked token prediction | Embeddings (512-dim) | ❌ No | Add task head + custom DPO |
| **HyenaDNA** | Next-token prediction | Embeddings (256-dim) | ❌ No | Add classifier + DPO training loop |
| **DNABERT** | Masked k-mer prediction | Embeddings (768-dim) | ❌ No | Add task head + DPO wrapper |
| **Meta-SpliceAI Multimodal** | Splice site classification | Logits (3-class) | ✅ **Yes** | Minimal—just preference pairs |

### Why Foundation Models Need Significant Work

#### 1. **No Task-Specific Heads**

Foundation models output **generic embeddings**, not splice site predictions:

```python
# What foundation models give you
from transformers import AutoModel

model = AutoModel.from_pretrained("chenkenbio/SpliceBERT")
embeddings = model(sequence_tokens)  # Shape: [batch, seq_len, 768]
# ❌ No splice site predictions!

# What you need for DPO
logits = model(sequence_tokens)  # Shape: [batch, 3] (donor, acceptor, neither)
# ✅ This requires adding a classification head
```

**Required work**:
- Design and add classification head
- Train head on splice site classification task
- Validate performance before applying DPO

#### 2. **No DPO Training Infrastructure**

DPO requires comparing model outputs on chosen vs. rejected examples:

```python
# Standard DPO (from trl library) expects text generation models
from trl import DPOTrainer

# ❌ Doesn't work out-of-the-box for classification tasks
trainer = DPOTrainer(
    model=splicebert_model,  # Not compatible
    train_dataset=preference_pairs,
    ...
)

# ✅ Need custom DPO implementation for classification
def custom_dpo_loss(model, chosen_data, rejected_data, beta=0.1):
    # Custom implementation required
    ...
```

**Required work**:
- Implement custom DPO loss for classification
- Adapt training loop for sequence inputs
- Handle preference pair batching

#### 3. **Data Format Mismatch**

Foundation models use different tokenization schemes:

```python
# SpliceBERT: 6-mer tokenization
"ATGCGTACG" → ["ATGCGT", "TGCGTA", "GCGTAC", "CGTACG"]

# Nucleotide Transformer: Character-level
"ATGCGTACG" → ["A", "T", "G", "C", "G", "T", "A", "C", "G"]

# Meta-SpliceAI: Direct sequence input (no tokenization)
"ATGCGTACG" → CNN/HyenaDNA encoder → embeddings
```

**Required work**:
- Implement tokenization for each model
- Handle variable-length sequences
- Ensure compatibility with DPO training loop

### Recommended Path: Meta-SpliceAI Multimodal Meta-Layer

**Why this is the most practical approach**:

#### ✅ Already DPO-Compatible

```python
# Meta-SpliceAI already outputs classification logits
from meta_spliceai.splice_engine.meta_layer import MetaSpliceModel

model = MetaSpliceModel(sequence_encoder='cnn', num_score_features=43)
logits = model(sequence, scores)  # [batch, 3] - ready for DPO!
```

#### ✅ Minimal Additional Work

**What you need to implement**:
1. **Preference pair generation** (50-100 lines of code)
   ```python
   def generate_preference_pairs(error_analysis_df):
       pairs = []
       for idx, row in error_analysis_df.iterrows():
           if row['is_false_positive']:
               pairs.append({
                   "sequence": row['sequence'],
                   "scores": row['scores'],
                   "chosen_label": 2,  # neither
                   "rejected_label": 0,  # donor (wrong)
               })
       return pairs
   ```

2. **DPO training loop** (100-200 lines of code)
   ```python
   def train_dpo(model, preference_pairs, beta=0.1):
       for batch in dataloader:
           chosen_logits = model(batch['sequence'], batch['scores'])
           rejected_logits = model(batch['sequence'], batch['scores'])
           
           loss = compute_dpo_loss(
               chosen_logits, batch['chosen_label'],
               rejected_logits, batch['rejected_label'],
               beta=beta
           )
           loss.backward()
           optimizer.step()
   ```

3. **Evaluation metrics** (50-100 lines of code)
   - PR-AUC on preference pairs
   - Top-k accuracy improvement
   - False positive/negative reduction

**Total implementation effort**: ~1-2 weeks

#### ✅ Leverages Existing Infrastructure

- ✅ Data pipeline: `ArtifactLoader` provides sequences + scores + labels
- ✅ Model architecture: Multimodal fusion already implemented
- ✅ Training infrastructure: PyTorch training loop exists
- ✅ Evaluation: Metrics and benchmarks already defined

### Alternative: Foundation Model + DPO (High Effort)

If you want to use SpliceBERT or Nucleotide Transformer:

**Estimated effort**: ~2-3 months

**Required components**:
1. **Task-specific head** (1-2 weeks)
   - Design classifier architecture
   - Train on splice site classification
   - Validate performance

2. **Custom DPO implementation** (2-3 weeks)
   - Adapt DPO loss for classification
   - Handle sequence tokenization
   - Implement training loop

3. **Data preprocessing** (1-2 weeks)
   - Tokenize sequences for each model
   - Create preference pairs
   - Handle variable-length inputs

4. **Integration and testing** (2-4 weeks)
   - Integrate with existing pipeline
   - Validate on benchmarks
   - Debug issues

5. **Optimization** (2-3 weeks)
   - Hyperparameter tuning
   - Memory optimization
   - GPU acceleration

### Summary: Implementation Readiness

| Approach | Readiness | Effort | Timeline | Recommended? |
|----------|-----------|--------|----------|--------------|
| **Meta-SpliceAI Multimodal + DPO** | ✅ High | Low | 1-2 weeks | ✅ **Yes** |
| **SpliceBERT + DPO** | ⚠️ Low | High | 2-3 months | ❌ No (unless research goal) |
| **Nucleotide Transformer + DPO** | ⚠️ Low | High | 2-3 months | ❌ No (unless research goal) |
| **HyenaDNA + DPO** | ⚠️ Low | High | 2-3 months | ❌ No (unless research goal) |

### Recommendation

**Start with Meta-SpliceAI's multimodal meta-layer** because:
1. ✅ Already outputs classification logits (DPO-ready)
2. ✅ Existing data pipeline and infrastructure
3. ✅ Minimal additional implementation work
4. ✅ Can demonstrate DPO effectiveness quickly
5. ✅ Production-ready architecture

**Consider foundation models only if**:
- You have 2-3 months for implementation
- You need to publish novel research on foundation model fine-tuning
- You have specific requirements that Meta-SpliceAI doesn't meet

For practical adaptive splice site prediction, **Meta-SpliceAI + DPO is the clear winner**.

---

## References

1. Rafailov et al. (2023). "Direct Preference Optimization"
2. Chen et al. (2024). "SpliceBERT: Self-supervised learning on RNA sequences"
3. Jaganathan et al. (2019). "Predicting Splicing from Primary Sequence with Deep Learning" (SpliceAI)
4. Cheng et al. (2021). "MMSplice: Modular modeling improves the predictions of genetic variant effects on splicing"
5. Vaquero-Garcia et al. (2016). "A new view of transcriptome complexity and regulation through the lens of local splicing variations" (MAJIQ)

---

**Related Documents**:
- `DPO_for_computational_biology.md` - General DPO use cases in biology
- `DPO_explainer.md` - Technical details of DPO algorithm
- `Lesson_5.ipynb` - DPO training example (identity shift)
