# Evaluation Guide - Face Clustering Quality Assessment

## Overview

This guide explains how to use the clustering evaluation system to assess and tune your face clustering parameters using ground truth labels.

**Purpose**: Measure clustering quality with 4 metrics (Purity, NMI, ARI, BCubed F1) to optimize parameters for better results.

---

## Quick Start

### 1. Prepare Ground Truth Dataset

Your ground truth dataset should already exist at `warehouse/labeled_faces/`:

```
warehouse/labeled_faces/
├── Trấn Thành/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── Kaity Nguyễn/
│   ├── image001.jpg
│   └── ...
└── [actor_name]/
    └── ...
```

**Requirements**:
- Each actor has their own folder (folder name = actor name)
- Minimum 10 images per actor recommended
- At least 3 actors for meaningful evaluation
- Images should be face crops similar to those in your clustering results

### 2. Validate Dataset

Before running evaluation, validate your dataset:

```bash
python utils/dataset_validator.py --dataset warehouse/labeled_faces --save-metadata
```

This will:
- Check folder structure
- Validate image files
- Report statistics
- Save metadata.json

### 3. Run Pipeline with Evaluation

Evaluation runs **by default** as Stage 12:

```bash
python flows/pipeline.py --movie your_movie_name
```

**To skip evaluation**:
```bash
python flows/pipeline.py --movie your_movie_name --skip-evaluation
```

### 4. Review Results

After pipeline completes, check `warehouse/evaluation/`:

```
warehouse/evaluation/
├── results.json              # Metrics scores
├── report.md                 # Detailed analysis
├── confusion_matrix.png      # Actor → Cluster heatmap
├── cluster_composition.png   # Cluster → Actor composition (NEW)
└── metrics_chart.png         # Bar chart of all metrics
```

**Console Output**: Metrics are printed with color codes:
- 🟢 Green: Excellent (>0.8)
- 🟡 Yellow: Fair (0.6-0.8)
- 🔴 Red: Poor (<0.6)

---

## Understanding the Metrics

### Purity (0-1, higher is better)
- **What it measures**: Are clusters homogeneous (contain mostly one actor)?
- **Good score**: > 0.85
- **High purity but low recall**: Over-segmentation (one actor split into many clusters)

### NMI - Normalized Mutual Information (0-1, higher is better)
- **What it measures**: Overall clustering quality vs ground truth
- **Good score**: > 0.75
- **Use case**: Standard metric for research papers
- **Advantage**: Not affected by number of clusters

### ARI - Adjusted Rand Index (-1 to 1, higher is better)
- **What it measures**: Pairwise agreement, adjusted for chance
- **Good score**: > 0.70
- **Interpretation**: 
  - 1.0 = perfect
  - 0.0 = random
  - <0 = worse than random

### BCubed F1 (0-1, higher is better)
- **What it measures**: Balance between precision and recall
- **Precision**: How pure are clusters? (no mixing different actors)
- **Recall**: Are same-actor images grouped together? (no splitting)
- **Good score**: > 0.80

---

## Parameter Tuning Based on Metrics

### Scenario 1: High Purity, Low Recall
**Problem**: Actors are split into too many small clusters

**Example**:
```
Purity: 0.92
BCubed Recall: 0.55
```

**Solutions**:
1. Increase `clustering.distance_threshold` (e.g., 0.85 → 0.90)
2. Increase `post_merge.distance_threshold` (e.g., 0.60 → 0.70)
3. Decrease `merge.within_movie_threshold` (e.g., 0.55 → 0.50)

### Scenario 2: High Recall, Low Precision
**Problem**: Different actors are being merged together

**Example**:
```
BCubed Precision: 0.58
BCubed Recall: 0.88
```

**Solutions**:
1. Decrease `clustering.distance_threshold` (e.g., 0.90 → 0.85)
2. Decrease `post_merge.distance_threshold` (e.g., 0.70 → 0.60)
3. Increase `merge.within_movie_threshold` (e.g., 0.50 → 0.55)

### Scenario 3: Low NMI Overall
**Problem**: General quality issues

**Example**:
```
NMI: 0.42
All metrics low
```

**Solutions**:
1. Check `quality_filters.min_score_hard_cutoff` (might be too low, allowing bad embeddings)
2. Review video quality settings (lighting, clarity auto-tuning)
3. Ensure labeled_faces images match video quality
4. Consider adjusting all 3 merge stages

---

## Workflow Example

### Typical Tuning Session

```bash
# 1. Run with current config
python flows/pipeline.py --movie test_video

# Check: NMI = 0.65, Precision = 0.62, Recall = 0.85
# Analysis: Over-merging (recall too high, precision too low)

# 2. Edit configs/config.yaml
# Decrease clustering.distance_threshold: 0.90 → 0.85
# Decrease post_merge.distance_threshold: 0.70 → 0.60

# 3. Re-run
python flows/pipeline.py --movie test_video --skip-ingestion --skip-embedding

# Check: NMI = 0.72, Precision = 0.75, Recall = 0.78
# Analysis: Better balance!

# 4. Fine-tune more if needed
# Adjust merge.within_movie_threshold for final optimization
```

---

## Adding New Actors to Dataset

To expand your ground truth dataset:

```bash
# 1. Create folder with actor name
mkdir "warehouse/labeled_faces/New Actor"

# 2. Add face crop images (10-20 images recommended)
# - Use high-quality face crops
# - Ensure diverse angles and expressions
# - Match the quality of your clustering results

# 3. Validate
python utils/dataset_validator.py --dataset warehouse/labeled_faces

# 4. Run evaluation
python flows/pipeline.py --movie your_movie --skip-ingestion --skip-embedding
```

---

## Reading the Confusion Matrix

The evaluation generates **two complementary matrices**:

### 1. Standard Confusion Matrix (Actor → Cluster)

Rows = Actors (ground truth), Columns = Clusters (predicted)

Shows how test images are distributed across clusters:

```
                C0   C1   C2   C3
Trấn Thành      45   2    0    0  ← Most images correctly in C0
Kaity Nguyễn    1    38   0    0  ← Most in C1, 1 misplaced in C0
Will            0    0    29   1  ← Most in C2
```

**Use case**: Verify test set matching quality

### 2. Cluster Composition Matrix (Cluster → Actor) **NEW**

Rows = Clusters, Columns = Actors + Unknown

Shows what each cluster contains:

```
           Trấn Thành   Kaity   Will   Unknown
C0         145          2       0      8         ← Mostly Trấn Thành + 8 unknowns
C1         3            98      0      12        ← Mostly Kaity + some contamination
C2         0            0       89     45        ← Will + many supporting actors
C3         0            0       0      127       ← All unknown (supporting cast)
```

**Use case**: 
- Detect dirty clusters (mixing multiple actors)
- Identify supporting actors (Unknown column)
- Assess cluster purity directly

**Ideal pattern**: 
- Each row has one dominant number (pure cluster)
- Unknown column shows supporting actors not in test set

**Common issues**:
- Row with multiple large numbers → Dirty cluster (mixed actors)
- High Unknown count → Many supporting actors OR poor test set coverage

---

## Troubleshooting

### No matches found between labeled faces and clustering results

**Cause**: Image filenames don't match

**Solution**: 
- Labeled faces images should have the same filenames as the crop images in your clustering results
- Check `Data/face_crops/[movie]/` to see crop filenames
- Rename labeled faces images to match

### Evaluation shows very low scores (<0.4)

**Possible causes**:
1. Labeled faces quality doesn't match video frames
2. Config parameters are far from optimal
3. Auto-tuning rule is incorrectly applied

**Solutions**:
1. Verify labeled faces are representative
2. Start with baseline config and tune gradually
3. Check `SESSION_CONTEXT.md` for current auto-tuning rules

### Dataset validation fails

**Common issues**:
- Empty folders: Remove or add images
- Invalid image files: Check file corruption
- Wrong folder structure: Each actor needs their own subfolder

---

## Integration with Research

### For Scientific Papers/Posters

When reporting evaluation results:

1. **Include all 4 metrics** in a table:
   ```
   | Metric    | Score |
   |-----------|-------|
   | Purity    | 0.87  |
   | NMI       | 0.78  |
   | ARI       | 0.72  |
   | BCubed F1 | 0.82  |
   ```

2. **Include confusion matrix** visualization

3. **Report** dataset statistics:
   - Number of actors: 6
   - Total images: 127  
   - Matched images for evaluation: 98

4. **Describe parameter tuning** process in methods section

### Example Text for Methods Section

> "Clustering quality was evaluated using a ground truth dataset of 6 Vietnamese celebrities with 127 labeled face images. We computed four metrics: Purity (0.87), Normalized Mutual Information (0.78), Adjusted Rand Index (0.72), and BCubed F1 (0.82). Parameters were iteratively tuned to maximize NMI while maintaining balanced precision and recall."

---

## Configuration Reference

Current evaluation settings in `configs/config.yaml`:

```yaml
evaluation:
  enable: true
  labeled_faces_path: "warehouse/labeled_faces"
  output_dir: "warehouse/evaluation"
  visualizations:
    confusion_matrix: true
    metrics_bar_chart: true
  metrics:
    - purity
    - nmi
    - ari
    - bcubed_f1
```

---

## Tips for Best Results

1. **Dataset Quality**:
   - Use at least 10 images per actor
   - Include diverse poses and expressions
   - Match the quality of your video frames

2. **Tuning Strategy**:
   - Start with NMI as primary metric
   - Use Precision/Recall to diagnose issues
   - Make small incremental changes

3. **Iteration**:
   - Use `--skip-ingestion --skip-embedding` for faster re-runs
   - Document your tuning experiments
   - Compare results across different videos

4. **Validation**:
   - Always validate dataset before evaluation
   - Check confusion matrix visually
   - Compare metrics across multiple runs

---

## Related Documentation

- `docs/EVALUATION_METRICS.md` - Detailed mathematical explanations
- `SESSION_CONTEXT.md` - Current parameter settings and tuning history
- `implementation_plan.md` - Technical implementation details

---

**Last Updated**: 2025-12-16
