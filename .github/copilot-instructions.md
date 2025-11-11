# AI Coding Agent Instructions - Assignment 3: N-grams, FastText, and GloVe

## Project Overview
This is an educational assignment exploring NLP methods (FastText and GloVe) for text classification using the AG News dataset. The project is structured as a Jupyter notebook with 10 exercises that build understanding of word embeddings, character n-grams, and text classification.

## Development Environment Setup
- **Python Version**: 3.11 (strict requirement, <3.12)
- **Environment Manager**: uv (preferred over pip/conda)
- **Platform**: Windows with Visual Studio Build Tools required for FastText compilation

### Critical Setup Commands
```bash
# Install VS 2022 Build Tools first (required for FastText)
setx CL "/std:c++17 /D ssize_t=Py_ssize_t"
uv python install
uv venv
uv sync
```

## Data Dependencies (Missing Files)
The notebook expects these files that must be obtained separately:
- `news_data.npz` - AG News dataset (train/test texts and labels)
- `glove.6B.50d.txt` - GloVe word vectors (download from Stanford NLP)
- `important_stuff.pkl` - Seed data for reproducible randomization

## Code Patterns & Architecture

### Exercise Structure Convention
- **Markdown cells** with colored backgrounds indicate exercise sections
- **💻 symbols** mark implementation tasks requiring code completion
- **Green answer boxes** (`<span style="background-color: #00590D">`) for student responses
- Functions often have incomplete implementations marked with `...` or `NotImplementedError`

### Key Function Templates
```python
# Typical incomplete function pattern:
def function_name(params):
    """Detailed docstring with Args and Returns"""
    # Implementation hints in comments
    ...  # Replace with actual implementation
    return result
```

### FastText Integration Patterns
- Supervised training requires `.txt` files with `__label__` prefixes
- Models expect lowercase, alphanumeric-only text preprocessing
- Character vs word models controlled by `minn`/`maxn` parameters (0 = word-only)

### Data Processing Workflow
1. Load AG News data from `.npz` files
2. Convert to FastText format with `txtify_data()` function
3. Train both character and word-based models
4. Compare performance on clean vs corrupted text (dyslexibot)

## Testing & Evaluation Patterns
- Model accuracy testing uses custom `test_fasttext_model()` function
- Text corruption testing with `dyslexibot()` for robustness evaluation
- PCA visualization for embedding space analysis
- Word similarity using cosine distance in embedding space

## Critical Dependencies & Integration Points
- **FastText**: CPU-only library requiring Visual Studio Build Tools
- **GloVe**: Pre-trained vectors loaded as dictionary `{word: vector}` 
- **AG News Dataset**: 4-class classification (Business, Sci/Tech, World, Sports)
- **Scikit-learn**: PCA for dimensionality reduction and visualization

## Student Workflow
1. Complete function implementations marked with `💻`
2. Answer theoretical questions in green response boxes
3. Run cells sequentially to build up model state
4. Experiment with hyperparameters and text corruption levels
5. Generate custom text for final classification exercise

## Debugging Notes
- FastText compilation issues: Ensure Visual Studio Build Tools installed
- Missing data files: Download GloVe vectors and create/obtain dataset files
- Memory issues: GloVe loading is memory-intensive, consider smaller vector files
- Text preprocessing: Functions expect clean, lowercase, alphanumeric text

## Exercise Completion Indicators
- Look for `NotImplementedError` or functions returning `...`
- Incomplete theoretical answers in green boxes
- Comments like "# TODO:" or "# Insert your training loop here"
- Assert statements requiring non-empty implementations

When helping with this codebase, focus on educational value and gradual complexity building rather than production-ready optimizations.