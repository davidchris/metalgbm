# MetalGBM - Gradient Boosting on Metal

This is a library that implements gradient boosting which can run on Apple Silicon GPU.

I'm missing such a library since XGBoost and LightGBM won't support Apple Silicon GPU.

## Features (Goal)

- feature parity with XGBoost and LightGBM for gradient boosted trees
- Python scikit-learn interface
- uses Metal for GPU acceleration on Apple Silicon
