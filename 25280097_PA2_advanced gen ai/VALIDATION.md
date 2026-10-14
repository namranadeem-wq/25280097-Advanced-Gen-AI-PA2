# Validation and remaining submission actions

Verified while assembling this package:

- Root core and adapter Python syntax trees match the training notebook's implementation cells.
- All standalone Python files parse successfully.
- The 622 judgments have unique IDs and matching SHA-256 digests for their anonymous pair contents and mappings.
- Recomputed outcomes agree with the CSV/JSON evaluation summary.
- Report renders as seven A4 pages and stays within the eight-page limit.
- Large model weights, recovery states and backup archives are excluded.

The conversation records 7 public smoke tests passing in Kaggle before training. The transcribed checks in this package were not rerun during assembly because the authoring environment has no PyTorch. Rerun in Kaggle before submitting:

```bash
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```

The exported notebooks preserve original stage ordering and recovery history. They are not a clean Run All pipeline. Use the README's cell map and stage dependencies. Full training datasets, original persona data/splits and adapter weights are not in this source/results archive; the preparation code and private backup identifiers are provided. Arrange accessible checkpoint/data locations for staff as required.

Before final submission:

1. Review the report, especially the AI assistance disclosure and documented limitations.
2. Run the seven adapter smoke checks in Kaggle and retain the output.
3. Upload the extracted contents to the intended GitHub repository root.
4. Preserve the starter repository's existing tests and metadata when merging; compare any existing test file against this transcribed suite before replacing it.
5. Check that the report link and notebooks open from GitHub and that staff can access required checkpoint locations.

No model retraining or additional judging is required for these actions.
