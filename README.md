# Words or Vision

**Words or Vision: Do Vision-Language Models Have Blind Faith in Text?** [[Paper](https://arxiv.org/abs/2402.15300)]

[Ailin Deng](https://d-ailin.github.io), [Tri Cao](https://caothientri2001vn.github.io/), [Zhirui Chen](https://zchen42.github.io/), [Bryan Hooi](https://bhooi.github.io/)

CVPR 2025

--------

## Data

The evaluation data is available on [Hugging Face](https://huggingface.co/datasets/dal-289/word_or_vision).

## Code

We provide the development code under the `raw_code` directory, which contains raw data and development scripts.  
We will provide the evaluation code to process the Hugging Face dataset directly in a later update.

### Key Files and Steps

- **Run test and generate output**
  ```bash
  bash scripts/eval_vqa.sh
    ```
- **Get performance and visualize analysis**

    Open and run the following notebook:
    `raw_code/notebooks/performance_models.ipynb`
 


