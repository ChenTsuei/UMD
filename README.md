# UMD

**If you have any questions about the code, please feel free to create an issue or contact me by email `chentsuei@gmail.com`.**

Official implementation of "User Attention-guided Multimodal Dialog Systems"

**The code is under refactoring and the new code will be published soon.**

## Data

The crawled images can be downloaded [here](https://icloud.qd.sdu.edu.cn:7777/#/link/5019AA633FE05A348F8C1D2E1A4A7087). Other data provided by MMD can be downloaded [here](https://drive.google.com/drive/folders/1JOGHzideeAsmykMUQD3z7aGFg-M4QlE2?usp=sharing).

## Prerequisite

- Python 3.5+
- PyTorch 1.0
- NLTK 3.4
- PIL 5.3.0

## How to run

Please place the data files to the appropriate path and set it in `options/dataset_option.py`, then run `python train <task> <saved_model_file>`.
