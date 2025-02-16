# MIND Training and Evaluation Guide

This guide provides instructions for training and evaluating the MIND model on the Bayer server.

## Training

To train the MIND model with memory usage, use the following command:

```sh
python main.py -c ../config/MIND_2wiki.json --use_memory True
```

- `-c ../config/MIND_2wiki.json`: Specifies the configuration file for the training.
- `--use_memory True`: Enables memory usage during training.

## Evaluation

To evaluate the trained model, first, ensure that you move out of the directory where training was performed. Then, run the following command:

```sh
python src/evaluate_.py --dir result/llama3.1_8b_hotpotqa/2025-01-23_09-17_use_new
```

- `--dir result/llama3.1_8b_hotpotqa/2025-01-23_09-17_use_new`: Specifies the directory containing the results from training.

## Notes

- Ensure that all dependencies and necessary libraries are installed before running the training or evaluation scripts.
- Adjust configurations in `MIND_2wiki.json` as needed to fine-tune training parameters.
- For troubleshooting, check logs and output messages generated during execution.

