# LLM-Sparsity Experiments

## Models:
- Decoder-Only models: GPT2-small, GPT2-XL (115M params, 1.5B params)
- Encoder-Only models: Electra-small, Electra-large (14M params, 335M params)
- Encoder-Decoder models: T5-small, T5-3B (100M params, 3B params)

## Runtimes:
- GPT2-small, GPT2-XL:  0.04s/example, 0.1s/example
- Electra-small, Electra-large: 0.005s/example, 0.06s/example
- T5-small, T5-3B: 0.05s/example, 0.21s/example

## Method:

These models were evaluated on 10000 examples of the CBT dataset. The process was as follows:
1) Pull example from CBT dataset.
2) Reconfigure example so that each input to model is context + option.
3) Tokenize example.
4) Find output logits and take softmax over each token.
5) Index "option" token and measure softmax entry associated with token (if option was multiple tokens, average SM entry 
was taken)
6) Take argmax over option softmax entries, highest value is network's prediction.
7) Network is correct if prediction matches answer, wrong otherwise.

To run an experiment:
```markdown
python3 main.py --first_and_sec True --large True --model_type e-o --num_examples 10000
```

## Results:

| Model       | Sparsity | Runtime | Evaluation on CBT |
|-------------|----------|---------|-------------------|
| GPT2-small  | 0%       | 410s    | 13%               |
|             | 10%      |         | 13%               |
|             | 50%      |         | 14%               |
|             | 90%      |         | 12%               |
|             | 99%      |         | 10%               |

| Model   | Sparsity | Runtime | Evaluation on CBT |
|---------|----------|---------|-------------------|
| GPT2-XL | 0%       | 1120s   | 19%               |
|         | 10%      |         | 17%               |
|         | 50%      |         | 19%               |
|         | 90%      |         | 12%               |
|         | 99%      |         | 10%               |

| Model         | Sparsity | Runtime | Evaluation on CBT |
|---------------|----------|---------|-------------------|
| Electra-small | 0%       | 110s    | 09%               |
|               | 10%      |         | 11%               |
|               | 50%      |         | 10%               |
|               | 90%      |         | 09%               |
|               | 99%      |         | 09%               |

| Model         | Sparsity | Runtime | Evaluation on CBT |
|---------------|----------|---------|-------------------|
| Electra-large | 0%       | 670s    | 13%               |
|               | 10%      |         | 14%               |
|               | 50%      |         | 13%               |
|               | 90%      |         | 11%               |
|               | 99%      |         | 11%               |

| Model    | Sparsity | Runtime | Evaluation on CBT |
|----------|----------|---------|-------------------|
| T5-small | 0%       | 510s    | 15%               |
|          | 10%      |         | 17%               |
|          | 50%      |         | 16%               |
|          | 90%      |         | 11%               |
|          | 99%      |         | 09%               |

| Model | Sparsity | Runtime | Evaluation on CBT |
|-------|----------|---------|-------------------|
| T5-3B | 0%       | 3400s   | 15%               |
|       | 10%      |         | 17%               |
|       | 50%      |         | 16%               |
|       | 90%      |         | 11%               |
|       | 99%      |         | 09%               |

The models need to be fine-tuned. Without fine-tuning, they do not perform well on these tasks and it is hard to assess 
performance. If there was more time, the models would be fine-tuned for the dataset they are being evaluated on. That 
being said, these were the experiment results.