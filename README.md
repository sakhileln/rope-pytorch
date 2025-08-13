# PyTorch RoPE
Implementation of Rotary Positional Embeddings (RoPE) — the clever trick behind positional encoding in modern transformers like LLaMA, Qwen, and GPT-NeoX.

### Run locally
```bash
# Clone the repo
git clone https://github.com/sakhileln/rope-pytorch.git
cd rope-pytorch
```

### Repo Structure
```bash
rope-pytorch/
├── README.md              # Intro + explanation
├── rope.py                # Core RoPE functions
├── transformer.py         # Tiny Transformer with RoPE
├── visualize_rope.py      # 2D/3D visualizations of rotation
├── train_demo.py          # Train on toy data (e.g., copy task)
└── requirements.txt
```
### Possible Extras
- Support for NTK scaling (used in LLaMA 2 for longer context).
- **Benchmark**: Compare perplexity with/without RoPE on a small language modeling dataset (like TinyShakespeare).
- A notebook version for Google Colab.

### Sources
- [RoPE: Rotary Positional Embeddings](https://arxiv.org/abs/2006.10029)
- [LLaMA: Pre-training Text Encoders as Discrete Transformers](https://arxiv.org/abs/2006.16236)
- [Qwen: A Simple and Efficient Transformer for Language Modeling](https://arxiv.org/abs/2006.04768)
- [GPT-NeoX: Improving Language Understanding by Generative Pre-Training](https://arxiv.org/abs/2106.03751)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Official Implementation](https://github.com/facebookresearch/rotary-embedding)
- [Reference](https://arxiv.org/abs/2006.10738)

## Contact
- Sakhile L. Ndlazi
- [LinkedIn Profile](https://www.linkedin.com/in/sakhile-)