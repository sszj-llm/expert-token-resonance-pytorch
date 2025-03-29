# expert-token-resonance-pytorch

Unofficial implementation of Expert Token Resonance in PyTorch (and later Triton)

Based on the paper:

### Expert-Token Resonance MoE: Bidirectional Routing with Efficiency Affinity-Driven Active Selection
#### Status:
##### 1 - An initial working version for PyTorch eager. I have finally been able to implement equation 4 properly,
and verified it results in cosine similarity of [-1,1]. (matmuls' didn't want to match earlier).

##### 2 - Next: Planning to use it in llm-base test bed to make sure it trains as expected.
