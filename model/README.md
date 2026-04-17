# Model

## Weight initialization

```
Embedding.linear.weight                         xavier_uniform_
Embedding.linear.bias                           zeros
Deembedding.linear.weight                       xavier_uniform_
Deembedding.linear.bias                         zeros
TransformerEncoderLayer.linear{1,2}.weight      xavier_uniform_
TransformerEncoderLayer.linear{1,2}.bias        zeros
MultiheadAttention.in_proj_weight               xavier_uniform_
MultiheadAttention.in_proj_bias                 zeros
MultiheadAttention.out_proj.weight              xavier_uniform_
MultiheadAttention.out_proj.bias                zeros
LayerNorm.weight   (all instances)              ones
LayerNorm.bias     (all instances)              zeros
```
