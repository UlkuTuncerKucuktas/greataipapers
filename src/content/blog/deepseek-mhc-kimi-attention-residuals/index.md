---
title: 'DeepSeek mHC and Kimi Attention Residuals'
description: 'Two papers, one problem: information dilutes as it flows through depth. DeepSeek widens the residual stream into constrained parallel channels. Kimi replaces uniform accumulation with learned attention over previous layers.'
pubDate: 'Apr 12 2026'
---

In 2015, Kaiming He and colleagues at Microsoft Research tried to train very deep convolutional networks for image recognition. Networks with 20 layers worked. Networks with 56 layers performed worse than their 20-layer counterparts, not because they overfit, but because they failed to train at all.

![Training error comparison](/blog/deepseek-mhc-kimi-attention-residuals/12.png)

The training error itself was higher for the deeper network, which ruled out overfitting as the explanation. The problem was optimization. Gradients that carry learning signal from the loss back through the network were passing through dozens of nonlinear transformations, shrinking at each one. By the time they reached the early layers, they were effectively zero. The network could not update its own weights.

![Residual connection](/blog/deepseek-mhc-kimi-attention-residuals/5.png)

Their fix was the residual connection. Instead of asking each layer to learn the full mapping from input to output, they asked it to learn only the *difference* from its input:

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l)
$$

The input $\mathbf{x}_l$ passes through unchanged via the skip connection, and the layer only needs to learn a small correction $\mathcal{F}(\mathbf{x}_l)$. If a layer has nothing useful to contribute, it can learn $\mathcal{F} \approx 0$ and pass information through without distortion. The gradient benefit is direct: the derivative of $\mathbf{x}_{l+1}$ with respect to $\mathbf{x}_l$ always contains an identity term, so the gradient flows regardless of what $\mathcal{F}$ does. ResNet trained networks with 152 layers, then 1,000 layers.

Around the same time, Sergey Ioffe and Christian Szegedy introduced Batch Normalization to address internal covariate shift, where the statistics of a layer's input change as earlier layers update during training. BatchNorm normalizes each feature *across the batch* to zero mean and unit variance. It worked well for convolutional networks with large, fixed-size batches, but it depended on batch statistics, which made it awkward for variable-length sequences and small batches. Jimmy Lei Ba, Jamie Kiros, and Geoffrey Hinton proposed Layer Normalization in 2016 as an alternative: normalize across the *feature dimension* for each individual example, independent of batch size or sequence length. LayerNorm became the default for recurrent networks and later for transformers.

When Vaswani et al. published "Attention Is All You Need" in 2017, they used LayerNorm with residual connections:

$$
\mathbf{x}_{l+1} = \text{LayerNorm}(\mathbf{x}_l + \mathcal{F}(\mathbf{x}_l))
$$

![PostNorm vs PreNorm](/blog/deepseek-mhc-kimi-attention-residuals/6.png)

The normalization sits *after* the residual addition. This is PostNorm. It controls signal magnitude at every layer. The hidden state gets rescaled back to a stable range after each addition, so nothing grows unboundedly regardless of depth. But gradients must pass through a normalization at every layer during backpropagation, and each one distorts them slightly. Stack sixty of these and the gradient reaching layer one has been warped by sixty consecutive rescalings. Training deep PostNorm transformers required careful warmup schedules, small learning rates, and some runs diverged anyway.

Within a few years, practitioners converged on a different placement. Move the normalization *before* the sublayer:

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\text{LayerNorm}(\mathbf{x}_l))
$$

This is PreNorm. The residual stream now has a clean identity path. $\mathbf{x}_l$ passes through the addition without touching any normalization. Gradients flow through this path unmodified, regardless of depth. Training becomes stable almost for free: no warmup needed, larger learning rates work, fewer runs diverge. PreNorm became the default for every large language model.

But PreNorm trades one problem for a quieter one.

![PreNorm dilution](/blog/deepseek-mhc-kimi-attention-residuals/9.png)

With PostNorm, the normalization wraps around the sum, so it controls the magnitude of the hidden state directly. With PreNorm, the normalization is *inside* the sublayer. Nothing controls the sum itself. And the sum just keeps growing. Every layer adds its output with a weight of one. After ten layers, the hidden state is the embedding plus ten layer outputs. After sixty layers, it is the embedding plus sixty layer outputs. The magnitude increases with depth.

This would be fine if each layer's contribution remained meaningful. But it does not. The normalization inside $\mathcal{F}$ divides by the magnitude of the hidden state before processing it. At layer 60, that magnitude is large, having accumulated sixty terms. So the layer sees a normalized, unit-scale input, does its computation, and produces output of some fixed magnitude. That output then gets added to a sum that is already sixty times larger. Layer 60's contribution to the total representation is roughly one sixtieth. Its cosine similarity with the hidden state before it ran is above 0.999. It barely moved anything.

This cascades. If a layer's contribution is negligible relative to the accumulated sum, then the gradient signal flowing back to that layer's parameters is also negligible. The layer cannot learn effectively. Its output stays small. Which means it contributes even less. Researchers confirmed this empirically. You can delete the last half of the layers from a trained LLM and performance barely changes. The model has sixty layers of parameters, but only the first thirty contribute meaningfully.

This is sometimes called representation collapse, sometimes PreNorm dilution. PostNorm does not have this problem, because it rescales after every addition. But PostNorm has the gradient problem instead. For a decade the field picked PreNorm's side of this tradeoff, because training stability was the more pressing concern, and the dilution went largely unnoticed.

In early 2026, DeepSeek published *Manifold-Constrained Hyper-Connections* and Kimi published *Attention Residuals*. Both papers target PreNorm dilution without going back to PostNorm. DeepSeek's approach widens the residual stream into multiple parallel streams and constrains how they mix. Kimi's approach keeps a single stream but lets each layer attend over all previous layer outputs instead of accumulating them uniformly.

---

The standard residual connection carries a single vector of dimension $d$ through the entire network. Every layer reads from and writes to this one vector. If the model has a hidden dimension of 4096, then all information at every layer must be packed into those 4096 numbers. Two layers that need to preserve different information compete for space in the same vector.

ByteDance's Hyper-Connections paper proposed a direct fix: expand the single vector into $n$ parallel copies. Instead of one stream of dimension $d$, carry $n$ streams of dimension $d$, typically $n = 4$. The state at each layer is no longer a vector $\mathbf{x}_l \in \mathbb{R}^d$ but a matrix $\mathbf{x}_l \in \mathbb{R}^{n \times d}$, where each row is a separate stream.

The transformer layer itself, the attention head or the MLP, still operates on a single $d$-dimensional vector. It does not change. The Hyper-Connections machinery sits around it, handling the translation between the multi-stream residual and the single-stream layer. Three learned matrices control this translation.

![Hyper-Connections architecture](/blog/deepseek-mhc-kimi-attention-residuals/1.png)

$\mathcal{H}^{\text{pre}}$ is a vector of $n$ weights that controls how the streams are combined into a single input for the layer. If $\mathcal{H}^{\text{pre}} = [0.5, 0.3, 0.15, 0.05]$, the layer receives 50% of stream one, 30% of stream two, 15% of stream three, and 5% of stream four, mixed into one vector of dimension $d$. This is the input the attention head or MLP actually sees. Different layers can learn different mixing weights, so one layer might read mostly from stream one while another reads mostly from stream three.

The layer processes this combined input and produces a single output vector $\mathbf{h}^{\text{out}} \in \mathbb{R}^d$.

$\mathcal{H}^{\text{post}}$ is another vector of $n$ weights that controls how this output gets distributed back to the streams. If $\mathcal{H}^{\text{post}} = [1.2, 0.8, 0.6, 0.4]$, stream one receives the layer output scaled by 1.2, stream two receives it scaled by 0.8, and so on. This is how the layer writes back to the multi-stream residual.

$\mathcal{H}^{\text{res}}$ is an $n \times n$ matrix that mixes the streams independently of the layer. This is the residual path. While $\mathcal{H}^{\text{pre}}$ and $\mathcal{H}^{\text{post}}$ handle the interaction between the streams and the layer, $\mathcal{H}^{\text{res}}$ handles the interaction between the streams and each other. Entry $(i, j)$ of this matrix says how much of stream $j$'s content gets mixed into stream $i$ at this layer.

The full update is:

$$
\mathbf{x}_{l+1} = \underbrace{\mathcal{H}^{\text{res}}_l \, \mathbf{x}_l}_{\text{streams mix with each other}} \;+\; \underbrace{(\mathcal{H}^{\text{post}}_l)^\top \, \mathcal{F}(\mathcal{H}^{\text{pre}}_l \, \mathbf{x}_l)}_{\text{layer reads from and writes to streams}}
$$

The first term takes the $n$ input streams and remixes them through $\mathcal{H}^{\text{res}}$. The second term takes the combined input ($\mathcal{H}^{\text{pre}}$ applied to the streams), runs it through the layer $\mathcal{F}$, and distributes the result back to the streams via $\mathcal{H}^{\text{post}}$. The two terms are added together to produce the $n$ streams for the next layer.

Compare this with the standard residual, where $n = 1$ and all three matrices are the scalar 1:

$$
\mathbf{x}_{l+1} = 1 \cdot \mathbf{x}_l + 1 \cdot \mathcal{F}(1 \cdot \mathbf{x}_l) = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l)
$$

Hyper-Connections are a strict generalization. Set $n = 1$ and you recover the standard residual exactly.

Hyper-Connections showed real improvements in small-scale experiments. The wider residual stream reduced representation collapse. Features across layers became more diverse, with lower cosine similarity between consecutive hidden states. But when DeepSeek tried to scale HC to a 27-billion parameter model, training broke.

The problem is $\mathcal{H}^{\text{res}}$. In HC, this matrix is unconstrained. Its entries can take any value, positive or negative. At each layer, the streams are multiplied by this matrix. After $L$ layers, the effective transformation is the product of all $L$ mixing matrices:

$$
\prod_{i=1}^{L} \mathcal{H}^{\text{res}}_{L-i}
$$

If any individual matrix has eigenvalues with magnitude greater than one, the product grows exponentially with depth. An eigenvalue of 1.15 at each layer gives $1.15^{60} \approx 3000\times$ amplification after 60 layers. DeepSeek measured signal gains exceeding $3000\times$ in their 27B model, and training diverged with a loss spike around step 12,000.

![Signal gain comparison](/blog/deepseek-mhc-kimi-attention-residuals/3.png)

The right panel shows this directly. The composite mapping gain, the product of all mixing matrices from layer 1 to layer $l$, reaches $10^3$ to $10^4$ for HC. Individual layers may each have small amplification, but sixty small amplifications multiply into a large one.

DeepSeek's fix is to constrain $\mathcal{H}^{\text{res}}$ so that this cannot happen. They force each mixing matrix to be *doubly stochastic*: all entries are non-negative, every row sums to one, and every column sums to one. A doubly stochastic matrix cannot amplify signals. Its largest eigenvalue is exactly one, and all others have magnitude at most one. Multiply any number of doubly stochastic matrices together and the result is still doubly stochastic. The product can never explode, regardless of depth.

What does this mean for the streams? Every row summing to one means each output stream receives a weighted average of the input streams, with weights that sum to one. No stream can grow by pulling in more total signal than exists. Every column summing to one means each input stream's content is fully distributed across the output streams, with nothing created or destroyed. Signal is redistributed, not amplified.

The constraint is enforced by the Sinkhorn-Knopp algorithm, a procedure from 1967. Start with the learned parameters, exponentiate them to ensure positivity, then alternate: divide each row by its row sum, divide each column by its column sum. After about twenty iterations, the matrix is doubly stochastic to numerical precision. The operation is differentiable, so gradients flow through it during training.

![Birkhoff polytope and Sinkhorn-Knopp](/blog/deepseek-mhc-kimi-attention-residuals/7.png)

The set of all doubly stochastic matrices forms a convex region called the Birkhoff polytope. Its corners are permutation matrices, which rearrange streams without mixing them. Every point inside the polytope is a weighted average of permutations. The four panels above show the Sinkhorn-Knopp projection at work: the top-left starts with an unconstrained matrix (high row and column sum error), and across iterations the point is pulled onto the polytope surface (bottom-right, errors near zero). The learned mixing can range from near-identity to near-permutation to uniform blending, but it cannot leave the polytope, and nothing inside the polytope amplifies signal.

$\mathcal{H}^{\text{pre}}$ and $\mathcal{H}^{\text{post}}$ are also constrained, but more simply: their entries are passed through a sigmoid, which keeps them between 0 and 1 for $\mathcal{H}^{\text{pre}}$ and between 0 and 2 for $\mathcal{H}^{\text{post}}$. This prevents sign cancellation. Negative entries in the pre- or post-projection could cause streams to cancel each other out, destabilizing training at scale.

This is the difference between panel (b) and panel (c) in the architecture figure. The structure is identical, same three matrices, same data flow. The only change is that the matrices are projected onto constrained manifolds. The orange boxes in (b) become green boxes in (c).

![Learned mixing matrices](/blog/deepseek-mhc-kimi-attention-residuals/4.png)

The effect is visible in the learned matrices themselves. The top row shows HC: individual mixing matrices have entries ranging from $-21$ to $+22$, and the composite over 60 layers has entries in the hundreds. The bottom row shows mHC: individual matrices have entries between 0 and 1, rows and columns sum to approximately 1, and the composite over 60 layers converges toward a uniform matrix with entries near $0.25$.

![Training curves](/blog/deepseek-mhc-kimi-attention-residuals/2.png)

The training curves confirm this. On a 27B parameter model, HC (blue) shows persistent loss instability and erratic gradient norms. mHC (gray) trains smoothly, with gradient norms that settle to a stable level. The final loss for mHC is 0.021 lower than the baseline. Across benchmarks, mHC outperforms both the baseline and unconstrained HC on every evaluation. The training overhead is about 6.7%, mostly from the wider residual stream's memory access cost, since the model now reads and writes $4\times$ as many values at each residual step.

---

Kimi's response to PreNorm dilution starts from a different angle. Where DeepSeek saw a stream that was too narrow, Kimi saw an accumulation that was too uniform. The hidden state at layer $L$ is the sum of all previous outputs, each weighted at one. There is no mechanism for a later layer to decide that some earlier outputs matter more than others.

The Kimi team drew an analogy to a problem that was solved in 2017. Before transformers, recurrent neural networks processed sequences by compressing all previous tokens into a single fixed-size vector, updating it one step at a time. Early tokens got washed out as the sequence grew longer. The transformer fixed this by replacing the sequential compression with attention: every position can look back at every previous position, with learned, content-dependent weights.

Standard residual connections have the same structure, but along depth instead of sequence length. They compress all previous layer outputs into a single running sum, updated one layer at a time. Attention Residuals replace that running sum with attention over all previous layer outputs.

![Attention Residuals architecture](/blog/deepseek-mhc-kimi-attention-residuals/8.png)

In the standard transformer on the left, each layer adds its output to the residual with the $\oplus$ operator. In Full Attention Residuals in the middle, each sublayer instead attends over all previous outputs using a learned query $\mathbf{w}$ and produces attention weights $\alpha$. The hidden state at layer $l$ becomes:

$$
\mathbf{h}_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot \mathbf{v}_i
$$

where $\mathbf{v}_0 = \mathbf{x}_0$ is the token embedding and $\mathbf{v}_i = f_i$ is the output of sublayer $i$. The weights $\alpha$ come from a softmax, so they are non-negative and sum to one.

This is a separate mechanism from the transformer's self-attention and much simpler. There are no Q, K, V projection matrices. Each layer $l$ has a single learned query vector $\mathbf{w}_l \in \mathbb{R}^d$, a fixed parameter that does not depend on the input. The keys are the previous layer outputs, normalized by RMSNorm:

$$
\alpha_{i \to l} = \frac{\exp(\mathbf{w}_l^\top \, \text{RMSNorm}(\mathbf{v}_i))}{\sum_{j=0}^{l-1} \exp(\mathbf{w}_l^\top \, \text{RMSNorm}(\mathbf{v}_j))}
$$

The query is fixed per layer, but the keys change with every input, because they are the actual layer outputs for the current token sequence. This is where the input-dependence comes from. When the model processes a sentence about geography, the outputs of each layer are different from when it processes code, so the attention weights shift. Layer 30 might attend heavily to layer 2's output for one input and to layer 20's output for another.

There are two properties worth noting about this formulation, because they directly address the two symptoms of PreNorm dilution.

First, magnitude control. In standard residuals, the hidden state is a sum that grows with depth. In AttnRes, the hidden state is a convex combination. The softmax weights are non-negative and sum to one, so $\mathbf{h}_l$ is a weighted average of the source vectors. It cannot exceed the largest source in magnitude, regardless of how many sources there are. This is the same property that makes mHC's doubly stochastic matrices stable: both approaches ensure that the aggregation operation cannot amplify signal. mHC enforces this through the Sinkhorn constraint on mixing matrices. AttnRes enforces it through the softmax normalization on attention weights.

Second, selective access. Each sublayer output is stored separately in memory rather than added to a running sum. No information is lost to accumulation. Layer 50 can retrieve layer 3's output at full fidelity, without it having been diluted by the forty-six additions that happened in between. In a standard transformer, those forty-six additions are irreversible: once a layer's output is merged into the running sum, no later layer can recover it individually.

![Attention weight heatmaps](/blog/deepseek-mhc-kimi-attention-residuals/11.png)

The attention weight heatmaps show what the trained model actually learns. The vertical axis is the layer index, the horizontal axis is the source index. The top-left panel shows Full AttnRes weights before attention sublayers. Two patterns are visible. First, there is strong diagonal weight: each layer attends most heavily to its immediate predecessor, which means the model partially recovers standard residual behavior where it is useful. Second, source 0 (the token embedding, at the far left) receives persistent attention at every depth. In a standard transformer, the embedding would be one sixtieth of the hidden state at layer 60. Here, a layer can allocate 30% or 40% of its attention weight to the embedding if the current input calls for it.

The top-right panel shows weights before MLP sublayers. The pattern has more variation: some layers develop strong off-diagonal attention, skipping back to specific earlier layers. The bottom row shows Block AttnRes, which groups layers into blocks and attends over block-level summaries instead of individual outputs. The structure is coarser but the same patterns hold.

![Scaling law heatmaps](/blog/deepseek-mhc-kimi-attention-residuals/10.png)

The Kimi team tested AttnRes on five model sizes and found consistent improvements. The scaling law heatmaps above compare validation loss across different width-to-depth ratios. AttnRes (right) is lower loss across the grid, and the optimal configuration shifts: the baseline favors $d_{\text{model}} / L_b = 60$, while AttnRes favors $d_{\text{model}} / L_b = 45$, a deeper and narrower architecture for the same parameter count. If each layer contributes more because of selective aggregation, additional depth becomes more useful relative to additional width.

On downstream benchmarks, the largest gains are on multi-step reasoning tasks. The Kimi team reports +7.5 on GPQA-Diamond (graduate-level science questions requiring multi-hop reasoning), +3.6 on MATH, and +3.1 on HumanEval (code generation). These are tasks where later layers need to combine outputs from multiple earlier processing stages. Tasks that depend mostly on local, single-step pattern matching see smaller improvements.

The memory cost of Full AttnRes is $O(Ld)$, since every sublayer output must be stored. For a 60-layer model with $d = 4096$, this is 60 vectors of 4096 floats per token, compared to one vector in the standard case. Block AttnRes reduces this to $O(Nd)$ by attending over roughly 8 block summaries instead of all individual outputs, recovering most of the benefit with much less memory. Training overhead for Block AttnRes is under 4%, and inference overhead is under 2%.

---

These two papers address the same problem through different mechanisms. mHC changes the residual signal itself by widening it into multiple constrained streams. AttnRes changes how signals from previous layers are aggregated by replacing uniform accumulation with learned attention. They are not mutually exclusive: a model could carry multiple streams and attend selectively over depth within each stream, though no published results exist on the combination. There are real tradeoffs between them. mHC requires $n\times$ the memory bandwidth at each residual step, since the model reads and writes $n$ vectors instead of one. AttnRes requires storing all previous layer outputs, which scales with depth. mHC has been validated on models up to 27B parameters; AttnRes on a 48B mixture-of-experts model with 3B active parameters. Neither paper benchmarks against the other, so direct comparison on the same model and data does not yet exist. Both are drops in a larger space of residual connection variants, including DenseFormer, Value Residual Learning, and the hybrid normalization approaches that try to get PostNorm's magnitude control without its gradient problems. The question of how information should flow through depth in deep networks is open, and these two papers are among the more concrete recent answers.

---

**References**

1. Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou. *Hyper-Connections*. ICLR 2025. [arxiv.org/abs/2409.19606](https://arxiv.org/abs/2409.19606)

2. Zhenda Xie, Yixuan Wei, Huanqi Cao, et al. *mHC: Manifold-Constrained Hyper-Connections*. DeepSeek-AI, January 2026. [arxiv.org/abs/2512.24880](https://arxiv.org/abs/2512.24880)

3. Kimi Team: Guangyu Chen, Yu Zhang, Jianlin Su, et al. *Attention Residuals*. MoonshotAI, March 2026. [arxiv.org/abs/2603.15031](https://arxiv.org/abs/2603.15031)
