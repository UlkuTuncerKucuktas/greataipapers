---
title: 'Mechanistic Interpretability'
description: 'Taking neural networks apart to understand what their pieces do, how information moves between them, and how they wire together into circuits that carry out specific behaviors.'
pubDate: 'Apr 11 2026'
---

Large language models learn their capabilities during training, but what exactly they learn is hard to say. The parameters encode some set of algorithms for language, reasoning, and recall. We just can't read them directly. A trained model is, in a sense, a compiled program running on the unusual virtual machine of a neural network architecture. Mechanistic interpretability is the effort to decompile it: to take a neural network apart and understand what its pieces do, how information moves between them, and how they wire together into circuits that carry out specific behaviors.

The field has roots in [Chris Olah's](https://colah.github.io/) work on vision model circuits at OpenAI, where researchers found that convolutional networks learn curve detectors in early layers and compose them into progressively higher-level features. Anthropic's Transformer Circuits thread, starting in 2021, brought this approach to language models. They started with the smallest possible transformers and built a mathematical framework for understanding attention heads as readable circuits.

## The Residual Stream

The first move in the Transformer Circuits framework is to look at the transformer not as a chain of layers that transform a hidden state, but as a collection of components that all read from and write to a shared communication channel. That channel is the residual stream.

![The residual stream](/blog/mechanistic-interpretability/1.png)

When a token enters the model, it gets embedded into a vector $\mathbf{x}_0 = W_E t$. This vector is the residual stream. From here, every component (each attention head, each MLP layer) reads from the stream, computes something, and adds its result back in. Nothing is overwritten, nothing is replaced. The stream just accumulates. The final prediction is the sum of everything that was written along the way:

$$
\mathbf{x}_{\text{final}} = \underbrace{\mathbf{x}_0}_{\text{embedding}} + \underbrace{\sum_{\ell=1}^{L} \sum_{h \in H_\ell} \mathrm{Attn}^{\ell,h}(\mathbf{x})}_{\text{all attention heads}} + \underbrace{\sum_{\ell=1}^{L} \mathrm{MLP}^\ell(\mathbf{x})}_{\text{all MLP layers}}
$$

If you have seen ARIMA, the classical time series framework, this structure might look familiar. The parallel is loose, but it is a good way to build intuition about what each type of component contributes. In ARIMA, a forecast is also built as a sum of terms: a baseline, autoregressive terms that look back at recent history, and correction terms:

$$
\hat{y}_t = \underbrace{\mu}_{\text{baseline}} + \underbrace{\sum_{i=1}^{p} \phi_i \, y_{t-i}}_{\text{autoregressive}} + \underbrace{\sum_{j=1}^{q} \theta_j \, \epsilon_{t-j}}_{\text{corrections}}
$$

![One-layer attention-only transformer](/blog/mechanistic-interpretability/2.png)

The embedding $\mathbf{x}_0 = W_E t$ is the baseline. With no attention and no MLP, the model computes $T(t) = W_U W_E \, t$, a direct lookup from input token to output distribution. This is a bigram model. It knows that "the" is usually followed by a noun, that a period is usually followed by a capital letter, but it has no ability to condition on anything beyond the single previous token. Like $\mu$ in ARIMA, this term is always present, always contributing, and it carries a surprising amount of the prediction. Much of language is locally predictable.

The attention heads are the autoregressive terms. They let the model look back into its context and pull forward relevant information. A head might attend to the previous token, or to the subject of the sentence, or to a token that appeared in a similar context earlier. The point is that it reaches back and copies something useful. Each head reads from the stream, decides which earlier position to attend to (via the QK circuit), decides what to copy from that position (via the OV circuit), and writes the result back.

The MLP layers play a roughly analogous role to the correction terms: they refine what has been assembled so far, though the parallel is looser here. They read the residual stream after attention has already written to it, so they see both the original token and whatever context the attention heads gathered. They apply nonlinear transformations that go beyond copying tokens, like combining features, applying stored factual knowledge ("Paris is the capital of France"), and suppressing irrelevant signals.

But the ARIMA analogy is not really the point. What matters for mechanistic interpretability is the consequence of the additive structure. Because the output is a sum of independent contributions, we can isolate each component and measure its effect on the prediction. We can zero out a single head and check what the model loses. We can amplify a single head and check what the model gains. This ability to attribute the model's behavior to individual components is the foundation that everything else in this field builds on.

## QK and OV Circuits

![QK and OV circuits](/blog/mechanistic-interpretability/6.png)

Each attention head does two things: it decides which earlier token to look at, and it decides what to do with what it finds. The Transformer Circuits framework shows that these two jobs are handled by two separate circuits that share no parameters with each other.

To see where they come from, start with a one-layer attention-only transformer written as a single expression. The embedding $W_E$ maps tokens into the residual stream, the attention layer adds each head's contribution (weighted by the attention pattern $A^h$ and transformed by $W_V^h W_O^h$), and the unembedding $W_U$ maps the result to logits:

$$
T = \mathrm{Id} \otimes W_U \cdot \left( \mathrm{Id} + \sum_{h \in H} A^h \otimes W_V^h W_O^h \right) \cdot \mathrm{Id} \otimes W_E
$$

Holding the attention patterns fixed, everything here is linear, so we can multiply out and distribute:

$$
T = \underbrace{\mathrm{Id} \otimes W_U W_E}_{\text{direct path}} + \sum_{h \in H} \underbrace{A^h \otimes (W_U W_V^h W_O^h W_E)}_{\text{attention head terms}}
$$

The direct path is the bigram baseline from before. Each attention head term factors into two independent pieces, and this factorization is the whole point.

The first piece is $A^h$, the attention pattern. It comes from the **QK circuit**:

$$
A^h = \mathrm{softmax}^* \left( t^T W_E^T \, W_Q^{h\,T} W_K^h \, W_E \, t \right)
$$

Each token gets projected into a query vector (through $W_Q$) and a key vector (through $W_K$). The dot product between a query at one position and a key at another gives the attention score, which is how much the first token wants to look at the second. After softmax, this becomes the attention pattern $A^h$. The QK circuit decides *where to look*, and it only involves $W_Q$ and $W_K$.

The second piece is $W_U W_V^h W_O^h W_E$, the **OV circuit**. Once the QK circuit has decided to attend to some token, that token gets projected through $W_V$ (extracting its "value"), then through $W_O$ (writing the result back to the residual stream). The OV circuit decides *what to copy*, and it only involves $W_V$ and $W_O$.

The two circuits share an attention head but touch completely separate weight matrices. You can study one without thinking about the other. This is what makes individual attention heads interpretable.

![QK and OV circuits traced through the weights](/blog/mechanistic-interpretability/5.png)

The figure above traces both circuits through the actual weight matrices, showing how the QK path (pink) and OV path (gold) take completely separate routes through the model.

What makes this practically useful is that both circuits can be expanded into explicit matrices over the full vocabulary. The expanded QK matrix $W_E^T W_Q^T W_K W_E$ and the expanded OV matrix $W_U W_O W_V W_E$ are both $|V| \times |V|$, where $|V|$ is the vocabulary size, typically around 50,000 tokens. Every entry has a concrete, readable meaning.

In the **QK matrix**, entry $(i, j)$ says how much token $i$ wants to attend to token $j$. You can read off the head's routing rule directly.

In the **OV matrix**, entry $(i, j)$ says: if this head attends to token $i$, how much does it boost or suppress token $j$ in the output?

These matrices are enormous, with 2.5 billion entries each at a 50,000-token vocabulary. You cannot inspect every entry. But you can sort by magnitude and ask what the strongest patterns are. This is how researchers turn opaque weight matrices into readable descriptions of what each head does.

Researchers at Anthropic and elsewhere have catalogued a growing zoo of attention head types, each with a distinct job. There are previous-token heads that just look one step back, induction heads that detect and continue repeated patterns, duplicate-token heads that find earlier copies of the current token, name-mover heads that copy names into prediction slots, and many others. But not every head is interpretable, and not every model develops the same set of types. Some heads are messy, responding to overlapping or context-dependent patterns that resist simple description. The point of the QK and OV matrix analysis is that it gives you a universal tool to look: expand any head into its vocabulary-by-vocabulary matrices and read what the strongest patterns are. If a head has a clean function, the matrices will show it. If it does not, that is useful information too. The figures below show simplified toy examples to illustrate what these matrices look like for two of the cleanest head types, but the same inspection technique applies to any head in any model.

![QK matrices for two attention heads](/blog/mechanistic-interpretability/7.png)

The previous-token head on the left has a clean subdiagonal: "cat" attends to "the", "sat" attends to "cat", "on" attends to "sat", and so on. This is one of the simplest and most common head types. It just copies the identity of the previous token forward. The pronoun-resolution head on the right is more interesting. The row for "she" lights up almost exclusively at the "Mary" column. The head has learned a matching rule that connects pronouns to their referents.

![OV matrices for two attention heads](/blog/mechanistic-interpretability/8.png)

The copying head on the left is the complement to the previous-token head. Once it attends to a token, it boosts that exact token in the output. Attending to "Paris" makes the model more likely to predict "Paris" next. The association head on the right does something more sophisticated: attending to "France" boosts "Paris", attending to "Spain" boosts "Madrid", attending to "King" boosts "Queen". The light pink entries show crosstalk between related tokens. The head distinguishes primary associations from secondary ones.

Together, these two matrices give a complete description of what a single attention head does. The QK matrix tells you its routing rule (which tokens it connects), and the OV matrix tells you its effect (what happens to the prediction when those connections fire). A head with a previous-token QK pattern and a copying OV pattern is a "repeat the last token" head. A head with a pronoun-resolution QK pattern and a name-boosting OV pattern is doing coreference resolution.

This gives us a clean framework for understanding attention. But attention is only half the model. The MLP layers remain dense, nonlinear, and hard to decompose, and they are where most of the model's stored knowledge lives. Worse, the features that MLPs operate on are not neatly aligned with individual neurons. A single neuron might respond to French text, recipes, and the color blue at the same time. This is the problem of superposition, and it requires a different tool.

## Superposition and Sparse Autoencoders

In a vision model like Inception v1, a single neuron might respond to both cat faces and car fronts. In a small language model, a single neuron might fire on academic citations, English dialogue, HTTP requests, and Korean text all at once. This is called **polysemanticity**: one neuron, many unrelated meanings. If you try to understand what a polysemantic neuron "does", there is no clean answer. It does several things depending on context, and there is no way to tell which meaning is active just by looking at whether the neuron fired.

![Polysemanticity](/blog/mechanistic-interpretability/9.png)

This is not a bug in the model. It is a compression strategy. A typical transformer layer might have 4,096 dimensions in its residual stream, but the model needs to keep track of far more than 4,096 concepts. It needs features for "French text" and "code with a bug" and "the Golden Gate Bridge" and "closing parenthesis" and tens of thousands of other things. There simply are not enough neurons to give each concept its own dedicated dimension.

![Superposition](/blog/mechanistic-interpretability/10.png)

The solution the model finds is called **superposition**. Instead of assigning each feature to a single neuron, the model encodes features as *directions* in the high-dimensional activation space. In a 4,096-dimensional space, you can only fit 4,096 perfectly orthogonal directions. But if you allow a little bit of interference between them, you can pack in far more. The key insight from Anthropic's Toy Models of Superposition paper is that this trade-off depends on sparsity. If a feature is rarely active (most tokens are not about the Golden Gate Bridge), then the interference it causes is also rare, and the model can afford to pack it in at an angle to other features. The sparser the features, the more of them you can fit.

This is geometrically visible. In a 2D space with no sparsity, you can represent at most 2 features (one per axis). At 80% sparsity, the model starts using antipodal pairs to squeeze in 4 features. At 90% sparsity, all 5 features get represented as a pentagon of directions, accepting some interference as the cost of not losing any feature entirely.

The result is that the model's internal representations are rich but tangled. The features are in there, encoded as directions, but they overlap with each other in ways that make individual neurons unreadable. To do mechanistic interpretability on the MLP layers, we need a way to untangle them.

![Sparse autoencoder](/blog/mechanistic-interpretability/11.png)

This is where **sparse autoencoders** (SAEs) come in. The idea is straightforward. Take the model's internal activations (say, 4,096 dimensions from the residual stream or an MLP layer) and project them into a much larger space (say, 65,000 or even 131,000 dimensions) using a learned encoder. Apply a sparsity constraint so that only a small number of these dimensions are active at once. Then decode back to the original 4,096 dimensions and train the whole thing to minimize reconstruction error.

The sparsity constraint is what makes this work. Without it, the autoencoder could learn any 65,000-dimensional representation, most of which would be as tangled as the original. With sparsity, each input activates only a handful of features (typically around 50 out of 65,000), which pushes the encoder to find directions where each feature corresponds to a single, coherent concept. The expansion gives room for every concept to have its own dimension. The sparsity makes sure they actually use it that way.

When Anthropic trained SAEs on Claude 3 Sonnet with roughly 34 million learned features, they found features that fired specifically on mentions of the Golden Gate Bridge, on brain science terminology, on popular tourist attractions, on transit infrastructure, on code quality issues, on emotional expressions, and on potential safety concerns. These were not hand-labeled. They emerged from the unsupervised decomposition.

![Feature activation distribution for the Golden Gate Bridge feature](/blog/mechanistic-interpretability/13.png)

The Golden Gate Bridge feature is a good example of what a clean SAE feature looks like. Most inputs produce zero activation (the feature is off). When it does fire, the activation strength correlates with how directly the input relates to the bridge. Vaguely related content (San Francisco, orange-colored things) produces weak activations. Direct mentions of the Golden Gate Bridge produce strong activations. At the highest activation levels, the specificity is near-perfect: the feature fires on exactly and only the thing it represents.

This lets us validate features in several ways. We can look at the top-activating examples and check that they share a coherent theme. We can use a language model to automatically generate a description of what the feature detects, and then test whether that description predicts the feature's activations on new inputs. And most powerfully, we can intervene: amplify a feature and see if the model's behavior changes in the way we would expect.

![Feature steering](/blog/mechanistic-interpretability/12.png)

When researchers clamped the Golden Gate Bridge feature to 10 times its maximum activation, Claude stopped saying "I don't have a physical form" and started describing itself as the bridge, with its "beautiful orange color, towering towers, and sweeping suspension cables." Clamping a brain sciences feature made the model answer "neuroscience" when asked about the most interesting science, even though it would normally say "physics." A transit infrastructure feature made the model confabulate a bridge when giving walking directions to a grocery store.

If amplifying a feature labeled "Golden Gate Bridge" makes the model obsess about the Golden Gate Bridge and nothing else, the label is almost certainly correct. If suppressing a feature makes the model lose a specific ability while leaving everything else intact, you have causal evidence for what that feature does. This kind of intervention is the strongest form of validation available.

---

We started with a simple observation: because the transformer's output is a sum of independent contributions, we can pull the model apart and ask what each piece does. The residual stream gives us the additive structure. The QK/OV decomposition gives us readable attention heads. Sparse autoencoders give us readable features in the MLP layers. Each tool opens up a different part of the model, and together they make it possible to trace a computation from input tokens through attention patterns and feature activations to the output.

There is a lot more to the field than what we covered here. Induction heads show how attention heads compose across layers to implement in-context learning. Attribution graphs trace full circuits through production-scale models. And there are real open problems with all of these tools, from SAE reconstruction error to the question of whether the features we find are the "right" decomposition at all. But the core ideas are the ones in this post, and they are enough to start reading the primary literature. A good next step is Anthropic's [Transformer Circuits thread](https://transformer-circuits.pub/) and Neel Nanda's [concrete problems list](https://www.neelnanda.io/mechanistic-interpretability/getting-started), both of which pick up where this post leaves off.
