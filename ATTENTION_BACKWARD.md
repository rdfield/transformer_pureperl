# Backward pass of multi-head attention

This is the derivation of every gradient `ML::MHA` computes, in the order `backward()` computes
them, with each step tied to the line that implements it. `test_mha.pl` checks the result against
central finite differences (ε = 1e-4): 656/656 pass, for self- and cross-attention, all four weight
matrices, the input gradient and `gradient_kv`. The mask and dropout steps are not in that test; see
[What the gradient check covers](#what-the-gradient-check-covers).

The pages the code was originally written from are gone, so this derivation is now the reference.
[Further reading](#further-reading) lists standard sources for the same maths.

## Notation

- **Row vectors.** A sequence is a matrix $X$ of shape $S \times E$, one row per token. A projection
  is $XW$ with $W$ of shape $E \times E$. `ML::MHA` has no biases.
- **Gradients.** $L$ is the loss. For any tensor $T$, $dT$ means $\partial L / \partial T$ and has
  the shape of $T$.
- **Batch and heads.** Every step runs per batch item $b$; the weight gradients sum over $b$. There
  are $H$ heads of width $d_k = E/H$, and per-head equations drop the $h$ subscript where that is
  unambiguous.
- $\odot$ is the elementwise product, $\mathbf{1}$ a column of ones, and
  $\operatorname{rowsum}(A) = A\mathbf{1}$.

Two rules carry almost the whole derivation:

```math
\text{Rule 1:}\quad Y = AB \;\Rightarrow\; dA = dY\,B^\top,\quad dB = A^\top dY
\qquad\qquad
\text{Rule 2:}\quad Y = A \odot M \;(M \text{ fixed}) \;\Rightarrow\; dA = dY \odot M
```

Rule 1 follows from $Y_{ij} = \sum_k A_{ik} B_{kj}$: then
$\partial L/\partial A_{ik} = \sum_j dY_{ij} B_{kj} = (dY\,B^\top)_{ik}$, and symmetrically for $B$.
The only other ingredient is the softmax Jacobian, derived in step 5.

## Forward pass

The backward undoes these steps in reverse, so this is the list to keep in view. $X_q$ is the query
stream and $X_{kv}$ the key/value stream: the same tensor for self-attention, the encoder output for
cross-attention.

| step | equation | code |
|---|---|---|
| 1. project | $Q = X_q W_q,\; K = X_{kv} W_k,\; V = X_{kv} W_v$ | `MHA.pm:111`, `:121`, `:131` |
| 2. split heads | $Q_h$ = head $h$'s $d_k$ columns of $Q$; the same for $K_h$, $V_h$ | `MHA.pm:150` |
| 3. scores | $S = Q_h K_h^\top / \sqrt{d_k}$ | `Util.pm:516`, `:528` |
| 4. mask | $\tilde S = S - 10^9 M$, with $M_{ij} = 1$ where key $j$ is hidden from query $i$ | `Util.pm:545` (2-D, key padding), `:557` (3-D, causal + padding) |
| 5. clamp | $\tilde S \leftarrow \min(\tilde S, 10^4)$; NaN also becomes $10^4$ | `Util.pm:574` |
| 6. softmax by row | $P_{ij} = e^{\tilde S_{ij} - m_i} / \sum_k e^{\tilde S_{ik} - m_i}$, with $m_i = \max_k \tilde S_{ik}$ | `Util.pm:583` |
| 7. dropout | $\tilde P = P \odot D$, with $D_{ij} \in \{0,\ 1/(1-p)\}$ | `Util.pm:592`–`601` |
| 8. attend | $O_h = \tilde P V_h$ | `Util.pm:612` |
| 9. concat heads | $C = [\,O_1 \cdots O_H\,]$ | `MHA.pm:177` |
| 10. output projection | $Y = C W_o$ | `MHA.pm:189` |

Kept for the backward: $P$ (`softmax_weights`, `MHA.pm:195`), $\tilde P$ (`attention_weights`,
`:194`), $D$ (`attention_dropout_mask`), the per-head $Q_h$, $K_h$, $V_h$, $C$ (`concat`), $X_q$
(`input`) and $X_{kv}$ (`input_kv`, `:100`).

## Backward pass

`backward()` starts from $dY$, the gradient from whatever consumed this layer's output
(`MHA.pm:223`).

### 1. Output projection

Rule 1 on $Y = C W_o$:

```math
dW_o = \sum_b C_b^\top\, dY_b \qquad\qquad dC = dY\, W_o^\top
```

`MHA.pm:252` stacks the batch's rows into one $(B \cdot S) \times E$ matrix and does a single
product, which is the same sum. $dC$ is `MHA.pm:236`.

### 2. Undo the concat

Concatenation only moves numbers, so its gradient moves them back: $dO_h$ is head $h$'s block of
columns of $dC$ (`MHA.pm:269`).

### 3. Through $O_h = \tilde P V_h$

Rule 1 again:

```math
dV_h = \tilde P^\top dO_h \qquad\qquad d\tilde P = dO_h\, V_h^\top
```

$dV_h$ is `MHA.pm:291`. It uses the post-dropout weights, because those are what multiplied $V$.
$d\tilde P$ is `MHA.pm:300`. Line 277 computes the same product into `$d_attn`, which nothing reads.

### 4. Through dropout

Rule 2, with $D$ fixed:

```math
dP = d\tilde P \odot D
```

`MHA.pm:312`. $D$ already holds $1/(1-p)$ for the kept entries, so the inverted-dropout scale comes
back with no extra term. With $p = 0$ there is no mask and $dP = d\tilde P$.

### 5. Through the softmax

Take one row and drop the row index, so $p_j = e^{s_j} / \sum_k e^{s_k}$. Differentiating,

```math
\frac{\partial p_j}{\partial s_k} = p_j\,(\delta_{jk} - p_k)
```

and the chain rule over the row gives

```math
ds_k = \sum_j dp_j\, \frac{\partial p_j}{\partial s_k}
     = p_k\, dp_k - p_k \sum_j p_j\, dp_j
     = p_k \Big( dp_k - \sum_j p_j\, dp_j \Big)
```

For every row at once:

```math
d\tilde S = P \odot \big( dP - \operatorname{rowsum}(P \odot dP)\,\mathbf{1}^\top \big)
```

The row sum is `MHA.pm:333` (`$AdA`) and the product is `MHA.pm:344`. Three things to note:

- It uses $P$, the **pre-dropout** weights. The Jacobian belongs to the softmax, whose output is $P$;
  dropout was dealt with in step 4. The comment at `MHA.pm:323`–`324` records this.
- The max subtraction needs no backward step. Subtracting the same $m_i$ from a whole row leaves the
  softmax unchanged, so it contributes nothing to the derivative.
- The $S \times S$ Jacobian of each row is never built. The row-sum form costs $O(S^2)$ per head
  instead of $O(S^3)$.

### 6. Through the clamp, the mask and the scale

- **Clamp.** The backward treats it as the identity. That is exact wherever no score exceeded
  $10^4$; at a clamped entry the true derivative is 0. It only fires on runaway or NaN scores.
- **Mask.** $\tilde S = S - 10^9 M$ with $M$ constant, so $dS = d\tilde S$ and the backward needs no
  mask of its own. Masked entries already received zero gradient in step 5: $e^{-10^9}$ underflows
  to exactly 0 in double precision, so $P = 0$ there and $d\tilde S = P \odot (\cdots) = 0$.
- **Scale.** $S = Q_h K_h^\top / \sqrt{d_k}$, so the gradient with respect to the raw product is
  $G = d(Q_h K_h^\top) = dS / \sqrt{d_k}$. The code folds the division into the same line
  (`MHA.pm:321`, `:344`), so `$d_scores` is $G$, not $dS$.

One edge case: if every key in a row were masked, the $10^9$ would cancel in the max subtraction and
that row of $P$ would be uniform, not zero. Neither mask builder produces such a row in practice.
`create_src_mask` hides only PAD keys, so a row is fully masked only if the whole source is PAD, and
`create_tgt_mask` leaves position 0 (the SOS token) visible to every row.

### 7. Through $Q_h K_h^\top$

Rule 1 with the second factor transposed: $Y = A B^\top$ gives $dA = dY\,B$ and $dB = dY^\top A$.
So

```math
dQ_h = G\, K_h \qquad\qquad dK_h = G^\top Q_h
```

`MHA.pm:353` and `:360`.

### 8. Merge the heads

The inverse of forward step 2: each head's $dQ_h$, $dK_h$ and $dV_h$ goes back into its block of
columns of $dQ$, $dK$ and $dV$ (`MHA.pm:369`, `:380`, `:391`).

### 9. Input projections

Rule 1 on each projection:

```math
dW_q = \sum_b X_{q,b}^\top\, dQ_b \qquad dW_k = \sum_b X_{kv,b}^\top\, dK_b \qquad dW_v = \sum_b X_{kv,b}^\top\, dV_b
```

```math
dX_q = dQ\, W_q^\top \qquad\qquad dX_{kv} = dK\, W_k^\top + dV\, W_v^\top
```

The weight gradients are `MHA.pm:409`–`411` and the input gradients `:433`, `:436` and `:438`.
$X_{kv}$ gets a sum because it feeds two projections.

### 10. Where the input gradients go

- **Self-attention:** $X_q = X_{kv} = X$, so all three paths add into one gradient,
  $dX = dQ\,W_q^\top + dK\,W_k^\top + dV\,W_v^\top$ (`MHA.pm:435`, `:445`, `:446`), returned by
  `gradient()`.
- **Cross-attention:** $X_q$ is the decoder stream and $X_{kv}$ the encoder output, so the two stay
  separate. $dX_q$ goes to `gradient()` and $dX_{kv}$ to `gradient_kv()` (`MHA.pm:440`–`451`). That
  is the first link of the channel, described in `CLAUDE.md`, that carries the encoder's gradient.

## All of it in one place

Given $dY$ (the middle three lines run for each head $h$):

```math
\begin{aligned}
dW_o &= \textstyle\sum_b C_b^\top dY_b, & dC &= dY\,W_o^\top, & dO_h &= \text{head } h\text{'s columns of } dC \\
dV_h &= \tilde P^\top dO_h, & dP &= \big(dO_h V_h^\top\big) \odot D \\
G &= \tfrac{1}{\sqrt{d_k}}\; P \odot \big(dP - \operatorname{rowsum}(P \odot dP)\,\mathbf{1}^\top\big), & dQ_h &= G\,K_h, & dK_h &= G^\top Q_h \\
dW_q &= \textstyle\sum_b X_q^\top dQ, & dW_k &= \textstyle\sum_b X_{kv}^\top dK, & dW_v &= \textstyle\sum_b X_{kv}^\top dV \\
dX_q &= dQ\,W_q^\top, & dX_{kv} &= dK\,W_k^\top + dV\,W_v^\top
\end{aligned}
```

## What the gradient check covers

`perl test_mha.pl` compares the analytic gradients of $W_q$, $W_k$, $W_v$, $W_o$ and of the inputs
against central differences with ε = 1e-4: 656/656 pass, for self-attention and for cross-attention,
including `gradient_kv`.

It runs with dropout `"0.0"` and no mask, so **steps 4 and 6 are derived here but not checked
numerically**. A check for the mask only needs a fixed mask passed to `forward`. A check for dropout
needs the same $D$ on every forward call, for example by calling `srand` with the same seed before
each one.

## Assumptions in the code

- **Equal lengths.** `ML::MHA` takes `seq_len` from the query and uses it for the K and V head split
  (`MHA.pm:150`) and for the $S \times S$ loops in the backward. Cross-attention therefore needs the
  source and target to be the same length. That holds here because both are padded to
  `max_seq_length`.
- **K is projected from `v`.** `MHA.pm:121` computes $K$ from `$args{v}`, while `input_kv`, which
  $dW_k$ uses, is copied from `$args{k}` (`MHA.pm:100`). Every caller passes the same tensor for both
  (`DecoderBlock.pm:36`, `test_mha.pl:191`), so the two agree today. A caller that passed different
  `k` and `v` would get wrong gradients for $W_k$.
- **No biases.** The four projections are plain matrices, so there are no bias gradients.

## Further reading

The pages the code was first written from are lost. These cover the same maths:

- Terence Parr and Jeremy Howard,
  [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528): rule 1 and the
  rest of the matrix calculus.
- Eli Bendersky,
  [The Softmax function and its derivative](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/):
  step 5.
- Justin Johnson,
  [Derivatives, Backpropagation, and Vectorization](http://cs231n.stanford.edu/handouts/derivatives.pdf)
  (CS231n): why the matrix forms are right, with the shapes worked through.
- Tri Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135), backward-pass appendix: the
  same equations, recomputed block by block so that $P$ is never stored. The C++ port's
  FlashAttention kernels are built on it.
