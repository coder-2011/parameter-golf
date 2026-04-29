# Frozen-QK

`FROZEN_QK=1` freezes the query and key projection weights in each Parcae
attention block at initialization while leaving value projections, output
projections, MLPs, embeddings, and scalar controls trainable.

The source paper is Dong et al., "Attention Retrieves, MLP Memorizes:
Disentangling Trainable Components in the Transformer":

https://arxiv.org/abs/2506.01115

The paper studies transformer variants that freeze either MLPs or attention
components. Its Frozen-QK variant keeps query/key projectors fixed at their
random initialization. The value path remains trainable, so attention can still
retrieve through input-dependent Q/K features while the trainable value/MLP path
does the remaining adaptation.

In `train_gpt_parcae.py`, Q/K/V are packed into one `attn.c_qkv.weight` tensor.
The flag therefore freezes only the Q/K rows of that packed matrix:

- a backward hook zeros Q/K gradient rows;
- after each optimizer step, Q/K rows are restored from their initialization;
- the V rows in the same packed tensor remain trainable.

The post-step restore is intentional because Muon weight decay or optimizer
state could otherwise move packed Q/K rows even with zero gradients.

This flag does not reduce checkpoint size or optimizer-state memory. Q/K rows
are still stored inside the packed `attn.c_qkv.weight` tensor, and optimizers
still see that full packed parameter. Turning Frozen-QK into an artifact-budget
or optimizer-memory optimization would require a different module layout, such
as generated Q/K projections plus a separate trainable V projection.

The frozen Q/K copy is kept on the same device as the model so the required
post-step restore is a device-local copy rather than a CPU-to-GPU transfer.
`QK_NORM` in this trainer is parameter-free, so there are no q/k norm parameters
to freeze.
