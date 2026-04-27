# Weight Tying Papers

Selected for evaluating whether RWKV-LM-V7 should support tying the input
embedding and output projection weights.

## Core Papers

- `2017_press_wolf_using_output_embedding_to_improve_language_models.pdf`
  - Ofir Press and Lior Wolf, EACL 2017.
  - Foundational paper recommending input/output embedding tying for language
    models and showing parameter reduction with improved perplexity in their
    experiments.
  - Source: https://aclanthology.org/E17-2025/

- `2017_inan_khosravi_socher_tying_word_vectors_and_word_classifiers.pdf`
  - Hakan Inan, Khashayar Khosravi, Richard Socher, ICLR 2017.
  - Independent theoretical framing that leads to tying input embeddings and
    output classifiers in language models.
  - Source: https://arxiv.org/abs/1611.01462

## Architecture Context

- `2017_vaswani_attention_is_all_you_need.pdf`
  - Ashish Vaswani et al., 2017.
  - Transformer reference architecture; included because its embedding/softmax
    section adopts shared embedding and pre-softmax weights.
  - Source: https://arxiv.org/abs/1706.03762

## Cautionary / Recent

- `2026_lopardo_weight_tying_biases_token_embeddings_towards_output_space.pdf`
  - Antonio Lopardo, Avyukth Harish, Catherine Arnett, Akshat Gupta, 2026.
  - Recent mechanistic argument that tied embeddings can become biased toward
    the output role, which is relevant when deciding whether tying is a good
    tradeoff for this parameter-limited RWKV experiment.
  - Source: https://arxiv.org/abs/2603.26663
