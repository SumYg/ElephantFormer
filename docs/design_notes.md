# Design Notes and Considerations

## Positional Embeddings vs. Game State Relevance

**Date Raised:** 2025-05-31

**Concern:**
The user rightly pointed out that in Elephant Chess (and similar board games), the absolute number of moves already played to reach a certain board state (which absolute positional embeddings might implicitly encode) is usually irrelevant to the optimal next move. What primarily matters is the current board configuration and whose turn it is. Using standard absolute positional embeddings might lead the model to learn spurious correlations based on game length.

**Why Positional Embeddings are Still Necessary (for the move sequence):**
Despite the above, positional embeddings are crucial for the Transformer to understand:
1.  The internal structure of each move (i.e., distinguishing `fx`, `fy`, `tx`, `ty` components within their 4-token block).
2.  The sequential order of moves, as the current board state is a direct result of this sequence.

**Potential Impact & Mitigation Strategies:**
*   **Potential Issue:** If not handled carefully, the model might generalize poorly if it over-relies on absolute game length cues from positional embeddings.
*   **Mitigation/Solutions:**
    1.  **Start with Standard Absolute Positional Embeddings:** Implement these first due to their simplicity and observe model behavior.
    2.  **Prioritize Board State Conditioning (Step 7 of README.md plan):** This is the most robust solution. By explicitly feeding the current board state as an input to the model (alongside or in combination with the recent move history), the model can directly learn to make decisions based on the strategically relevant board configuration. The influence of "absolute move number" from positional embeddings on game strategy should significantly diminish.
    3.  **Consider Relative Positional Embeddings (Future Improvement):** If issues persist even with board state conditioning, exploring relative positional embeddings could be a valuable next step. These focus on the relationship/distance between tokens rather than their absolute position.
    4.  **Limited Sequence Length / Recency Bias:** Training on fixed-length windows of recent moves can also reduce the impact of absolute positions over very long games, but this limits historical context.

**Plan:**
- Proceed with standard absolute positional embeddings initially.
- Emphasize the implementation of board state conditioning as a core part of the model architecture to ensure strategic decisions are based on the current game situation.
- Revisit the choice of positional embeddings if performance indicates an issue related to game length sensitivity. 