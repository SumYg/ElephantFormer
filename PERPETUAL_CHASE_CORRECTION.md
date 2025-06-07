# Corrected Perpetual Chase Implementation

## Research Summary

After extensive research of official Xiangqi sources, including:
- Wikipedia Xiangqi articles
- Club Xiangqi (CXQ) rules
- Asian Xiangqi Federation rules
- World Xiangqi Federation references
- XQBase (象棋百科全书)

**No official rule was found requiring pieces to only perpetually chase pieces of equal or greater value.**

## Official Rules Found

The actual perpetual chase rules focus on:

1. **Protection Status**: Whether the threatened piece is protected or unprotected
   - "Continuously chasing one unprotected opponent piece using one or more pieces is not allowed"

2. **Specific Piece Interactions**: Rules for particular piece combinations
   - "A Cannon cannot perpetually chase a Rook"
   - "Rook can perpetually chase a protected Cannon"
   - Various specific piece-type rules

3. **Position-Based Rules**: Context matters more than simple value comparisons

## Changes Made

### Before (Incorrect Implementation)
```python
# Check if this is a valid chase (threatening piece of equal or greater value)
moving_piece_value = self._get_piece_value(self.board[ty, tx])
threatened_piece_value = self._get_piece_value(threatened_piece)

# Chase is when threatening piece of equal or greater value
if threatened_piece_value >= moving_piece_value:
```

### After (Corrected Implementation)
```python
# Check if this is a valid chase according to official rules
# Official rules focus on whether the piece is protected, not piece values
# A chase occurs when continuously threatening an unprotected piece
if self._is_piece_unprotected(threat_x, threat_y):
```

## New Helper Method Added

```python
def _is_piece_unprotected(self, x: int, y: int, board_state: Optional[Board] = None) -> bool:
    """
    Checks if a piece at (x,y) is unprotected according to official Xiangqi rules.
    A piece is unprotected if no friendly piece can capture an attacker that captures it.
    """
```

## Impact

This correction aligns the implementation with actual official Xiangqi rules:
- Removes the non-existent "piece value comparison" requirement
- Implements protection-based chase detection as found in official sources
- More accurately reflects the nuanced nature of perpetual chase rules

## Sources Consulted

- CXQ Chinese Chess Rules
- Asian Chinese Chess Rules (detailed scenarios)
- Multiple Wikipedia articles on Xiangqi
- XQBase rule documentation
- World Xiangqi Federation materials

The corrected implementation now follows the documented official rules rather than a misconception about piece value requirements.
