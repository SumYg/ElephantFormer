"""MCTS at inference for the board-state ElephantFormer (Phase 1).

Search wraps the policy (priors) and value head (leaf evaluation) of a trained
:class:`BoardLightningModule`. Design, kept correctness-first:

* Node values are the expected score in ``[0, 1]`` **for the side to move at
  that node**; backups negate per ply (``parent view = 1 - child value``).
* Terminals are exact: a side with no legal reply loses (checkmate or 困毙
  stalemate, matching the engine), and a position seen three or more times
  along game history + search path scores as a draw (claim-based repetition,
  approximated). Each node owns an engine copy, whose ``position_history``
  already accumulates the game *and* the path to the node.
* Root selection is Gumbel-style sequential halving over the ``root_top_m``
  moves ranked by ``gumbel + policy logit`` (Danihelka et al.): the sim budget
  is spent in halving rounds, survivors ranked by ``g + logit + sigma(q)`` with
  ``sigma(q) = (c_visit + max_child_visits) * c_scale * q``. With
  ``gumbel_scale = 0`` (the default) play is deterministic — right for eval
  matches; nonzero scale gives the sampling behaviour self-play wants.
* Below the root, plain PUCT with unvisited children initialised to 0.5.

Leaf evaluations within a halving round are batched through the network.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from elephant_former.data_utils import board_features as bf
from elephant_former.engine.elephant_chess_game import ElephantChessGame, Move

DRAW_SCORE = 0.5
REPETITION_DRAW_COUNT = 3


class Node:
    """One search node; ``state`` is the engine position it represents."""

    __slots__ = ("state", "prior", "visits", "value_sum", "children", "legal_indices",
                 "child_priors", "terminal_value")

    def __init__(self, state: ElephantChessGame, prior: float = 0.0) -> None:
        self.state = state
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children: Dict[int, "Node"] = {}          # policy index -> child
        self.legal_indices: Optional[List[int]] = None  # set on expansion
        self.child_priors: Optional[Dict[int, float]] = None
        self.terminal_value: Optional[float] = None     # score for this node's stm

    @property
    def q(self) -> float:
        """Mean value for this node's side to move (0.5 when unvisited)."""
        return self.value_sum / self.visits if self.visits else DRAW_SCORE

    @property
    def expanded(self) -> bool:
        return self.legal_indices is not None or self.terminal_value is not None


class MCTS:
    """Search driver around a model with ``_forward_features`` (see ModelBot)."""

    def __init__(
        self,
        forward_features,
        num_simulations: int = 200,
        root_top_m: int = 16,
        c_puct: float = 1.5,
        gumbel_scale: float = 0.0,
        c_visit: float = 50.0,
        c_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self._forward = forward_features
        self.num_simulations = num_simulations
        self.root_top_m = root_top_m
        self.c_puct = c_puct
        self.gumbel_scale = gumbel_scale
        self.c_visit = c_visit
        self.c_scale = c_scale
        self._rng = np.random.default_rng(seed)

    # ---- expansion / evaluation -------------------------------------------------

    def _check_terminal(self, node: Node, check_repetition: bool = True) -> bool:
        """Set ``terminal_value`` for mates/困毙 and (in-tree) repetition draws."""
        state = node.state
        if (
            check_repetition
            and state.position_history[state.position_sequence[-1]] >= REPETITION_DRAW_COUNT
        ):
            node.terminal_value = DRAW_SCORE
            return True
        if not state.get_all_legal_moves_basic(state.current_player):
            node.terminal_value = 0.0  # no reply: the side to move loses
            return True
        return False

    def _evaluate_batch(self, nodes: Sequence[Node], check_repetition: bool = True) -> List[float]:
        """Expand ``nodes`` (priors + children stubs) and return leaf values.

        Terminal nodes get their exact value; the rest share one forward pass.
        """
        values: List[Optional[float]] = [None] * len(nodes)
        pending: List[int] = []
        feats = []
        legal_lists: List[List[Move]] = []
        for i, node in enumerate(nodes):
            if node.expanded:
                values[i] = node.terminal_value if node.terminal_value is not None else node.q
                continue
            if self._check_terminal(node, check_repetition=check_repetition):
                values[i] = node.terminal_value
                continue
            legal = node.state.get_all_legal_moves_basic(node.state.current_player)
            legal_lists.append(legal)
            feats.append(bf.extract_features(node.state, stm_legal_moves=legal))
            pending.append(i)

        if pending:
            policy_logits, value_logits = self._forward(feats)
            probs = torch.softmax(value_logits.float().cpu(), dim=-1)
            # (loss, draw, win) for the node's side to move -> expected score.
            leaf_values = (probs[:, 2] + 0.5 * probs[:, 1]).tolist()
            policy_logits = policy_logits.float().cpu()
            for k, i in enumerate(pending):
                node = nodes[i]
                legal = legal_lists[k]
                idx = [bf.move_to_policy_index(m) for m in legal]
                logits = policy_logits[k][idx]
                pri = torch.softmax(logits, dim=-1).tolist()
                node.legal_indices = idx
                node.child_priors = dict(zip(idx, pri))
                values[i] = float(leaf_values[k])
        return [float(v) for v in values]  # type: ignore[arg-type]

    def _child(self, node: Node, index: int) -> Node:
        child = node.children.get(index)
        if child is None:
            state = node.state.copy()
            state.apply_move(bf.policy_index_to_move(index))
            child = Node(state, prior=node.child_priors[index] if node.child_priors else 0.0)
            node.children[index] = child
        return child

    # ---- simulation --------------------------------------------------------------

    def _select_puct(self, node: Node) -> int:
        """PUCT over the node's legal moves (children may be unvisited)."""
        sqrt_n = math.sqrt(max(1, node.visits))
        best_index, best_score = -1, -math.inf
        for idx in node.legal_indices:  # type: ignore[union-attr]
            child = node.children.get(idx)
            prior = node.child_priors[idx]  # type: ignore[index]
            if child is None or child.visits == 0:
                q_parent = DRAW_SCORE
                n_child = 0
            else:
                q_parent = 1.0 - child.q
                n_child = child.visits
            score = q_parent + self.c_puct * prior * sqrt_n / (1 + n_child)
            if score > best_score:
                best_index, best_score = idx, score
        return best_index

    def _simulate_from(self, root_child: Node) -> float:
        """One descent starting below a root child; returns its backed-up value
        **from the root child's side-to-move perspective**."""
        path = [root_child]
        node = root_child
        while node.expanded and node.terminal_value is None:
            node = self._child(node, self._select_puct(node))
            path.append(node)
        value = self._evaluate_batch([node])[0]
        # Back up along the path; ``value`` is for path[-1]'s side to move.
        for depth, visited in enumerate(reversed(path)):
            visited.visits += 1
            visited.value_sum += value if depth % 2 == 0 else 1.0 - value
        return path[0].q

    # ---- root: Gumbel sequential halving -----------------------------------------

    def _sigma(self, q: float, max_visits: int) -> float:
        return (self.c_visit + max_visits) * self.c_scale * q

    def search(self, game: ElephantChessGame) -> Tuple[Move, Dict[int, float]]:
        """Run the budgeted search; returns (chosen move, root child Q by index)."""
        root = Node(game.copy())
        # The real game may legitimately sit on a repeated position (draws are
        # claim-based); only in-tree nodes treat repetition as terminal.
        self._evaluate_batch([root], check_repetition=False)
        if root.terminal_value is not None or not root.legal_indices:
            raise ValueError("search() called on a terminal position")
        if len(root.legal_indices) == 1:
            only = root.legal_indices[0]
            return bf.policy_index_to_move(only), {only: DRAW_SCORE}

        logits = {i: math.log(max(p, 1e-12)) for i, p in root.child_priors.items()}  # type: ignore[union-attr]
        gumbel = {
            i: (self._rng.gumbel() * self.gumbel_scale if self.gumbel_scale > 0 else 0.0)
            for i in root.legal_indices
        }

        m = min(self.root_top_m, len(root.legal_indices))
        survivors = sorted(root.legal_indices, key=lambda i: gumbel[i] + logits[i], reverse=True)[:m]
        # Ensure every survivor is expanded/evaluated once, batched together.
        children = [self._child(root, i) for i in survivors]
        first_values = self._evaluate_batch(children)
        for child, v in zip(children, first_values):
            if child.visits == 0:
                child.visits += 1
                child.value_sum += v
        budget = max(0, self.num_simulations - len(survivors))

        rounds = max(1, math.ceil(math.log2(m)))
        while len(survivors) > 1:
            per_action = max(1, budget // (rounds * len(survivors))) if budget > 0 else 0
            for idx in survivors:
                child = self._child(root, idx)
                for _ in range(per_action):
                    if budget <= 0:
                        break
                    self._simulate_from(child)
                    budget -= 1
            max_visits = max(self._child(root, i).visits for i in survivors)

            def rank(i: int) -> float:
                child = self._child(root, i)
                return gumbel[i] + logits[i] + self._sigma(1.0 - child.q, max_visits)

            survivors = sorted(survivors, key=rank, reverse=True)[: max(1, len(survivors) // 2)]

        best = survivors[0]
        q_by_index = {
            i: 1.0 - root.children[i].q for i in root.children if root.children[i].visits > 0
        }
        return bf.policy_index_to_move(best), q_by_index
