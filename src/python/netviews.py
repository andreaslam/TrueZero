"""Static views of a TrueNet's internals for one position (`visualisenet.py --mode inspect`).

Each `fig_*` function returns a matplotlib figure:

  overview   board with policy arrows, from/to-square heatmaps, value/WDL/moves-left,
             value-hidden layer and per-piece value contribution
  inputs     the 21 input planes
  trunk      per-layer activation maps: mean, max, residual update and the most varied
             channels, plus a layer x channel strip
  policy     the 73 raw policy-head logit planes (AlphaZero move-type layout) and the
             probability mass per move type
  value      value-head conv maps, their contribution to the value, hidden units and WDL
  saliency   which squares and pieces the value and the best move depend on
  weights    first-layer filters, per-channel weight norms, BN scales, policy 1x1 conv

Orientation: every board is drawn from White's side (rank 8 at the top) regardless of the
side to move, like the rest of visualisenet. The network sees the board rank-flipped when
Black is to move, so activation planes go through `to_display_grid`, and anything indexed
by physical square through `square_grid`. Values are shown from White's perspective unless
a title says "side to move". Static weights have no position, so they are drawn in the
network's own frame: side to move at the bottom.
"""

import chess
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from decoder import _flip_rank, convert_board, decode_nn_output, extract_policy
from lib.mapping.mapping import CHESS_FLAT_TO_CONV

SEQ = "magma"  # unsigned quantities (activations, norms, probabilities)
DIV = "RdBu_r"  # signed quantities, red = positive (good for White / side to move)
BOARD_CMAP = ListedColormap(["#e8dcc4", "#9c7b5b"])  # light, dark squares
TOP_CHANNELS = 8  # channels per layer in the trunk view
TOP_MOVES = 5  # arrows on the overview board
KERNEL_CHANNELS = 8  # output channels in the first-layer kernel view

INPUT_NAMES = [
    "white to move", "black to move",
    "us O-O", "us O-O-O", "them O-O", "them O-O-O",
    "repetitions", "halfmove clock",
    "us P", "us N", "us B", "us R", "us Q", "us K",
    "them P", "them N", "them B", "them R", "them Q", "them K",
    "en passant",
]
N_SCALARS = 8

# AlphaZero 73-plane policy layout (checked against lib/mapping/chess_flat_to_conv.txt):
# planes 0..55 are 8 directions x 7 distances, 56..63 knight jumps, 64..72 underpromotions
# (capture left, push, capture right) x (rook, bishop, knight). Directions are in the side
# to move's frame: "up" is towards the opponent.
DIRECTIONS = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖"]
KNIGHTS = ["↑↑→", "→→↑", "→→↓", "↓↓→", "↓↓←", "←←↓", "←←↑", "↑↑←"]
UNDERPROMOS = [f"{d}={p}" for d in ("↖", "↑", "↗") for p in ("R", "B", "N")]
POLICY_PLANE_NAMES = (
    [f"{d} {n}" for d in DIRECTIONS for n in range(1, 8)] + KNIGHTS + UNDERPROMOS
)
GLYPHS = {
    chess.PAWN: "♟", chess.KNIGHT: "♞", chess.BISHOP: "♝",
    chess.ROOK: "♜", chess.QUEEN: "♛", chess.KING: "♚",
}
_FONT = FontProperties(family="DejaVu Sans")
_GLYPH_PATHS = {}


# ---------------------------------------------------------------------------
# orientation helpers
# ---------------------------------------------------------------------------


def to_display_grid(plane, board):
    """Orient an 8x8 plane in the network's frame (row = canonical rank) as White sees it.

    Row r of the input is rank r for White to move and rank 7-r for Black to move (the
    encoder rank-flips for Black). imshow puts row 0 at the top, so flip White's planes to
    get rank 8 at the top; Black's are already in that order.
    """
    return np.flipud(plane) if board.turn == chess.WHITE else plane


def network_frame(plane):
    """Draw a plane in the network's own frame: side to move's first rank at the bottom."""
    return np.flipud(plane)


def square_grid(values):
    """64 values indexed by physical square -> 8x8 grid with rank 8 at the top."""
    grid = np.zeros((8, 8), dtype=np.float32)
    for sq, v in values.items() if isinstance(values, dict) else enumerate(values):
        grid[7 - chess.square_rank(sq), chess.square_file(sq)] = v
    return grid


def to_physical(move, board):
    """Map a canonical (side-to-move frame) move back to physical squares."""
    if board.turn == chess.WHITE:
        return move
    return chess.Move(
        _flip_rank(move.from_square), _flip_rank(move.to_square), promotion=move.promotion
    )


def white_sign(board):
    """+1 if side-to-move quantities already read from White's side, else -1."""
    return 1.0 if board.turn == chess.WHITE else -1.0


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------


def _glyph_path(ch):
    if ch not in _GLYPH_PATHS:
        path = TextPath((0, 0), ch, size=1, prop=_FONT)
        ext = path.get_extents()
        _GLYPH_PATHS[ch] = Affine2D().translate(
            -(ext.x0 + ext.x1) / 2, -(ext.y0 + ext.y1) / 2
        ).transform_path(path)
    return _GLYPH_PATHS[ch]


def clean(ax, title=None, fontsize=8):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=fontsize)


def symmetric_limit(*arrays):
    m = max(float(np.abs(a).max()) for a in arrays)
    return m if m > 0 else 1.0


def draw_board(ax, board, heat=None, cmap=SEQ, vmin=None, vmax=None, alpha=0.9,
               title=None, colorbar=False, annotate=None):
    """Board from White's side, optional 8x8 heatmap (rank 8 at the top) under the pieces.

    `annotate` maps physical squares to short strings printed in the square's corner.
    Pieces are vector paths in data coordinates, so they scale with the axes.
    """
    ax.imshow(np.indices((8, 8)).sum(0) % 2, cmap=BOARD_CMAP, vmin=0, vmax=1)
    im = None
    if heat is not None:
        # opacity follows magnitude, so quiet squares show the board through
        heat = np.asarray(heat, dtype=np.float32)
        norm = Normalize(vmin=heat.min() if vmin is None else vmin,
                         vmax=heat.max() if vmax is None else vmax)
        im = ScalarMappable(norm=norm, cmap=cmap)
        rgba = im.to_rgba(heat)
        t = np.clip(norm(heat), 0, 1)
        strength = np.abs(2 * t - 1) if norm.vmin < 0 < norm.vmax else t
        rgba[..., 3] = alpha * (0.12 + 0.88 * strength)
        ax.imshow(rgba)
    for sq, piece in board.piece_map().items():
        x, y = chess.square_file(sq), 7 - chess.square_rank(sq)
        white = piece.color == chess.WHITE
        # imshow's y axis points down, so flip the glyph vertically
        tr = Affine2D().scale(0.78, -0.78).translate(x, y) + ax.transData
        ax.add_patch(PathPatch(
            _glyph_path(GLYPHS[piece.piece_type]), transform=tr,
            facecolor="#fafafa" if white else "#151515",
            edgecolor="#151515" if white else "#fafafa", lw=0.6,
        ))
    for sq, text in (annotate or {}).items():
        ax.text(chess.square_file(sq) + 0.45, 7 - chess.square_rank(sq) + 0.45, text,
                ha="right", va="bottom", fontsize=6, color="white",
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground="black")])
    ax.set_xticks(range(8))
    ax.set_xticklabels("abcdefgh", fontsize=7)
    ax.set_yticks(range(8))
    ax.set_yticklabels("87654321", fontsize=7)
    ax.tick_params(length=0)
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(7.5, -0.5)
    if title:
        ax.set_title(title, fontsize=9)
    if colorbar and im is not None:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    return im


def draw_arrow(ax, move, prob, scale, colour="#39d353"):
    x0, y0 = chess.square_file(move.from_square), 7 - chess.square_rank(move.from_square)
    x1, y1 = chess.square_file(move.to_square), 7 - chess.square_rank(move.to_square)
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.8",
                        lw=1.5 + 6 * prob / scale, color=colour, alpha=0.85,
                        shrinkA=0, shrinkB=4),
    )


def tile(ax, grid, title=None, cmap=SEQ, vmin=None, vmax=None, fontsize=6):
    ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
    clean(ax, title, fontsize)


# ---------------------------------------------------------------------------
# network probe
# ---------------------------------------------------------------------------


class Probe:
    """One forward pass over a position with every interesting tensor captured.

    `acts` keys: input, start, res0..resN-1, trunk_norm, policy_hidden, policy_planes,
    value_conv, value_hidden (all batch-squeezed numpy arrays).
    """

    def __init__(self, net, board):
        self.net, self.board = net, board
        self.device = next(net.parameters()).device
        x, _ = convert_board(board)
        self.x = x.to(self.device)

        named = [("start", net.startBlock)]
        named += [(f"res{i}", b) for i, b in enumerate(net.backBone)]
        named += [
            ("trunk_norm", net.backBoneNorm),
            ("policy_hidden", net.policyHead[2]),
            ("policy_planes", net.policyHead[3]),
            ("value_conv", net.valueHead[2]),
            ("value_hidden", net.valueHead[5]),
        ]
        self.n_res = len(net.backBone)
        raw = {}

        def keep(name):
            def hook(_m, _i, o):
                if name == "value_conv":
                    o.retain_grad()
                raw[name] = o
            return hook

        handles = [m.register_forward_hook(keep(n)) for n, m in named]
        inp = self.x.unsqueeze(0).clone().requires_grad_(True)
        try:
            value, policy = net(inp)
        finally:
            for h in handles:
                h.remove()

        (self.value, self.wdl, self.moves_left, self.legal, self.lookup,
         self.probs, best) = decode_nn_output(value.detach(), policy.detach(), board)
        self.best_move = chess.Move.from_uci(best)
        _, self.legal_idx = extract_policy(board)

        # gradient of the side-to-move value (tanh of the scalar head) ...
        torch.tanh(value[0, 0]).backward(retain_graph=True)
        self.value_grad = inp.grad[0].detach().cpu().numpy()
        vc = raw["value_conv"]
        self.value_conv_attr = (vc.grad * vc)[0].detach().cpu().numpy()
        # ... and of the best move's logit
        inp.grad = None
        best_idx = self.legal_idx[int(np.argmax(self.probs))]
        policy[0, best_idx].backward()
        self.best_grad = inp.grad[0].detach().cpu().numpy()

        self.acts = {k: v.detach()[0].cpu().numpy() for k, v in raw.items()}
        self.acts["input"] = self.x.cpu().numpy()
        self.policy_logits = policy.detach()[0].cpu().numpy()

    def white_value(self):
        return white_sign(self.board) * self.value

    def white_wdl(self):
        w, d, l = self.wdl
        return (w, d, l) if self.board.turn == chess.WHITE else (l, d, w)

    def physical_moves(self):
        """[(physical move, probability, canonical flat index)] sorted by probability."""
        rows = [
            (to_physical(mv, self.board), p, i)
            for mv, p, i in zip(self.legal, self.probs, self.legal_idx)
        ]
        return sorted(rows, key=lambda r: -r[1])

    def layers(self):
        return ["start"] + [f"res{i}" for i in range(self.n_res)] + ["trunk_norm"]

    def occlusion(self):
        """White-perspective value change when each non-king piece is removed.

        Positive = the piece is worth that much to White. Evaluated on stack-free copies
        (no repetition history) so baseline and occluded positions are comparable.
        """
        base = self.board.copy(stack=False)
        squares = [sq for sq, p in base.piece_map().items() if p.piece_type != chess.KING]
        boards = [base]
        for sq in squares:
            b = base.copy(stack=False)
            b.remove_piece_at(sq)
            boards.append(b)
        xs = torch.stack([convert_board(b)[0] for b in boards]).to(self.device)
        with torch.no_grad():
            value, _ = self.net(xs)
        v = torch.tanh(value[:, 0]).cpu().numpy() * white_sign(self.board)
        return {sq: float(v[0] - v[k + 1]) for k, sq in enumerate(squares)}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _wdl_panel(ax, probe):
    w, d, l = probe.white_wdl()
    for left, width, colour, label in (
        (0, w, "#f0f0f0", f"W {w:.2f}"),
        (w, d, "#808080", f"D {d:.2f}"),
        (w + d, l, "#202020", f"L {l:.2f}"),
    ):
        ax.barh(0, width, left=left, color=colour, edgecolor="#555")
        if width > 0.08:
            ax.text(left + width / 2, 0, label, ha="center", va="center", fontsize=8,
                    color="black" if colour != "#202020" else "white")
    ax.set_xlim(0, 1)
    clean(ax)
    side = "White" if probe.board.turn == chess.WHITE else "Black"
    ax.set_title(
        f"Value (White) {probe.white_value():+.3f}   "
        f"moves left {probe.moves_left:.1f}   {side} to move",
        fontsize=9,
    )


def fig_overview(probe, title):
    board = probe.board
    moves = probe.physical_moves()
    from_heat, to_heat = np.zeros(64), np.zeros(64)
    for mv, p, _ in moves:
        from_heat[mv.from_square] += p
        to_heat[mv.to_square] += p

    fig = plt.figure(figsize=(17, 9.5))
    fig.suptitle(title)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.12], width_ratios=[1.35, 1, 1, 1])

    ax = fig.add_subplot(gs[0:2, 0])
    draw_board(ax, board, square_grid(from_heat), vmin=0, colorbar=True,
               title="Policy: probability by from-square, top moves")
    for mv, p, _ in moves[:TOP_MOVES][::-1]:
        draw_arrow(ax, mv, p, moves[0][1])

    draw_board(fig.add_subplot(gs[0, 1]), board, square_grid(to_heat), vmin=0,
               colorbar=True, title="Policy: probability by to-square")

    occ = probe.occlusion()
    lim = symmetric_limit(np.array(list(occ.values()) or [0.0]))
    draw_board(fig.add_subplot(gs[0, 2]), board, square_grid(occ), cmap=DIV,
               vmin=-lim, vmax=lim, colorbar=True,
               title="Piece worth to White\n(value drop when removed)",
               annotate={sq: f"{v:+.2f}" for sq, v in occ.items()})

    sal = (probe.value_grad[N_SCALARS:] * probe.acts["input"][N_SCALARS:]).sum(0)
    sal = to_display_grid(sal, board) * white_sign(board)
    lim = symmetric_limit(sal)
    draw_board(fig.add_subplot(gs[0, 3]), board, sal, cmap=DIV, vmin=-lim, vmax=lim,
               colorbar=True, title="Value saliency (input × gradient, White)")

    mean_act = to_display_grid(probe.acts["trunk_norm"].mean(0), board)
    draw_board(fig.add_subplot(gs[1, 1]), board, mean_act, cmap="viridis", alpha=0.75,
               colorbar=True, title="Trunk output: mean activation")

    hidden = probe.acts["value_hidden"]
    side = int(np.ceil(np.sqrt(hidden.size)))
    padded = np.full(side * side, np.nan)
    padded[: hidden.size] = hidden
    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow(padded.reshape(side, side), cmap=SEQ)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    clean(ax, f"Value hidden layer ({hidden.size} units)\n"
              f"{int((hidden > 0).sum())} active", 9)

    ax = fig.add_subplot(gs[1, 3])
    top = moves[:12]
    ax.barh(range(len(top)), [p for _, p, _ in top], color="#39d353")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([board.san(m) for m, _, _ in top], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1e-6, top[0][1]) * 1.15)
    ax.tick_params(labelsize=7)
    ax.set_title(f"Top policy moves ({len(moves)} legal)", fontsize=9)

    _wdl_panel(fig.add_subplot(gs[2, :]), probe)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return fig


def fig_inputs(probe, title):
    board, x = probe.board, probe.acts["input"]
    fig, axes = plt.subplots(3, 8, figsize=(16, 6.6))
    fig.suptitle(f"{title}\nInput planes (drawn from White's side; the network sees "
                 "ranks flipped when Black is to move)")
    for k, ax in enumerate(axes.flat):
        if k >= len(INPUT_NAMES):
            ax.axis("off")
            continue
        if k < N_SCALARS:
            v = float(x[k, 0, 0])
            ax.imshow(np.full((8, 8), min(v, 1.0)), cmap=SEQ, vmin=-0.15, vmax=1)
            ax.text(3.5, 3.5, f"{v:g}", ha="center", va="center", fontsize=14,
                    color="white" if v < 0.5 else "black")
            clean(ax, INPUT_NAMES[k], 9)
        else:
            draw_board(ax, chess.Board(None), to_display_grid(x[k], board), cmap=SEQ,
                       vmin=0, vmax=1, alpha=1.0, title=INPUT_NAMES[k])
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return fig


def fig_trunk(probe, title):
    board, acts, layers = probe.board, probe.acts, probe.layers()
    fixed = ["mean", "max", "|residual update|"]
    n_cols = len(fixed) + TOP_CHANNELS
    fig = plt.figure(figsize=(1.35 * n_cols + 1, 1.35 * len(layers) + 3.2))
    fig.suptitle(f"{title}\nTrunk activations (each tile scaled to its own range; "
                 f"top {TOP_CHANNELS} channels by spatial spread)")
    gs = fig.add_gridspec(len(layers) + 1, n_cols + 1,
                          height_ratios=[1] * len(layers) + [2.2],
                          width_ratios=[0.6] + [1] * n_cols)
    prev = None
    for row, name in enumerate(layers):
        a = acts[name]
        lab = fig.add_subplot(gs[row, 0])
        lab.axis("off")
        lab.text(1, 0.5, {"start": "start", "trunk_norm": "final BN"}.get(
            name, f"res {name[3:]}"), ha="right", va="center", fontsize=9)
        panels = [a.mean(0), a.max(0), None]
        if name.startswith("res") and prev is not None:
            panels[2] = np.linalg.norm(a - prev, axis=0)
        spread = a.reshape(a.shape[0], -1).std(1)
        top = np.argsort(-spread)[:TOP_CHANNELS]
        for col in range(n_cols):
            ax = fig.add_subplot(gs[row, col + 1])
            if col < len(fixed):
                grid, t = panels[col], fixed[col]
                cmap = "viridis" if col == 2 else SEQ
            else:
                c = top[col - len(fixed)]
                grid, t, cmap = a[c], f"ch {c}", SEQ
            if grid is None:
                ax.axis("off")
                continue
            tile(ax, to_display_grid(grid, board), t if row == 0 or col >= len(fixed)
                 else None, cmap=cmap)
        prev = a

    strip = np.stack([acts[n].mean((1, 2)) for n in layers])
    ax = fig.add_subplot(gs[-1, 1:])
    lim = symmetric_limit(strip)
    im = ax.imshow(strip, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto",
                   interpolation="nearest")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(["start"] + [str(i) for i in range(probe.n_res)] + ["BN"],
                       fontsize=7)
    ax.set_xlabel("channel", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("Mean activation per channel per layer", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01).ax.tick_params(labelsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return fig


def fig_policy(probe, title):
    board = probe.board
    planes = probe.acts["policy_planes"]  # (73, 8, 8) raw logits, network frame
    lo, hi = np.percentile(planes, [1, 99])

    # legal moves per (plane, canonical square), and probability per plane
    conv = CHESS_FLAT_TO_CONV.cpu().numpy()
    legal_at = {}
    mass = np.zeros(73)
    for i, p in zip(probe.legal_idx, probe.probs):
        plane, sq = divmod(int(conv[i]), 64)
        legal_at.setdefault(plane, []).append(sq)
        mass[plane] += p

    hid = probe.acts["policy_hidden"]
    n_hid = min(hid.shape[0], 9)
    fig = plt.figure(figsize=(12.5, 19))
    fig.suptitle(f"{title}\nPolicy head: raw logit planes at each from-square "
                 f"(shared scale {lo:.1f}..{hi:.1f}; dots = legal moves; "
                 "arrows in the side to move's frame)")
    gs = fig.add_gridspec(12, 10, height_ratios=[1] * 11 + [3.2],
                          width_ratios=[0.5] + [1] * 9)

    def plane_tile(ax, k, label):
        grid = to_display_grid(planes[k], board)
        ax.imshow(grid, cmap=SEQ, vmin=lo, vmax=hi)
        for sq in legal_at.get(k, []):
            r, c = divmod(sq, 8)
            r = 7 - r if board.turn == chess.WHITE else r
            ax.plot(c, r, "o", ms=2.5, color="#39d353")
        clean(ax, label, 6)

    for d in range(8):
        lab = fig.add_subplot(gs[d, 0])
        lab.axis("off")
        lab.text(0.5, 0.5, DIRECTIONS[d], ha="center", va="center", fontsize=14)
        for n in range(7):
            plane_tile(fig.add_subplot(gs[d, n + 1]), d * 7 + n, f"dist {n + 1}" if d == 0
                       else None)
    for row, (start, names, label) in enumerate(
        ((56, KNIGHTS, "N"), (64, UNDERPROMOS, "=")), start=8
    ):
        lab = fig.add_subplot(gs[row, 0])
        lab.axis("off")
        lab.text(0.5, 0.5, label, ha="center", va="center", fontsize=12)
        for j, name in enumerate(names):
            plane_tile(fig.add_subplot(gs[row, j + 1]), start + j, name)

    lab = fig.add_subplot(gs[10, 0])
    lab.axis("off")
    lab.text(0.5, 0.5, "hid", ha="center", va="center", fontsize=10)
    for c in range(n_hid):
        tile(fig.add_subplot(gs[10, c + 1]), to_display_grid(hid[c], board),
             f"head conv ch {c}")

    sub = gs[11, 1:].subgridspec(1, 2, wspace=0.6)
    ax = fig.add_subplot(sub[0])
    im = ax.imshow(mass[:56].reshape(8, 7), cmap=SEQ, vmin=0)
    ax.set_yticks(range(8))
    ax.set_yticklabels(DIRECTIONS, fontsize=11)
    ax.set_xticks(range(7))
    ax.set_xticklabels(range(1, 8), fontsize=7)
    ax.set_xlabel("distance", fontsize=8)
    ax.set_title("Probability by direction × distance", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.05).ax.tick_params(labelsize=6)

    ax = fig.add_subplot(sub[1])
    ax.barh(range(17), mass[56:], color="#39d353")
    ax.set_yticks(range(17))
    ax.set_yticklabels([f"N {k}" for k in KNIGHTS] + UNDERPROMOS, fontsize=7)
    ax.invert_yaxis()
    ax.tick_params(labelsize=7)
    ax.set_title("Probability: knight jumps and underpromotions", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return fig


def fig_value(probe, title):
    board, net = probe.board, probe.net
    conv = probe.acts["value_conv"]  # (C, 8, 8) after ReLU
    attr = probe.value_conv_attr * white_sign(board)  # contribution to value, White
    w1 = net.valueHead[4].weight.detach().cpu().numpy()  # (hidden, C*64)
    n = conv.shape[0]
    readout = np.linalg.norm(w1.reshape(w1.shape[0], n, 8, 8), axis=0)

    fig = plt.figure(figsize=(3.2 * n + 3.5, 10.5))
    fig.suptitle(f"{title}\nValue head")
    gs = fig.add_gridspec(4, n + 1, height_ratios=[1, 1, 1, 0.25],
                          width_ratios=[1] * n + [1.3])
    lim = symmetric_limit(attr)
    for c in range(n):
        draw_board(fig.add_subplot(gs[0, c]), board, to_display_grid(conv[c], board),
                   vmin=0, colorbar=True, title=f"conv ch {c} activation")
        draw_board(fig.add_subplot(gs[1, c]), board, to_display_grid(attr[c], board),
                   cmap=DIV, vmin=-lim, vmax=lim, colorbar=True,
                   title=f"ch {c} contribution to value (White)")
        ax = fig.add_subplot(gs[2, c])
        im = ax.imshow(network_frame(readout[c]), cmap=SEQ)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
        clean(ax, f"ch {c} readout weight norm\n(network frame, side to move below)", 8)

    total = attr.sum(0)
    draw_board(fig.add_subplot(gs[0, n]), board, to_display_grid(total, board),
               cmap=DIV, vmin=-lim * 1.5, vmax=lim * 1.5, colorbar=True,
               title="all channels: contribution (White)")
    hidden = probe.acts["value_hidden"]
    side = int(np.ceil(np.sqrt(hidden.size)))
    padded = np.full(side * side, np.nan)
    padded[: hidden.size] = hidden
    ax = fig.add_subplot(gs[1, n])
    im = ax.imshow(padded.reshape(side, side), cmap=SEQ)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    clean(ax, f"hidden layer ({int((hidden > 0).sum())}/{hidden.size} active)", 9)

    w2 = net.valueHead[6].weight.detach().cpu().numpy()  # (5, hidden)
    ax = fig.add_subplot(gs[2, n])
    lim2 = symmetric_limit(w2)
    ax.imshow(w2[:, :64], cmap=DIV, vmin=-lim2, vmax=lim2, aspect="auto")
    ax.set_yticks(range(5))
    ax.set_yticklabels(["value", "W", "D", "L", "moves left"], fontsize=7)
    ax.set_xticks([])
    ax.set_title("output weights (first 64 hidden units)", fontsize=8)

    _wdl_panel(fig.add_subplot(gs[3, :]), probe)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return fig


def fig_saliency(probe, title):
    board = probe.board
    sign = white_sign(board)
    piece_grad = probe.value_grad[N_SCALARS:]
    piece_in = probe.acts["input"][N_SCALARS:]
    panels = [
        ("Value: input × gradient (White)",
         to_display_grid((piece_grad * piece_in).sum(0), board) * sign, DIV),
        ("Value: |gradient| (where a piece would matter)",
         to_display_grid(np.abs(piece_grad).sum(0), board), SEQ),
        (f"Best move {board.san(probe.best_move)}: |gradient| of its logit",
         to_display_grid(np.abs(probe.best_grad[N_SCALARS:]).sum(0), board), SEQ),
    ]
    occ = probe.occlusion()
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.4))
    fig.suptitle(f"{title}\nSaliency")
    lim = symmetric_limit(np.array(list(occ.values()) or [0.0]))
    draw_board(axes[0], board, square_grid(occ), cmap=DIV, vmin=-lim, vmax=lim,
               colorbar=True, title="Occlusion: value drop when piece removed (White)",
               annotate={sq: f"{v:+.2f}" for sq, v in occ.items()})
    for ax, (t, grid, cmap) in zip(axes[1:], panels):
        lim = symmetric_limit(grid)
        draw_board(ax, board, grid, cmap=cmap, vmin=-lim if cmap == DIV else 0, vmax=lim,
                   colorbar=True, title=t)
    if probe.best_move:
        draw_arrow(axes[3], probe.best_move, 1, 1)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return fig


def fig_weights(net, title):
    def w(t):
        return t.detach().cpu().numpy()

    start = w(net.startBlock[0].weight)  # (H, 21, 3, 3)
    n_res = len(net.backBone)
    fig = plt.figure(figsize=(18, 15))
    fig.suptitle(f"{title}\nWeights (static; network frame)")
    gs = fig.add_gridspec(4, 2, height_ratios=[1.1, 1.6, 1, 1])

    ax = fig.add_subplot(gs[0, :])
    norms = np.linalg.norm(start.reshape(*start.shape[:2], -1), axis=2).T
    im = ax.imshow(norms, cmap=SEQ, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(INPUT_NAMES)))
    ax.set_yticklabels(INPUT_NAMES, fontsize=7)
    ax.set_xlabel("start conv output channel", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("Start conv: kernel L2 norm per (input plane, output channel)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01).ax.tick_params(labelsize=6)

    # 3x3 kernels of the strongest output channels on the 13 board planes
    top = np.argsort(-np.linalg.norm(start.reshape(start.shape[0], -1), axis=1))
    top = top[:KERNEL_CHANNELS]
    sub = gs[1, :].subgridspec(KERNEL_CHANNELS, 14, wspace=0.08, hspace=0.12)
    lim = symmetric_limit(start[top][:, N_SCALARS:])
    for r, c in enumerate(top):
        lab = fig.add_subplot(sub[r, 0])
        lab.axis("off")
        lab.text(1, 0.5, f"ch {c}", ha="right", va="center", fontsize=8)
        for j in range(13):
            ax = fig.add_subplot(sub[r, j + 1])
            ax.imshow(network_frame(start[c, N_SCALARS + j]), cmap=DIV, vmin=-lim,
                      vmax=lim)
            clean(ax, INPUT_NAMES[N_SCALARS + j] if r == 0 else None, 7)

    conv_norms = np.stack([
        np.linalg.norm(w(getattr(b, conv).weight).reshape(
            b.conv1.weight.shape[0], -1), axis=1)
        for b in net.backBone for conv in ("conv1", "conv2")
    ])
    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(conv_norms, cmap=SEQ, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(2 * n_res))
    ax.set_yticklabels([f"{i}.{k}" for i in range(n_res) for k in (1, 2)], fontsize=6)
    ax.set_xlabel("output channel", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_title("Residual convs: L2 norm per output channel", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03).ax.tick_params(labelsize=6)

    gammas = np.stack(
        [w(net.startBlock[1].weight)]
        + [w(getattr(b, bn).weight) for b in net.backBone for bn in ("bn1", "bn2")]
        + [w(net.backBoneNorm.weight)]
    )
    ax = fig.add_subplot(gs[2, 1])
    lim = symmetric_limit(gammas)
    im = ax.imshow(gammas, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto",
                   interpolation="nearest")
    ax.set_yticks(range(len(gammas)))
    ax.set_yticklabels(["start"] + [f"{i}.{k}" for i in range(n_res) for k in (1, 2)]
                       + ["final"], fontsize=6)
    ax.set_xlabel("channel", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_title("BatchNorm scale γ (≈0 = channel switched off)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03).ax.tick_params(labelsize=6)

    pw = w(net.policyHead[3].weight)[:, :, 0, 0]  # (73, head_ch)
    ax = fig.add_subplot(gs[3, 0])
    lim = symmetric_limit(pw)
    im = ax.imshow(pw.T, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(range(0, 73, 7))
    ax.set_xticklabels([POLICY_PLANE_NAMES[k] for k in range(0, 73, 7)], fontsize=7)
    ax.set_ylabel("policy conv ch", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("Policy 1×1 conv: weight per (move plane, hidden channel)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03).ax.tick_params(labelsize=6)

    vw = w(net.valueHead[0].weight)[:, :, 0, 0]  # (value_ch, H)
    ax = fig.add_subplot(gs[3, 1])
    lim = symmetric_limit(vw)
    im = ax.imshow(vw, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto",
                   interpolation="nearest")
    ax.set_xlabel("trunk channel", fontsize=8)
    ax.set_ylabel("value conv ch", fontsize=8)
    ax.set_yticks(range(vw.shape[0]))
    ax.tick_params(labelsize=7)
    ax.set_title("Value 1×1 conv: weight per (value channel, trunk channel)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03).ax.tick_params(labelsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return fig


def all_figures(net, board, title):
    """Yield (name, figure) for every view of `board`."""
    probe = Probe(net, board)
    yield "overview", fig_overview(probe, title)
    yield "inputs", fig_inputs(probe, title)
    yield "trunk", fig_trunk(probe, title)
    yield "policy", fig_policy(probe, title)
    yield "value", fig_value(probe, title)
    yield "saliency", fig_saliency(probe, title)
    yield "weights", fig_weights(net, title)
