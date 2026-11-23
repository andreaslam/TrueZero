"""Visualise TrueZero network activations, value and policy.

Modes:
  evolution  one frame per checkpoint in nets/ for a fixed FEN, stitched into a GIF
  play       the latest checkpoint plays itself (greedy policy); GIF + MP4 of the game

Boards are always drawn from White's point of view (rank 8 at the top), whichever
side is to move, so frames stay comparable across a game. The network itself sees
the board from the side to move's perspective (ranks flipped when Black is to move),
so activations and policy moves are mapped back to physical squares for display.
"""

import argparse
import os
import re
import shutil
import subprocess
from multiprocessing import Pool

import chess
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from decoder import _flip_rank, convert_board, decode_nn_output
from network import TrueNet
from paths import data_dir, data_path


def extract_number_tz(filename):
    return int(os.path.basename(filename).split("tz_")[1].split(".")[0])


def build_net_from_checkpoint(model_path, device):
    """Rebuild an eager TrueNet matching a TorchScript checkpoint and load its weights.

    Forward hooks need an eager module, and the checkpoint can be any TrueNet-shaped
    net (TrueNet, TrueNetXS, ...), so the sizes are read from the weights.
    """
    scripted = torch.jit.load(model_path, map_location=device)
    sd = scripted.state_dict()
    net = TrueNet(
        num_resBlocks=sum(
            1 for k in sd if k.startswith("backBone.") and k.endswith(".conv1.weight")
        ),
        num_hidden=sd["startBlock.0.weight"].shape[0],
        head_channel_policy=sd["policyHead.0.weight"].shape[0],
        head_channel_values=sd["valueHead.0.weight"].shape[0],
        head_hidden_value=sd["valueHead.4.weight"].shape[0],
        policy_conv_channels=sd["policyHead.3.weight"].shape[0],
    )
    net.load_state_dict(sd)
    return net.to(device).to(torch.float32).eval()


def to_physical_move(uci, board):
    """Map a canonical (side-to-move) UCI move back to physical squares."""
    if board.turn == chess.WHITE:
        return uci
    mv = chess.Move.from_uci(uci)
    return chess.Move(
        _flip_rank(mv.from_square), _flip_rank(mv.to_square), promotion=mv.promotion
    ).uci()


def to_display_grid(plane, board):
    """Orient an 8x8 activation plane (row = canonical rank) as White sees the board.

    Row r of the input is rank r for White to move and rank 7-r for Black to move
    (the encoder rank-flips for Black). imshow puts row 0 at the top, so flip White's
    planes to get rank 8 at the top; Black's are already in that order.
    """
    return np.flipud(plane) if board.turn == chess.WHITE else plane


def fig_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def save_gif(frames, path, fps):
    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def save_video(frames, path, fps, size):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg not found; skipping video")
        return
    h, w, _ = frames[0].shape
    out_w, out_h = (s - s % 2 for s in size)  # yuv420p needs even dimensions
    proc = subprocess.Popen(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "-",
            "-vf", f"scale={out_w}:{out_h}", "-pix_fmt", "yuv420p", path,
        ],
        stdin=subprocess.PIPE,
    )
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()


class ChessVisualiser:
    def __init__(self, model_path, i_d=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_net_from_checkpoint(model_path, self.device)
        self.i_d = i_d
        self.activations_startblock = []
        self.activations_backbone = []

    def _run(self, board):
        """Forward pass with hooks; returns decode_nn_output for the position."""
        input_data, _ = convert_board(board)
        input_data = input_data.unsqueeze(0).to(self.device)
        self.activations_startblock = []
        self.activations_backbone = []
        handles = [
            self.net.startBlock.register_forward_hook(
                lambda m, i, o: self.activations_startblock.append(o)
            )
        ] + [
            block.register_forward_hook(
                lambda m, i, o: self.activations_backbone.append(o)
            )
            for block in self.net.backBone
        ]
        try:
            with torch.no_grad():
                value, policy = self.net(input_data)
        finally:
            for h in handles:
                h.remove()
        return decode_nn_output(value, policy, board)

    def max_policy(self, fen):
        board = chess.Board(fen)
        _, _, _, _, legal_lookup, _, _ = self._run(board)
        return max(legal_lookup.values())

    def render(self, board, title, max_pol=None):
        """Render one figure for `board` and return (rgb frame, best physical move)."""
        value, wdl, _, _, legal_lookup, _, best_move = self._run(board)

        planes = [("Start Block", self.activations_startblock[0])] + [
            (f"Backbone {i}", a) for i, a in enumerate(self.activations_backbone)
        ]
        n_cols = len(planes)

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(max(12, 1.6 * n_cols), 7))
        fig.suptitle(title)
        gs = fig.add_gridspec(2, n_cols, height_ratios=[1, 1.4])

        # top row: mean activation per square, oriented as White sees the board
        for col, (name, act) in enumerate(planes):
            ax = fig.add_subplot(gs[0, col])
            grid = act[0].mean(dim=0).detach().cpu().numpy()
            ax.imshow(to_display_grid(grid, board), cmap="plasma")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name, fontsize=8)

        # value from White's perspective (the net outputs side-to-move)
        white_value = value if board.turn == chess.WHITE else -value
        w, d, l = wdl if board.turn == chess.WHITE else wdl[::-1]
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(
            np.full((8, 8), white_value, dtype=np.float32),
            cmap="plasma",
            vmin=-1,
            vmax=1,
        )
        ax.axis("off")
        ax.set_title(
            f"Value (White): {white_value:+.3f}\nW {w:.2f}  D {d:.2f}  L {l:.2f}",
            fontsize=8,
        )

        # policy over legal moves, labelled with physical squares
        ax = fig.add_subplot(gs[1, 1:])
        labels = [to_physical_move(m, board) for m in legal_lookup]
        probs = list(legal_lookup.values())
        ax.bar(range(len(labels)), probs)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        if max_pol:
            ax.set_ylim(0, max_pol)
        side = "White" if board.turn == chess.WHITE else "Black"
        ax.set_title(f"Policy ({side} to move, best {best_move})")

        fig.tight_layout()
        frame = fig_to_rgb(fig)
        plt.close(fig)
        return frame, chess.Move.from_uci(best_move)

    def play(self):
        board = chess.Board()
        frames = []
        while not board.is_game_over():
            frame, move = self.render(
                board, f"TrueZero self-play, ply {board.ply()}"
            )
            frames.append(frame)
            board.push(move)
        print(f"game over: {board.result()} after {board.ply()} plies")
        return frames


def process_net(args):
    net, max_pol, fen, output_dir, i_d = args
    vis = ChessVisualiser(net, i_d=i_d)
    frame, _ = vis.render(
        chess.Board(fen),
        f"TrueZero visualisation: {os.path.basename(net)}",
        max_pol,
    )
    path = os.path.join(output_dir, f"temp_frame{i_d}.png")
    Image.fromarray(frame).save(path)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Chess Visualisation with Neural Networks"
    )
    parser.add_argument("--mode", choices=["evolution", "play"], required=True)
    parser.add_argument(
        "--nets_dir", default="nets", help="Directory containing neural network models"
    )
    parser.add_argument(
        "--output_dir", default="frames", help="Directory for output frames"
    )
    parser.add_argument(
        "--fen",
        default="rnbqkbnr/pppp1p1p/6p1/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR w KQkq - 0 3",
        help="FEN string for chess board",
    )
    parser.add_argument("--video_name", default="chess_game.mp4")
    parser.add_argument("--gif_name", default="chess_game.gif")
    parser.add_argument(
        "--video_frame_size", type=int, nargs=2, default=(2000, 1000)
    )
    parser.add_argument("--video_fps", type=float, default=2)
    args = parser.parse_args()

    output_dir = data_dir(args.output_dir)
    nets_dir = data_dir(args.nets_dir)
    all_nets = sorted(
        (
            os.path.join(nets_dir, net)
            for net in os.listdir(nets_dir)
            if re.fullmatch(r"tz_\d+\.pt", net)
        ),
        key=extract_number_tz,
    )
    if not all_nets:
        raise SystemExit(f"no tz_<n>.pt checkpoints in {nets_dir}")

    if args.mode == "evolution":
        max_pol = max(ChessVisualiser(net).max_policy(args.fen) for net in all_nets)
        pool_args = [
            (net, max_pol, args.fen, output_dir, i) for i, net in enumerate(all_nets)
        ]
        with Pool() as pool:
            paths = pool.map(process_net, pool_args)
        frames = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
        gif_path = os.path.join(output_dir, "evolution.gif")
        save_gif(frames, gif_path, args.video_fps)
        print(f"wrote {gif_path}")

    elif args.mode == "play":
        frames = ChessVisualiser(all_nets[-1]).play()
        gif_path = os.path.join(output_dir, args.gif_name)
        video_path = os.path.join(output_dir, args.video_name)
        save_gif(frames, gif_path, args.video_fps)
        save_video(frames, video_path, args.video_fps, args.video_frame_size)
        print(f"wrote {gif_path} and {video_path}")


if __name__ == "__main__":
    main()
