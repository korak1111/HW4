"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

# Created by me, with assitance from notes + Copilot
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
import torch.optim as optim

from .models import load_model, save_model
from .datasets.road_dataset import load_data
import torch.nn as nn

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

def directional_loss(pred: torch.Tensor, target: torch.Tensor, alpha=2.0, beta=1.0):
    """
    Custom loss penalizing longitudinal (x) errors more than lateral (y).
    Args:
        pred: (B, N, 2) predicted waypoints
        target: (B, N, 2) ground-truth waypoints
        alpha: weight for longitudinal (x-axis)
        beta: weight for lateral (y-axis)
    Returns:
        scalar loss
    """
    x_diff = (pred[..., 0] - target[..., 0]) ** 2  # (B, N)
    y_diff = (pred[..., 1] - target[..., 1]) ** 2  # (B, N)

    loss = alpha * x_diff.mean() + beta * y_diff.mean()
    return loss



def train(args):
    # Setup logging
    log_dir = Path(args.log_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = tb.SummaryWriter(log_dir)

    # Load train and validation datasets
    train_loader = load_data("drive_data/train", batch_size=args.batch_size)
    val_loader   = load_data("drive_data/val", batch_size=args.batch_size)
    # Initialize model, optimizer, and loss function
    model = load_model(args.model_name)
    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # loss_fn = nn.MSELoss()
    # loss_fn = lambda pred, target: directional_loss(pred, target, alpha=2.0, beta=1.0)
    loss_fn = nn.HuberLoss(delta=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


    step = 0
    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            track_left = batch["track_left"].to(args.device)    # shape (B, 10, 2)
            track_right = batch["track_right"].to(args.device)  # shape (B, 10, 2)
            waypoints_gt = batch["waypoints"].to(args.device)   # shape (B, 3, 2)
            
            center = (track_left[:, 0] + track_right[:, 0]) / 2  # (B, 2)
            track_left -= center.unsqueeze(1)
            track_right -= center.unsqueeze(1)
            waypoints_gt = waypoints_gt - center.unsqueeze(1)  # (B, 3, 2)

            optimizer.zero_grad()
            waypoints_pred = model(track_left=track_left, track_right=track_right)

            loss = loss_fn(waypoints_pred, waypoints_gt)
            loss.backward()
            optimizer.step()

            logger.add_scalar("train/loss", loss.item(), step)
            total_train_loss += loss.item()
            step += 1

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                center = (track_left[:, 0] + track_right[:, 0]) / 2
                track_left = track_left - center.unsqueeze(1)
                track_right = track_right - center.unsqueeze(1)
                waypoints_gt = waypoints_gt - center.unsqueeze(1)

                waypoints_pred = model(track_left=track_left, track_right=track_right)
                loss = loss_fn(waypoints_pred, waypoints_gt)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{args.num_epochs} - Val Loss: {avg_val_loss:.4f}")
        logger.add_scalar("val/loss", avg_val_loss, epoch)
        scheduler.step()

        # Save model
        save_model(model)

    logger.close()



if __name__ == "__main__":
    print("Time to train")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mlp_planner", help="Path to save the model")
    parser.add_argument("--log_dir", type=str, default="logs", help="TensorBoard log directory")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    args = parser.parse_args()

    # Manually set device here
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args)
    print("Training complete.")



