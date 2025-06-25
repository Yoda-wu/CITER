import argparse
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from src.models.conf_model import ConfidenceModel
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="MLP model fine-tuning script with early stopping and dynamic saving."
    )

    # data and model paths
    parser.add_argument(
        "--data_path",
        type=str,
        default="path/to/train_iter_1.pkl",
        help="Path to the training data (pkl file).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/path/to/model",
        help="Path where the mlp model will be saved.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-1.5B",
        help="Pretrained model name (used for mlp config).",
    )

    # training configurations
    parser.add_argument("--batch_size", type=int, default=80, help="Batch size.")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-7, help="Learning rate."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1000, help="Number of epochs."
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for early stopping."
    )

    # CUDA
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default="0,1",
        help="Specify which CUDA devices to use (e.g., '0,1').",
    )

    # wandb configuration
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use WandB for logging (default: False).",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mlp_training",
        help="WandB project name (if using WandB).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    if args.use_wandb:
        import wandb

        wandb.init(project=args.wandb_project)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    class TokenDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]["hidden_states"], self.data[idx]["label"]

    with open(args.data_path, "rb") as f:
        training_data = pickle.load(f)

    train_size = int(0.8 * len(training_data))
    val_size = len(training_data) - train_size
    train_dataset, val_dataset = random_split(
        TokenDataset(training_data), [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    config = AutoConfig.from_pretrained(args.model_name)
    mlp_model = ConfidenceModel(config).to(device)

    # training settings
    loss_fun = nn.BCELoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=args.learning_rate)

    num_epochs = args.num_epochs
    patience = args.patience
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        running_loss = 0.0
        mlp_model.train()

        # training
        for hidden_states, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"
        ):
            hidden_states = hidden_states.to(device)
            labels = labels.float().to(device)

            outputs = mlp_model(hidden_states.unsqueeze(1))
            loss = loss_fun(outputs.squeeze(), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}")

        # validation
        mlp_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for hidden_states, labels in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
            ):
                hidden_states = hidden_states.to(device)
                labels = labels.float().to(device)

                outputs = mlp_model(hidden_states.unsqueeze(1))
                loss = loss_fun(outputs.squeeze(), labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")

        if args.use_wandb:
            wandb.log(
                {"Training Loss": avg_train_loss, "Validation Loss": avg_val_loss}
            )

        # check if validation loss improved, use early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_model_state = mlp_model.state_dict().copy()
            print(
                f"New best model found at Epoch {epoch+1} with Validation Loss: {best_val_loss:.4f}"
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    print("Training complete.")

    # model name will be, e.g. mlp_iter_2876.pth with best validation loss 0.2876
    formatted_loss = f"{best_val_loss:.4f}".split(".")[1]
    os.makedirs(args.output_path, exist_ok=True)
    best_model_path = os.path.join(args.output_path, f"mlp_iter_{formatted_loss}.pth")

    torch.save(best_model_state, best_model_path)
    print(f"Best model saved as {best_model_path}")


if __name__ == "__main__":
    main()
