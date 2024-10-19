import os
import argparse

from trainer import Trainer


# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

# Get an arg for task type
parser.add_argument("--task_type",
                    default="page",
                    type=str,
                    choices=["page", "question"],
                    help="Type of task type.")

# Get an arg for yolo model_size
parser.add_argument("--model_type",
                    default='m',
                    type=str,
                    help="Size of yolo segmentation model (n, s, m, l, x).")

# Get an arg for pretrained model path
parser.add_argument("--pretrained_model_path",
                    type=str,
                    default=None,
                    help="Previously trained model for this task type.")

# Get an arg for epochs
parser.add_argument("--num_epochs",
                    default=100,
                    type=int,
                    help="The number of training epochs.")

# Get an arg for image size
parser.add_argument("--image_size",
                    default=640,
                    type=int,
                    help="The size of image.")


# Get an arg for batch size
parser.add_argument("--batch_size",
                    default=16,
                    type=int,
                    help="The size of batch.")

# Get an arg for overlap mask
parser.add_argument("--overlap_mask",
                    default=False,
                    action="store_true",
                    help="Determines whether segmentation masks should overlap during training.")


# Get an arg for patience
parser.add_argument("--patience",
                    default=50,
                    type=int,
                    help="Number of epochs to wait without improvement in validation metrics before early stopping the training.")


# Get an arg for seed
parser.add_argument("--seed",
                    default=42,
                    type=int,
                    help="Random seed to reproduce training results.")


# Get arguments from the parser
args = parser.parse_args()

HOME = os.getcwd()
DATA_DIR = os.path.join(HOME, "data", args.task_type)
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')


def main() -> None:
    trainer = Trainer(dataset_dir=DATASET_DIR,
                      model_type=args.model_type,
                      pretrained_model_path=args.pretrained_model_path,
                      num_epochs=args.num_epochs,
                      image_size=args.image_size,
                      batch_size=args.batch_size,
                      overlap_mask=args.overlap_mask,
                      patience=args.patience,
                      seed=args.seed)
    # Train
    trainer.train()
    # Validate
    trainer.validate()

    return


if __name__ == "__main__":
    main()
