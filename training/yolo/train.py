import os
import argparse

from trainer import Trainer


# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")


# Get an arg for epochs
parser.add_argument("--num_epochs",
                    default=2,
                    type=int,
                    help="the number of epochs to train for")

# Get an arg for image size
parser.add_argument("--image_size",
                    default=640,
                    type=int,
                    help="the size of image to train for")


# Get an arg for batch size
parser.add_argument("--batch_size",
                    default=16,
                    type=int,
                    help="the number of batch size to train for")


# Get an arg for yolo model_size
parser.add_argument("--model_type",
                    default='m',
                    type=str,
                    help="Size of yolo segmentation model (n, s, m, l, xl)")

# Get an arg for seed
parser.add_argument("--seed",
                    default=42,
                    type=int,
                    help="Random seed of yolo training")


# Get our arguments from the parser
args = parser.parse_args()

# Define constant variables
NUM_EPOCHS = args.num_epochs
IMAGE_SIZE = args.image_size
BATCH_SIZE = args.batch_size
MODEL_TYPE = args.model_type
SEED = args.seed

HOME = os.getcwd()
DATA = os.path.join(HOME, "data")
DATASET_DIR = os.path.join(DATA, 'dataset')


def train():
    "Trains YOLO model"
    trainer = Trainer(dataset_dir=DATASET_DIR,
                      num_epochs=NUM_EPOCHS,
                      image_size=IMAGE_SIZE,
                      batch_size=BATCH_SIZE,
                      seed=SEED,
                      model_type=MODEL_TYPE)

    # Current device
    print(f"Current device is {trainer.device}.")

    # Training
    result = trainer.train()
    metrics = trainer.validate()


def main():
    train()


if __name__ == "__main__":
    main()
