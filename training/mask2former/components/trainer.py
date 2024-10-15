from typing import Mapping

import os
import json
import math

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor, get_scheduler

from accelerate import Accelerator
from accelerate.utils import set_seed

from .dataset import Mask2FormerDataset


class Mask2FormerTrainer:
    def __init__(self,
                 dataset_dir: str,
                 runs_dir: str,
                 pretrained_model_name: str = None,
                 image_height: int = 384,
                 image_width: int = 384,
                 batch_size: int = 4,
                 num_workers: int = 0,
                 learning_rate: float = 0.0001,
                 lr_scheduler_type: str = "constant",
                 mixed_precision: str = "no",
                 train_epochs: int = 10,
                 train_steps: int = None,
                 gradient_accumulation_steps: int = 1,
                 warmup_steps: int = 5,
                 checkpoint_steps: int = 25,
                 seed: int = 42) -> None:
        # Paths
        self.dataset_dir = dataset_dir

        self.train_dir = os.path.join(dataset_dir, "train")
        self.val_dir = os.path.join(dataset_dir, "val")
        self.test_dir = os.path.join(dataset_dir, "test")

        self.runs_dir = runs_dir
        self.__output_dir = None
        self.best_model_dir = os.path.join(self.output_dir, "best")
        self.last_model_dir = os.path.join(self.output_dir, "last")

        self.classes_path = os.path.join(dataset_dir, 'classes.txt')

        # ID and label
        self.__id2label = None
        self.__label2id = None

        # Seed
        self.seed = seed

        # Image processor
        self.image_processor = Mask2FormerImageProcessor.from_pretrained(pretrained_model_name,
                                                                         do_resize=True,
                                                                         do_normalize=True,
                                                                         do_reduce_labels=False)
        # ignore_index=255)

        self.image_processor.size = {
            "height": image_height, "width": image_width}

        # Model
        self.pretrained_model_name = pretrained_model_name
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(pretrained_model_name,
                                                                         label2id=self.label2id,
                                                                         id2label=self.id2label,
                                                                         ignore_mismatched_sizes=True)

        # Training parameters
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Data
        self.dataloader_args = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": self.collate_fn,
            "persistent_workers": False}
        self.__setup_data()

        # Epochs and steps
        self.train_epochs = train_epochs

        self.train_steps = train_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.checkpoint_steps = checkpoint_steps
        self.__setup_steps()

        # LR, Optimizer
        self.accelerator = Accelerator(mixed_precision=mixed_precision,
                                       gradient_accumulation_steps=gradient_accumulation_steps)
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.__setup_optimizers()

        # Setup accelerator
        self.__setup_accelerator()

    def __setup_data(self) -> None:
        train_dataset = Mask2FormerDataset(set_dir=self.train_dir,
                                           image_processor=self.image_processor)
        self.train_dataloader = DataLoader(train_dataset,
                                           shuffle=True,
                                           **self.dataloader_args)

        val_dataset = Mask2FormerDataset(set_dir=self.val_dir,
                                         image_processor=self.image_processor)
        self.val_dataloader = DataLoader(val_dataset,
                                         shuffle=False,
                                         **self.dataloader_args)

        test_dataset = Mask2FormerDataset(set_dir=self.test_dir,
                                          image_processor=self.image_processor)
        self.test_dataloader = DataLoader(test_dataset,
                                          shuffle=False,
                                          **self.dataloader_args)

    def __setup_steps(self) -> None:
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.gradient_accumulation_steps)

        self.overrode_train_steps = False
        if self.train_steps is None:
            self.train_steps = self.train_epochs * num_update_steps_per_epoch
            self.overrode_train_steps = True

    def __setup_optimizers(self) -> None:
        # Optimizer
        self.optimizer = AdamW(list(self.model.parameters()),
                               lr=self.learning_rate)

        # LR scheduler
        num_training_steps = self.train_steps if self.overrode_train_steps else self.train_steps * \
            self.accelerator.num_processes
        self.lr_scheduler = get_scheduler(name=self.lr_scheduler_type,
                                          optimizer=self.optimizer,
                                          num_warmup_steps=self.warmup_steps * self.accelerator.num_processes,
                                          num_training_steps=num_training_steps)

    def __setup_accelerator(self) -> None:
        # Prepare everything with our `accelerator`.
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.test_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
            self.lr_scheduler)

        # Recalculate total training steps
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.gradient_accumulation_steps)

        if self.overrode_train_steps:
            self.train_steps = self.train_epochs * num_update_steps_per_epoch

        # Recalculate number of training epochs
        self.train_epochs = math.ceil(
            self.train_steps / num_update_steps_per_epoch)

        # Set seed
        set_seed(self.seed, device_specific=True)

    @property
    def output_dir(self) -> None:
        if self.__output_dir is None:
            # Create runs dir
            os.makedirs(self.runs_dir, exist_ok=True)

            listdir = os.listdir(self.runs_dir)
            if len(listdir) == 0:
                output_dir = "train"
            elif len(listdir) == 1:
                output_dir = "train2"
            else:
                listdir.sort(key=lambda x: 0 if x ==
                             "train" else int(x.lstrip("train")))

                number = int(listdir[-1].lstrip("train")) + 1
                output_dir = "train" + str(number)

            output_dir = os.path.join(self.runs_dir, output_dir)
            os.makedirs(output_dir)

            self.__output_dir = output_dir

        return self.__output_dir

    @property
    def id2label(self) -> dict[int, str]:
        if self.__id2label is None:
            self.__id2label = self.__create_id2label()

        return self.__id2label

    @property
    def label2id(self) -> dict[str, int]:
        if self.__label2id is None:
            self.__label2id = {v: k for k, v in self.id2label.items()}

        return self.__label2id

    def __create_id2label(self) -> dict[int, str]:
        with open(self.classes_path, 'r') as file:
            # Set the names of the classes
            classes = [i.split('\n')[0] for i in file.readlines()]
            id2label = {k: v for k, v in enumerate(classes, start=1)}

        # Set background as 0
        id2label[0] = "background"

        return id2label

    def collate_fn(self, examples) -> dict:
        batch = {}
        batch['pixel_values'] = torch.stack(
            [item['pixel_values'] for item in examples])
        batch['pixel_mask'] = torch.stack(
            [item['pixel_mask'] for item in examples])
        batch['mask_labels'] = [item['mask_labels'] for item in examples]
        batch['class_labels'] = [item['class_labels'] for item in examples]

        return batch

    def nested_cpu(self, tensors):
        if isinstance(tensors, (list, tuple)):
            return type(tensors)(self.nested_cpu(t) for t in tensors)
        elif isinstance(tensors, Mapping):
            return type(tensors)({k: self.nested_cpu(t) for k, t in tensors.items()})
        elif isinstance(tensors, torch.Tensor):
            return tensors.cpu().detach()
        else:
            return tensors

    def evaluation_loop(self, dataloader: DataLoader):
        metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)

        for inputs in tqdm(dataloader,
                           total=len(dataloader)):
            with torch.no_grad():
                outputs = self.model(**inputs)

            inputs = self.accelerator.gather_for_metrics(inputs)
            inputs = self.nested_cpu(inputs)

            outputs = self.accelerator.gather_for_metrics(outputs)
            outputs = self.nested_cpu(outputs)

            post_processed_targets = []
            post_processed_predictions = []
            target_sizes = []

            # Collect targets
            for masks, labels in zip(inputs["mask_labels"], inputs["class_labels"]):
                post_processed_targets.append(
                    {
                        "masks": masks.to(dtype=torch.bool),
                        "labels": labels,
                    })
                target_sizes.append(masks.shape[-2:])

            # Collect predictions
            post_processed_output = self.image_processor.post_process_instance_segmentation(
                outputs,
                threshold=0.0,
                target_sizes=target_sizes,
                return_binary_maps=True)

            for image_predictions, target_size in zip(post_processed_output, target_sizes):
                if image_predictions["segments_info"]:
                    post_processed_image_prediction = {
                        "masks": image_predictions["segmentation"].to(dtype=torch.bool),
                        "labels": torch.tensor([x["label_id"] for x in image_predictions["segments_info"]]),
                        "scores": torch.tensor([x["score"] for x in image_predictions["segments_info"]]),
                    }
                else:
                    # for void predictions, we need to provide empty tensors
                    post_processed_image_prediction = {
                        "masks": torch.zeros([0, *target_size], dtype=torch.bool),
                        "labels": torch.tensor([]),
                        "scores": torch.tensor([]),
                    }
                post_processed_predictions.append(
                    post_processed_image_prediction)

            # Update metric for batch targets and predictions
            metric.update(post_processed_predictions, post_processed_targets)

        # Compute metrics
        metrics = metric.compute()

        # Replace list of per class metrics with separate metric for each class
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_name = self.id2label[class_id.item(
            )] if self.id2label is not None else class_id.item()
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        return metrics

    def train(self) -> None:
        # Progress bar
        progress_bar = tqdm(range(self.train_steps),
                            disable=not self.accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        best_map = 0

        for epoch in range(starting_epoch, self.train_epochs):

            self.model.train()

            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(pixel_values=batch["pixel_values"],
                                         pixel_mask=batch["pixel_mask"],
                                         mask_labels=batch["mask_labels"],
                                         class_labels=batch["class_labels"])
                    loss = outputs.loss
                    # We keep track of the loss at each epoch
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                # Save on checkpoint steps
                if completed_steps % self.checkpoint_steps == 0 and self.accelerator.sync_gradients:
                    save_dir = os.path.join(
                        self.output_dir, f"step_{completed_steps}")
                    self.save_model(save_dir=save_dir)

                # Break if all steps
                if completed_steps >= self.train_steps:
                    break

            metrics = self.evaluation_loop(self.val_dataloader)
            print(f"epoch {epoch}: {metrics}")

            # Save model with best metrics
            if metrics["map"] > best_map:
                best_map = metrics["map"]
                self.save_model(save_dir=self.best_model_dir)

        # Run test evaluation
        metrics = self.evaluation_loop(self.test_dataloader)
        print("Test metrics:", metrics)

        # Save model and image processor
        self.save_model(save_dir=self.last_model_dir)
        if self.accelerator.is_main_process:
            with open(os.path.join(self.output_dir, "all_results.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            self.image_processor.save_pretrained(self.output_dir)

    def save_model(self, save_dir: str) -> None:
        self.accelerator.save_state(save_dir)
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(save_dir,
                                        is_main_process=self.accelerator.is_main_process,
                                        save_function=self.accelerator.save)
