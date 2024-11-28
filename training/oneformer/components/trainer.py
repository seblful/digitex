from typing import Mapping

import os
import json
import math

from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import AdamW

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from transformers import OneFormerForUniversalSegmentation, OneFormerImageProcessor, OneFormerProcessor, CLIPTokenizer, get_scheduler
from transformers.image_processing_base import BatchFeature

from accelerate import Accelerator
from accelerate.utils import set_seed

from .dataset import OneFormerDataset


class OneFormerTrainer:
    def __init__(self,
                 dataset_dir: str,
                 runs_dir: str,
                 preprocessor_config_path: str,
                 class_info_file_path: str,
                 pretrained_model_name: str = None,
                 width: int = 2048,
                 height: int = 512,
                 batch_size: int = 4,
                 num_workers: int = 0,
                 learning_rate: float = 0.0001,
                 lr_scheduler_type: str = "constant",
                 mixed_precision: str = "no",
                 train_epochs: int = 10,
                 train_steps: int = None,
                 gradient_accumulation_steps: int = 1,
                 warmup_steps: int = 5,
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

        self.preprocessor_config_path = preprocessor_config_path
        self.class_info_file_path = class_info_file_path

        # ID and label
        self.__id2label = None
        self.__label2id = None

        # Seed
        self.seed = seed

        # Image processor
        self.width = width
        self.height = height

        # Processor and model
        self.pretrained_model_name = pretrained_model_name
        self.__processor = None
        self.model = OneFormerForUniversalSegmentation.from_pretrained(pretrained_model_name,
                                                                       is_training=True,
                                                                       id2label=self.id2label,
                                                                       label2id=self.label2id,
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
        train_dataset = OneFormerDataset(set_dir=self.train_dir,
                                         processor=self.processor)
        self.train_dataloader = DataLoader(train_dataset,
                                           shuffle=True,
                                           **self.dataloader_args)

        val_dataset = OneFormerDataset(set_dir=self.val_dir,
                                       processor=self.processor)
        self.val_dataloader = DataLoader(val_dataset,
                                         shuffle=False,
                                         **self.dataloader_args)

        test_dataset = OneFormerDataset(set_dir=self.test_dir,
                                        processor=self.processor)
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
        with open(self.class_info_file_path, "r") as class_info_file:
            id2label = json.load(class_info_file)
            id2label = {int(k): v["name"] for k, v in id2label.items()}

        return id2label

    @property
    def processor(self) -> OneFormerImageProcessor:
        if self.__processor is None:
            image_processor = OneFormerImageProcessor.from_json_file(
                self.preprocessor_config_path)
            image_processor.size = {"width": self.width,
                                    "height": self.height}

            tokenizer = CLIPTokenizer.from_pretrained(
                self.pretrained_model_name)
            self.__processor = OneFormerProcessor(image_processor=image_processor,
                                                  tokenizer=tokenizer)
            self.__processor.image_processor.num_text = self.model.config.num_queries - \
                self.model.config.text_encoder_n_ctx
            self.__processor.save_pretrained(self.output_dir)

        return self.__processor

    def collate_fn(self, batch:  list[BatchFeature]) -> BatchFeature:
        new_batch = {}

        new_batch["pixel_values"] = torch.stack(
            [item['pixel_values'] for item in batch])
        new_batch["pixel_mask"] = torch.stack(
            [item['pixel_mask'] for item in batch])
        new_batch["mask_labels"] = [item["mask_labels"] for item in batch]
        new_batch["class_labels"] = [item["class_labels"] for item in batch]
        new_batch["text_inputs"] = torch.stack(
            [item['text_inputs'] for item in batch])
        new_batch["task_inputs"] = torch.stack(
            [item['task_inputs'] for item in batch])

        return new_batch

    def nested_cpu(self, tensors):
        if isinstance(tensors, (list, tuple)):
            return type(tensors)(self.nested_cpu(t) for t in tensors)
        elif isinstance(tensors, Mapping):
            return type(tensors)({k: self.nested_cpu(t) for k, t in tensors.items()})
        elif isinstance(tensors, torch.Tensor):
            return tensors.cpu().detach()
        else:
            return tensors

    def convert_segmentation_to_mask(self,
                                     image_predictions: dict,
                                     target_size: list[torch.Size]):
        # Find unique instances
        segments_info = image_predictions["segments_info"]
        unique_ids = [x["id"] for x in segments_info]

        # Create zeros mask
        height, width = target_size

        masks = torch.zeros(
            (len(unique_ids), *target_size), dtype=torch.bool)

        # Populate the 3D tensor
        segmentation = image_predictions["segmentation"]
        for idx, instance in enumerate(unique_ids):
            masks[idx] = (segmentation == instance)

        return masks

    def evaluation_loop(self, dataloader: DataLoader) -> dict:
        metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
        loss = 0.0

        for batch in tqdm(dataloader,
                          total=len(dataloader)):
            with torch.no_grad():
                outputs = self.model(**batch)
            loss += outputs.loss

            batch = self.accelerator.gather_for_metrics(batch)
            batch = self.nested_cpu(batch)

            outputs = self.accelerator.gather_for_metrics(outputs)
            outputs = self.nested_cpu(outputs)

            post_processed_targets = []
            post_processed_predictions = []
            target_sizes = []

            # Collect targets
            for masks, labels in zip(batch["mask_labels"], batch["class_labels"]):
                post_processed_targets.append(
                    {
                        "masks": masks.to(dtype=torch.bool),
                        "labels": labels,
                    })
                target_sizes.append(tuple(masks.shape[-2:]))

            # Collect predictions
            post_processed_output = self.processor.post_process_instance_segmentation(outputs,
                                                                                      threshold=0.0,
                                                                                      target_sizes=target_sizes)

            for image_predictions, target_size in zip(post_processed_output, target_sizes):
                if image_predictions["segments_info"]:
                    masks = self.convert_segmentation_to_mask(image_predictions=image_predictions,
                                                              target_size=target_size)
                    post_processed_image_prediction = {
                        "masks": masks,
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
        loss = loss / int(len(dataloader) / self.batch_size)

        # Replace list of per class metrics with separate metric for each class
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_name = self.id2label[class_id.item(
            )] if self.id2label is not None else class_id.item()
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        # Loss
        metrics["loss"] = loss

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        return metrics

    def train(self) -> None:
        # Iterating through epochs
        for epoch in range(self.train_epochs):
            self.model.train()

            progress_bar = tqdm(total=self.train_steps // self.train_epochs,
                                desc=f'Epoch {epoch + 1}/{self.train_epochs}',
                                position=0, leave=True)
            postfix = {"loss": None, "train_loss": None, "val_loss": None}

            completed_steps = 0

            min_loss = float('inf')
            train_loss = 0.0

            # Iterating through batches
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(pixel_values=batch["pixel_values"],
                                         pixel_mask=batch["pixel_mask"],
                                         mask_labels=batch["mask_labels"],
                                         class_labels=batch["class_labels"],
                                         task_inputs=batch["task_inputs"],
                                         text_inputs=batch["text_inputs"])
                    loss = outputs.loss
                    train_loss += loss.item()

                    # Backpropagation and optimization steps
                    self.optimizer.zero_grad()
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    completed_steps += 1
                    progress_bar.update(1)
                    postfix["loss"] = loss.item()
                    progress_bar.set_postfix(postfix, refresh=True)

                # Break if all steps
                if completed_steps >= self.train_steps:
                    break

            # Calculate epoch loss
            postfix["train_loss"] = train_loss / completed_steps
            progress_bar.set_postfix(postfix, refresh=True)

            # Validation
            metrics = self.evaluation_loop(self.val_dataloader)
            print("Val metrics:", metrics)

            # Val loss
            postfix["val_loss"] = metrics["loss"]
            progress_bar.set_postfix(postfix, refresh=True)

            # Save model with min loss
            if metrics["loss"] < min_loss:
                min_loss = metrics["loss"]
                self.save_model(save_dir=self.best_model_dir,
                                metrics=metrics)

        # Run test evaluation
        metrics = self.evaluation_loop(self.test_dataloader)
        print("Test metrics:", metrics)

        # Save last model
        self.save_model(save_dir=self.last_model_dir,
                        metrics=metrics)

    def save_model(self,
                   save_dir: str,
                   metrics: dict) -> None:
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save_state(save_dir)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(save_dir,
                                            is_main_process=self.accelerator.is_main_process,
                                            save_function=self.accelerator.save)
            with open(os.path.join(self.output_dir, "all_results.json"), "w") as f:
                json.dump(metrics, f, indent=2)
