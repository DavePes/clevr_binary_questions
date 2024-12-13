#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise
import random
import keras
import numpy as np
import torch
from torchvision.transforms import v2
from cifar10 import CIFAR10
import matplotlib.pyplot as plt
import multiprocessing

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")


class TorchTensorBoardCallback(keras.callbacks.Callback):
    def __init__(self, path):
        self._path = path
        self._writers = {}

    def writer(self, writer):
        if writer not in self._writers:
            import torch.utils.tensorboard
            self._writers[writer] = torch.utils.tensorboard.SummaryWriter(os.path.join(self._path, writer))
        return self._writers[writer]

    def add_logs(self, writer, logs, step):
        if logs:
            for key, value in logs.items():
                self.writer(writer).add_scalar(key, value, step)
            self.writer(writer).flush()

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            if isinstance(getattr(self.model, "optimizer", None), keras.optimizers.Optimizer):
                logs = logs | {"learning_rate": keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)}
            self.add_logs("train", {k: v for k, v in logs.items() if not k.startswith("val_")}, epoch + 1)
            self.add_logs("val", {k[4:]: v for k, v in logs.items() if k.startswith("val_")}, epoch + 1)

transformation = v2.Compose([
v2.RandomResize(30, 36),
v2.RandomRotation(degrees=(-8, 8)),
v2.Pad(4),
v2.RandomCrop(32),
v2.RandomHorizontalFlip(),
v2.ColorJitter(brightness=(0.9,1.1),contrast=(0.9,1.1),saturation=(0.7,1.3),hue=(-0.2,0.2))
#    v2.ColorJitter(brightness=(0.8,1.2),contrast=(0.8,1.2),saturation=(0.6,1.4),hue=(-0.2,0.2))
])
def augmentation_fn(image: np.ndarray) -> np.ndarray:
    image = torch.from_numpy(image).to(torch.uint8)
    image = image.clone().detach().permute(2, 0, 1)
    image = transformation(image).permute(1, 2, 0)
    image = image.numpy()
    return image
        
def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    cifar = CIFAR10()
    cifar.train.data['labels'] = keras.ops.one_hot(cifar.train.data['labels'],10) #[:, 2:30, 2:30, :]
    cifar.dev.data['labels'] = keras.ops.one_hot(cifar.dev.data['labels'],10)
    #cifar.test.data['images'] = cifar.test.data['images'][:, 2:30, 2:30, :]
    def se_block(input, ratio):
        init = input
        filters = init.shape[-1] # Number of filters

        # Squeeze
        se = keras.layers.GlobalAveragePooling2D()(init)
        se = keras.layers.Reshape((1, 1, filters))(se)

        # Excitation
        se = keras.layers.Dense(filters // ratio, activation='relu')(se)  
        se = keras.layers.Dense(filters, activation='sigmoid')(se) 

        # Rescaling
        se = keras.layers.multiply([init, se])
        return se
    # TODO: Create the model and train it
    
    # Define the model architecture
    def create_model():
        inputs = keras.layers.Input([32, 32, 3])
        x = keras.layers.Rescaling(1 / 255)(inputs)
        x = keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = se_block(x,2)
        x = keras.layers.Conv2D(64, (3, 3),padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = se_block(x,2)
        x = keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = se_block(x,4)
        x = keras.layers.Conv2D(128, (3, 3),padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = se_block(x,4)
        x = keras.layers.Conv2D(162, (3, 3), strides=(2, 2), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = se_block(x,6)
        x = keras.layers.Conv2D(162, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = se_block(x,6)
        x = keras.layers.Conv2D(256, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = se_block(x,8)
        x = keras.layers.Conv2D(256, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = se_block(x,8)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(600, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        return model

    model = create_model()
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")])
    # Train the model
    pool = multiprocessing.Pool(8)
    cifar.train.data['images'] = cifar.train.data['images']
    cifar.train.data['labels'] = np.array(cifar.train.data['labels'])
    for i in range(0,100):
        augmented_data = np.array(pool.map(augmentation_fn, cifar.train.data['images']))
        model.fit(augmented_data, cifar.train.data['labels'], epochs=args.epochs, batch_size=args.batch_size,
                validation_data=(cifar.dev.data['images'], cifar.dev.data['labels']),
                callbacks=[TorchTensorBoardCallback(args.logdir)])

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data.
        for probs in model.predict(cifar.test.data['images']):
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
