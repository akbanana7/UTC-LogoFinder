import os
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import torch
import torch.xpu

print(torch.xpu.is_available())
if not torch.xpu.is_available():
    raise RuntimeError("An Intel XPU is required to run this detector")


import numpy
import keras
from keras import layers


# Fill these in before calling train().
imageDirectory = r"dataset/train/image"
labelDirectory = r"dataset/train/label"
weightsPath = "utcDetector.weights.h5"
trainEpochs = 20 # The small training set needs enough updates to overfit.

def getModel():
    """Return a detector whose output is [x, y, width, height, confidence].

    Image inputs are converted to NumPy arrays shaped (height, width, 3), or a
    batch shaped (batch, height, width, 3). Box values are normalized to 0..1.
    """
    inputs = keras.Input(shape=(640, 640, 3), name="image")
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    # Preserve spatial information. GlobalAveragePooling2D makes it
    # impossible for the head to learn where the object is in the image.
    x = layers.MaxPooling2D()(x)
    # Reduce the feature map before flattening.  Flattening an 80x80x128 map
    # creates a very large dense layer and can exhaust XPU memory.
    x = layers.MaxPooling2D()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    # Use logits internally.  A sigmoid output layer can saturate at 0 or 1,
    # leaving almost no gradient when a bad checkpoint is loaded.
    # Keep the initial predictions near zero.  With the default initializer,
    # the very large Flatten -> Dense projection can produce saturated logits
    # on the first update; sigmoid then gives almost no useful gradient and
    # training appears to stop after one epoch.
    outputs = layers.Dense(
        5,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer="zeros",
        name="yoloOutput",
    )(x)

    model = keras.Model(inputs, outputs, name="utcDetector")
    
    return model


def detectorLoss(yTrue, yPred):
    """Compute a differentiable per-batch loss from the model's logits."""
    boxTarget = yTrue[:, :4]
    objectTarget = yTrue[:, 4]
    objectMask = objectTarget[:, None]

    # Decode only for the box regression loss.  Do not detach or convert to
    # NumPy here: Keras must be able to backpropagate through these ops.
    boxPrediction = keras.ops.sigmoid(yPred[:, :4])
    boxError = boxPrediction - boxTarget
    absoluteError = keras.ops.abs(boxError)
    huber = keras.ops.where(
        absoluteError < 1.0,
        0.5 * keras.ops.square(boxError),
        absoluteError - 0.5,
    )
    # Mean over coordinates, then normalize over labeled samples.  This
    # avoids reducing the masked tensor by an unrelated batch-size factor.
    boxLoss = keras.ops.mean(huber, axis=-1) * objectTarget
    boxLoss = keras.ops.sum(boxLoss) / keras.ops.maximum(
        keras.ops.sum(objectTarget), 1.0
    )

    # Stable BCE-with-logits; using probabilities here would lose gradients
    # when the confidence logit becomes large.
    confidenceLogits = yPred[:, 4]
    confidenceLoss = keras.ops.mean(
        keras.ops.maximum(confidenceLogits, 0.0)
        - objectTarget * confidenceLogits
        + keras.ops.log1p(keras.ops.exp(-keras.ops.abs(confidenceLogits)))
    )
    return 5.0 * boxLoss + confidenceLoss


def decodedMae(yTrue, yPred):
    """Mean absolute error after converting logits to detector outputs."""
    prediction = keras.ops.sigmoid(yPred)
    return keras.ops.mean(keras.ops.abs(prediction - yTrue))


def imageInput(image):
    """Convert an image to the model's NumPy input format."""
    image = numpy.asarray(image, dtype="float32")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    return image[numpy.newaxis, ...]


# Keep model creation explicitly inside the Intel XPU device scope.  This is
# needed by the Keras torch backend for placing layer variables on the iGPU.
with torch.device("xpu"):
    model = getModel()
    if not os.path.isfile(weightsPath):
        print(f"Could not find exported weights: {weightsPath}")
    else:
        try:
            model.load_weights(weightsPath)
        except ValueError as error:
            print(f"Could not load incompatible weights {weightsPath}: {error}")




def exportWeights(path=weightsPath):
    """Export the detector weights and return the path used."""
    model.save_weights(path)
    return path


def detect(frame):
    """Return ``[x, y, width, height, confidence]`` for an OpenCV frame.

    ``frame`` must be a BGR image as returned by ``cv2.VideoCapture``. Box
    coordinates are normalized to 0..1, matching the model's output.
    """
    if frame is None or not isinstance(frame, numpy.ndarray):
        raise ValueError("frame must be a non-empty OpenCV image")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape (height, width, 3)")

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = imageInput(image)
    # The model variables are created on XPU.  Move inference input to the
    # same device explicitly; otherwise the torch backend may leave this
    # NumPy-derived tensor on CPU and fail with a device mismatch.
    image = torch.as_tensor(image, device="xpu")
    with torch.no_grad():
        prediction = model(image, training=False)
    prediction = keras.ops.sigmoid(prediction).detach().cpu().numpy()[0]
    return prediction

def train(index):
    """Train using 1.jpg..index.jpg and matching YOLO label files.

    Returns:
        keras.callbacks.History: The history from ``model.fit``. Its ``history``
        attribute is a dictionary mapping metric names such as ``loss`` and
        ``mae`` to lists containing the value recorded after each epoch. The
        final list item is the metric value from the last epoch, and the
        object can also be used to inspect ``epoch`` and ``params``.
    """
    if not imageDirectory or not labelDirectory:
        raise ValueError("Set imageDirectory and labelDirectory first")
    if index < 1:
        raise ValueError("index must be at least 1")
    
    images = []
    targets = []
    for number in range(1, index + 1):
        imagePath = os.path.join(imageDirectory, f"{number}.jpg")
        labelPath = os.path.join(labelDirectory, f"{number}.txt")

        image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {imagePath}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(imageInput(image)[0])

        # Ignore the YOLO object index (the first value); this detector has one
        # class and only needs the first box's x, y, width, and height.
        target = numpy.zeros(5, dtype="float32")
        with open(labelPath, "r", encoding="utf-8") as labelFile:
            for line in labelFile:
                values = line.split()
                if len(values) >= 5:
                    _, x, y, width, height = values[:5]
                    try:
                        box = numpy.asarray([x, y, width, height], dtype="float32")
                    except ValueError as error:
                        raise ValueError(f"Invalid label: {labelPath}") from error
                    if not numpy.isfinite(box).all() or numpy.any(box < 0.0) or numpy.any(box > 1.0):
                        raise ValueError(
                            f"{labelPath} must contain normalized YOLO coordinates (0..1)"
                        )
                    target[:4] = box
                    target[4] = 1.0
                    break
        if target[4] == 0.0:
            raise ValueError(
                f"No valid object label found in {labelPath}; "
                "expected: class x_center y_center width height"
            )
        targets.append(target)

    images = numpy.asarray(images, dtype="float32")
    targets = numpy.asarray(targets, dtype="float32")
    print(
        f"Loaded {len(images)} samples; target range "
        f"{targets[:, :4].min():.3f}..{targets[:, :4].max():.3f}"
    )
    # Avoid CPU/XPU transfers for every batch and use one sample per update;
    # with this small data set, larger batches can average away the position
    # gradient and leave the model at its near-constant initial prediction.
    images = torch.as_tensor(images, device="xpu")
    targets = torch.as_tensor(targets, device="xpu")
    # A clipped, relatively large update can saturate the output logits after
    # the first pass through this small data set.  Use smaller unclipped steps
    # so localization continues improving instead of becoming stuck.
    optimizer = keras.optimizers.Adam(learning_rate=5e-4)
    # Do not let the torch backend capture the first batch into a compiled
    # training step.  On XPU this can make subsequent calls to fit reuse the
    # first step's state and appear to stop learning after epoch one.
    # Running one batch at a time also makes optimizer updates happen on the
    # same device as the model variables.
    model.compile(
        optimizer=optimizer,
        loss=detectorLoss,
        metrics=[decodedMae],
        run_eagerly=True,
        steps_per_execution=1,
    )
    # Return the History object containing per-epoch training metrics.
    history = model.fit(
        images,
        targets,
        epochs=trainEpochs,
        # Keep activation memory bounded on the XPU.
        batch_size=1,
        shuffle=True,
    )
    exportWeights()
    cv2.destroyAllWindows(  )
    return history

if __name__ == "__main__":
    train(14)