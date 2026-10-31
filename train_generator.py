import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers


MODEL_PATH = Path(__file__).resolve().parent / "mnist_generator.keras"
LATENT_DIM = 100
NUM_CLASSES = 10
BATCH_SIZE = 128

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)


def build_generator():
    noise = layers.Input(shape=(LATENT_DIM,), name="noise")
    label = layers.Input(shape=(), dtype="int32", name="digit")
    label_embedding = layers.Embedding(NUM_CLASSES, 50)(label)
    x = layers.Concatenate()([noise, label_embedding])
    x = layers.Dense(7 * 7 * 128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Reshape((7, 7, 128))(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    image = layers.Conv2DTranspose(
        1, 4, strides=2, padding="same", activation="tanh", name="generated_image"
    )(x)
    return tf.keras.Model([noise, label], image, name="mnist_generator")


def build_discriminator():
    image = layers.Input(shape=(28, 28, 1), name="image")
    label = layers.Input(shape=(), dtype="int32", name="digit")
    label_image = layers.Embedding(NUM_CLASSES, 28 * 28)(label)
    label_image = layers.Reshape((28, 28, 1))(label_image)
    x = layers.Concatenate(axis=-1)([image, label_image])
    x = layers.Conv2D(64, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    validity = layers.Dense(1, name="validity_logit")(x)
    return tf.keras.Model([image, label], validity, name="mnist_discriminator")


def train(epochs: int) -> None:
    (images, labels), _ = tf.keras.datasets.mnist.load_data()
    images = (images.astype("float32") - 127.5) / 127.5
    images = images[..., None]
    dataset = (
        tf.data.Dataset.from_tensor_slices((images, labels))
        .shuffle(60_000)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    generator = build_generator()
    discriminator = build_discriminator()
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

    @tf.function
    def train_step(real_images, real_labels):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal((batch_size, LATENT_DIM))
        generated_labels = tf.random.uniform(
            (batch_size,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32
        )

        with tf.GradientTape() as discriminator_tape:
            generated_images = generator([noise, generated_labels], training=True)
            real_logits = discriminator([real_images, real_labels], training=True)
            fake_logits = discriminator(
                [generated_images, generated_labels], training=True
            )
            discriminator_loss = cross_entropy(
                tf.ones_like(real_logits) * 0.9, real_logits
            ) + cross_entropy(tf.zeros_like(fake_logits), fake_logits)

        discriminator_gradients = discriminator_tape.gradient(
            discriminator_loss, discriminator.trainable_variables
        )
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables)
        )

        noise = tf.random.normal((batch_size, LATENT_DIM))
        generated_labels = tf.random.uniform(
            (batch_size,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32
        )
        with tf.GradientTape() as generator_tape:
            generated_images = generator([noise, generated_labels], training=True)
            fake_logits = discriminator(
                [generated_images, generated_labels], training=True
            )
            generator_loss = cross_entropy(tf.ones_like(fake_logits), fake_logits)

        generator_gradients = generator_tape.gradient(
            generator_loss, generator.trainable_variables
        )
        generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables)
        )
        return generator_loss, discriminator_loss

    for epoch in range(epochs):
        generator_losses = tf.keras.metrics.Mean()
        discriminator_losses = tf.keras.metrics.Mean()
        for real_images, real_labels in dataset:
            generator_loss, discriminator_loss = train_step(real_images, real_labels)
            generator_losses.update_state(generator_loss)
            discriminator_losses.update_state(discriminator_loss)
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"generator loss: {generator_losses.result():.4f}, "
            f"discriminator loss: {discriminator_losses.result():.4f}"
        )

    generator.save(MODEL_PATH)
    print(f"Saved conditional generator to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a conditional MNIST generator.")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    train(args.epochs)
