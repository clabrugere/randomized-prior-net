import tensorflow as tf
from sklearn.model_selection import train_test_split


def to_dataset(X, y, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
