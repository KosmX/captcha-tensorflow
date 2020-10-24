
import math
import numpy as np
import tensorflow as tf

def evaluate(model, data_generator, test_gen1, test_gen2):
    test_gen1 = data_generator(df, test_idx, for_training=False, batch_size=128)
    dict(zip(model.metrics_names, model.evaluate(test_gen1, steps=len(test_idx)//128)))

    x_test, y_test = next(test_gen2)

    y_pred = model.predict_on_batch(x_test)

    y_true = tf.math.argmax(y_test, axis=-1)
    y_pred = tf.math.argmax(y_pred, axis=-1)

    n = 30
    random_indices = np.random.permutation(n)
    n_cols = 5
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 20))
    for i, img_idx in enumerate(random_indices):
        ax = axes.flat[i]
        ax.imshow(x_test[img_idx])
        ax.set_title('pred: {}'.format(
            ''.join(map(str, y_pred[img_idx].numpy()))))
        ax.set_xlabel('true: {}'.format(
            ''.join(map(str, y_true[img_idx].numpy()))))
        ax.set_xticks([])
        ax.set_yticks([])