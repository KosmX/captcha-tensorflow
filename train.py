
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

def  plot_train_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    axes[0].plot(history.history['accuracy'], label='Train accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].legend() 

    axes[1].plot(history.history['loss'], label='Training loss')
    axes[1].plot(history.history['val_loss'], label='Validation loss')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()

#train_step ~ 10 000, valid = train/2
def train_ai(model, train_gen, train_step, valid_gen, valid_step):
    callbacks = [
        ModelCheckpoint("./model_checkpoint", monitor='val_loss')
    ]

    history = model.fit(train_gen,
                        steps_per_epoch=train_step,
                        epochs=5,
                        callbacks=callbacks,
                        validation_data=valid_gen,
                        validation_steps=valid_step)
    plot_train_history(history)
