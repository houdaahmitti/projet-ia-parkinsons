from keras.callbacks import EarlyStopping

def train_model(model, x_train, y_train, epochs=100, batch_size=16, validation_split=0.2):
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    
    history = model.fit(
        x_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop]
    )
    
    return model, history
