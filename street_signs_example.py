import datetime
import os
import glob
from sklearn.model_selection import train_test_split
import shutil
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from my_utils import split_data, order_test_set, create_generators
from deeplearning_models_2D import streetsigns_model
    
if __name__=="__main__":
    
    # path_to_data = '/Users/zharec/groking-computer-vision/archive/Train'
    # path_to_train = '/Users/zharec/groking-computer-vision/training_data/train'
    # path_to_val = '/Users/zharec/groking-computer-vision/training_data/val'
    
    # split_data(path_to_data, path_to_train, path_to_val, 0.1)
    
    # path_to_images = '/Users/zharec/groking-computer-vision/archive/Test'
    # path_to_csv = '/Users/zharec/groking-computer-vision/archive/Test.csv'
    # order_test_set(path_to_images, path_to_csv)
    
    path_to_train = '/Users/zharec/groking-computer-vision/training_data/train'
    path_to_val = '/Users/zharec/groking-computer-vision/training_data/val'
    path_to_test = '/Users/zharec/groking-computer-vision/archive/Test'
    batch_size = 64
    epochs = 15
    
    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes
    
    TRAIN=False
    TEST=True
    
    if TRAIN:
        path_to_save_model = './Models'
        ckpt_saver = ModelCheckpoint(
            path_to_save_model,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )
        
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=10
        )
        
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        model = streetsigns_model(nbr_classes)
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        model.fit(train_generator,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_generator,
                callbacks=[ckpt_saver, early_stop])
        
        model.evaluate(test_generator)
    
    if TEST:
        model = tf.keras.models.load_model('./Models')
        model.summary()
        
        print("Evaluating validation set:")
        model.evaluate(val_generator)
        
        print("Evaluating test set:")
        model.evaluate(test_generator)
    
    
    