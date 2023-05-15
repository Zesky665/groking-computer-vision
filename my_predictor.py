import tensorflow as tf
import numpy as np

def predict_with_model(model, img_path):
    
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [60, 60])
    image = tf.expand_dims(image, axis=0)
    
    predictions = model.predict(image)
    prediction = np.argmax(predictions)
    
    return prediction
    
if __name__=="__main__":
    
    img_path = "/Users/zharec/groking-computer-vision/archive/Meta/11.png"
    
    model = tf.keras.models.load_model("Models")
    predction = predict_with_model(model, img_path)
    
    print(f"Prediction: {predction}")