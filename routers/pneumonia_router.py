import io
import tensorflow as tf
import uvicorn
from fastapi import APIRouter, File, FastAPI
from PIL import Image
from keras.preprocessing.image import img_to_array
import keras

router = APIRouter()

app = FastAPI(title= 'Pneumonia Detection API', version = '1.0', description = "Amar's MTU.")
@app.post('/predict')
def pnuemonia_router(image_file: bytes = File(...)):
    model = keras.models.load_model(r'C:\Users\alika\Desktop\pneumoniadetection\classifier\models\model.h5')
    model.load_weights(r'C:\Users\alika\Desktop\pneumoniadetection\classifier\models\weights.h5')

    image = Image.open(io.BytesIO(image_file))

    if image.mode != 'L':
        image = image.convert('L')

    image = image.resize((64, 64))
    image = img_to_array(image)/255.0
    image = image.reshape(1, 64, 64, 1)

    graph = tf.get_default_graph()

    with graph.as_default():
        prediction = model.predict_proba(image)

    predicted_class = 'pneumonia' if prediction[0] > 0.5 else 'normal'

    return {'predicted_class': predicted_class,
            'pneumonia_probability': str(prediction[0])}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8006)