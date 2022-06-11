import gradio as gr
import tensorflow as tf

from kedro_gradio.pipelines.data_processing.nodes import preprocess_mnist


# Change the file path to specific path.
model_fp = '../data/06_models/dnn.h5/2022-06-11T06.49.02.453Z/dnn.h5'
model = tf.keras.models.load_model(model_fp)

lables = [i for i in range(10)]

def predict(img):
    (r, c )= img.shape
    # Add the dimension to make it acceptable.
    img = img.reshape((1, r, c))
    x = preprocess_mnist(images=img)
    p = model.predict(x)
    # Converting numpy.float32 to float to render predict probabilty. 
    confidences = {lables[i]: float(v) for i, v in enumerate(p[0])}
    return confidences


# Create a Gradio app with sketchpad, for handwritten digit.
demo = gr.Interface(
    fn=predict, 
    inputs="sketchpad",
    outputs=gr.outputs.Label(num_top_classes=5),
    live=True
)

demo.launch()