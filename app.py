import gradio as gr
import tensorflow as tf
import numpy as np

# --- 1. LOAD THE TRAINED MODELS ---
# Make sure the .h5 file names match what you have saved
try:
    model_pneumonia = tf.keras.models.load_model('pneumonia_classifier.h5')
    model_covid19 = tf.keras.models.load_model('covid19_classifier.h5')
    model_tuberculosis = tf.keras.models.load_model('turberculosis_classifier.h5')
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please ensure 'pneumonia_classifier.h5', 'covid19_classifier.h5', and 'turberculosis_classifier.h5' are in the same directory as this script.")
    # Exit if models can't be loaded
    exit()

# Store models in a dictionary for easy access
models = {
    "Pneumonia": model_pneumonia,
    "COVID-19": model_covid19,
    "Tuberculosis": model_tuberculosis
}

# --- 2. DEFINE THE PREDICTION FUNCTION (Corrected Version) ---
def classify_image(model_choice, image_array):
    """
    This function takes a model name and an image, preprocesses the image,
    makes a prediction, and returns the raw probabilities for Gradio to format.
    """
    if image_array is None:
        return "Please upload an image."

    # Preprocess the image
    image_batch = tf.image.resize(image_array, [150, 150])
    image_batch = np.expand_dims(image_batch, axis=0)
    image_batch = image_batch / 255.0

    # Select the model and make a prediction
    model = models[model_choice]
    prediction = model.predict(image_batch)
    
    # The model's output ('probability') is always the confidence for the disease
    disease_probability = prediction[0][0]
    normal_probability = 1 - disease_probability

    # Return a dictionary with the raw probabilities (values between 0 and 1).
    # Gradio will automatically convert these to percentages for display.
    result = {
        f"{model_choice}": float(disease_probability),
        "Normal": float(normal_probability)
    }
    
    return result

# --- 3. CREATE THE GRADIO INTERFACE ---
# gr.Interface will build the UI for us
app_ui = gr.Interface(
    fn=classify_image,
    inputs=[
        gr.Radio(
            choices=["Pneumonia", "COVID-19", "Tuberculosis"],
            label="1. Select the Model to Use",
            value="Pneumonia" # Default selection
        ),
        gr.Image(
            type="numpy", # We want the image as a NumPy array
            label="2. Upload a Chest X-Ray Image"
        )
    ],
    outputs=gr.Label(
        num_top_classes=2, # We want to show probabilities for 2 classes
        label="Prediction Results"
    ),
    title="X-Ray Diagnostic Assistant",
    description="An Automated tool to classify chest X-rays for Pneumonia, COVID-19, and Tuberculosis. Upload an image and select a model to get a prediction.",
    allow_flagging="never",
    examples=[
        ["Pneumonia", "X-Ray_Dataset/test/PNEUMONIA/person1_virus_6.jpeg"],
        ["COVID-19", "X-Ray_Dataset/test/COVID19/COVID19(461).jpg"],
        ["Tuberculosis", "X-Ray_Dataset/test/TURBERCULOSIS/Tuberculosis-666.png"]
    ]
)

# --- 4. LAUNCH THE APP ---
if __name__ == "__main__":
    print("Launching...")
    app_ui.launch(share=True)


    
