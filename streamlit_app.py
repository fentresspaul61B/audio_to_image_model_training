import streamlit as st  # Used to create UI.
import mlflow  # Used to track ML experiments.
import numpy as np  # Used for arrays and rounding numbers.

st.title("Train and Track Audio Classification Model")

# Directs user to MLFlow UI.
st.write("MLFlow Artifacts are stored: http://localhost:5000.")

# Upload nested data folder.
st.header("Step #1: Load Data")
uploaded_files = st.file_uploader("Upload a Data Folder", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    st.write(bytes_data)

# Choose the model parameter for transfer learning.
st.header("Step #2: Choose Model Parameters")
model_spec = st.selectbox(
    'model_spec',
    ('efficientnet_lite0',
     'efficientnet_lite1',
     'efficientnet_lite2',
     'efficientnet_lite3',
     'efficientnet_lite4',
     'mobilenet_v2',
     'resnet_50'))

# Choose parameter for batch size.
batch_size = st.slider('Batch Size: Default = 64', 16, 256, step=16)

# Choose parameter for epochs.
epochs = st.slider('Training Epochs: Default = 10', 1, 50, step=1)

# Choose parameter for learning rate.
learning_rate = st.select_slider(
    'Learning Rate: Default = 0.004',
    options=[round(x, 3) for x in list(np.arange(0.001, .101, 0.001))])

# Choose parameter for dropout rate.
dropout_rate = st.slider('Dropout Rate: Default = 0.2', 0.0, 1.0, step=.01)

# Choose parameter for dropout rate.
shuffle = st.checkbox('Shuffle Data?')
use_augmentation = st.checkbox('Use Data Augmentation?')

# Train model and log parameters.
st.header("Step #3: Train Model")
if st.button("Train Model"):
    mlflow.log_param("model_spec", model_spec)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("dropout_rate", dropout_rate)
    mlflow.log_param("shuffle", shuffle)
    mlflow.log_param("use_augmentation", use_augmentation)
    mlflow.end_run()
    pass

# Save model to MLFlow Model Artifacts Register
st.header("Step #3: MLFlow Model Artifacts")
st.download_button('Download Model', "some_model.tflite")

# Adding image.
image_url = "https://github.com/fentresspaul61B/Image_Files/raw/main/Screenshot 2023-01-03 at 10.28.32 AM.png"
st.image(image_url, width=None)

if __name__ == "__main__":
    pass
