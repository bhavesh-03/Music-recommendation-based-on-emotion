
In the provided code, a pre-trained emotion detection model is used to classify facial expressions into different emotions. The specific model used is named 'fer2013_mini_XCEPTION.102-0.66.hdf5'. This model is likely a deep learning model trained on the FER2013 dataset, which is a dataset of facial expressions.

Here's the relevant part of the code that loads the model:

python
Copy code
### Load the pre-trained emotion detection model
emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
The model is loaded using the load_model function from Keras. The file 'fer2013_mini_XCEPTION.102-0.66.hdf5' should contain the weights and architecture of the pre-trained model.

If you need information about the architecture of this specific model or details about the FER2013 dataset, you would typically refer to the documentation or source where you obtained the model file.


# Instruction from here
## first of all it is pretrained model so model train nahi kiya hai !!!!
## we need to train the model in our project but i was just trying to play with pre train model so made this with chat gpt
# Ways to run the project :
## install all dependency required 
## run ::
    streamlit run main.py
