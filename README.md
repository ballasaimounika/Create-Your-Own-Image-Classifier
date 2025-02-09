# Create Your Own Image Classifier 

In this project, you'll train an image classifier to recognize different species of flowers. Imagine using this in a phone app that tells you the name of the flower your camera is looking at. In practice, you would train this classifier and then export it for use in your application. We'll be using this [dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories. Below are a few examples of flowers.

![](assets/Flowers.png) 

---

## Project Structure

The project is split into two main parts:


### Part 1 - Developing an Image Classifier with Deep Learning

This part of the project is broken down into several steps:

1. Load the image dataset and create a pipeline.
2. Build and train an image classifier on this dataset.
3. Use your trained model to perform inference on flower images.
4. Save the trained model for later use in applications.
   
   
### Part 2 - Building the Command Line Application

After building and training the deep neural network on the flower dataset, we will convert it into a Python script that runs from the command line. For testing, we’ll use the saved Keras model from Part 1.

---

## Requirements

The project requires **Python 3.x** and the following Python libraries installed :

- PyTorch
- ArgParse
- PIL
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
  
These libraries can be installed using Anaconda.

*Note*: To complete this project, you will need to use a GPU, as training the model on a local CPU may not work efficiently. You should only enable the GPU when necessary.


### GPU

Due to the complexity of the deep convolutional neural network, the training process cannot be done effectively on a common laptop. You have three options to run your training:

1. **Cuda** -- If you have an NVIDIA GPU then you can install CUDA from [here](https://developer.nvidia.com/cuda-downloads). CUDA will allow you to train your model, though the process may still be time-consuming.
2. **Cloud Services** -- There are many paid cloud services like [AWS](https://aws.amazon.com/fr/) or [Google Cloud](https://cloud.google.com/) where you can train your models.
3. **Coogle Colab** -- [Google Colab](https://colab.research.google.com/) provides free access to a Tesla K80 GPU for 12 hours at a time. After 12 hours, you can reload and continue. However, you will need to upload the dataset to Google Drive, and large datasets may cause space limitations.
   
Once a model is trained, a normal CPU can be used for the `predict.py` script, and the predictions will be made within a few seconds., once a model is trained then a normal CPU can be used for the `predict.py` file and you will have an answer within some seconds.


### JSON File

For the network to print the name of the flower, a `.json` file is required. If you’re unfamiliar with JSON, you can read more about it [here](https://www.json.org/). The `.json` file will map numerical labels to actual flower names, which will correspond to the folder structure containing the dataset. 

---

## Training the classifier

`train.py` will train the classifier. The user must specify one **mandatory argument:** `'data_dir'`, which contains the path to the training data directory.

Optional arguments:

- `--save_dir`: Path to the directory where the model will be saved.
- `--arch`: Architecture of the neural network. The default is AlexNet, but you can also specify VGG16.
- `--learning_r`: Learning rate for gradient descent. Default is 0.001.
- `--hidden_units`: Specifies how many neurons an extra hidden layer will contain (if chosen).
- `--epochs`: Specifies the number of epochs. Default is 5.
- `--GPU`: If a GPU is available, specify this argument.


## Using the classifier

`predict.py` accepts an image as input and outputs a probability ranking of predicted flower species. The only **mandatory argument** is `'image_dir'`, the path to the input image.

Options:

- `--load_dir`: The path to the checkpoint directory.
- `--top_k`: Specifies the number of top K classes to output. Default is 5.
- `--category_names`: Path to the JSON file mapping categories to names.
- `--GPU`: If a GPU is available, specify this argument.

---

## Run

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/ballasaimounika/Create-Your-Own-Image-Classifier.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd Create-Your-Own-Image-Classifier
   ```
   
3. Run the following command to open the Jupyter notebook:
   ```bash
   jupyter notebook Image-Classifier-Project.ipynb
   ```
   
   This will open the project file in your browser using Jupyter Notebook.
   
---

## Command Line Application

* Train a new network on the dataset using ```train.py```
  
  * Basic Usage :
    ```python train.py data_directory```
  * This will print out the current epoch, training loss, validation loss, and validation accuracy as the network trains.

  * Options:
    
    * Save checkpoints to a specified directory:
      ```python train.py data_dir --save_dir save_directory```
      
    * Choose architecture (AlexNet, VGG16, etc.):
      ```python train.py data_dir --arch "vgg16"```
      
    * Set hyperparameters:
      ```python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20```
      
    * Use GPU for training:
      ```python train.py data_dir --gpu```

    
* Predict flower name from an image using ```predict.py``` 
  
  * Basic usage:
    ```python predict.py /path/to/image checkpoint```
    
  * Options:
    
    * Return top **K** most likely classes:
      ```python predict.py input checkpoint --top_k 3```
      
    * Use a mapping of categories to real names:
      ```python predict.py input checkpoint --category_names cat_To_name.json```
      
    * Use GPU for inference:
      ```python predict.py input checkpoint --gpu```
