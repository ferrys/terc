# terc
CS 542 Group Project

To reproduce our training results:
1. Install all of the required packages
2. Create a folder named Terc_images with all of the tagged images that you would like to use to train the model
3. Run the extract_tags.py program which will analyze the images and create a csv file with each image name and their respective tags
4. Run resize.py which will resize the images to be 224x224
5. Run resnet_50.py which will split the images into Training, Validation, and Test data and then will train on the training data and then run this trained model and return the predictions on the validation data

To test the trained model:
1. Follow the above steps to train the model and acquire the required files
2. Run test_model.py, which will load the trained model and run it on the test data, and will return a csv file with the accuracies obtained on the test data

To run the trained model on real untagged images:
1. Download the terc_model folder from the GitHub: https://github.com/ferrys/terc/tree/master/terc_model
2. Create a folder FOLDER_NAME within the terc_model folder that contains all of the untagged images you would like to tag
3. Run terc_model.py with FOLDER_NAME as an argument in the command line. (Example: "python terc_model.py FOLDER_NAME") would run the neural network on the folder FOLDER_NAME
4. This will run the trained model on all of the images in the folder and will insert tags into the EXIF metadata of each image based on the tags that the model predicts


Packages and Installations required:

Anaconda   
Exiftool   
bleach                    1.5.0                    
ca-certificates           2017.7.27.1  
certifi                   2016.2.28               
cycler                    0.10.0                    
enum34                    1.1.6                     
h5py                      2.7.1                     
html5lib                  0.9999999                 
icu                       58.2                   
jpeg                      9b                      
Keras                     2.1.1                    
libpng                    1.6.28                   
libtiff                   4.0.7                    
libwebp                   0.5.2                    
Markdown                  2.6.9                    
matplotlib                2.1.0                    
mkl                       2017.0.3                 
numpy                     1.13.3                   
numpy                     1.13.1                   
olefile                   0.44                    
opencv                    3.3.0                  
openssl                   1.0.2l                 
pandas                    0.21.0                    
Pillow                    4.3.0                     
pip                       9.0.1                    
protobuf                  3.4.0                    
pyparsing                 2.2.0                    
python                    3.6.2                    
python-dateutil           2.6.1                    
pytz                      2017.3                   
PyYAML                    3.12                     
qt                        5.6.2                    
scikit-learn              0.19.1                   
scipy                     1.0.0                     
setuptools                36.4.0                   
setuptools                36.7.2                   
six                       1.11.0                   
sklearn                   0.0                      
tensorflow-gpu            1.4.0                    
tensorflow-tensorboard    0.4.0rc2                 
vc                        14                       
vs2015_runtime            14.0.25420              
Werkzeug                  0.12.2                   
wheel                     0.29.0                   
wheel                     0.30.0                   
wincertstore              0.2                      
zlib                      1.2.11                   



