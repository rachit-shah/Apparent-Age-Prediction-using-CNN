# Apparent-Age-Prediction-using-CNN
Predict the apparent age of a person possibly in real time. The model used is pre-trained on VGG16 architecture and then trained using a convolutional neural network on the ChaLearn LAP dataset which consists of 8000 image samples. Test Accuracy achieved – 84%.

Introduction
We tried to solve apparent age prediction problem from facial images using deep learning. We have used various deep learning architectures with transfer learning. We have tried to implement this problem on LAP (Looking At People) dataset and imdb-wiki dataset.We have taken this problem as classification problem of 101 classes for age 0 to 100. Our method first extracts faces from the given images and aligns them. Thenwe extracted the features of images using different architectures. We have tried various architectures such as VGG16, Xception, ResNet for feature extraction and then finetuned them on our images. We achieved 87% accuracy using ResNet architecture on LAP dataset.


