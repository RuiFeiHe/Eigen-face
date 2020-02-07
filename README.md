# Eigenface

### face recognition and rebuild using Eigenvalue decomposition

Course Project, CS 21191070 Computer Vision, ZJU, 2019 Winter. Instructor: Mingli Song

浙江大学 计算机学院 计算机视觉 课程作业



## Results

Top 10 eigenface:
![image](https://github.com/RuiFeiHe/Eigenface/blob/master/top10.png)


recognition and rebuild results:
<img width="350" height="350" src="https://github.com/RuiFeiHe/Eigenface/blob/master/Eigenface.png"/>


## Functions

1. It is assumed that each face image has only one face, and the positions of two eyes are known (i.e. manually labeled). The eye position of each image exists in a text file with the same name as the image file but the suffix is TXT in the corresponding directory. The text file is represented by four numbers separated by a line and a space, respectively corresponding to the position of the two eye centers in the image;

2. Implement two program processes (two executive files), corresponding to training and identification respectively;

3. Build a personal face database by yourself (at least 40 people, including myrself). 

4. It is not allowed to directly call some functions related to eigenface in opencv, eigenvalue and eigenvector solving functions can be called; only C / C + +, not other programming languages can be used; GUI can only use opencv's own highgui, not QT or other; platform can use win / Linux / MacOS, it is recommended that win takes precedence;

5. The training program format is roughly: "mytrain.exe energy percentage model filename other parameters..." The energy percentage is used to determine how many feature faces are selected, and the training results are saved to the model file. At the same time, the first 10 feature faces are assembled into an image, and then displayed;

6. The format of recognition program is roughly: "mytest.exe face image filename model filename other parameters..." After the model file is loaded in, the input face image is recognized, and the recognition results are superimposed on the input face image to display. At the same time, the most similar image with the face image in the face database is displayed.



## Methodology

1. For data selection, I chose the att face data set, with 40 people in total, each with 10 expressions. During my training, for each person I used 5 pictures, and added 5 pictures of myself, with 205 pictures in total. For the eye position calibration, I use the open-source Python code on the Internet to achieve.

2. I designed a class for each image and the whole dataset. For the class of picture, I designed the class `Per_Face_ATT`. I first load the desired txt file and picture, perform transform() function to center the eye position and align the connection line between the two eyes. For the class of dataset, I designed the class `Face_ATT`. Read 205 pictures respectively according to the form of dataset directory and store them in the container.

3. When training, first calculate the average face, and then subtract the average face from each face to achieve de-averaging. After taking away the average, we do the calculation of covariance matrix. There is a trick to calculate the covariance matrix, which can be proved mathematically. For the matrix of mnxk, the characteristic matrix is required, and the covariance matrix of KxK can be obtained, which can greatly accelerate the process of finding the characteristic matrix later. Then for 1xk eigenvector, the original eigenvector of 1xmn can be obtained by multiplying with samples matrix. The covariance matrix is obtained, and eigen values and eigenvectors are obtained by Eigen function.

4. For how many eigenvectors to retain, the ratio of the sum of K eigenvalues to the sum of the total eigenvalues of the eigenvalue matrix is used as the energy ratio to get the number of eigenvectors that should be retained. Take the first k eigenvalues and eigenvectors as the model, and save the model.

5. For the first 10 feature vectors, reshape into the size of the picture. After hconcat(), imshow(), which shows an image made of the first 10 feature faces.

6. In the test, first multiply the read in test photo and the original training face by the transposition of the transformation matrix, convert them to the coordinates under the eigenvector, and then find the one with the closest coordinates as the most similar picture; when reconstructing the face, multiply the coordinates under the eigenvector of the picture to be tested by the transformation matrix, and then add the average face, reshape, and then display them, that is, to rebuild the face.
