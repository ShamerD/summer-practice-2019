# summer-practice-2019
A summer practice project on solving 3 tasks using Python.

The tasks are following:
  ## 1. Implementation of Prokudyn-Gorsky method.
  Given a black-and-white picture representing photos taken with blue, green and red filters, and a fixed green point, find an original colored image and blue and red points that refer to the given point. MSE and Cross-Correlation metrics were used to find the best original picture.
  
  ## 2. Seam carving for content-aware image resizing.
  Given a picture, expand or shrink it so that important areas remain unchanged. Seam carving method is based on idea of deleting carve with the least "energy", while there are various ways to define "energy" (Here, absolute value of gradient was used). Possibility of using masks (to mark things that are certainly to be or not to be deleted) was also implemented.
  
  ## 3. Traffic sign classification using HOG and SVM.
  Histograms Of Gradients feature extraction algorithm was implemented. Support Vector Machine was trained and tested to classify traffic signs. 
