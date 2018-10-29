import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import time
from sklearn.linear_model import LogisticRegression

from imgslicer import imgslicer
from histogram import preProcess
from kmeans import kmeans
from watershed import mywatershed
from otsu import otsu

#imgslicer('inverted_msword1.png',64)

t0 = time.clock()
preP = preProcess(10)
train_images, train_class, train_files = preP.GetImagesAndLabels('.\\img\\train')
test_images, test_class, test_files = preP.GetImagesAndLabels('.\\img\\test')

print('train:',train_images.shape, train_class.shape, 'average time:',(time.clock()-t0)/len(train_images))
for i in range(len(train_images)):
    np.set_printoptions(precision=3,suppress=True)
    print(train_files[i], 'class:', train_class[i])#, 'data:', train_images[i])
    print('')

# logistic regression
t0 = time.clock()
model = LogisticRegression()
print(train_images.shape,train_class.shape)
model.fit(train_images, train_class)
# make predictions
expected = test_class
predicted = model.predict(test_images)
print('expected:', expected)
print('predicted:', predicted)
# summarize the fit of the model
print('logistic regression time:',time.clock()-t0)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
error = np.zeros(4)
for i in range(len(test_images)):
    if expected[i] != predicted[i]:
        error[expected[i]] += 1
print('errors:',error)

kmeans('.\\img\\segmentation\\mix_ie.png')

mywatershed('.\\img\\segmentation\\mix_ie.png')

otsu('.\\img\\segmentation\\mix_ie.png')
