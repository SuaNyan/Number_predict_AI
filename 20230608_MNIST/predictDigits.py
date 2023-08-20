import sklearn.datasets
import sklearn.svm
import PIL.Image
import numpy

def imageToData(filename):
    greyImage = PIL.Image.open(filename).convert("L")
    greyImage = greyImage.resize((8,8),PIL.Image.Resampling.LANCZOS)
    
    numImage = numpy.asarray(greyImage, dtype = float)
    numImage = 16 - numpy.floor(17 * numImage / 256)
    numImage = numImage.flatten()
    return numImage
 
def predictDigits(data):
    digits = sklearn.datasets.load_digits()
    clf = sklearn.svm.SVC(gamma = 0.001)
    clf.fit(digits.data, digits.target)
    n = clf.predict([data])
    print("予測=",n)
i = 1
while i <= 9:
    data = imageToData("digits_"+str(i)+".png")
    predictDigits(data)
    i+=1
    
