import numpy as np
from sklearn.svm import SVC
from PIL import Image
import warnings

def imageToData(filename):
    greyImage = Image.open(filename).convert("L")
    greyImage = greyImage.resize((8, 8), Image.Resampling.LANCZOS)

    numImage = np.asarray(greyImage, dtype=float)
    numImage = 16 - np.floor(17 * numImage / 256)
    numImage = numImage.flatten()
    return numImage

def predictDigits(data):
    # MNISTデータセットの読み込み
    from sklearn.datasets import fetch_openml
    warnings.filterwarnings("ignore", category=FutureWarning)
    mnist = fetch_openml('mnist_784', version=1, cache=True)

    # データを制限する
    sample_indices = np.random.choice(len(mnist.data), size=4000, replace=False)
    data_subset = mnist.data[sample_indices]
    target_subset = mnist.target[sample_indices]

    # モデルの初期化とトレーニングデータの設定
    clf = SVC(gamma=0.001)
    clf.fit(data_subset, target_subset)

    # データの予測
    n = clf.predict([data])
    print("予測=", n)

data = imageToData("digits_1.png")
print(data)

i = 1
while i <= 9:
    data = imageToData("digits_" + str(i) + ".png")
    predictDigits(data)
    i += 1
