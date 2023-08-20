import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# MNISTデータセットをダウンロード
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

# 特徴量とラベルを取得
X = mnist.data
y = mnist.target

# データを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの構築と訓練
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# テストデータセットでの予測
y_pred = model.predict(X_test)


# 手書き画像の読み込みと前処理
i = 1
j = 0
while i <= 6:
    while j <= 9:
        image = Image.open('images/digits' + str(i) + '_' + str(j) +'.PNG').convert('L')  # 手書き画像のパスを指定して読み込む
        image = image.resize((28, 28))  # 28x28にリサイズ
        image = np.array(image)  # NumPy配列に変換
        image = image.reshape(1, -1)  # 1次元の特徴量ベクトルに変換
        #image = image / 255.0  # スケーリング (0から1の範囲に正規化)
        image = 255.0 - image  # 色の反転（白黒反転）

        # ラベルの作成
        label = np.array([j])

        # データを追加
        X_train = np.vstack((X_train, image))
        y_train = np.concatenate((y_train, label))
        j += 1
    i += 1

y_train = np.array(y_train, dtype=int)

# 新しい学習データでモデルを再訓練
model.fit(X_train, y_train)

# 自作画像の学習

i = 0
predictionNum=[]
pred = [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9] #正解ラベル20枚の時
#pred = [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9]#正解レベル30枚の時
while i <= 9:
   y=0
   while y < 2:#テストデータが30枚の時は"y < 3"とする
      image = Image.open("images/digits_"+str(i) +'-'+str(y)+ '.png').convert('L')  # 手書き画像のパスを指定して読み込む
      image = image.resize((28, 28))  # 28x28にリサイズ
      image = np.array(image)  # NumPy配列に変換
      image = image.reshape(1, -1)  # 1次元の特徴量ベクトルに変換
      #image = image / 255.0  # スケーリング (0から1の範囲に正規化)
      image = 255.0 - image #色の反転
      # モデルによる予測
      prediction = model.predict(image)
      predictionNum.append(int(prediction[0]))
      #testown = np.array([])
      #test = np.append(testown,prediction[0])
      print(prediction)#予想した数字が画像と一緒に出力される
    
      y += 1
       
      # 結果の表示
      plt.imshow(image.reshape(28, 28), cmap='gray')
      plt.title("number:"+str(i)+" Predicted Label: " + str(prediction[0]))#実際の数字、予測した数字の表示
      plt.axis('off')
      plt.show()
   i += 1

# 正答率の計算
accuracy = accuracy_score(predictionNum, pred)
print("Accuracy:", accuracy)
