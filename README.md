# Dataminig_createAI
# 手書き数字認識AI

このリポジトリには、手書き数字を認識するためのニューラルネットワークモデルをトレーニングおよび使用するPythonスクリプトが含まれています。モデルはMNISTデータセットでトレーニングされ、グレースケール画像に描かれた数字を予測できます。

## 目次

- [はじめに](#はじめに)
  - [前提条件](#前提条件)
  - [インストール](#インストール)
- [使用方法](#使用方法)

## はじめに

### 前提条件

- Python 3.x
- 必要なパッケージ: numpy, matplotlib, Pillow, scikit-learn

次のコマンドを使用して必要なパッケージをインストールできます。

```sh
pip install numpy matplotlib pillow scikit-learn
```

リポジトリのインストール
```sh
git clone https://github.com/your-username/handwritten-digit-recognition.git
```
ディレクトリに移動
```sh
cd handwritten-digit-recognition
```

## 使用方法
1.train_model.py スクリプトを実行してモデルをトレーニングします。これにより、MNISTデータセットが読み込まれ、モデルがトレーニングされ、テストセットでの正確度が計算されます。
```sh
python train_model.py
```
2.トレーニング後、predict_digits.py スクリプトを使用して手書き数字を予測します。スクリプトは数字の画像を読み込み、前処理を行い、トレーニング済みモデルを使用して予測します。
```sh
python predict_digits.py
```









