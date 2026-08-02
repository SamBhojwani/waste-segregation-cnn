# ♻️ Waste Segregation CNN

> A convolutional neural net that sorts waste photographs into **organic** and **recyclable**,
> served through a Streamlit app you can drop an image into.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

---

Municipal waste sorting is mostly a classification problem wearing overalls. This project trains
a small CNN from scratch on 25,000+ labelled waste images and wraps it in an interface simple
enough for someone standing over a bin.

## The dataset

Kaggle's [Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data),
reorganised into a two-class split:

| Split | Organic | Recyclable | Total |
| :-- | --: | --: | --: |
| Train | 12,565 | 9,999 | 22,564 |
| Test | 1,401 | 1,112 | 2,513 |

The images themselves are **not** committed here. Download them from Kaggle and place them as
`clean_dataset/train/{O,R}` and `clean_dataset/test/{O,R}`.

## The model

A three-block convolutional stack, trained from scratch rather than fine-tuned, so the whole
thing stays legible:

```
Input 224 x 224 x 3
  Conv2D(32, 3x3) + ReLU  ->  MaxPool 2x2
  Conv2D(64, 3x3) + ReLU  ->  MaxPool 2x2
  Conv2D(128, 3x3) + ReLU ->  MaxPool 2x2
  Flatten -> Dense(128) + ReLU -> Dropout(0.5)
  Dense(2) + Softmax
```

Adam optimiser, categorical cross-entropy, 10 epochs, batch size 32, pixels rescaled to `[0, 1]`.

## Running it

```bash
pip install -r requirements.txt

# 1) train (writes waste_classifier.h5)
python model.py

# 2) serve the classifier
streamlit run app.py
```

Trained weights are not committed. `model.py` regenerates them from the Kaggle dataset in one
run; expect a few minutes per epoch on CPU.

## Files

```
model.py        dataset loading, the CNN definition, training, and export
preprocess.py   the loading + normalisation pipeline on its own
app.py          Streamlit interface: upload an image, get a label
```

## Honest limitations

- **Two classes only.** Real waste streams need hazardous, e-waste, glass and paper separated
  out; this collapses everything into organic vs. recyclable.
- **No held-out evaluation beyond the test split**, and no per-class precision/recall recorded
  from the training run. The next useful step is a confusion matrix rather than another epoch.
- **Trained from scratch on a small architecture.** Fine-tuning a pretrained backbone
  (MobileNetV3, EfficientNet) would almost certainly beat it for less compute.
- **Clean product photography.** The dataset does not look like the inside of an actual bin.

## Where it would go next

- Confusion matrix and per-class metrics on the test split
- Transfer learning from a pretrained backbone as a baseline comparison
- More waste categories, starting with e-waste and hazardous
- Raspberry Pi deployment for real-time sorting at the bin

---

<sub>Built as a college mini project by Samarth Bhojwani.</sub>
