# CMPE 591: Deep Learning in Robotics - Hmw 1

---

## 📌 Introduction

This report presents the results of training Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) models using a dataset collected from a simulated environment. The models aim to predict positional changes after taking specific actions.

---

## 📊 Dataset Collection

Random actions were taken, and the corresponding before-action images, actions, and resulting positions were stored in a dataset file (`dataset.pkl`). Each data sample consists of:

- **img_before**: Initial state of the environment as an image.
- **action_id**: Randomly chosen action (one of four possible actions).
- **pos_after**: The position of the agent after taking the action.
- **img_after**: Final state of the environment as an image.

A total of **1000 data points** were generated for training and testing.

---

## 🏗️ Machine Learning Models

### 🔹 Multi-Layer Perceptron (MLP)

#### **Architecture**
- **Input**: One-hot encoded action (4-dimensional).
- **Hidden Layers**:
  - First hidden layer: 128 neurons (ReLU activation).
  - Second hidden layer: 64 neurons (ReLU activation).
- **Output**: 2 neurons representing final position.

#### **Training**
- **Loss function**: Mean Squared Error (MSE).
- **Optimizer**: Adam.
- **Epochs**: 100.
- **Final test loss**: **0.00478**.

---

### 🔹 Convolutional Neural Network (CNN)

#### **Architecture**
- **Input**: Initial environment image (3x128x128).
- **Convolutional Layers**:
  - Conv2D (32 filters, kernel size = 3x3) + ReLU + MaxPool(2x2).
  - Conv2D (64 filters, kernel size = 3x3) + ReLU + MaxPool(2x2).
- **Fully Connected Layers**:
  - 128 neurons (ReLU activation).
  - 2 neurons (predicting `pos_after`).

#### **Training**
- **Loss function**: Mean Squared Error (MSE).
- **Optimizer**: Adam.
- **Epochs**: 100.
- **Final test loss**: **0.0529**.

---

## 📈 Results

### 🔹 Training Loss Over 100 Epochs

| Epoch | MLP Loss | CNN Loss |
|-------|---------|---------|
| 1  | 0.2324  | 37,867,314.16 |
| 2  | 0.0714  | 0.1197 |
| 3  | 0.0748  | 0.1098 |
| 4  | 0.0484  | 0.1213 |
| 5  | 0.0198  | 0.2623 |
| 10 | 0.0096  | 0.6071 |
| 20 | 0.0063  | 0.4838 |
| 50 | 0.0047  | 0.0951 |
| 100 | 0.0047  | 0.0515 |

---

### 🔹 Comparison of MLP vs. CNN Performance

| Model | Input | Output | Training Time | Test Loss |
|-------|-------|--------|--------------|-----------|
| MLP  | One-hot encoded action (4D) | Final position (2D) | Fast | **0.00478** |
| CNN  | Initial environment image (3x128x128) | Final position (2D) | Slow | **0.0529** |

---

## 🔍 Conclusion

The **MLP model** outperformed the **CNN model** in terms of final test loss. This suggests that using direct action input might be a more effective approach for predicting positional changes compared to processing image inputs. The CNN model had a very high initial loss, but it improved significantly after training.
