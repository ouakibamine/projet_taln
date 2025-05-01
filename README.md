# Emotion Classification using CNN-GRU-SVM

This project implements an emotion classification system using a hybrid **CNN + GRU + SVM** model. It is structured into multiple folders, each containing code or data necessary to train and evaluate the model.

## 📁 Project Structure

```
.
├── prepared_data/           # Preprocessed training and validation datasets
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_val.npy
│   └── y_val.npy
├── checkpoints/             # Directory to save model checkpoints
├── logs/                    # Training logs
├── results_cnn/             # Model evaluation results
├── cnn_gru_svm_main.py      # Main training script
├── main.ipynb               # Jupyter notebook for training and testing
└── requirements.txt         # List of required Python packages
```

## 📦 Setup & Installation

1. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Training the Model

You can run the training process either through the notebook (`main.ipynb`) or using the command line:

```bash
python cnn_gru_svm_main.py \
  --operation train \
  --train_dataset ./prepared_data/X_train.npy \
  --train_labels ./prepared_data/y_train.npy \
  --validation_dataset ./prepared_data/X_val.npy \
  --validation_labels ./prepared_data/y_val.npy \
  --checkpoint_path ./checkpoints \
  --log_path ./logs \
  --model_name emotion_cnn_gru_svm \
  --result_path ./results_cnn
```

## 📊 Output

- **Checkpoints**: The model is saved in the `./checkpoints/` folder.
- **Logs**: Training logs are stored in `./logs/`.
- **Results**: Evaluation results are saved in `./results_cnn/`.

## 🔤 Emotion Classes

The model classifies input text into the following emotion categories:

- `joy` (0)
- `sadness` (1)
- `anger` (2)
- `fear` (3)
- `disgust` (4)
- `surprise` (5)

## 📝 Notes

- Make sure the `.npy` files in `prepared_data/` are correctly formatted.
- The `main.ipynb` notebook allows for interactive testing and visualization.

## 📬 Contact

If you have any questions or issues, feel free to open an issue or reach out.
