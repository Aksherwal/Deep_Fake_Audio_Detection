# Deep_Fake_Audio_Detection

## Overview
This project demonstrates a lightweight, real-time deepfake audio detection system combining **Retrieval-Augmented Detection (RADD)**, **Generative Adversarial Networks (GANs)**, and **Variational Autoencoders (VAEs)**. It processes audio live from a microphone, identifies potential deepfakes, and adapts to new threats through continuous learning. Built for quick testing, it uses 100 real and 100 fake audio samples from the ASVspoof2019 LA dataset and runs a 10-second real-time detection demo.

#### Read the [Complete Document](https://github.com/Aksherwal/Deep_Fake_Audio_Detection/blob/3ed07a0793dca2bf6fb2ad5e7b2007492901f553/Deepfake%20Audio%20Detection%20System_%20A%20Hybrid%20Real-Time%20Solution.pdf) for more details.

### Key Features
- **Real-Time Detection**: Analyzes audio in 128-ms chunks for instant deepfake flagging.
- **Adaptability**: Uses GANs to simulate new fakes and a database to update itself.
- **Robustness**: VAEs enhance data with variations, preparing for real-world conditions.
- **Speed**: Caches retrieval results for efficiency, inspired by [arXiv:2403.11778](https://arxiv.org/abs/2403.11778).

### Applications
- Call center fraud prevention
- Social media audio verification
- Legal audio authenticity checks

---

## Prerequisites

### Dataset
- **ASVspoof2019 LA**: Place `ASVspoof2019_LA_train/flac/` files in a `flac/` folder and `ASVspoof2019.LA.cm.train.trn.txt` in the root directory.
  - Download from [ASVspoof Challenge](https://www.asvspoof.org/).

### Environment
- **Python**: Version 3.9
- **Dependencies**: Install via pip:
  ```bash
  pip install librosa numpy faiss-cpu transformers torch tensorflow pyaudio mysql-connector-python scikit-learn
  ```
- **Hardware**: Microphone required for real-time detection.

---

## Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd deepfake-audio-detection
   ```

2. **Prepare Dataset**:
   - Copy `flac/` folder and `ASVspoof2019.LA.cm.train.trn.txt` into the project directory.

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt  # Create this file with the above dependencies if needed
   ```

4. **Database Configuration**:
   - Update `DB_CONFIG` in the notebook with your MySQL credentials if using a different setup:
     ```python
     DB_CONFIG = {
         "host": "your-host",
         "port": your-port,
         "user": "your-user",
         "password": "your-password",
         "database": "your-database"
     }
     ```
---

## Usage

1. **Open the Notebook**:
   ```bash
   jupyter notebook Deepfake_Audio_Detection_Demo.ipynb
   ```

2. **Run All Cells**:
   - Execute sequentially to:
     - Load and preprocess 200 audio files (100 real, 100 fake).
     - Train GAN (50 epochs) and VAE (10 epochs).
     - Set up RADD and train the detector (50 epochs).
     - Run a 10-second real-time detection test with your microphone.

3. **Check Outputs**:
   - Logs show training progress, real-time probabilities (e.g., “Deepfake Probability: 0.50”), and evaluation metrics (e.g., “Accuracy: 0.50”).

---

## How It Works
1. **Data Loading**: Reads audio, converts to spectrograms.
2. **RADD**: Compares audio to a feature library using Wav2Vec2 and FAISS.
3. **GAN**: Generates synthetic deepfakes for training.
4. **VAE**: Augments real audio for robustness.
5. **Detector**: Classifies audio in real time with a CNN.
6. **Continuous Learning**: Stores new fakes in MySQL and retrains.

---

## Current Performance
- **Training**: Detector accuracy ~0.50–0.56, validation stuck at 0.50 (needs more data/epochs).
- **Real-Time**: Outputs ~0.50 probability (undecisive, requires tuning).
- **Metrics**: Accuracy 0.50, Recall 1.00, F1 0.67—promising but improvable.

---

## Future Enhancements
- Increase dataset size (`DEMO_FILES > 100`).
- Use a lighter model (e.g., Wav2Vec2-small) for speed.
- Add platform-specific artifacts (compression, noise) to VAE.
- Extend real-time detection to continuous monitoring (`while True`).

---

## References
- Tak, H., et al. (2024). "Real-Time Deepfake Detection Using Retrieval-Augmented Methods." *arXiv:2403.11778*. [Link](https://arxiv.org/abs/2403.11778)
- Goodfellow, I., et al. (2014). "Generative Adversarial Nets." *Advances in Neural Information Processing Systems*. [Link](https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)
- Kingma, D. P., & Welling, M. (2013). "Auto-Encoding Variational Bayes." *arXiv:1312.6114*. [Link](https://arxiv.org/abs/1312.6114)

---

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing
Feel free to open issues or submit pull requests with improvements!
