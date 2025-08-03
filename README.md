# Space Station Object Detection Hackathon

## Overview

This repository contains the complete solution for the **Duality AI Space Station Hackathon** focused on training an object detection model for space station objects using synthetic data from the Falcon digital twin simulation platform. The challenge involved detecting and classifying objects such as Toolboxes, Oxygen Tanks, and Fire Extinguishers under varied conditions like lighting, occlusions, and camera angles.

The project utilizes the YOLOv8 model for multi-class object detection trained exclusively on the provided synthetic dataset.

## Repository Contents

- `train.py` : Script to train the YOLOv8 object detection model.
- `predict.py` : Script to run inference and evaluate the model on test data.
- `config.yaml` : Configuration file defining model and training parameters.
- `runs/` : Directory containing training logs, outputs, performance visualizations, and checkpoints.
- `dataset/` : Folder containing the synthetic dataset divided into train, validation, and test splits with YOLO-format annotations.
- `README.md` : This documentation file.
- `setup_env.bat` : Windows environment setup script (also provide equivalent for Mac/Linux).

## Getting Started

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) installed.
- Python 3.8+ (environment managed via conda).
- Recommended: NVIDIA GPU with CUDA support for faster training.

### Environment Setup

1. Open Anaconda Prompt and navigate to the `ENV_SETUP` folder inside the project directory.
2. Run the environment setup script:

   For Windows:
setup_env.bat


  For Mac/Linux (create and run `setup_env.sh` with equivalent commands):
bash setup_env.sh

  
3. Activate the environment:
conda activate EDU


## How to Run

### Training the Model

1. Make sure you are in the project root directory with the activated `EDU` environment.
2. Run the training script:
python train.py
3. Training logs, checkpoints, and performance metrics will be saved to the `runs/` directory.

### Evaluating the Model

1. Run the prediction and evaluation script:
python predict.py
2. This will generate performance metrics including:
- mAP@0.5 (Mean Average Precision at IoU=0.5)
- Precision and Recall scores
- Confusion matrix visualizations
- Failure case analyses with example images

## Project Highlights

- **Dataset**: Synthetic data generated from the Falcon digital twin platform simulating space station conditions.
- **Target Objects**: Toolbox, Oxygen Tank, Fire Extinguisher.
- **Challenges Tackled**: Varied lighting, occlusions, and multiple viewpoints.
- **Model**: YOLOv8 fine-tuned on the provided synthetic dataset.
- **Performance Metrics**:
- Achieved mAP@0.5 of [Insert your final score]% on the test set.
- Precision and recall above [Insert values]% for critical classes.

## Troubleshooting & Tips

- If training is slow:
- Reduce batch size in `config.yaml`.
- Close background applications.
- Verify GPU utilization with `nvidia-smi`.
- Avoid using test images during training to prevent disqualification.
- Back up model weights and logs regularly.

## Reproducing the Results

1. Follow the environment setup instructions.
2. Download and place the dataset as per folder structure under `dataset/`.
3. Run the training and evaluation scripts as described.
4. Use results from `runs/` for analyzing model performance and producing visualizations.

## Bonus: Use Case Application

*If your team implemented a bonus application:*

- Description of the application functionality and real-world impact.
- Instructions to run the app (if applicable).
- Methodology to update the model continuously using Falcon to adapt to real-world changes (e.g., object appearance changes or introduction of new objects).

## Team Information

- Team name: [Your Team Name]
- Members: [List of team members]

## References and Resources

- [Falcon Digital Twin Platform](https://www.duality.ai)
- YOLOv8 Documentation
- Duality AI Community Discord Server

## License

Specify your license here (e.g., MIT License).

## Contact

For questions, issues, or collaboration, please reach out to [Your Contact Information].

