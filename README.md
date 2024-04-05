# Adversarial Attacks: Comparing Classical and Quantum-Classical (or Hybrid) Neural Network Models

This repository provides an implementation of a representative Convolutional Neural Network (CNN) as a Classical Neural Network and a Quantum-Classical or Hybrid Neural Network (HNN) for the classification of handwritten digits from the MNIST dataset. The primary objective is to evaluate the adversarial robustness of these models against a comprehensive set of adversarial attacks, including compound attacks formed by combining multiple individual attack methods. The HNN model combines a classical CNN with a parameterized quantum circuit, leveraging potential advantages of quantum computing.

## Overview

The code includes the following key components:

1. **CNN Model**: A classical Convolutional Neural Network for MNIST digit classification.
2. **HNN Model**: A hybrid model that combines the classical CNN with a parameterized quantum circuit.
3. **Quantum Circuit**: Functions to create and simulate a parameterized quantum circuit with adjustable rotation angles and entanglement gates.
4. **Hybrid Forward Pass**: A function that combines the outputs of the classical CNN and the simulated quantum circuit.
5. **Data Loading and Preprocessing**: Functions to load and preprocess the MNIST dataset, including options for filtering specific digits.
6. **Model Training and Evaluation**: Functions to train and evaluate the CNN and HNN models on the MNIST dataset.
7. **Attack Generation**: Functions to generate individual adversarial attacks and compound adversarial attacks formed by combining multiple individual attack methods.
8. **Attack Evaluation**: Functions to evaluate the impact of attacks on the models' performance by measuring pre-attack and post-attack accuracy and loss.
9. **Result Visualization**: Functions to visualize the training curves, misclassified samples, and a tabular summary of attack results.

## White-box, Targetted Adversarial Attacks

The code implements and evaluates a comprehensive set of individual adversarial attacks, each with configurable parameters, including:

- **GN (Gaussian Noise)**: Adds random Gaussian noise to the input image.
- **FGSM (Fast Gradient Sign Method)**: A one-step attack that perturbs the input in the direction of the gradient.
- **BIM (Basic Iterative Method)**: An iterative version of FGSM that applies multiple small perturbations.
- **CW (Carlini-Wagner)**: An attack that finds the smallest perturbation causing misclassification.
- **RFGSM (Random Fast Gradient Sign Method)**: A variant of FGSM that starts from a randomly perturbed input.
- **PGD (Projected Gradient Descent)**: An iterative attack that applies perturbations and projects the result onto the allowed perturbation space.
- **MIFGSM (Momentum Iterative Fast Gradient Sign Method)**: An iterative attack that incorporates momentum.
- **TPGD (Tamed Projected Gradient Descent)**: A variant of PGD that applies a taming function to the gradients.
- **EOTPGD (Expectation Over Transformation Projected Gradient Descent)**: An extension of TPGD that incorporates random transformations.
- **APGD (Autoregressive Projected Gradient Descent)**: An iterative attack that adjusts the perturbation based on the current prediction and a target class.
- **DIFGSM (Diverse Input Fast Gradient Sign Method)**: An extension of FGSM that applies diverse transformations to the input.
- **Jitter**: An attack that applies random jitter to the input.

## Results
!["Results"](https://github.com/ericyoc/cnn_hnn_comparison_analysis/blob/main/adv_attack_results/results_output.jpg)

## White-box, Targetted Compound Adversarial Attacks

In addition to individual adversarial attacks, the code generates white-box, targeted compound attacks by combining multiple individual attack methods. These compound attacks are created by iteratively applying one attack method followed by another, forming a sequence of perturbations that can potentially bypass the defenses of the target model.

A white-box attack assumes that the attacker has complete knowledge of the model's architecture, parameters, and decision boundaries. This information is leveraged to craft adversarial examples that are tailored to the specific model being attacked.

Targeted attacks, on the other hand, aim to cause the model to misclassify the adversarial example as a specific target class chosen by the attacker, rather than simply causing a misclassification to any incorrect class.

The code generates white-box, targeted compound attacks by considering all possible pairs of the implemented individual attack methods, which include both white-box and targeted attacks. For each compound attack, the code evaluates the models' performance by measuring the pre-attack and post-attack accuracy and loss, allowing for a comprehensive analysis of the adversarial robustness against these combined attack strategies.

By combining multiple attack methods in a sequential manner, the compound attacks can potentially exploit different vulnerabilities of the target model, making them more effective at evading the model's defenses and causing targeted misclassifications.

## Measuring Attack Success

To determine the success of an adversarial attack, the code measures the pre-attack and post-attack accuracy and loss for both the CNN and HNN models. By comparing these metrics, it is possible to assess the impact of the attack on each model's performance.

Measuring the accuracy and loss for both models is crucial because the HNN incorporates a quantum circuit component, which may exhibit different vulnerabilities or robustness compared to the classical CNN. This approach allows for a comprehensive evaluation of the adversarial robustness of each model architecture against various attack strategies.

Additionally, the code provides a function to visualize a sample of misclassified images, displaying the original image, the true label, and the predictions made by both the CNN and HNN models. This visual inspection can provide valuable insights into the types of adversarial examples that successfully fooled the models and can aid in understanding the strengths and weaknesses of each approach.

!["CNN Pre-attack and Post Attack Accuracy Results"](https://github.com/ericyoc/cnn_hnn_comparison_analysis/blob/main/adv_attack_results/cnn_comparison_analysis.jpg)

## MNIST Dataset Preprocessing

The MNIST dataset consists of grayscale images (28x28 pixels) of handwritten digits from 0 to 9. The code provides two options for preprocessing the dataset:

1. **Digits 0 and 1**: The dataset is filtered to include only images of digits 0 and 1, creating a balanced subset for each digit.
2. **All Digits (0 to 9)**: The dataset is filtered to include all digits from 0 to 9, creating a balanced subset with an equal number of samples for each digit.

The preprocessed dataset is then split into training and test loaders for model training and evaluation.

## Quantum Circuit

The HNN model incorporates a parameterized quantum circuit to introduce non-linearities and potential advantages of quantum computing. The quantum circuit consists of the following elements:

1. **Qubit Initialization**: The circuit begins with a set of qubits initialized to the |0> state.
2. **Rotation Gates**: Rotation gates (Ry) are applied to each qubit, with adjustable rotation angles controlled by learnable parameters.
3. **Entanglement Gates**: Controlled-NOT (CNOT) gates are applied to introduce entanglement between the qubits.

The output of the quantum circuit is a final state vector, which is combined with the output of the classical CNN through a hybrid forward pass.

## Evaluation Metrics

For each attack, the code reports the following metrics:

- Pre-Attack CNN Accuracy (%)
- Pre-Attack CNN Loss
- Post-Attack CNN Accuracy (%)
- Post-Attack CNN Loss
- Pre-Attack HNN Accuracy (%)
- Pre-Attack HNN Loss
- Post-Attack HNN Accuracy (%)
- Post-Attack HNN Loss

These metrics allow for a comprehensive comparison of the adversarial robustness of the CNN and HNN models against different types of attacks.

## Requirements

To run this code, you'll need the following dependencies:

- Python 3.6 or higher
- PyTorch
- Cirq
- TorchAttacks
- NumPy
- Matplotlib
- Tabulate

## Usage

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies.
3. Run the `main.py` script to train the CNN and HNN models, evaluate their performance against various attacks, and visualize the results.

```bash
python main.py
