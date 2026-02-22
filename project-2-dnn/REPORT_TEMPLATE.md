# Project 2 Report: Forest Cover Type Classification with DNNs

## Your baseline results
I used a baseline dense neural network with one hidden layer of 64 neurons, ReLU activation, no dropout, learning rate 0.001, batch size 64, and 20 epochs. This model gave me a useful starting point for comparison because it was simple and easy to interpret. The baseline reached a training accuracy of **[fill]**, validation accuracy (last/best) of **[fill]/[fill]**, and test accuracy of **[fill]**. From the learning curves, the model showed **[fill: stable convergence / mild overfitting / underfitting]**, which helped me decide what to change next.

## The effects of each architectural or hyperparameter change
I changed one main factor at a time to keep comparisons fair. Increasing depth from one hidden layer to two or three hidden layers changed performance by **[fill]** and affected convergence speed by **[fill]**. Switching the activation function from ReLU to ELU resulted in **[fill]**, especially in early training behavior. Adding dropout (0.3) changed generalization by **[fill]**, while L2 regularization (1e-4) produced **[fill]**. Finally, changing optimization settings (learning rate 5e-4 and batch size 128) gave **[fill]**, which suggested that **[fill: smaller updates were more stable / larger batch slowed adaptation / etc.]**.

### Experiment summary table
| Experiment | Change from Baseline | Best Val Accuracy | Test Accuracy | Key Observation |
|---|---|---:|---:|---|
| Baseline | 1x64 ReLU, lr=1e-3, bs=64 | [fill] | [fill] | [fill] |
| Exp 1 | 128-64 ReLU | [fill] | [fill] | [fill] |
| Exp 2 | 256-128-64 ReLU | [fill] | [fill] | [fill] |
| Exp 3 | 128-64 ELU | [fill] | [fill] | [fill] |
| Exp 4 | 128-64 ReLU + Dropout(0.3) | [fill] | [fill] | [fill] |
| Exp 5 | 128-64 ReLU + L2(1e-4) | [fill] | [fill] | [fill] |
| Exp 6 | 128-64 ReLU, lr=5e-4, bs=128 | [fill] | [fill] | [fill] |

## Key charts (loss/accuracy plots, confusion matrix, classification reports)
The training and validation accuracy/loss curves were the main tool for diagnosing learning behavior. The baseline curves showed **[fill]**, while the best model curves showed **[fill]**, indicating **[fill: improved generalization / faster convergence / reduced overfitting]**. The confusion matrix highlighted where the model performed well and where it confused similar classes; in my case, the most common confusion was **[fill]**. The classification report gave per-class precision, recall, and F1-score, and showed strongest performance on **[fill]** and weaker performance on **[fill]**.

## Your best model configuration in terms of performance
The best-performing model in this project was **[fill: experiment name]** with configuration **[fill architecture + activation + regularization + lr + batch size]**. I selected this model because it had the highest validation performance and maintained strong test accuracy, with curves that looked more stable than the alternatives. Its final test accuracy was **[fill]**, and its class-level metrics were more balanced than the baseline, especially for **[fill]**.

## Reflections on what worked and what did not
What worked best was balancing model capacity and regularization rather than simply making the network deeper. Some changes improved training accuracy but did not improve validation/test performance, which reminded me that generalization matters more than fitting the training set. Regularization and careful optimization settings helped control overfitting, but overly aggressive settings could slow learning or reduce final accuracy. If I continued this work, I would add early stopping and a learning-rate scheduler, and I would test a few more architecture sizes to further improve the hardest classes.

