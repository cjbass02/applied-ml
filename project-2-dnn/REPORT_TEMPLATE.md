# Project 2 Report: Forest Cover Type Classification with DNNs

## Your baseline results

I used a baseline dense neural network with one hidden layer of 64 neurons, ReLU activation, no dropout, learning rate 0.001, batch size 64, and 20 epochs. This model gave me a useful starting point for comparison because it was simple and easy to interpret. The baseline reached a training accuracy of **0.823**, validation accuracy (last/best) of **0.820 / 0.820** (from `dnn_experiment_results.csv`), and test accuracy was not computed separately for the baseline (only the best model was evaluated on the test set). From the learning curves, the model showed **stable convergence with room to improve**—validation and training accuracy were close, suggesting the model was not overfitting but also had limited capacity. That helped me decide to try deeper architectures and other changes next.

## The effects of each architectural or hyperparameter change

I changed one main factor at a time to keep comparisons fair. **Increasing depth** from one hidden layer to two (128–64) gave a clear jump: best validation accuracy went from 0.820 to **0.889** (exp1: `val_accuracy_best` 0.8891). Going to three layers (256–128–64) improved further to **0.918** (exp2: 0.9176), the best of all experiments, with train accuracy about 0.929 and a small gap between train and val, so the deeper network generalized well. **Switching to ELU** (128–64) gave validation **0.877** (exp3: 0.8774)—slightly below the same-sized ReLU model (0.889), so ReLU was at least as good for this setup. **Adding dropout 0.3** (128–64) reduced overfitting (train 0.805, val best 0.838) but also lowered validation accuracy compared to no dropout (0.889), so the regularization was a bit strong for 20 epochs. **L2 (1e-4)** on 128–64 gave val **0.880** (exp5: 0.8802)—a small gain over baseline and similar in spirit to dropout, slightly constraining the model. **Lower learning rate (5e-4) and larger batch (128)** gave val **0.866** (exp6: 0.8655), so slower updates and bigger batches did not help here and likely slowed adaptation.

### Experiment summary table

(Values from `dnn_experiment_results.csv`; test accuracy from notebook best-model evaluation.)

| Experiment | Change from Baseline | Best Val Accuracy | Train Acc (last) | Test Accuracy | Key Observation |
|---|---|---:|---:|---:|---|
| Baseline | 1×64 ReLU, lr=1e-3, bs=64 | 0.820 | 0.823 | — | Lowest val; stable, limited capacity. |
| Exp 1 | 128–64 ReLU | 0.889 | 0.892 | — | Big gain from depth; good generalization. |
| Exp 2 | 256–128–64 ReLU | **0.918** | 0.929 | **0.916** | Best val and test; deeper helped further. |
| Exp 3 | 128–64 ELU | 0.877 | 0.880 | — | Slightly worse than same-size ReLU. |
| Exp 4 | 128–64 ReLU + Dropout(0.3) | 0.838 | 0.805 | — | Less overfitting but lower val accuracy. |
| Exp 5 | 128–64 ReLU + L2(1e-4) | 0.880 | 0.883 | — | Mild regularization; small improvement over baseline. |
| Exp 6 | 128–64 ReLU, lr=5e-4, bs=128 | 0.866 | 0.872 | — | Slower/larger-batch training hurt performance. |

(Test set was evaluated only for the best model, Exp 2; notebook output: `Test Accuracy (exp2_deeper_256_128_64_relu): 0.9160`.)

## Key charts (loss/accuracy plots, confusion matrix, classification reports)

The training and validation accuracy/loss curves were the main tool for diagnosing learning behavior. The **baseline** curves showed steady improvement with train and val close together, indicating no serious overfitting but limited peak performance. The **best model (exp2: 256–128–64)** curves showed higher accuracy and lower loss, with a small gap between train and val, indicating **improved generalization** without much overfitting. The confusion matrix (in the notebook) highlights where the model performs well and where it confuses classes; the classification report (below) shows that **classes 3 and 4** (small support: 824 and 2,848) had the lowest precision/recall/F1, so the most common confusions likely involve those minority classes. The classification report gave per-class precision, recall, and F1-score: **strongest performance** on classes 0, 1, 2, and 6 (F1 around 0.90–0.93), and **weaker performance** on classes 3 and 4 (F1 ~0.76).

**Classification report (best model, test set)** — from notebook `classification_report(y_test, test_preds, digits=4)` for `exp2_deeper_256_128_64_relu`:

| Class | Precision | Recall | F1-score | Support |
|------:|----------:|-------:|--------:|--------:|
| 0 | 0.8979 | 0.9344 | 0.9158 | 63,552 |
| 1 | 0.9368 | 0.9207 | 0.9287 | 84,991 |
| 2 | 0.9177 | 0.9018 | 0.9097 | 10,726 |
| 3 | 0.7762 | 0.7451 | 0.7604 | 824 |
| 4 | 0.8140 | 0.7068 | 0.7566 | 2,848 |
| 5 | 0.8323 | 0.8658 | 0.8487 | 5,210 |
| 6 | 0.9638 | 0.8480 | 0.9022 | 6,153 |
| **accuracy** | | | **0.9160** | 174,304 |
| macro avg | 0.8770 | 0.8461 | 0.8603 | 174,304 |
| weighted avg | 0.9165 | 0.9160 | 0.9159 | 174,304 |

## Your best model configuration in terms of performance

The best-performing model was **exp2_deeper_256_128_64_relu** (CSV: best val 0.9176, train 0.9294) with configuration: **three hidden layers (256, 128, 64 units), ReLU activation, no dropout, no L2, learning rate 0.001, batch size 64**, trained for 20 epochs. I selected this model because it had the highest validation accuracy (0.918) and maintained strong test accuracy (**0.916**, from notebook). The curves looked more stable than the baseline and other experiments, with a small train–val gap. Its final test accuracy was **0.916**, and class-level metrics were more balanced than the baseline where we had them; the main weakness was on the smallest classes (3 and 4), which is expected with a multi-class dataset and no class weighting.

## Reflections on what worked and what did not

**What worked:** Deeper architecture (256–128–64) gave the biggest gain without hurting generalization. Going from 1 to 2 to 3 hidden layers improved validation (and test for the best model) consistently. ReLU was at least as good as ELU in the 128–64 comparison. The best model did not need dropout or L2 for this dataset and epoch budget—the extra capacity was used well.

**What did not help (or hurt):** Dropout 0.3 was too strong and lowered validation accuracy. Lower learning rate and larger batch size (Exp 6) reduced performance. L2 gave a small improvement over baseline but did not beat the deeper ReLU model without regularization.

**Next steps:** I would add early stopping and a learning-rate scheduler, and possibly class weights or oversampling for classes 3 and 4 to improve their recall. Trying one or two more deeper/wider variants and more epochs with early stopping could also help.
