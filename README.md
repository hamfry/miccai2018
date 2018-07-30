# Miccai 2018

Pytorch implementation for models in https://arxiv.org/pdf/1806.04259.pdf

### Requirements

This code works on Pytorch 0.4.0.

### Instructions

0. Download the breast dataset [here](https://drive.google.com/open?id=1x6KhBa14T8LqER5oO2yeTXTPm9qPMYC0)


1. To train the model

   ```python
   python train_lstm_cnn.py
   ```

2. To test the model

   ```
   python lstm_cnn_test.py
   ```

### Reference

- Sirinukunwattana K, Alham NK, Verrill C, Rittscher J. Improving Whole Slide Segmentation Through Visual Context-A Systematic Study. arXiv preprint arXiv:1806.04259. 2018 Jun 11.

### Todo

- [ ] add other models