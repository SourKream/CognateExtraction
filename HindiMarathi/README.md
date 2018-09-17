# Hindi-Marathi Tests

This repo extracts pair of different Part-of-Speach (PoS) (N_NN, V_VM) from Hindi and Marathi PoS-tagged sentence-aligned data, and produces a score for the pairs based on similarity model trained for cognate detection on supervised data for various language pairs.

### Files

- extractPairs.py : Loads the hindi/marathi test pairs of the mentioned PoS tags into a dict `data_ipa` in memory, from the files in the Data folder.
- LoadModel.py : Loads the model architecture and weights
