# Hindi-Marathi Tests

This repo extracts pair of different Part-of-Speach (PoS) (N_NN, V_VM) from Hindi and Marathi PoS-tagged sentence-aligned data, and produces a score for the pairs based on similarity model trained for cognate detection on supervised data for various language pairs.

### Files
 
- Data/ : This folder contains the sentence-aligned PoS-tagged dataset for Hindi-Marathi
- FirstTest/ : This folder contains the results from the first test run on the data, including the extracted pairs, the manually annotated labels and the results produced by the models on them.
- Models/ : This folder contains the weights from the trained model, that was used for testing.

- AttentionLayer.py : This file contains code for the structure of the custom Attention Layer used in the neural network model.
- LoadModel.py : Loads the model architecture and weights from the Models/ folder.
- PRvals.txt : To be ignored
- Utils.py : Util functions used in the code.
- extractPairs.py : Loads the hindi/marathi test pairs of the mentioned PoS tags into a dict `data_ipa` in memory, from the files in the Data/ folder.
- hindi2ipa.txt : Mapping from devnagri characters to IPA characters
- hindi2ipa_Pruned.txt : Prunned mapping according to the input vocab of the trained model
