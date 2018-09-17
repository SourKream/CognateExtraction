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

### Steps to run the code

#### Load the data
- Open `ipython` in interactive mode
- Run extractPairs.py using `execfile('extractPairs.py')`
- This loads a dict object `data_ipa` with 2 keys `['N_NN', 'V_VM']`. To load more PoS tags from the data, modify the file at `line 10`. Each key in the dict gives a list of all the pairs. The pairs are created by looking at the first occurence of the PoS in the Hindi sentence and the Marathi sentence, and subsequently translating them into IPA.
- To see any word in the pair, use the `print` command. `print data_ipa['N_NN'][0][0]` will return `d̪rʃn` instead of `u'd\u032ar\u0283n'`.

##### Load Your Own Data
If you wish to use your own data, then replace `data_ipa['N_NN']` in the following instructions with the list of your own pairs. The format for this list should be similar to `data_ipa['N_NN']`.

#### Load the model
- Next load the model using `execfile('LoadModel.py')`

#### Run the model on the data
- `X, Y = load_test_data(data_ipa['N_NN'], vocab)`
- `result = model.predict([X,Y])`
- If the above command is taking time, you can run it on a smaller set using `model.predict([X[:100,:], Y[:100,:]])`
- `output = [(data_ipa['N_NN'][i][0], data_ipa['N_NN'][i][1], result[i,0]) for i in range(len(result))]` consolidates the results in the variable `output`.
- The output can be written to a file using `writeToFile(output, <filename>)`

### Dependencies

- Python 2.7
- IPython 5.5.0
- Keras 2.1.2
- Theano 0.9.0
- Numpy 1.13.3

This was the last version status of the dependencies when I wrote the code. Newer versions might need modification to the existing code in order to run.
