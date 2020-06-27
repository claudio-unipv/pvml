TODO
====
- Classification trees
  + pruning
  + categorical variables?
- Other clustering?
- MLP
  + refactor the class with activations as attributes (?)
  + use attributes for activation instead of methods (?)
  + try subclasses (e.g. autoencoder)
- CNN
  + padding
  + refactor to look more similar to MLP
  + Improve PVMLNet (mostly with padding)
- RNN
  + Basic
  + LSTM (?)
  + GRU (?)
- Input checks + better error messages
- Unit test
  + everything


BUGS
====
- resize update_b and update_w for transfer learning in pvmlnet
  + fix: refactor pvmlnet so that it is not a separate class but just
    an instance of CNN
- error loading jpegs
  + fix: add pillow to the dependencies
- review the use of np.nan_to_num


DOCS
====
- turn lab activities into tutorials?  (long term)
