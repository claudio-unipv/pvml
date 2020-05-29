TODO
====
- Classification trees
  + pruning
  + categorical variables?
- KMeans
  + Check
  + Add to demo
- Other clustering?
- MLP
  + refactor the class with activations as attributes
  + use attributes for activation instead of methods
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
- Extra bin in edge histogram (sometimes 65 bins)
  + fix: clip before bincount
- resize update_b and update_w for transfer learning in pvmlnet
  + fix: refactor pvmlnet so that it is not a separate class but just
    an instance of CNN
- error loading jpegs
  + fix: add pillow to the dependencies


DOCS
====
- turn lab activities into tutorials?  (long term)
