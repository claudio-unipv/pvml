TODO
====
- Other clustering?
- KSVM
  + better optimization algorithm?
- OneHotVector class (if needed by RNN)
- MLP
  + refactor the class with activations as attributes (?)
  + use attributes for activation instead of methods (?)
  + try subclasses (e.g. autoencoder)
  + refactor load/save to not use pickle
  + reset_momentum method
- CNN
  + padding
  + refactor to look more similar to MLP
  + Improve PVMLNet (mostly with padding)
  + refactor load/save to not use pickle
  + reset_momentum method
- RNN
  + Basic
  + LSTM (peephole?)
  + GRU (?)
- Reinforcement learning
  + Better example with CNN or MLP (use Open AI gym?)
- Unit test
  + Mostly NN


BUGS
====
- resize update_b and update_w for transfer learning in pvmlnet
  + fix: refactor pvmlnet so that it is not a separate class but just
    an instance of CNN
  + fix pad mode in PVML (or update to numpy >= 1.17)
- error loading jpegs
  + fix: add pillow to the dependencies


DOCS
====
- turn lab activities into tutorials?  (long term)
