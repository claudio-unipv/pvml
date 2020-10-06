TODO
====
- Other clustering?
- KSVM
  + better optimization algorithm?
- OneHotVector class (if needed by RNN)
- MLP
  + try subclasses (e.g. autoencoder)
- CNN
  + Improve PVMLNet (mostly with padding)
- RNN
  + LSTM?
  + RNN example
- Reinforcement learning
  + Better example with CNN or MLP (use Open AI gym?)
- Unit test
  + pvmlnet
  + dataset

BUGS
====
- resize update_b and update_w for transfer learning in pvmlnet (rename attributes?)
  + fix: refactor pvmlnet so that it is not a separate class but just
    an instance of CNN
  + fix pad mode in PVML (or update to numpy >= 1.17)
- error loading jpegs
  + fix: add pillow to the dependencies


DOCS
====
- turn lab activities into tutorials?  (long term)
