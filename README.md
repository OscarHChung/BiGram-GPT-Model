# BiGram-GPT-Model

This project takes a text file containing human speech from either the internet or the user's local computer and
emulates the linguistics of the file using the BiGram GPT Model.

The program makes use of multi-head attention, feedforwards, dropouts, transformer blocks, layer norms, and residual layers.
It also has the ability to run on a GPU with CUDA capabilities. It uses the Adam optimizer and of course, BiGram model.

This repository has 2 versions of a BiGram GPT model. The more newer version is the bigram-gpt.py file, which runs in about 15 min.
The older version is the model.py file, which contains a simpler, non fine-tuned version that runs in 1 min. The input.txt
file contains all of Shakespeare's published work (which was originally parsed from the internet), and is used as a default input.
