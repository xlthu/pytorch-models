# Pytorch Models

A set of pytorch models from my undergraduate thesis, they are:

- VGG
- ResNet
- RNN LM
- Seq2seq translation
- Attention-over-Attention Reader
- Deep Q-Learning for Flappy Bird

## About the repo

My undergraduate thesis is to design a deep learning benchmark set, these models are part of it. However, in this repo, I exclude the profiling part, so you won't see any code about profiling. I want these code to be a good resource for beginners to learn Pytorch.

These files are almost independent with each other except the common part, *utils* and *torchseq*. You can focus on few files if you're only interested in some certain model.

- VGG: `vgg.py`, `run_ic.py`, `utils/`
- ResNet: `resnet.py`, `run_ic.py`, `utils/`
- RNN LM: `birnn.py`, `run_lm.py`, `torchseq/`, `utils/`
- Seq2seq translation: `seq2seq.py`, `run_mt.py`, `torchseq/`, `utils/`
- Attention-over-Attention Reader: `aoa.py`, `run_qa.py`, `torchseq/`, `utils/`
- Deep Q-Learning for Flappy Bird: `dqn.py`, `run_rl.py`, `game/`, `utils/`

These models are implemented according to their paper and sometimes referring to public popular repos on Github. I'm trying to make the codes easy to understand, they are similar in code architecture, so it won't take you too long time to read the second model code if you have been familiar to the first one.

The most difficult part to understand will be the model part and the data loading part, but certainly not the training and test logic. You can conquer the model part by reading with its paper together, and I add some key comments to help understand. For data loading, I use `torchvision` for image loading, which a popular library for pytorch user, and I write a simple and easy-to-revise `torchseq` for sequence loading, especially for text, which I will explain more in next section.

## Design for torchseq

A universal and uniform library is best for development and learning, as I was programming the RNN models, I searched and researched lots of libraries including the official `torchtext`, however, they're either too complex and heavy, not easy to understand and revise or too exclusive, only for a certain dataset.

So I start to design a new library which follows the principle of `torchvision` and is uniform with it, then ending up to `torchseq`. I write several datasets and some utilities (some are revised from official code) targeting for padding problem in sequence. As I have no much time on it, I want to leave it to someone who want to evolve it.

You can use them whatever you want, I'm new to license, but I think MIT license can allow you do almost everything.

## Future work

I'll find a time to add the origins of all the repos and codes that I refer to. Thanks a lot to them.
