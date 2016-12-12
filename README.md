# char-rnn-tensorflow

Based on @karpathy's [min-char-rnn](https://gist.github.com/karpathy/d4dee566867f8291f086).

## Sample from the min-char-rnn
```
SEBASTIAN.
And his leainstsunzasS!

FERDINAND.
I lost, and sprery
RS]
How it ant thim hand
```

This is a sample what the RNN learnt from a [Shakespeare's 'The Tempest'](http://www.gutenberg.org/cache/epub/1540/pg1540.txt), using a hidden layer of 100 neurons and a sequence length of 25, over 500 epochs.

## Usage
You might have to install `tensorflow`, `numpy` and `matplotlib` via `pip`. Once those are done, feel free to run:
```
python min-char-rnn.py
```

## Coming Up
* Using RNN-Cell instead of coding an RNN from hand.
* LSTM-based implementation.
* word-rnn using LSTM.
