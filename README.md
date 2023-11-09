# Cello-0.0.1
</br>
Generate inhibitory molecule from protein sequence <br/>

## 1 - Sequence to Sequence Learning with Neural Networks
<br/>
<br/>
In this project, we'll start simple to understand the general concepts by implementing the model from the [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) paper. <br/>
<br/>
<br/>
The most common sequence-to-sequence (seq2seq) models are *encoder-decoder* models, which commonly use a *recurrent neural network* (RNN) to *encode* the source (input) sentence into a single vector. In this notebook, we'll refer to this single vector as a *context vector*. We can think of the context vector as being an abstract representation of the entire input sentence. This vector is then *decoded* by a second RNN which learns to output the target (output) sentence by generating it one word at a time.

![](https://www.google.com/url?sa=i&url=https%3A%2F%2Fyjjo.tistory.com%2F35&psig=AOvVaw2aK9s91U4unSKaIqxpndP_&ust=1699623079860000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCLjr7caDt4IDFQAAAAAdAAAAABAD)
<br/>
The above image shows an example of seq2seq model. The input/source is passed through the embedding layer and then input into the encoder. We also append a *start of sequence* (`<sos>`) and *end of sequence* (`<eos>`) token to the start and end of sentence, respectively. At each time-step, the input to the encoder RNN is both the embedding, $e$, of the current word, $e(x_t)$, as well as the hidden state from the previous time-step, $h_{t-1}$, and the encoder RNN outputs a new hidden state $h_t$. We can think of the hidden state as a vector representation of the sentence so far. The RNN can be represented as a function of both of $e(x_t)$ and $h_{t-1}$: <br/>
<br/>
$$h_t = \\text{EncoderRNN}(e(x_t), h_{t-1})$$
<br/>
We're using the term RNN generally here, it could be any recurrent architecture, such as an *LSTM* (Long Short-Term Memory) or a *GRU* (Gated Recurrent Unit).
<br/>
Here, we have $X = \\{x_1, x_2, ..., x_T\\}$, where $x_1 = \\text{<sos>}, x_2 = \\text{guten}$, etc. The initial hidden state, $h_0$, is usually either initialized to zeros or a learned parameter.
<br/>
Once the final word, $x_T$, has been passed into the RNN via the embedding layer, we use the final hidden state, $h_T$, as the context vector, i.e. $h_T = z$. This is a vector representation of the entire source sentence.
<br/>
Now we have our context vector, $z$, we can start decoding it to get the output/target sentence, \"good morning\". Again, we append start and end of sequence tokens to the target sentence. At each time-step, the input to the decoder RNN (blue) is the embedding, $d$, of current word, $d(y_t)$, as well as the hidden state from the previous time-step, $s_{t-1}$, where the initial decoder hidden state, $s_0$, is the context vector, $s_0 = z = h_T$, i.e. the initial decoder hidden state is the final encoder hidden state. Thus, similar to the encoder, we can represent the decoder as:
<br/>
</br>
$$s_t = \\text{DecoderRNN}(d(y_t), s_{t-1})$$
<br/>
Although the input/source embedding layer, $e$, and the output/target embedding layer, $d$, are both shown in yellow in the diagram they are two different embedding layers with their own parameters.
<br/>
"In the decoder, we need to go from the hidden state to an actual word, therefore at each time-step we use $s_t$ to predict (by passing it through a `Linear` layer, shown in purple) what we think is the next word in the sequence, $\\hat{y}_t$.
<br/>
</br>
$$\\hat{y}_t = f(s_t)$$
<br/>
The words in the decoder are always generated one after another, with one per time-step. We always use `<sos>` for the first input to the decoder, $y_1$, but for subsequent inputs, $y_{t>1}$, we will sometimes use the actual, ground truth next word in the sequence, $y_t$ and sometimes use the word predicted by our decoder, $\\hat{y}_{t-1}$. This is called *teacher forcing*, see a bit more info about it [here](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/).
<br/>
When training/testing our model, we always know how many words are in our target sentence, so we stop generating words once we hit that many. During inference it is common to keep generating words until the model outputs an `<eos>` token or after a certain amount of words have been generated.
<br/>
Once we have our predicted target sentence, $\\hat{Y} = \\{ \\hat{y}_1, \\hat{y}_2, ..., \\hat{y}_T \\}$, we compare it against our actual target sentence, $Y = \\{ y_1, y_2, ..., y_T \\}$, to calculate our loss. We then use this loss to update all of the parameters in our model.
<br/>
<br/>
We'll be coding up the models in PyTorch and using torchtext to help us do all of the pre-processing required. We'll also be using deepchem to assist in the tokenization of the data."
