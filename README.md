# Cello
Generate inhibitory molecule from protein sequence



based on https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3 </br>
![](https://miro.medium.com/max/1400/1*qN2Pj5J4VqAFf7dsA2dHpA.png)
</br>

The overarching seq2seq model has a few changes for packed padded sequences, masking and inference. </br>

We need to tell it what the indexes are for the pad token and also pass the source sentence lengths as input to the `forward` method. </br>

We use the pad token index to create the masks, by creating a mask tensor that is 1 wherever the source sentence is not equal to the pad token. This is all done within the `create_mask` function. </br>

The sequence lengths as needed to pass to the encoder to use packed padded sequences. </br>

The attention at each time-step is stored in the `attentions` </br>
