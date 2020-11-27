---
layout: post
title: "My recurrent network doesn’t work with data parallelism"
date: 2020-11-28 06:15:55 +0300
image: pack_rnn.jpg
tags: Debug
---  
  
# My recurrent network doesn’t work with data parallelism
  
There is a subtlety in using the `pack sequence -> recurrent network -> unpack sequence` pattern in a `Module` with `DataParallel` or `data_parallel()`. Input to each the `forward()`
on each device will only be part of the entire input. Because the unpack operation `torch.nn.utils.rnn.pad_packed_sequence()` 
by default only pads up to the longest input it sees, i.e., the longest on that particular device, 
size mismatches will happen when results are gathered together. Therefore, you can instead take 
advantage of the `total_length` argument of `pad_packed_sequence()` to make sure that the `forward()` 
calls return sequences of same length. For example, you can write:  
  
```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MyModule(nn.Module):
    # ... __init__, other methods, etc.

    # padded_input is of shape [B x T x *] (batch_first mode) and contains
    # the sequences sorted by lengths
    #   B is the batch size
    #   T is max sequence length
    def forward(self, padded_input, input_lengths):
        total_length = padded_input.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(padded_input, input_lengths,
                                            batch_first=True)
        packed_output, _ = self.my_lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                        total_length=total_length)
        return output


m = MyModule().cuda()
dp_m = nn.DataParallel(m)
```  
  
Additionally, extra care needs to be taken when batch dimension is dim 1 (i.e., batch_first=False) with data parallelism. In this case, the first argument of pack_padded_sequence padding_input will be of shape [T x B x *] and should be scattered along dim 1, but the second argument input_lengths will be of shape [B] and should be scattered along dim 0. Extra code to manipulate the tensor shapes will be needed.
