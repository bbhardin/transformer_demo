class MultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = DotProductAttention() # Scaled dot product attention
        self.heads = h # Number of attention heads ot use
        self.d_k = d_k # Dimensionality of the linearly projected queries and keys
        self.d_v = d_v # Dimensionality of the linearly projected values
        self.W_q = Dense(d_k) # Learned projection matrix for the queries
        self.W_k = Dense(d_k) # Learned projection matrix for the keys
        self.W_v = Dense(d_v) # Learned projection matrix for the values
        self.W_o = Dense(d_model) # Learned projection matrix for the multi-head output

        # Batch size is a hyperparameter of the training process
        # Sequence length defines the maximum length of the input/output phrases
        # Model dimensionality is the dimensionality of the outputs produced by all sub-layers of the model

    def reshape_tensor(self, x, heads, flag):
        # Reshape the linearly projected queries, keys, and values to allow
        #    attention heads to be computed in parallel
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], -1))
        return -1
    

    def call(self, queries, keys, values, mask=None):
        # Feed the linearly projected queries, keys, and values 
        #   into the reshape_tensor method then feed into the scaled
        #   dot product attention function
        # Mask is the look ahead mask if desired
        # Padding mask also needs ot be introduced to prevent the zero values
        #      from being processed along with the input
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)

        # Rearrange back into concatenated form
        output = self.reshape_tensor(o_reshaped, self.heads, False)

        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.W_o(output)