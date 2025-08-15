Notes for now:
Use HuggingFace transformer as template for nn Modules

Self Causal Attention:
Have to ensure that the n_embd is divisble by the n_heads to make sure we arent using too many or too little embeddings
Use nn.Linear to create linear layers where the transformations actually happen within our attention
Use Register_buffer to create and store the mask to make sure there is no looking ahead at future tokens
Multiply number of embedings within a Linear layer in order to create query, key, and value vectors. Remember, this is not copying the vectors, q, k, and v are all different because we are just repeating the linear transformations 3 times
