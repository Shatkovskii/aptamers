import torch
from model_1 import causal_mask

# Define function to obtain the most probable next token
def greedy_decode(model, source, tokenizer_tgt, max_len, device):
    # Retrieving the indices from the start and end of sequences of the target tokens
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Initializing the decoder input with the Start of Sentence token
    decoder_input = torch.empty(1,1).fill_(sos_idx).to(device)

    if source.dim() == 2:
        source = source.unsqueeze(1)  # make shape [batch, seq_len=1, d_model]

    # Looping until the 'max_len', maximum length, is reached
    while True:
        if decoder_input.size(1) == max_len:
            break
            
        # Building a mask for the decoder input
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)
        # Calculating the output of the decoder
        decoder_input = decoder_input.type(torch.int64)
        out = model.decode(source, decoder_input, decoder_mask)  #source is sequence embedding, "encoder_input"
        
        # Applying the projection layer to get the probabilities for the next token
        prob = model.project(out[:, -1])
        
        # Selecting token with the highest probability
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        
        # If the next token is an End of Sentence token, we finish the loop
        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)