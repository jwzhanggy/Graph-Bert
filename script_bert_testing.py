
import torch
import logging
from MethodGraphBert import MethodGraphBert
from MethodGraphBertNodeConstruct import MethodGraphBertNodeConstruct
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertConfig

#---- GraphBert functionality test 1: hidden state output ----
if 0:
    logging.basicConfig(level=logging.INFO)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize input
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = tokenizer.tokenize(text)

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'
    assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a',
                              'puppet', '##eer', '[SEP]']

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model = MethodGraphBert.from_pretrained('bert-base-uncased')
    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
        # See the models docstrings for the detail of the inputs
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        # Transformers models always output tuples.
        # See the models docstrings for the detail of all the outputs
        # In our case, the first element is the hidden state of the last layer of the Bert model
        encoded_layers = outputs[0]
    # We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
    print(tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size))


#---- GraphBert functionality test 2: masked langauge modeling ----
if 1:
    logging.basicConfig(level=logging.INFO)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize input
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = tokenizer.tokenize(text)

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'
    assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a',
                              'puppet', '##eer', '[SEP]']

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model = MethodGraphBertNodeConstruct.from_pretrained('bert-base-uncased')
    model.eval()

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs[0]

    # confirm we were able to predict 'henson'
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(predicted_token)







