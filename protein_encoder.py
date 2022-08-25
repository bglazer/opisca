from transformers import T5Tokenizer, T5Model
import re
import torch

tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_bfd', do_lower_case=False)

model = T5Model.from_pretrained("Rostlab/prot_t5_xl_bfd")

# TODO find file with sequences
# mapping to graph?
sequences_Example = ["A E T C Z A O","S K T Z P"]

sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)

input_ids = torch.tensor(ids['input_ids'])
attention_mask = torch.tensor(ids['attention_mask'])

with torch.no_grad():
    embedding = model(decoder_input_ids=input_ids, input_ids=input_ids, attention_mask=attention_mask)

# For feature extraction we recommend to use the encoder embedding
encoder_embedding = embedding[2].cpu().numpy()
breakpoint()
# TODO export results
