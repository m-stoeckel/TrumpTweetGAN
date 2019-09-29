from typing import Any

from torch import nn

from run_generation import *


class GPT2_G(nn.Module):
    def __init__(self, model_name_or_path, max_seq_len, temperature=1.0, top_k=0, top_p=0.9, gpu=True):
        super(GPT2_G, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_seq_len = max_seq_len
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.gpt2: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(self.device)


    def forward(self, *input: torch.LongTensor, **kwargs: Any):
        # TODO: move this block to instructor
        # context = self.tokenizer.encode(input)
        # context = torch.tensor(input, dtype=torch.long, device=self.device)
        for _ in range(self.max_seq_len):
            inputs = {'input_ids': input}
            # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            outputs = self.gpt2(**inputs)
            next_token_logits = outputs[0][0, -1, :] / self.gpt2
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=self.top_k, top_p=self.top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            input = torch.cat((input, next_token.unsqueeze(0)), dim=1)
        return input[0, len(input):]

    def sample(self, num_samples, batch_size):
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        generated = torch.randint_like(torch.tensor((batch_size, 1)), low=0, high=self.gpt2.config.num_labels).long()
        generated = generated.to(self.device)
        for b in trange(num_batch):
            sequence = self.forward(*generated)

        return None
