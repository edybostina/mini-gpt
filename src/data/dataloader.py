import tiktoken
import torch

class DataLoader:
    def __init__(self, logger, encoding, batch_size, seq_length, filepath="input.txt"):
        self.batch_size = batch_size
        self.seq_length = seq_length

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        self.enc = tiktoken.get_encoding(encoding)
        ids = self.enc.encode(text)
        self.tokens = torch.tensor(ids, dtype=torch.long)
        self.num_tokens = len(self.tokens)
        logger.info(f"loaded {self.num_tokens} tokens from {filepath}")

        # rough idea of how many steps make an epoch
        self.steps_per_epoch = max(1, self.num_tokens // (self.batch_size * self.seq_length))
        logger.info(f"~{self.steps_per_epoch} steps per epoch (rough)")

    def next_batch(self):
        # sample independent random starts for each item in batch
        starts = torch.randint(
            low=0,
            high=self.num_tokens - self.seq_length - 1,
            size=(self.batch_size,),
        )
        x = torch.stack([self.tokens[s:s+self.seq_length] for s in starts])
        y = torch.stack([self.tokens[s+1:s+self.seq_length+1] for s in starts])
        return x, y