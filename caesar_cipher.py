#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import torch, random


#------------------------------------------------------------------------------
#	Main execution
#------------------------------------------------------------------------------
class CaesarCipher():
	def __init__(self, vocab="ABCDEFGHIJKLMNOPQRSTUVWXYZ-", shift=13):
		self.vocab = [char for char in vocab]
		self.shift = shift
		self.n_vocab = len(vocab)

	def normalize(self, word):
		word_ = word.upper()
		word_ = [char for char in word_]
		for i, char in enumerate(word_):
			if char not in self.vocab:
				word_[i] = '-'
		return "".join(word_)

	def encrypt(self, word):
		indices = [self.vocab.index(char) for char in word]
		indices = [(index+self.shift)%self.n_vocab for index in indices]
		word_ = [self.vocab[index] for index in indices]
		word_ = "".join(word_)
		return word_

	def gen_data(self, n_samples, mess_len):
		dataset = []
		for _ in range(n_samples):
			ex_out = ''.join([random.choice(self.vocab) for _ in range(mess_len)])
			ex_in = self.encrypt(''.join(ex_out))
			ex_in = [self.vocab.index(x) for x in ex_in]
			ex_out = [self.vocab.index(x) for x in ex_out]
			dataset.append((torch.tensor(ex_in), torch.tensor(ex_out)))
		return dataset


#------------------------------------------------------------------------------
#   Test bench
#------------------------------------------------------------------------------
if __name__=="__main__":
	# Create instance
	cipher = CaesarCipher(vocab="ABCDEFGHIJKLMNOPQRSTUVWXYZ-", shift=13)

	# Test normalization and encryption
	text = "aBCDEFGHIJKLMNoPQRSTUVWXYZ-/"
	text_norm = cipher.normalize(text)
	text_encr = cipher.encrypt(text_norm)
	print(text_encr)

	# Test gen_data
	dataset = cipher.gen_data(n_samples=20, mess_len=15)
	print(dataset)