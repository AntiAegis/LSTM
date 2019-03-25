#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
from caesar_cipher import CaesarCipher


#------------------------------------------------------------------------------
#  Network
#------------------------------------------------------------------------------
class Network(nn.Module):
	def __init__(self, input_sz, embedding_dim, hidden_sz, output_sz):
		super(Network, self).__init__()
		self.embed = nn.Embedding(input_sz, embedding_dim)
		self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_sz, batch_first=False)
		self.linear = nn.Linear(hidden_sz, output_sz)
		self.hidden_in = (torch.zeros(1, 1, HIDDEN_SIZE), torch.zeros(1, 1, HIDDEN_SIZE))

	def forward(self, x):
		lstm_in = self.embed(x)
		lstm_in = lstm_in.unsqueeze(1)
		lstm_out, _ = self.lstm(lstm_in, self.hidden_in)
		logits = self.linear(lstm_out)
		return logits


#------------------------------------------------------------------------------
#  Parameters
#------------------------------------------------------------------------------
# Network
N_TRAIN_SAMPLES = 100
N_TEST_SAMPLES = 50
MESS_LEN = 50
EMBEDDING_DIM = 5
HIDDEN_SIZE = 10

# Training
LR = 1e-3
N_EPOCHS = 15


#------------------------------------------------------------------------------
#	Main execution
#------------------------------------------------------------------------------
# Create dataset
cipher = CaesarCipher()
train_data = cipher.gen_data(n_samples=N_TRAIN_SAMPLES, mess_len=MESS_LEN)
test_data = cipher.gen_data(n_samples=N_TEST_SAMPLES, mess_len=MESS_LEN)

# Build model
network = Network(
	input_sz=cipher.n_vocab,
	embedding_dim=EMBEDDING_DIM,
	hidden_sz=HIDDEN_SIZE,
	output_sz=cipher.n_vocab,
)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=LR)

# Train
# network.train()
for epoch in range(N_EPOCHS):
	print('Epoch: {}'.format(epoch))
	for data, label in train_data:
		# optimizer.zero_grad()
		logits = network(data)
		logits = logits[:,0,:]

		loss = loss_fn(logits, label)
		loss.backward()
		optimizer.step()

	print('Loss: {:6.4f}'.format(loss.item()))

# Test
with torch.no_grad():
	network.eval()
	matches, total = 0, 0
	for data, label in test_data:
		logits = network(data)
		_, predictions = logits.max(dim=2)
		predictions = predictions.squeeze(1)
		matches += torch.eq(predictions, label).sum().item()
		total += torch.numel(predictions)
	accuracy = matches / total
	print('Accuracy: {:4.2f}%'.format(accuracy * 100))