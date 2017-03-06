#Character level RNN 
import numpy as np 

file = open('alice.txt' , 'r').read()
chars = list(set(file))
file_size , char_size = len(file) , len(chars)
print "File has %d characters and %d unique characters" %(file_size,char_size) 


#Builds the Vector model 
char_ix = {ch:i for i,ch in enumerate(chars)}
ix_char = {i:ch for i,ch in enumerate(chars)}

# Example Vector for 'x'

vector_for_char = np.zeros((char_size , 1))
vector_for_char[char_ix['x']] = 1

# print the sample vector

print vector_for_char.ravel()

#param
neuron_size = 100 
seq_length = 25 # number of steps to unroll RNN
learn_rate = 1e-1

#weights
Wxh = np.random.randn(neuron_size , char_size)*0.01  
Whh = np.random.randn(neuron_size , neuron_size)*0.01 
Why = np.random.randn(char_size , neuron_size)*0.01
bh = np.zeros((neuron_size , 1))
by = np.zeros((char_size,1))

# Creating a sample text model without any optimization

def text_sample(h , start_ix , n):
	'''
	h -> memory state
	start_ix -> start char

	'''
	x = np.zeros((char_size , 1))
	x[start_ix] = 1
	ixs = []
	for t in xrange(n):
		h = np.tanh(np.dot(Wxh , x) + np.dot(Whh ,h) + bh)
		y = np.dot(Why , h) + by
		
		p = np.exp(y) / np.sum(np.exp(y))
		ix = np.random.choice(range(char_size) , p = p.ravel())
		x = np.zeros((char_size , 1))
		x[ix] = 1
		ixs.append(ix)
	txt = ''.join(ix_char[ix] for ix in ixs)
	print '------ \n %s \n ------'%(txt,)

hprev = np.zeros((neuron_size , 1)) # reset the memory state
text_sample(hprev , char_ix['a'] , 2000)

# The above model makes no sense

#Now defining the loss function
def lossFunc(inputs , targets , hprev):
	'''
	input , target -> list of integers
	hprev -> initial memory state of hidden neurons (we start with zeros)
	returns the loss , gradients on model parameters , last neuron hidden state
	'''

	n = len(inputs)
	xs , hs , ys , ps = {} , {}, {} ,{}
	hs[-1] = np.copy(hprev)
	loss= 0

	#Forward Pass

	for t in xrange(n):
		xs[t] = np.zeros((char_size,1)) 
		xs[t][inputs[t]] = 1

		hs[t] = np.tanh(np.dot(Wxh , xs[t] + np.dot(Whh , hs[t-1] + bh))) #Hidden neuron state
		ys = np.dot(Why , hs[t]) + by # Probalibity fro next char
		ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
		loss += -np.log(ps[t][targets[t] , 0]) #softmax
	
	# Backward Pass

	dWxh ,dWhh , dWhy = np.zeros_like(Wxh) , np.zeros_like(Whh) , np.zeros_like(Why)
	dbh , dby = np.zeros_like(bh) , np.zeros_like(by)

	dhnext = np.zeros_like(hs[0])

	# Going backwards so reversed( xrange() ) **

	for t in reversed(xrange(n)):
		dy = np.copy[ps[t]]
		dy[targets[t]] -= 1 # Backprop into y
		dWhy += np.dot(dy,hs[t].T)
		