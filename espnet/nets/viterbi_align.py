import numpy as np
import pdb

# logits is the probo f output of each phoneme at time t (L, T) labels x time
# sequence is the index of the phoneme expected in the output sequence. 

# http://deeplearning.cs.cmu.edu/slides.spring19/lec14.CTC.pdf
# slide 158
def viterbi_align(logits, sequence):
	blank_id = 0
	sequence = list(sequence)
	for i in range(len(sequence)):
		sequence.insert(2*i, blank_id)
	sequence.append(blank_id)

	T = logits.shape[1]
	N = len(sequence)
	# print(N, T)
	if (N > T):
		raise Exception("Number of expected symbols more than the time stamps")
	
	s = np.zeros((T, N))
	bp = np.zeros((T, N), dtype=np.int)
	bp.fill(-1)
	bscr = np.zeros((T, N))
	aligned_seq1 = np.zeros((T), dtype=np.int)
	aligned_seq2 = np.zeros((T), dtype=np.int)

	# filling S
	# print(sequence)
	for i in range(N):
		s[:, i] = logits[sequence[i]] 

	# s = np.log(np.array([[0.1, 0.5, 0.4, 0.1, 0.2], [0.2, 0.2, 0.2, 0.3, 0.7], [0.4, 0.1, 0.1, 0.2, 0.6],\
	# 	 [0.2, 0.3, 0.3, 0.1, 0.6], [0.3, 0.1, 0.4, 0.4, 0.7]])).T

	# base case
	bp[0, 0] = 0 # made this 0 instead of -1. 
	bp[0, 1] = 1 
	bscr[0, 0] = s[0, 0]
	bscr[0, 1] = s[0, 1]
	bscr[0, 2:] = np.NINF

	# filling over the rest time stamps
	for t in range(1, T):
		bp[t, 0] = bp[t-1,0]
		bscr[t, 0] = bscr[t-1,0] + s[t, 0]
		bp[t, 1] = 1 if bscr[t-1,1] > bscr[t-1, 0] else 0
		bscr[t, 1] = bscr[t-1, bp[t, 1]] + s[t, 1]

		for i in range(2, N):
			# print("going in")
			if (i%2 == 0): # blank 
				bp[t, i] = i if bscr[t-1,i] > bscr[t-1, i-1] else i-1
			else:
				if (sequence[i] == sequence[i-2]):
					bp[t, i] = i if bscr[t-1,i] > bscr[t-1, i-1] else i-1
				else:
					bp[t, i] = i if (bscr[t-1,i] > bscr[t-1, i-1] and bscr[t-1,i] > bscr[t-1, i-2]) else\
						   (i-1 if (bscr[t-1,i-1] > bscr[t-1, i] and bscr[t-1,i-1] > bscr[t-1, i-2]) else\
						   	i-2)
			bscr[t, i] = bscr[t-1, bp[t, i]] + s[t, i]

	# print(bp.T)
	# print(np.exp(bscr).T)

	aligned_seq1[T-1], path_score_1 = N-1, 0
	for t in range(T-1, 0, -1):
		aligned_seq1[t-1] = bp[t, aligned_seq1[t]] 
		path_score_1 += bscr[t, aligned_seq1[t]]

	aligned_seq2[T-1], path_score_2 = N-2, 0
	for t in range(T-1, 0, -1):
		aligned_seq2[t-1] = bp[t, aligned_seq2[t]] 
		path_score_2 += bscr[t, aligned_seq2[t]]

	aligned_seq = aligned_seq1 if (path_score_1 > path_score_2) else aligned_seq2

	aligned_symbols_idx = []
	for i in range(len(aligned_seq)):
		if i > 0 and aligned_seq[i] == aligned_seq[i-1]:
   			aligned_symbols_idx.append(0)
		else:
			aligned_symbols_idx.append(sequence[aligned_seq[i]])
	aligned_idx = np.where(np.array(aligned_symbols_idx) != 0)

	return aligned_idx

if __name__ == "__main__":
	total_labels = 10 # including blank 
	T = 9
	sequence = [2,5]
	logits = np.random.rand(total_labels, T) #(total_labels, T)
	aligned_symbols_idx = viterbi_align(logits, sequence)
	print(aligned_symbols_idx)
