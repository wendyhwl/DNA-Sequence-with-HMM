import sys
import numpy as np
import difflib

def main(algorithm, filename):
	d = difflib.Differ()
	if algorithm == 'viterbi':
		reference = open('reference_viterbi.txt')
	else:
		reference = open('reference_posterior.txt')
	output    = open(filename)

	x = np.array("".join(reference.readlines()).split('\n'))
	ref = list(x[:-1].reshape((19, 3)))
	x = np.array("".join(output.readlines()).split('\n'))
	out = list(x[:-1].reshape((19, 3)))
	
	for index, observation in enumerate(ref):
		current = out[index]
		if observation[0] == current[0] and observation[1] == current[1] and observation[2] == current[2]:
			print("sequence " + str(index) + " passed.")
		else:
			print("sequence " + str(index) + " failed.")


	reference.close()
	output.close()


if __name__ == '__main__':
	
	algo = sys.argv[1]
	file = sys.argv[2]
	main(algo, file)