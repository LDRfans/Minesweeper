import numpy as np

num_train_rounds = 5
num_test_rounds = 2
num_succ = 1
weights_in_h = np.array([[1,1,1,1],[0,0,0,0],[2,2,2,2]])
weights_h_out = np.array([[0,0],[1,1],[9,9],[4,4]])

fo = open("output0.txt", "w")
fo.write(str(num_train_rounds)+" "+str(num_test_rounds)+" "+str(num_succ)+"\n")
fo.write("weights_in_h:\n")
fo.write(str(weights_in_h)+"\n")
fo.write("weights_h_out:\n")
fo.write(str(weights_h_out))
# print the winning rate
print("Done.")
fo.close()
