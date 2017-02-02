import sys

orig_stdout = sys.stdout
f = open('output.txt','w')
sys.out=f

i=15
print(i)


sys.stdout = orig_stdout
f.close()
