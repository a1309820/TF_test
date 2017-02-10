to create the ags98234/ft-test docker file use the following command:
docker build -t ags98234/tf-test .       <--- where the Dockerfile is in the current directory

to run the ags98234/ft-test docker file with the mnist_softmax.py Python code use the command:
docker run -i -t -v `pwd`:/code ags98234/tf-test python3 -u /code/myMnist_softmax_1.py > output.txt

Had to increase memory allocation for docker itself (moved to 12GB).  This allowd the final model accuracy measurement to be completed.

