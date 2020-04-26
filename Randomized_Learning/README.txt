I primarily used the ABAGAIL repository (https://github.com/pushkar/ABAGAIL).

Part 1

To preprocess and produce the MNIST data as a csv, run the mnist_to_csv.py file.
The hyperparameters are set at the top of file. Sklearn and matplotlib are required
python libraries that can be pip installed.

I adapted a tutorial for ABAGAIL for a poker dataset 
(https://gist.github.com/mosdragon/53edf8e69fde531db69e) to perform random optimization 
and hyperparameter search for the MNIST data. I changed the hyperparameters of the filename, 
num examples, and num_attributes (pca dimension) to fit my mnist data along with some 
debugging and variations on hyperparameter search. To run the code, you can add the 
java file with the changed parameters to the src repo of the ABAGAIL code and use ant 
to compile the jar. Then you can include the jar and execute the main class.

python mnist_to_csv.py
ant jar
java -cp ABAGAIL.jar MnistTest2

For ease of running, as I didn't want to include the whole ABAGAIL repo,
I have provided by jar file as well as an example post processed csv file.
just run:
java -cp ABAGAIL.jar project2.MnistTest2

Part 2

The code for the three theory problems are provided as FlipFlopTest.java, 
FourPeaksTest.java, and TravelingSalesmanTest.java and are adapted from the test 
folder under the src/ in the ABAGAIL repository (just edited for more trials).
To compile just take the ABAGAIL.jar file under the ABAGAIL repository and 
do the following:

javac -cp ABAGAIL.jar:. theory_problem.java
java -cp ABAGAIL.jar:. theory_problem

Flip flop test will take arguments for N (input string) and iterations
java -cp ABAGAIL.jar:. FlipFlopTest 180 20

Four Peaks test will take arguments for N, T, and iterations
java -cp ABAGAIL.jar:. FourPeaksTest 50 5 20

Traveling Salesman test will take arguments for N and iterations
java -cp ABAGAIL.jar:. TravelingSalesmanTest 50 20