This is an example of a simple <b>Convolutional Neural Network </b>. 
It consists of two files preprocessing.py and cnn.py for preparing the data and implementing cnn.
Here we use keras with a tensorflow backend.


This is how your project Directoy must be.<br><br>
<pre>
  ------/Workspace  
            |---/data-directory
            |        |---/Class A
            |               |---/IMG1
            |               |---/IMG2
            |        |---/Class B
            |                |---/IMG1
            |                |---/IMG2
            |---preprocessing.py
            |---cnn.py
           </pre>

 > We'll have to run preprocessing.py first so that all the images are first loaded, converted to required format, 
that is 200 X 200 resolution and grayscale. They are then pickled and stored.
 > Next we run cnn.py which loads the pickled features and labels vector. Then keras comes and tensorflow come into the play
 sucessfully running the Convolution Neural Network.
 
 
