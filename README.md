# burlap_caffe

### Caffe Dependencies

To make sure you have all the dependencies for Caffe, we recommend
cloning the [Caffe Github repo](https://github.com/BVLC/caffe)
and compiling it (with CUDA) by following their
[installation instructions](http://caffe.berkeleyvision.org/installation.html).

Additionally, make sure CUDA is added to the LD_LIBRARY_PATH. If CUDA
was installed with apt-get (on a 64-bit machine),
you can simply add this to your `.bashrc`
```sh
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
And then running
```sh
source ~/.bashrc
```


### JavaCPP Caffe with cuDNN

We are using Bytedeco's Java bindings for Caffe, the dependencies of which
are all handled through maven, and should download and run smoothly.
But, this repository compiles Caffe without cuDNN, which makes the GPU training much slower.
To compile with cuDNN, first clone the JavaCPP presets repo:

```sh
git clone https://github.com/bytedeco/javacpp-presets
```

By default, this library does not compile caffe with cuDNN, so we have to change the cppbuild.sh script.
Replace the javacpp-presets/caffe/cppbuild.sh with the file provided in the instalation_files directory.
Now run these commands to compile JavaCPP caffe and install it to maven:

```sh
./cppbuild.sh install caffe
mvn install --projects caffe
```

Now this library should be linked with maven.


### Example Code

We provide two sets of example code within the project.

The first, `AtariDQN`, is the same architecture as DeepMind's DQN
with the same hyper-parameters as specified in their [nature paper]
(http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html).
NOTE: You may need to increase the Java max memory size to run this example.
To do so, add `-Xmx8192m` as a Java VM argument (this sets the max memory
to 8GB).

The second, `GridWorldDQN`, is a simple DQN implementation of a
built-in BURLAP domain.
