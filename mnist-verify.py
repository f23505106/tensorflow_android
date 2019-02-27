import argparse 
import tensorflow as tf
import gzip
import numpy
import imageio

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def,name="prefix")
    return graph

def extract_data(filename):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(28 * 28 * 10000 * 1)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (255 / 2.0)) / 255
    data = data.reshape(10000, 28, 28, 1)
    return data


def extract_labels(filename):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(10000)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("model", default="frozen_model.pb", type=str, help="Frozen model file to import")
    parser.add_argument("-cvs", type=str, help="the result run the mode in cvs format")

    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.model)

    if not args.cvs:
        # We can verify that we can access the list of operations in the graph
        for op in graph.get_operations():
            print("name:",op.name,"  op:",op.type)
            # prefix/Placeholder/inputs_placeholder
            # ...
            # prefix/Accuracy/predictions

    # We access the input and output nodes 
    inp = graph.get_tensor_by_name('prefix/Placeholder_3:0')
    probabilities = graph.get_tensor_by_name('prefix/probabilities:0')
    classes = graph.get_tensor_by_name('prefix/classes:0')
    # We launch a Session
    with tf.Session(graph=graph) as sess:
        if args.cvs:
            data = extract_data('./data/t10k-images-idx3-ubyte.gz')
            labels = extract_labels('./data/t10k-labels-idx1-ubyte.gz')
            with open(args.cvs, 'w') as r:
                r.write('index\tprobabilities\tprediction\tlabel\n')
                # Note: we don't nee to initialize/restore anything
                # There is no Variables in this graph, only hardcoded constants 
                count = 0
                for i in range(10000):
                    prediction = sess.run([probabilities,classes], feed_dict={
                        inp: [data[i]] # < 45
                    })
                    # I taught a neural net to recognise when a sum of numbers is bigger than 45
                    # it should return False in this case
                    #print('prediction:',prediction,' label:',labels[i]) # [[ False ]] Yay, it works!
                    r.write("%d\t%s\t%d\t%d\n" % (i," ".join(str(e) for e in prediction[0][0]),prediction[1][0],labels[i]))
                    if prediction[1][0] == labels[i]:
                        count+=1
                r.write(str(count/10000))
        else:
            im = imageio.imread("models/official/mnist/example5.png")
            im = (im[:,:,0:1]).astype(numpy.float32)
            im = ((255 / 2.0) - im) / 255
            print(im.shape,im.dtype)
            prediction = sess.run([probabilities,classes], feed_dict={
                inp: [im] # < 45
            })
            print('prediction:',prediction)