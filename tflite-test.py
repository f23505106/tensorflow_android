import numpy
import tensorflow as tf
import gzip

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

# Load TFLite model and allocate tensors.
interpreter = tf.contrib.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("input_details",input_details)
print("output_details",output_details)
# Test model on random input data.

data = extract_data('./data/t10k-images-idx3-ubyte.gz')
labels = extract_labels('./data/t10k-labels-idx1-ubyte.gz')

with open("tflite.cvs", 'w') as r:
    r.write('index\tprobabilities\tprediction\tlabel\n')
    count = 0
    for i in range(10000):
        interpreter.set_tensor(input_details[0]['index'], [data[i]])
        interpreter.invoke()
        probabilities = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        # I taught a neural net to recognise when a sum of numbers is bigger than 45
        # it should return False in this case
        #print('prediction:',prediction,' label:',labels[i]) # [[ False ]] Yay, it works!
        r.write("%d\t%s\t%d\t%d\n" % (i," ".join(str(e) for e in probabilities[0]),classes[0],labels[i]))
        if classes[0] == labels[i]:
            count+=1
    r.write(str(count/10000))
