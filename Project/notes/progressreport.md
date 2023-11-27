# Progress Report : Convolution in GPU 

This project contains a convolution layer implemented in GPU.

## Implementation

See README.md in the project root directory.

## Optimization points.

### conv2mul

* Since the multiplication has been optimized over years, it's generally profitable to convert convolution to multiplication.
  
This involves several steps

1. Input matrix unrolling

2. Invoke Tensor Core

3. Put the result back to the output matrix