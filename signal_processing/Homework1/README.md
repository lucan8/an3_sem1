# JPEG encoder/decoder

## Description

Implemented a jpeg encoder/decoder that implements the whole jpeg algorithm without writitng/reading to/from  file, keeping everything in memory. It works for both gray and colored images, allows the user specify a mse
threshold to fine tune the compression and works on videos.

## Usage

This does not yet have a CLI, so just clone the repo, install the dependencies, and call task_i.  

- task1: Compression + decompression on a gray image
- task2: Compression + decompression on a colored image
- task3: Compression for any kind of image, with a given mse threshold
- task4: Compression for videos

All the tasks show the initial and compressed + decompressed images for comparison and print the mse.

## Dependencies

matplotlib==3.10.6
numpy==2.2.6
opencv-python==4.12.0.88
scipy==1.16.3

## Algorithm

Given an image, it first gets resized in order to be split in 8x8 blocks. For each block devide by the quantization matrix (multiplied by the quantization factor), apply cosine transform, convert the matrix to a zig-zag vector, convert that to a rule length encoding (zero_count, non_zero_value), take (zero_count, non_zero_value_bit_length) and encode it using huffman and append the binary non_zero_value to it. With this now we have something I will call bitstream, which in code is represented as a bytearray.  
This bytearray would normally be written to a file together with all the metadata(huffman table/tree, image initial size, quantization factor, etc), but I keep everything in memory for simplicity.  
