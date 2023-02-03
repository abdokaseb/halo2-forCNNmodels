# halo2 CNN Model

This repository is a fork of https://github.com/zcash/halo2 with CNN model example.

The added module is CNNmodel with the Halo2 API. This is a non-official implementation of [Scaling up Trustless DNN Inference with Zero-Knowledge Proofs](https://arxiv.org/abs/2210.08674).

## Usage

First cd to the correct directory

```
cd halo2_proofs
```

Then run python file to generate the model parameters and input image and dump them and the excepted result on a text files

```
python examples/build_model.py
```

Then run the halo2 verifier

```
cargo run --example dnn-model
```

Both python and rs should print the values of the final layer in hex represntation and both of them are equal each other

## Future Work

* Improve the implmentation of the lookup gate in ReLU (as it takes large verification time)
* Improve the implmentation of the dot product to be a gate by itself without calling add, sub, and mull gates (this will improve verification time)
* Reimplement the convolution by more optimized approch. As now it's O(C^2 * K^2 * H * W) but there are multiple optimized approahces, where C is the channel size, K is the kernel size, while H and W are the output height and width
