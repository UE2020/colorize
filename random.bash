#!/bin/bash
ls /home/tt/enhancer/data/truth|sort -R |tail -100 |while read file; do
    echo $file
    /home/tt/Documents/autoencoder/target/release/autoencoder test /home/tt/Documents/autoencoder/final.safetensors /home/tt/enhancer/data/truth/$file false
done