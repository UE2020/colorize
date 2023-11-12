#!/bin/bash
ls /home/tt/Documents/autoencoder/data/data/truth|sort -R |tail -100 |while read file; do
    echo $file
    /home/tt/Documents/autoencoder/target/release/autoencoder test /home/tt/Documents/autoencoder/face5.safetensors /home/tt/Documents/autoencoder/data/data/truth/$file false
    cp /home/tt/Documents/autoencoder/data/data/truth/$file truth.jpg
    /home/tt/Documents/autoencoder/target/release/autoencoder test /home/tt/Documents/autoencoder/final.safetensors /home/tt/Documents/autoencoder/data/data/truth/$file false
done