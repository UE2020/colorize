#!/bin/bash
ls /home/tt/enhancer/data/truth|sort -R |tail -100 |while read file; do
    echo $file
    /home/tt/Documents/autoencoder/target/release/autoencoder test /home/tt/Downloads/model06.safetensors /home/tt/enhancer/data/truth/$file
done