#!/bin/bash
ls /home/tt/Downloads/trailer/frames|sort -R |tail -100 |while read file; do
    echo $file
    /home/tt/Documents/autoencoder/target/release/autoencoder test /home/tt/Documents/autoencoder/colab9.safetensors /home/tt/Downloads/trailer/frames/$file
done