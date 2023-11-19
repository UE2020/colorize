#!/bin/bash
ls /home/tt/Downloads/trailer/frames/|sort -R |tail -100 |while read file; do
    echo $file
    /home/tt/Documents/autoencoder/target/release/autoencoder test /home/tt/Downloads/final.pt  /home/tt/Downloads/trailer/frames/$file false
    cp /home/tt/Downloads/trailer/frames/$file truth.jpg    
done