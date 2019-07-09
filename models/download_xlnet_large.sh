#!/usr/bin/env bash

if ! [ -d "cased_L-24_H-1024_A-16" ]; then
    wget https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip
    unzip cased_L-24_H-1024_A-16.zip
    rm -f cased_L-24_H-1024_A-16.zip
fi