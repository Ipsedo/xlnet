#!/usr/bin/env bash

DOWNLOADED_DIR="downloaded"

if ! [ -d ${DOWNLOADED_DIR} ]; then
    mkdir ${DOWNLOADED_DIR}
fi

cd ${DOWNLOADED_DIR}

if ! [ -d "aclImdb" ]; then
    wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    tar zxvf aclImdb_v1.tar.gz
    rm -f aclImdb_v1.tar.gz
fi

cd ..