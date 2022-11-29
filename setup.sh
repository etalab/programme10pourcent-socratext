#!/bin/bash
mc cp s3/projet-socratext/tickets/labeled_sample.json data/sample/
mc cp s3/projet-socratext/tickets/images.zip data/sample/
cd data/sample
unzip images.zip
