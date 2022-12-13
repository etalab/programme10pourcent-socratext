#!/bin/bash
FULL_S3_ENDPOINT="https://${AWS_S3_ENDPOINT}"
AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    AWS_REGION=us-east-1 \
    S3_ENDPOINT=$FULL_S3_ENDPOINT \
    S3_USE_HTTPS=0 \
    S3_VERIFY_SSL=0 \
    tensorboard --logdir s3://projet-socratext/logs --host 0.0.0.0
