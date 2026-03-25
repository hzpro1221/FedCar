#!/bin/bash

SOURCE="dataset"

PARENT_DIR=".."
DEST_NAME="results_$(date +%Y%m%d_%H%M).tar.gz"
DEST_PATH="$PARENT_DIR/$DEST_NAME"

echo "Compressing folder '$SOURCE' into '$DEST_PATH'..."
tar -czf "$DEST_PATH" "$SOURCE"

if [ $? -eq 0 ]; then
    echo "Compression complete: $DEST_PATH"
else
    echo "Error: Compression failed!"
    exit 1
fi

echo "Uploading to Google Drive..."
rclone copy "../results_20260324_2138.tar.gz" gdrive:FedCar_Project/Results \
    --progress \
    --buffer-size 16M \
    --drive-chunk-size 64M \
    --low-level-retries 10 \
    --stats 1m
    
if [ $? -eq 0 ]; then
    echo "Upload successful! Cleaning up temporary file from parent directory..."
    rm "$DEST_PATH"
    echo "Process finished."
else
    echo "Error: Upload failed! File preserved at $DEST_PATH for safety."
    exit 1
fi