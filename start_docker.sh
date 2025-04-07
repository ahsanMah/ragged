docker run -it \
--gpus=all --cap-add SYS_RESOURCE \
-e USE_MLOCK=0 -it -u 1000:1000 \
-v ./:/app \
-v /mnt/storage/:/mnt/storage \
-p 8000:8000 \
ragged bash -c "cd /app && uvicorn server:app --port 8000 --host 0.0.0.0"
