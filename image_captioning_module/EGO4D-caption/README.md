We run LLM server using gunicorn to generate captions.

1. Run LLM Server.
   - Install libraries ```pip install "uvicorn[standard]" gunicorn transformers fastapi```.
   - Start LLM server 
  ```cd PALM/image_captioning_module/EGO4D-caption/utils```
  ```gunicorn api_utils:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8964 -t 600 -c config.py```.


2. Run Captioning models.
  ```python generate_caption_api.py```.
