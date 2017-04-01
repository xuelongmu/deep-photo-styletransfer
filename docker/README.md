# deep-photo-styletransfer dockerfile
Build this image using something like:
```
docker build -t deep_photo .
```
To run the container you'll need recent nvidia drivers installed and nvidia-docker (from here: https://github.com/NVIDIA/nvidia-docker). Then run something like:
```
nvidia-docker run -it --name deep_photo deep_photo
```

