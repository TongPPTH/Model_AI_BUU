# Run Yolov8 model

```
python model.py

```

# Can't run?

_create your model for yourself_

- in IVUF-100-1 folder have a yolo_from.py

```
python yolo_from.py
```

_train model at epochs recommend 60-100 epochs_

```
yolo train model=yolov8n.pt data=/content/IVUF-100-1/data.yaml epochs={epochs} imgsz=640
```

_use model_

```
yolo task=detect mode=predict model={your path}/best.pt  source=/content/IVUF-100-1/test/images
```

_now you can Run Yolov8 model_

```
python model.py

```
