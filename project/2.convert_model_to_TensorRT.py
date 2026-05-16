from ultralytics import YOLO

model = YOLO("/home/ys9072/jetson_orin_nano_super_yolov11-master/weights/best.pt")

model.export(format="engine")  

trt_model = YOLO("best.engine")


results = trt_model("/home/ys9072/jetson_orin_nano_super_yolov11-master/test.jpg")

print(results)