from ultralytics import YOLO

# Ubicación del archivo best
model = YOLO('./runs/detect/train14/weights/last.pt')

# Ubicación de la imagen a detectar
source = 'C:/Users/Lucas Garcia/Desktop/LLVS/ProyectoAvioncito/datanet/commercial/test/Cessna 172/0890644.jpg'

model.predict(source, save=True, imgsz=320, conf=0.5)