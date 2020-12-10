import os
from imageai.Detection import ObjectDetection
print('===>Iniciando.')

model_path = "./models/yolo-tiny.h5"
input_path = "./input/"
output_path = "./output/"

file_paths = [os.path.join(input_path, file) for file in os.listdir(input_path)]
files = [file for file in file_paths if os.path.isfile(file) and file.lower().endswith(".jpg")]

# Load model
detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()
print('===> Modelo Yolo carregado')

# Object detection
for i, file in enumerate(files):
	detection = detector.detectObjectsFromImage(input_image=file, output_image_path=f"{output_path}image{i}.jpg")
	print(f"\n\n=====> ITEM {i} = {file} <======")
	for eachItem in detection:
		print("===> ",eachItem["name"] , " : ", eachItem["percentage_probability"])