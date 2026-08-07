

# Discrete3DOD

Repositorio oficial para IEEE Transaction on Intelligent Transportation Systems, titulado ["One Stage Monocular 3D Object Detection Utilizing Discrete Depth and Orientation Representation."](https://ieeexplore.ieee.org/abstract/document/9780191)

## Requisitos
1. Python 3.7
2. Pytorch 1.9
3. OpenCV > 3.0
4. Tqdm
5. Cityscapesscripts (https://github.com/mcordts/cityscapesScripts)
6. PyQuaternion

## Configurar los ajustes
Realice cambios en la configuración [aquí](cfg/settings.py):
1. Cambiar la ruta raíz del conjunto de datos
2. Cambiar el nombre/ubicación de los pesos
3. Cambiar la ubicación de la carpeta de guardado para almacenar los resultados de predicción
4. Descargue nuestros pesos preentrenados [aquí](https://drive.google.com/file/d/1daQsTJXrEGrW7ckDk6UmYc-2lTZgQ0Gr/view?usp=sharing)

## Ejecutar la evaluación
Para ejecutar el benchmark de cityscapes, simplemente 
```
python evaluate.py
```
Los resultados se guardarán en saves/results/results.json

## Visualizar resultados
```
python gen_vis.py
```
Los resultados se guardarán en la carpeta saves/vis/
