# You need to have roboflow installed in your virtual environment

from roboflow import Roboflow
rf = Roboflow(api_key="ieuVIgoSLJvfjf8y5cco")
project = rf.workspace("christiancella22").project("wara_bottles_detection")
version = project.version(1)
dataset = version.download("yolov8")
