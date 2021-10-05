import os
import shutil
import unittest
import cv2
print(os.path.exists("runs/detect/test_result"))
if os.path.exists("runs/detect/test_result"):
    shutil.rmtree("runs/detect/test_result")
    print("Remove file")
print("Start running ..")
os.system("python detect.py --weights detect_3.pt --img 640 --source test_images/hardhelmet_vest_images --name test_result")
print("Completed running ..")

class TestModel(unittest.TestCase):

    # Test suite for detection model
    def test_helmet(self):
        pass

    def test_vest(self):
        pass

    def test_gloves(self):
        pass

    def test_goggles(self):
        pass

    def test_mask(self):
        pass

    # Tet suite for the User Interface
