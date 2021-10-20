import os
import shutil
import unittest
import cv2
import numpy
import matplotlib.pyplot as plt
print(os.path.exists("runs/detect/test_result"))
if os.path.exists("runs/detect/test_result"):
    shutil.rmtree("runs/detect/test_result")
    shutil.rmtree("runs/detect/gloves_result")
    shutil.rmtree("runs/detect/mask_result")
    shutil.rmtree("runs/detect/vest_result")
    shutil.rmtree("runs/detect/goggle_result")
    print("Remove file")
print("Start running ..")
os.system("python detect.py --weights detect_4.pt --img 640 --source test_images/hardhelmet_vest_images --name test_result")
os.system("python detect.py --weights detect_4.pt --img 640 --source test_images/gloves_images --name gloves_result")
os.system("python detect.py --weights detect_4.pt --img 640 --source test_images/goggle_images --name goggle_result")
os.system("python detect.py --weights detect_4.pt --img 640 --source test_images/mask_images --name mask_result")
os.system("python detect.py --weights detect_4.pt --img 640 --source test_images/vest_images --name vest_result")
print("Completed running ..")

class TestModel(unittest.TestCase):

    # Test suite for detection model
    def extract (self,class_i,filename):
        if not os.path.isfile(filename):
            return []
        file = open(filename, "r")
        xywh = file.readlines()
        new = []
        for line in xywh:
            new.append([float(i) for i in line.replace('\n', "").split(" ")])
        rtn_val = []
        for i in new :
            if i[0] == class_i:
                rtn_val.append(i)
        file.close()
        return rtn_val

    def get_preprocessing_result(self,extract_list,width,height,img) :
        contours = []
        stencil  = numpy.zeros(img.shape).astype(img.dtype)
        for i in extract_list :
            (x1,y1) = (int(i[1]*width-0.5*i[3]*width), int(i[2]*height-0.5*i[4]*height))
            (x2,y2) = (int(i[1]*width+0.5*i[3]*width), int(i[2]*height+0.5*i[4]*height))

            bb_points = [[x2,y2],
                         [x1,y2],
                         [x1,y1],
                         [x2,y1]]
            contours.append(numpy.array(bb_points))
        color    = [255, 255, 255]
        cv2.fillPoly(stencil, contours, color)

        result1 = cv2.bitwise_and(img, stencil)
        result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2RGB)
        plt.imshow(result1)
        return result1

    def get_number_of_TP_and_FP_FN(self, truth_extract, extract_list, truth_width, truth_height, extract_width, extract_height,img) :
        TP = 0
        FP = 0
        FN = 0
        last_elm = len(truth_extract)
        for i in extract_list :
            contours = []
            stencil  = numpy.zeros(img.shape).astype(img.dtype)
            color    = [255, 255, 255]
            (x1,y1) = (int(i[1]*extract_width-0.5*i[3]*extract_width), int(i[2]*extract_height-0.5*i[4]*extract_height))
            (x2,y2) = (int(i[1]*extract_width+0.5*i[3]*extract_width), int(i[2]*extract_height+0.5*i[4]*extract_height))
            bb_points = [[x2,y2],
                         [x1,y2],
                         [x1,y1],
                         [x2,y1]]
            contours.append(numpy.array(bb_points))
            cv2.fillPoly(stencil, contours, color)
            result1 = cv2.bitwise_and(img, stencil)
            result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2RGB)
            for j in range(len(truth_extract)) :
                contours = []
                stencil  = numpy.zeros(img.shape).astype(img.dtype)
                color    = [255, 255, 255]
                (x1,y1) = (int(i[1]*truth_width-0.5*truth_extract[j][3]*truth_width), int(i[2]*truth_height-0.5*truth_extract[j][4]*truth_height))
                (x2,y2) = (int(i[1]*truth_width+0.5*truth_extract[j][3]*truth_width), int(i[2]*truth_height+0.5*truth_extract[j][4]*truth_height))
                bb_points = [[x2,y2],
                             [x1,y2],
                             [x1,y1],
                             [x2,y1]]
                contours.append(numpy.array(bb_points))
                cv2.fillPoly(stencil, contours, color)
                result2 = cv2.bitwise_and(img, stencil)
                result2 = cv2.cvtColor(result2, cv2.COLOR_BGR2RGB)

                intersection = numpy.logical_and(result1, result2)
                union = numpy.logical_or(result1, result2)
                iou_score = numpy.sum(intersection)/ numpy.sum(union)
                if iou_score >= 0.5 :
                    TP += 1
                    break
                elif j == last_elm - 1 :
                    FP += 1
        for i in truth_extract :
                contours = []
                stencil  = numpy.zeros(img.shape).astype(img.dtype)
                color    = [255, 255, 255]
                (x1,y1) = (int(i[1]*extract_width-0.5*i[3]*extract_width), int(i[2]*extract_height-0.5*i[4]*extract_height))
                (x2,y2) = (int(i[1]*extract_width+0.5*i[3]*extract_width), int(i[2]*extract_height+0.5*i[4]*extract_height))
                bb_points = [[x2,y2],
                             [x1,y2],
                             [x1,y1],
                             [x2,y1]]
                contours.append(numpy.array(bb_points))
                cv2.fillPoly(stencil, contours, color)
                result1 = cv2.bitwise_and(img, stencil)
                result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2RGB)
                # plt.imshow(result1)
                for j in range(len(extract_list)) :
                    contours = []
                    stencil  = numpy.zeros(img.shape).astype(img.dtype)
                    color    = [255, 255, 255]
                    (x1,y1) = (int(i[1]*truth_width-0.5*extract_list[j][3]*truth_width), int(i[2]*truth_height-0.5*extract_list[j][4]*truth_height))
                    (x2,y2) = (int(i[1]*truth_width+0.5*extract_list[j][3]*truth_width), int(i[2]*truth_height+0.5*extract_list[j][4]*truth_height))
                    bb_points = [[x2,y2],
                                 [x1,y2],
                                 [x1,y1],
                                 [x2,y1]]
                    contours.append(numpy.array(bb_points))
                    cv2.fillPoly(stencil, contours, color)
                    result2 = cv2.bitwise_and(img, stencil)
                    result2 = cv2.cvtColor(result2, cv2.COLOR_BGR2RGB)
                    # plt.imshow(result2)
                    # plt.show()
                    intersection = numpy.logical_and(result1, result2)
                    union = numpy.logical_or(result1, result2)
                    iou_score = numpy.sum(intersection)/ numpy.sum(union)
                    if iou_score >= 0.5 :
                        # print("terminated here")
                        break
                    elif j == len(extract_list) - 1 :
                        FN += 1
        return (TP,FP,FN)

    def get_iou(self, test_file_name, ground_truth_file_name, test_i, ground_truth_i,img_file):
        img = cv2.imread(img_file)
        (height,width,_) = img.shape
        extract_list = self.extract(test_i,test_file_name)
        truth_extract = self.extract(ground_truth_i,ground_truth_file_name)
        result1 = self.get_preprocessing_result(extract_list,width,height,img)
        result2 = self.get_preprocessing_result(truth_extract,width,height,img)

        #IOU calculation
        intersection = numpy.logical_and(result1, result2)
        union = numpy.logical_or(result1, result2)
        iou_score = numpy.sum(intersection)/ numpy.sum(union)
        if str(iou_score) == "nan" :
            return 1
        return iou_score

    def get_recall_precision(self,test_file_name, ground_truth_file_name, test_i, ground_truth_i, img_file):
        img = cv2.imread(img_file)
        (height, width, _) = img.shape
        extract_list = self.extract(test_i, test_file_name)
        truth_extract = self.extract(ground_truth_i,ground_truth_file_name)
        (TP,FP,FN) = self.get_number_of_TP_and_FP_FN(truth_extract, extract_list, width, height,width,height,img)
        if (TP+FP == 0) and (TP + FP == 0):
            #print("True Positive : ",TP, "False Positive : ", FP, "False Negative : " , FN ," Precision : " , 1, "Recall : ", 1)
            return (1.0,1.0)
        #print("True Positive : ",TP, "False Positive : ", FP, "False Negative : " , FN ," Precision : " , TP / (TP+FP), "Recall : ", TP/ (TP+FN))
        return(TP / (TP+FP) , TP/ (TP+FN))

    def test_helmet(self):
        total_iou = 0
        iou_score = self.get_iou("./runs/detect/test_result/labels/batch_1 (3).txt",
                                 "../truth/labels/batch_1--3-_jpg.rf.bc3132b0c80fadf5b876f52b0e726ce3.txt",2,0,
                                 "./test_images/hardhelmet_vest_images/batch_1 (3).jpg")
        total_iou += iou_score

        iou_score = self.get_iou("./runs/detect/test_result/labels/batch_1 (4).txt",
                                 "../truth/labels/batch_1--4-_jpg.rf.5e9643e02696291eb839abb50c23e091.txt",2,0,
                                 "./test_images/hardhelmet_vest_images/batch_1 (4).jpg")
        total_iou += iou_score

        iou_score = self.get_iou("./runs/detect/test_result/labels/batch_1 (6).txt",
                                 "../truth/labels/batch_1--6-_jpg.rf.88d81737cae01753b97b5e78b1117c92.txt",2,0,
                                 "./test_images/hardhelmet_vest_images/batch_1 (6).jpg")
        total_iou += iou_score

        iou_score = self.get_iou("./runs/detect/test_result/labels/batch_1 (7).txt",
                                 "../truth/labels/batch_1--7-_jpg.rf.8a41d55ae28e659878940307a8e8dbe4.txt",2,0,
                                 "./test_images/hardhelmet_vest_images/batch_1 (7).jpg")
        total_iou += iou_score

        iou_score = self.get_iou("./runs/detect/test_result/labels/batch_1 (8).txt",
                                 "../truth/labels/batch_1--8-_jpg.rf.f0f14f485e3493326bfc90448f306026.txt",2,0,
                                 "./test_images/hardhelmet_vest_images/batch_1 (8).jpg")
        total_iou += iou_score
        print("Average IOU for helmet : ", total_iou/5)
        pass

    def test_vest(self):
        total_iou = 0
        for file in os.listdir("./test_images/vest_images/"):
            filename = os.fsdecode(file)
            new = os.path.splitext(filename)[0]
            img_file = "./test_images/vest_images/" + filename
            result = "./runs/detect/vest_result/labels/" + new + ".txt"
            truth  = "../truth/vest_truth/label/" + new + ".txt"
            # print(result + "\n", truth + "\n", new)
            iou_score = self.get_iou(result,truth,5,6,img_file)
            total_iou += iou_score
            # print("result : " ,result, "iou_score" ,iou_score , "\n")
        print("Average IOU for vest : ", total_iou/len(os.listdir("./test_images/vest_images/")))
        pass

    def test_gloves(self):
        total_iou = 0
        for file in os.listdir("./test_images/gloves_images/"):
            filename = os.fsdecode(file)
            new = os.path.splitext(filename)[0]
            img_file = "./test_images/gloves_images/" + filename
            result = "./runs/detect/gloves_result/labels/" + new + ".txt"
            truth  = "../truth/gloves_truth/label/" + new + ".txt"
            # print(result + "\n", truth + "\n", new)
            iou_score = self.get_iou(result,truth,0,0,img_file)
            total_iou += iou_score
            # print("result : " ,result, "iou_score" ,iou_score , "\n")
        print("Average IOU for gloves : ", total_iou/len(os.listdir("./test_images/gloves_images/")))
        pass

    def test_goggles(self):
        total_iou = 0
        for file in os.listdir("./test_images/goggle_images/"):
            filename = os.fsdecode(file)
            new = os.path.splitext(filename)[0]
            img_file = "./test_images/goggle_images/" + filename
            result = "./runs/detect/goggle_result/labels/" + new + ".txt"
            truth  = "../truth/goggle_truth/label/" + new + ".txt"
            # print(result + "\n", truth + "\n", new)
            iou_score = self.get_iou(result,truth,1,1,img_file)
            total_iou += iou_score
            # print("result : " ,result, "iou_score" ,iou_score , "\n")
        print("Average IOU for goggle : ", total_iou/len(os.listdir("./test_images/goggle_images/")))
        pass

    def test_mask(self):
        total_iou = 0
        for file in os.listdir("./test_images/mask_images/"):
            filename = os.fsdecode(file)
            new = os.path.splitext(filename)[0]
            img_file = "./test_images/mask_images/" + filename
            result = "./runs/detect/mask_result/labels/" + new + ".txt"
            truth  = "../truth/mask_truth/label/" + new + ".txt"
            # print(result + "\n", truth + "\n", new)
            iou_score = self.get_iou(result,truth,3,3,img_file)
            total_iou += iou_score
            # print("result : " ,result, "iou_score" ,iou_score , "\n")
        print("Average IOU for mask : ", total_iou/len(os.listdir("./test_images/mask_images/")))
        pass

    def test_helmet_precision_recall(self):
        total_recall = 0
        total_precision = 0
        (recall,precision) = self.get_recall_precision("./runs/detect/test_result/labels/batch_1 (4).txt",
                                  "../truth/labels/batch_1--4-_jpg.rf.5e9643e02696291eb839abb50c23e091.txt",2,0,
                                  "./test_images/hardhelmet_vest_images/batch_1 (4).jpg")

        total_recall += recall
        total_precision += precision
        (recall,precision) = self.get_recall_precision("./runs/detect/test_result/labels/batch_1 (3).txt",
                                 "../truth/labels/batch_1--3-_jpg.rf.bc3132b0c80fadf5b876f52b0e726ce3.txt",2,0,
                                 "./test_images/hardhelmet_vest_images/batch_1 (3).jpg")
        total_recall += recall
        total_precision += precision
        (recall,precision) = self.get_recall_precision("./runs/detect/test_result/labels/batch_1 (6).txt",
                                 "../truth/labels/batch_1--6-_jpg.rf.88d81737cae01753b97b5e78b1117c92.txt",2,0,
                                 "./test_images/hardhelmet_vest_images/batch_1 (6).jpg")
        total_recall += recall
        total_precision += precision
        (recall,precision) = self.get_recall_precision("./runs/detect/test_result/labels/batch_1 (7).txt",
                                 "../truth/labels/batch_1--7-_jpg.rf.8a41d55ae28e659878940307a8e8dbe4.txt",2,0,
                                 "./test_images/hardhelmet_vest_images/batch_1 (7).jpg")
        total_recall += recall
        total_precision += precision
        (recall,precision) = self.get_recall_precision("./runs/detect/test_result/labels/batch_1 (8).txt",
                                 "../truth/labels/batch_1--8-_jpg.rf.f0f14f485e3493326bfc90448f306026.txt",2,0,
                                 "./test_images/hardhelmet_vest_images/batch_1 (8).jpg")
        total_recall += recall
        total_precision += precision
        print("Helmet :  Average Recall : ", total_recall/5 , " Average Precision : ",total_precision/5)

    def test_vest_precision_recall(self):
            total_recall = 0
            total_precision = 0
            for file in os.listdir("./test_images/vest_images/"):
                filename = os.fsdecode(file)
                new = os.path.splitext(filename)[0]
                img_file = "./test_images/vest_images/" + filename
                result = "./runs/detect/vest_result/labels/" + new + ".txt"
                truth  = "../truth/vest_truth/label/" + new + ".txt"
                # print(result + "\n", truth + "\n", new)
                (recall,precision) = self.get_recall_precision(result,truth,5,6,img_file)
                total_recall += recall
                total_precision += precision
                # print("result : " ,result, "iou_score" ,iou_score , "\n")
            print("Vest :  Average Recall : ", total_recall/len(os.listdir("./test_images/vest_images/")) , " Average Precision : ",total_precision/len(os.listdir("./test_images/vest_images/"))  )
            pass

    def test_gloves_precision_recall(self):
        total_recall = 0
        total_precision = 0
        for file in os.listdir("./test_images/gloves_images/"):
            filename = os.fsdecode(file)
            new = os.path.splitext(filename)[0]
            img_file = "./test_images/gloves_images/" + filename
            result = "./runs/detect/gloves_result/labels/" + new + ".txt"
            truth  = "../truth/gloves_truth/label/" + new + ".txt"
            # print(result + "\n", truth + "\n", new)
            (recall,precision) = self.get_recall_precision(result,truth,0,0,img_file)
            total_recall += recall
            total_precision += precision
            # print("result : " ,result, "iou_score" ,iou_score , "\n")
        print("Gloves :  Average Recall : ", total_recall/len(os.listdir("./test_images/gloves_images/")) ,
              " Average Precision : ",total_precision/len(os.listdir("./test_images/gloves_images/"))  )
        pass

    def test_goggles_precision_recall(self):
        total_recall = 0
        total_precision = 0
        for file in os.listdir("./test_images/goggle_images/"):
            filename = os.fsdecode(file)
            new = os.path.splitext(filename)[0]
            img_file = "./test_images/goggle_images/" + filename
            result = "./runs/detect/goggle_result/labels/" + new + ".txt"
            truth  = "../truth/goggle_truth/label/" + new + ".txt"

            (recall,precision) = self.get_recall_precision(result,truth,1,1,img_file)
            total_recall += recall
            total_precision += precision

        print("Goggles :  Average Recall : ", total_recall/len(os.listdir("./test_images/goggle_images/")) ,
              " Average Precision : ",total_precision/len(os.listdir("./test_images/goggle_images/"))  )
        pass

    def test_mask_precision_recall(self):
        total_recall = 0
        total_precision = 0
        for file in os.listdir("./test_images/mask_images/"):
            filename = os.fsdecode(file)
            new = os.path.splitext(filename)[0]
            img_file = "./test_images/mask_images/" + filename
            result = "./runs/detect/mask_result/labels/" + new + ".txt"
            truth  = "../truth/mask_truth/label/" + new + ".txt"

            (recall,precision) = self.get_recall_precision(result,truth,3,3,img_file)
            total_recall += recall
            total_precision += precision

        print("Mask :  Average Recall : ", total_recall/len(os.listdir("./test_images/mask_images/")) ,
              " Average Precision : ",total_precision/len(os.listdir("./test_images/mask_images/"))  )
        pass


    # Tet suite for the User Interface
