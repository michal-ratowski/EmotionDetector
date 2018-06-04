#include <cstdlib>

#include "FaceAnalyzer.h"
#include "/home/michal/Downloads/stasm4.1.0/stasm/stasm.h"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"

using namespace std;
using namespace cv;
using namespace dlib;

FaceAnalyzer::FaceAnalyzer(){
}
FaceAnalyzer::~FaceAnalyzer() {}


std::vector<float> FaceAnalyzer::extractLandmarks(Mat &inputImage, shape_predictor sp, frontal_face_detector detector, bool showLandmarks, bool showNumbers) {

    std::vector<float> faceLandmarks;
    array2d<rgb_pixel> inputImageArray;
    assign_image(inputImageArray, cv_image<bgr_pixel>(inputImage));

    std::vector<dlib::rectangle> dets = detector(inputImageArray);
    //cout << "Number of faces detected: " << dets.size() << endl;

    std::vector<full_object_detection> shapes;

    if(dets.size()>0)
    {

        full_object_detection shape = sp(inputImageArray, dets[0]);

        if(showLandmarks || showNumbers){
            for(int i=0;i<shape.num_parts();i++){
                if(showLandmarks){
                    circle(inputImage,Point(shape.part(i).x(),shape.part(i).y()),1,Scalar(255,255,255),1);
                }
                if(showNumbers){
                    string number = to_string(i);
                    putText(inputImage,number,Point(shape.part(i).x(),shape.part(i).y()),FONT_HERSHEY_DUPLEX,0.3,Scalar(255,255,255),0.5);
                }
            }
        }

        Point faceLeftSide = Point(shape.part(1).x(),shape.part(1).y());
        Point faceRightSide = Point(shape.part(15).x(),shape.part(15).y());

        Point upperLipCenter = Point(shape.part(57).x(),shape.part(57).y());
        Point lowerLipCenter = Point(shape.part(51).x(),shape.part(51).y());
        Point leftLipCorner = Point(shape.part(48).x(),shape.part(48).y());
        Point rightLipCorner = Point(shape.part(64).x(),shape.part(64).y());

        Point leftEyeLeftCorner = Point(shape.part(36).x(),shape.part(36).y());
        Point leftEyeRightCorner = Point(shape.part(39).x(),shape.part(39).y());

        Point rightEyeLeftCorner = Point(shape.part(42).x(),shape.part(42).y());
        Point rightEyeRightCorner = Point(shape.part(45).x(),shape.part(45).y());

        Point leftBrow = Point(shape.part(17).x(),shape.part(17).y());
        Point leftBrowCenter = Point(shape.part(19).x(),shape.part(19).y());
        Point leftBrowEnd = Point(shape.part(21).x(),shape.part(21).y());

        Point rightBrow = Point(shape.part(22).x(),shape.part(22).y());
        Point rightBrowCenter = Point(shape.part(24).x(),shape.part(24).y());
        Point rightBrowEnd = Point(shape.part(26).x(),shape.part(26).y());

        Point noseBottom = Point(shape.part(33).x(),shape.part(33).y());

        float faceWidth = faceRightSide.x-faceLeftSide.x;

        float mouthHeight = upperLipCenter.y-lowerLipCenter.y;
        float mouthWidth = rightLipCorner.x-leftLipCorner.x;

        float smile[2];
        smile[0] = noseBottom.y-leftLipCorner.y;
        smile[1] = noseBottom.y-rightLipCorner.y;

        float leftBrowRise[3];
        leftBrowRise[0] = leftBrow.y-leftEyeLeftCorner.y;
        leftBrowRise[1] = leftBrowCenter.y-leftEyeLeftCorner.y;
        leftBrowRise[2] = leftBrowEnd.y-leftEyeLeftCorner.y;

        float rightBrowRise[3];
        rightBrowRise[0] = rightBrow.y-rightEyeLeftCorner.y;
        rightBrowRise[1] = rightBrowCenter.y-rightEyeLeftCorner.y;
        rightBrowRise[2] = rightBrowEnd.y-rightEyeLeftCorner.y;

        mouthHeight/=faceWidth;
        mouthWidth/=faceWidth;

        smile[0]/=faceWidth;
        smile[1]/=faceWidth;

        leftBrowRise[0]/=faceWidth;
        leftBrowRise[1]/=faceWidth;
        leftBrowRise[2]/=faceWidth;

        rightBrowRise[0]/=faceWidth;
        rightBrowRise[1]/=faceWidth;
        rightBrowRise[2]/=faceWidth;

        faceLandmarks.push_back(mouthHeight);
        faceLandmarks.push_back(mouthWidth);
        faceLandmarks.push_back(smile[0]);
        faceLandmarks.push_back(smile[1]);
        faceLandmarks.push_back(leftBrowRise[0]);
        faceLandmarks.push_back(leftBrowRise[1]);
        faceLandmarks.push_back(leftBrowRise[2]);
        faceLandmarks.push_back(rightBrowRise[0]);
        faceLandmarks.push_back(rightBrowRise[1]);
        faceLandmarks.push_back(rightBrowRise[2]);
    }
    return faceLandmarks;
}

void FaceAnalyzer::printLandmarks(std::vector<float> faceLandmarks){
    cout<<"Mouth height = "<<faceLandmarks.at(0)<<endl;
    cout<<"Mouth width = "<<faceLandmarks.at(1)<<endl;
    cout<<"Smile = "<<faceLandmarks.at(2)<<endl;
    cout<<"Smile = "<<faceLandmarks.at(3)<<endl;
    cout<<"LeftBrowRise = "<<faceLandmarks.at(4)<<endl;
    cout<<"LeftBrowRise = "<<faceLandmarks.at(5)<<endl;
    cout<<"LeftBrowRise = "<<faceLandmarks.at(6)<<endl;
    cout<<"RightBrowRise = "<<faceLandmarks.at(7)<<endl;
    cout<<"RightBrowRise = "<<faceLandmarks.at(8)<<endl;
    cout<<"RightBrowRise = "<<faceLandmarks.at(9)<<endl;
}
