#include <cstdlib>
#include "math.h"
#include <time.h>
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

class FaceAnalyzer {
public:
    FaceAnalyzer();
    virtual ~FaceAnalyzer();
    std::vector<float> extractLandmarks(Mat &inputImage, shape_predictor sp, frontal_face_detector detector, bool showLandmarks, bool showNumbers);
    void printLandmarks(std::vector<float> faceLandmarks);
};

