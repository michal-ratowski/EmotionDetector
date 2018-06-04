#ifndef TRAINER_H
#define	TRAINER_H

#include <string>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"

#include "FaceAnalyzer.h"
#include "dialog.h"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

using namespace std;
using namespace cv;

class Trainer {
public:
    Trainer();
    virtual ~Trainer();

    void trainSVM(CvSVM &svm, int svm_type, int kernel_type, double degree, double gamma, double coef0, double C, double nu, double p, int max_iter, double epsilon);
    void trainKNN(CvKNearest &knn, int K);
    void trainMLP(CvANN_MLP &mlp, int train_method, double dw_scale, double moment_scale, double dw0, double dw_plus, double dw_minus, int layer1, int layer2, int max_iter, double epsilon);
    void trainBayes(CvNormalBayesClassifier &bayes);
    void trainDecisionTree(CvDTree &dtree, int min_sample_count, double regression_accuracy, bool use_surrogates, int max_categories, double cv_folds, bool use_1se_rule, bool truncate_pruned_tree);

    void start(shape_predictor sp, frontal_face_detector detector);
    void prepareData(int &picturesFound, int &facesFound, Dialog* dialog, bool training);
    void loadPictures(Dialog* dialog, bool training);
    void scaleData();

    void createAverageNeutralImage();
    void loadNeutralImage(string fileName);
    void saveNeutralImage(string fileName);

    void saveData(string fileName);
    void loadData(string fileName, Dialog* dialog);
    void saveEvaluationData(string fileName);
    void loadEvaluationData(string fileName, Dialog* dialog);

    Mat getTrainingDataMat();
    Mat getLabelsMat();

    std::string trainingImagePath, analysisImagePath;
    std::vector<float> neutralImage;
    shape_predictor sp;
    frontal_face_detector detector;

    std::vector<std::vector<float>> trainingDataVector;
    std::vector<std::vector<float>> trainingDataVectorScaled;
    std::vector<float> trainingLabelsVector;

    std::vector<std::vector<float>> evaluateDataVector;
    std::vector<float> evaluateLabelsVector;

    std::vector<Mat> trainingImages;
    std::vector<Mat> analysisImages;
    std::vector<Mat> neutralImages;

    std::vector<float> smallestScaleVector;
    std::vector<float> largestScaleVector;
    bool minmax;
    float scaleFactor;

};

#endif

