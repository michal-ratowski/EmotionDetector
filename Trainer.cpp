#include <cstdlib>
#include <QMessageBox>
#include "math.h"
#include "Trainer.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"

using namespace std;
using namespace cv;
using namespace dlib;

Trainer::Trainer(){}
Trainer::~Trainer() {}

void Trainer::start(shape_predictor sp, frontal_face_detector detector){
    this->sp = sp;
    this->detector = detector;
    this->minmax=false;
}

void Trainer::prepareData(int &picturesFound, int &facesFound, Dialog* dialog, bool training) {

    string logString;
    logString = "Loading pictures...";

    if(training) {dialog->appendLine(logString,1);}
    else {dialog->appendLine(logString,3);}
    loadPictures(dialog,training);

    int imagesSize;
    if(training){ imagesSize = trainingImages.size();}
    else{ imagesSize = analysisImages.size(); }

    cout<<"Images size: "<<imagesSize<<endl;
    picturesFound = imagesSize;

    FaceAnalyzer faceAnalyzer;

    if(training){
        logString = "Creating average neutral image...";
        dialog->appendLine(logString,1);
        createAverageNeutralImage();
        saveNeutralImage("neutralImage");
    }

    for(int i=0;i<imagesSize;i++){

        logString = "Analysing image "+to_string(i+1)+"/"+to_string(imagesSize);
        if(training){dialog->appendLine(logString,1);}
        else{dialog->appendLine(logString,3);}

        std::vector<float> landmarks;

        if(training){ landmarks = faceAnalyzer.extractLandmarks(trainingImages[i], sp, detector, true, false); }
        else{ landmarks = faceAnalyzer.extractLandmarks(analysisImages[i], sp, detector, true, false); }

        if(landmarks.size()>0){
            std::vector<float> landmarksDisplacements;   
            for(int j=0;j<10;j++){
                landmarksDisplacements.push_back(landmarks[j]-neutralImage[j]);
            }

            if(training){ trainingDataVector.push_back(landmarksDisplacements); }
            else{ evaluateDataVector.push_back(landmarksDisplacements); }
            facesFound++;

        }
        else{
            cout<<"Erasing image "<<i<<" - face not detected"<<endl;
            if(training) {trainingLabelsVector.erase(trainingLabelsVector.begin() + i);}
            else{evaluateLabelsVector.erase(trainingLabelsVector.begin() + i);}

        }
    }  
}

void Trainer::trainSVM(CvSVM &svm, int svm_type, int kernel_type, double degree, double gamma, double coef0, double C, double nu, double p, int max_iter, double epsilon) {

    float trainingDataArray[trainingDataVector.size()][10];
    float labelsArray[trainingDataVector.size()];

    for(int i=0;i<trainingDataVectorScaled.size();i++){
        for(int j=0;j<10;j++){
            trainingDataArray[i][j] = trainingDataVectorScaled[i][j];
            //cout<<"SVM Training data ["<<i<<"]["<<j<<"] = "<<trainingDataArray[i][j]<<endl;
        }
    }

    for(int i=0;i<trainingLabelsVector.size();i++){
        labelsArray[i] = trainingLabelsVector[i];
    }

    Mat trainingDataMat = Mat(trainingDataVectorScaled.size(), 10, CV_32FC1, trainingDataArray);
    Mat labelsMat(trainingLabelsVector.size(), 1, CV_32FC1, labelsArray);

    CvSVMParams params;
    params.svm_type = svm_type;
    params.kernel_type = kernel_type;
    params.degree = degree;
    params.gamma = gamma;
    params.coef0 = coef0;
    params.C = C;
    params.nu = nu;
    params.p = p;
    params.term_crit.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
    params.term_crit.max_iter = max_iter;
    params.term_crit.epsilon = epsilon;

    cout<<"Training SVM..."<<endl;
    svm.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
}

void Trainer::trainKNN(CvKNearest &knn, int K) {

    float trainingDataArray[trainingDataVectorScaled.size()][10];
    float labelsArray[trainingDataVectorScaled.size()];

    for(int i=0;i<trainingDataVectorScaled.size();i++){
        for(int j=0;j<10;j++){
            trainingDataArray[i][j] = trainingDataVectorScaled[i][j];
        }
    }

    for(int i=0;i<trainingLabelsVector.size();i++){
        labelsArray[i] = trainingLabelsVector[i];
    }

    Mat trainingDataMat = Mat(trainingDataVectorScaled.size(), 10, CV_32FC1, trainingDataArray);
    Mat labelsMat(trainingLabelsVector.size(), 1, CV_32FC1, labelsArray);

    cout<<"Training KNN..."<<endl;
    knn.train(trainingDataMat, labelsMat, Mat(), false, K);    
}

void Trainer::trainMLP(CvANN_MLP &mlp, int train_method, double dw_scale, double moment_scale, double dw0, double dw_plus, double dw_minus, int layer1, int layer2, int max_iter, double epsilon){

    float trainingDataArray[trainingDataVectorScaled.size()][10];
    float labelsArray[trainingDataVectorScaled.size()];

    for(int i=0;i<trainingDataVectorScaled.size();i++){
        for(int j=0;j<10;j++){
            trainingDataArray[i][j] = trainingDataVectorScaled[i][j];
        }
    }

    for(int i=0;i<trainingLabelsVector.size();i++){
        labelsArray[i] = trainingLabelsVector[i];
    }

    Mat trainingDataMat = Mat(trainingDataVectorScaled.size(), 10, CV_32FC1, trainingDataArray);
    Mat labelsMat(trainingLabelsVector.size(), 1, CV_32FC1, labelsArray);

    Mat layers = cv::Mat(4, 1, CV_32SC1);
    layers.row(0) = cv::Scalar(10);
    layers.row(1) = cv::Scalar(layer1);
    layers.row(2) = cv::Scalar(layer2);
    layers.row(3) = cv::Scalar(1);

    CvANN_MLP_TrainParams params;

    params.train_method = train_method;
    params.bp_dw_scale = dw_scale;
    params.bp_moment_scale = moment_scale;
    params.rp_dw0 = dw0;
    params.rp_dw_plus = dw_plus;
    params.rp_dw_minus = dw_minus;
    params.rp_dw_min = FLT_EPSILON;
    params.rp_dw_max = 50;
    params.term_crit.max_iter = max_iter;
    params.term_crit.epsilon = epsilon;
    params.term_crit.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

    mlp.create(layers);
    cout<<"Training MLP..."<<endl;

    theRNG().state = 12345679;
    mlp.train(trainingDataMat, labelsMat, cv::Mat(), cv::Mat(), params);

}

void Trainer::loadPictures(Dialog* dialog, bool training) {

    if(training){
        trainingImages.clear();
        trainingLabelsVector.clear();
        neutralImages.clear();
    }
    else{
        analysisImages.clear();
        evaluateLabelsVector.clear();
    }

    std::vector<cv::String> fn;
    string folderName;

    for(int i=0;i<6;i++){
        switch(i) {
            case 0:
                folderName = "neutral";
                break;
            case 1:
                folderName = "sad";
                break;
            case 2:
                folderName = "happy";
                break;
            case 3:
                folderName = "angry";
                break;
            case 4:
                folderName = "surprised";
                break;
            case 5:
                folderName = "disgusted";
                break;
            default:
                break;
        }

        string specificImagePath;
        if(training){ specificImagePath = trainingImagePath + "/" + folderName + "/*.jpg"; }
        else{ specificImagePath = analysisImagePath + "/" + folderName + "/*.jpg"; }

        glob(specificImagePath, fn, false);
        size_t count = fn.size();
        string logString = "Found "+to_string(count)+" pictures in folder "+folderName;

        if(training){ dialog->appendLine(logString,2); }
        else{ dialog->appendLine(logString,3); }

        for (size_t j=0;j<count;j++)
        {

            Mat img = imread(fn[j],1);
            if (img.empty()){
                cerr<<"Image "<<fn[j]<<" can't be loaded!"<<endl;
                continue;
            }
            else{
                if(training){
                    trainingImages.push_back(img);
                    trainingLabelsVector.push_back(i);
                    if(i==0){
                        neutralImages.push_back(img);
                    }
                }
                else{
                    analysisImages.push_back(img);
                    evaluateLabelsVector.push_back(i);
                }
            }
        }
    }
}

void Trainer::scaleData(){

    largestScaleVector.clear();
    smallestScaleVector.clear();    
    trainingDataVectorScaled.clear();

    for(int i=0;i<trainingDataVector.size();i++){
        std::vector<float> floatLine;
        for(int j=0;j<10;j++){
            floatLine.push_back(trainingDataVector[i][j]);
        }
        trainingDataVectorScaled.push_back(floatLine);
    }

    for(int i=0;i<trainingDataVector.size();i++){
        cout<<i<<": ";
        for(int j=0;j<10;j++){
            cout<<trainingDataVectorScaled[i][j]<<", ";
        }
        cout<<endl;
    }

    for(int i=0;i<10;i++){

        double largest = 0;
        double smallest = 0;

        for(int j=0;j<trainingDataVector.size();j++){
            if(trainingDataVector[j][i]>largest){
                largest = trainingDataVector[j][i];
            }
            else if(trainingDataVector[j][i]<smallest){
                smallest = trainingDataVector[j][i];
            }
        }

        largestScaleVector.push_back(largest);
        smallestScaleVector.push_back(smallest);

        cout<<"Training data vector "<<i<<" smallest: "<<smallest<<", largest: "<<largest<<endl;
    }

    for(int i=0;i<trainingDataVector.size();i++){
        for(int j=0;j<10;j++){
            if(minmax){
                trainingDataVectorScaled[i][j] = (trainingDataVectorScaled[i][j]-smallestScaleVector[j])/(largestScaleVector[j]-smallestScaleVector[j]);
            }
            else{
                trainingDataVectorScaled[i][j]*=scaleFactor;
            }
        }
    }
}


void Trainer::trainBayes(CvNormalBayesClassifier &bayes){

    float trainingDataArray[trainingDataVectorScaled.size()][10];
    float labelsArray[trainingDataVectorScaled.size()];

    for(int i=0;i<trainingDataVectorScaled.size();i++){
        for(int j=0;j<10;j++){
            trainingDataArray[i][j] = trainingDataVectorScaled[i][j];
        }
    }

    for(int i=0;i<trainingLabelsVector.size();i++){
        labelsArray[i] = trainingLabelsVector[i];
    }

    Mat trainingDataMat = Mat(trainingDataVectorScaled.size(), 10, CV_32FC1, trainingDataArray);
    Mat labelsMat(trainingLabelsVector.size(), 1, CV_32FC1, labelsArray);

    cout<<"Training Bayes..."<<endl;
    bayes.train(trainingDataMat, labelsMat);
}


void Trainer::trainDecisionTree(CvDTree &dtree, int min_sample_count, double regression_accuracy, bool use_surrogates, int max_categories, double cv_folds, bool use_1se_rule, bool truncate_pruned_tree) {

    float trainingDataArray[trainingDataVectorScaled.size()][10];
    float labelsArray[trainingDataVectorScaled.size()];

    for(int i=0;i<trainingDataVectorScaled.size();i++){
        for(int j=0;j<10;j++){
            trainingDataArray[i][j] = trainingDataVectorScaled[i][j];
        }
    }

    for(int i=0;i<trainingLabelsVector.size();i++){
        labelsArray[i] = trainingLabelsVector[i];
    }

    Mat trainingDataMat = Mat(trainingDataVectorScaled.size(), 10, CV_32FC1, trainingDataArray);
    Mat labelsMat(trainingLabelsVector.size(), 1, CV_32FC1, labelsArray);

    cv::Mat var_type(11, 1, CV_8U);

    for(int i=0;i<11;i++){
        var_type.at<unsigned char>(0,i) = CV_VAR_NUMERICAL;
    }

    CvDTreeParams params;
    params.max_depth = INT_MAX;
    params.min_sample_count = min_sample_count;
    params.regression_accuracy = regression_accuracy;
    params.use_surrogates = use_surrogates;
    params.max_categories = max_categories;
    params.cv_folds = cv_folds;
    params.use_1se_rule = use_1se_rule;
    params.truncate_pruned_tree = truncate_pruned_tree;
    params.priors = 0;

    cout<<"Training Decision Tree..."<<endl;
    dtree.train(trainingDataMat,CV_ROW_SAMPLE, labelsMat, cv::Mat(), cv::Mat(), var_type, cv::Mat(), params);
}

void Trainer::saveData(string fileName){

    cout<<"Saving training data..."<<endl;
    ofstream myFile;
    myFile.open (fileName);

    for(int i=0;i<trainingDataVector.size();i++){
        for(int j=0;j<10;j++){
            myFile << trainingDataVector[i][j]<<",";
        }
        myFile << trainingLabelsVector[i] << endl;
    }

    myFile.close();

}

void Trainer::saveEvaluationData(string fileName){

    cout<<"Saving evaluated data..."<<endl;
    ofstream myFile;
    myFile.open (fileName);

    for(int i=0;i<evaluateDataVector.size();i++){
        for(int j=0;j<10;j++){
            myFile << evaluateDataVector[i][j]<<",";
        }
        myFile << evaluateLabelsVector[i] << endl;
    }

    myFile.close();

}

void Trainer::loadData(string fileName, Dialog* dialog){

    loadNeutralImage("neutralImage");

    cout<<"Loading data from file..."<<endl;
    string line;
    ifstream myFile (fileName);

    if (myFile.is_open())
    {
        trainingDataVector.clear();
        trainingLabelsVector.clear();

        while(!myFile.eof())
        {
            getline(myFile,line);
            QString qstr = QString::fromStdString(line);
            QStringList pieces = qstr.split(",");

            std::vector<float> singleDataVector;

            if(pieces.size()==11){
                for(int i=0;i<11;i++){
                    float landmark = pieces.value(i).toFloat();
                    singleDataVector.push_back(landmark);
                }
            }

            if(singleDataVector.size()>0){
                trainingLabelsVector.push_back(singleDataVector[10]);
                singleDataVector.pop_back();
                trainingDataVector.push_back(singleDataVector);
            }

        }
        myFile.close();

        if(trainingDataVector.size()==0){
            QMessageBox::information(dialog,dialog->tr("Error"),"Invalid data file!");
        }
    }
    else{
        cerr << "Unable to open file!"<<endl;
        QString errorMessage = "File <<"+QString::fromStdString(fileName)+">> does not exist!";
        QMessageBox::information(dialog,dialog->tr("Error"),errorMessage);
    }

}

void Trainer::loadEvaluationData(string fileName, Dialog* dialog){

    cout<<"Loading data from file: "<<fileName<<endl;
    string line;
    ifstream myFile (fileName);

    if (myFile.is_open())
    {
        evaluateDataVector.clear();
        evaluateLabelsVector.clear();

        while(!myFile.eof())
        {
            getline(myFile,line);
            QString qstr = QString::fromStdString(line);
            QStringList pieces = qstr.split(",");

            std::vector<float> singleDataVector;

            if(pieces.size()==11){
                for(int i=0;i<11;i++){
                    float landmark = pieces.value(i).toFloat();
                    singleDataVector.push_back(landmark);
                }
            }

            if(singleDataVector.size()>0){
                evaluateLabelsVector.push_back(singleDataVector[10]);
                singleDataVector.pop_back();
                evaluateDataVector.push_back(singleDataVector);
            }

        }
        myFile.close();

        if(evaluateDataVector.size()==0){
            QMessageBox::information(dialog,dialog->tr("Error"),"Invalid data!");
        }
    }
    else{
        cerr << "Unable to open file!";
        QMessageBox::information(dialog,dialog->tr("Error"),"File does not exist!");
    }

}

void Trainer::createAverageNeutralImage(){
    neutralImage.clear();

    int goodPictures=0;
    FaceAnalyzer faceAnalyzer;

    float averageLandmarks[10] = {0};

    for(int i=0;i<neutralImages.size();i++){

        std::vector<float> landmarks;

        landmarks = faceAnalyzer.extractLandmarks(neutralImages[i], sp, detector, false, false);

        if(landmarks.size()>0){
            goodPictures++;
            for(int j=0;j<10;j++){
                averageLandmarks[j]+=landmarks[j];
            }
        }
    }

    for(int j=0;j<10;j++){
        averageLandmarks[j]/=goodPictures;
        neutralImage.push_back(averageLandmarks[j]);
    }
}

void Trainer::loadNeutralImage(string fileName){

    cout<<"Loading neutral image..."<<endl;
    string line;
    ifstream myFile (fileName);

    float averageLandmarks[10] = {0};

    if (myFile.is_open())
    {
        neutralImage.clear();

        while(!myFile.eof())
        {
            getline(myFile,line);
            QString qstr = QString::fromStdString(line);
            QStringList pieces = qstr.split(",");

            if(pieces.size()==11){
                for(int i=0;i<10;i++){
                    float landmark = pieces.value(i).toFloat();
                    neutralImage.push_back(landmark);
                }
            }
        }

        myFile.close();

        cout<<"Neutral image: ";
        for(int j=0;j<10;j++){
            cout<<neutralImage[j]<<", ";
        }
        cout<<endl;
    }

    else{
        cerr << "Unable to open neutral image file!"<<endl;
    }

}

void Trainer::saveNeutralImage(string fileName){

    cout<<"Saving neutral image..."<<endl;
    ofstream myFile;
    myFile.open (fileName);

    for(int i=0;i<10;i++){
        myFile << neutralImage[i]<<",";
    }

    myFile.close();
}


