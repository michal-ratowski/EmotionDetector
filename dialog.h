#ifndef DIALOG_H
#define DIALOG_H

#include <QDialog>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

using namespace cv;

namespace Ui {
class Dialog;
}

class Dialog : public QDialog
{
    Q_OBJECT

public:
    explicit Dialog(QWidget *parent = 0);
    ~Dialog();

public slots:
    void processFrame();
    bool pathIsCorrect(string fileName);
    void appendLine(string line, int num);
    void getFPS();

private slots:
    void trainSVM(CvSVM &svm);
    void trainKNN(CvKNearest &knn);
    void trainTree(CvDTree &dtree);
    void trainMLP(CvANN_MLP &mlp);
    void trainBayes(CvNormalBayesClassifier &bayes);

    void on_btnCamera_clicked();
    void on_loadPicturesButton_clicked();
    void on_saveDataButton_clicked();
    void on_loadDataButton_clicked();
    void on_browseDataButton_clicked();
    void on_browsePicturesButton_clicked();
    void on_trainButton_clicked();
    void on_loadRadioButton_clicked();
    void on_analyseRadioButton_clicked();

    void on_scaleMinMax_clicked();

    void on_browseButton2_clicked();

    void on_evaluateButton_clicked();

    void on_scaleButton_clicked();

    void on_loadPicturesButton2_clicked();

    void on_loadEvaluationRadioButton_clicked();

    void on_evaluationAnalyseRadioButton_clicked();

    void on_saveEvaluateDataButton_clicked();

    void on_browseLoadEvaluationData_clicked();

    void on_loadEvaluationDataButton_clicked();

    void on_saveResultsButton_clicked();

    void on_pictureBrowseButton_clicked();

    void on_analysePictureButton_clicked();

    void on_tabWidget_currentChanged(int index);

    void on_resetToDefaults_clicked();

private:

    VideoCapture capture;
    Mat cameraImage;
    Mat cameraImageSmall;

    QImage qtImage;
    std::vector<int>response;
    std::vector<float>classifierAccuracy;
    QTimer* timer;

    void checkBoxes();

    Ui::Dialog *ui;

};

#endif // DIALOG_H
