#include "dialog.h"
#include "ui_dialog.h"
#include <QtCore>
#include <QFileDialog>
#include <QMessageBox>

#include "Trainer.h"

using namespace std;
using namespace cv;
using namespace dlib;

bool camFirstOpened = false;
bool showLandmarks=false;
bool showLandmarksNumbers=false;

string emotions[6] = {"NEUTRAL","SADNESS","HAPPINESS","ANGER","SURPRISE","DISGUST"};

FaceAnalyzer faceAnalyzer;
frontal_face_detector detector;
shape_predictor sp;
Trainer trainer;
int K;

CvSVM svm;
CvANN_MLP mlp;

CvKNearest knn;
CvNormalBayesClassifier bayes;
CvDTree dtree;

time_t startTime, endTime;
time_t startTime2, endTime2;
double fps;
int counter = 0;
int countSkip=0;
double sec;

Dialog::Dialog(QWidget *parent): QDialog(parent), ui(new Ui::Dialog){
    ui->setupUi(this);

    QPalette* palette = new QPalette();
    palette->setColor(QPalette::WindowText,Qt::white);
    ui->fpsLabel->setPalette(*palette);

    timer = new QTimer(this);
    connect(timer,SIGNAL(timeout()),this,SLOT(processFrame()));

    detector = get_frontal_face_detector();
    deserialize("/home/michal/dlib-18.18/shape_predictor_68_face_landmarks.dat") >> sp;

    trainer.start(sp,detector);

}

Dialog::~Dialog()
{
    delete ui;
}

// METHODS //////////////////////////////////////////

void Dialog::processFrame(){
    countSkip++;

    if(capture.isOpened()){
        capture.read(cameraImage);
        if(cameraImage.empty()) return;

        getFPS();
        checkBoxes();

        if ( countSkip % ui->frameSkip->value() == 0 ){

            float inputData[10];

            std::vector<float> landmarks = faceAnalyzer.extractLandmarks(cameraImage, sp, detector, showLandmarks, showLandmarksNumbers);

            if(landmarks.size()>0){
                for(int i=0;i<10;i++){
                    inputData[i] = (landmarks[i]-trainer.neutralImage[i]);

                    float smallest = trainer.smallestScaleVector[i];
                    float largest = trainer.largestScaleVector[i];

                    if(trainer.minmax){
                        inputData[i] = (inputData[i]-trainer.smallestScaleVector[i])/(trainer.largestScaleVector[i]-trainer.smallestScaleVector[i]);
                    }
                    else {
                        inputData[i]*=trainer.scaleFactor;
                    }
                }

                Mat inputMat = Mat(1, 10, CV_32FC1, &inputData);

                float response;

                ui->txtLog->clear();

                response = svm.predict(inputMat);
                ui->txtLog->appendPlainText(QString::fromStdString("SVM response: "+to_string(response)+"("+emotions[static_cast<int>(response)]+")"));

                response = knn.find_nearest(inputMat, K);
                ui->txtLog->appendPlainText(QString::fromStdString("KNN response: "+to_string(response)+"("+emotions[static_cast<int>(response)]+")"));

                response = bayes.predict(inputMat);
                ui->txtLog->appendPlainText(QString::fromStdString("Bayes response: "+to_string(response)+"("+emotions[static_cast<int>(response)]+")"));

                cv::Mat mlpResponse(1, 1, CV_32FC1);
                mlp.predict(inputMat, mlpResponse);

                if(mlpResponse.at<float>(0,0)>=5.5){
                    response = 5;
                }
                else{
                    response = mlpResponse.at<float>(0,0);
                }

                ui->txtLog->appendPlainText(QString::fromStdString("MLP response: "+to_string(response)+"("+emotions[static_cast<int>(response+0.5)]+")"));

                CvDTreeNode* prediction = dtree.predict(inputMat);
                ui->txtLog->appendPlainText(QString::fromStdString("Tree response: "+to_string(prediction->value)+"("+emotions[static_cast<int>((prediction->value)+0.5)]+")"));

            }

            else{
                ui->txtLog->clear();
                ui->txtLog->appendPlainText("Face not detected!");
            }
        }

        cvtColor(cameraImage,cameraImage,CV_BGR2RGB);
        QImage qImgOriginal((uchar*)cameraImage.data,cameraImage.cols,cameraImage.rows,cameraImage.step,QImage::Format_RGB888);

        ui->camLabel->setPixmap(QPixmap::fromImage(qImgOriginal));
   }
}

void Dialog::checkBoxes(){
    if(ui->landmarksCheckBox->isChecked()){showLandmarks=true;}
    else{showLandmarks=false;}
    if(ui->numbersCheckBox->isChecked()){showLandmarksNumbers=true;}
    else{showLandmarksNumbers=false;}
}

// SLOTS //////////////////////////////////////////


void Dialog::on_btnCamera_clicked()
{
    if(timer->isActive()){
        if(capture.isOpened()){
            capture.release();
        }
        timer->stop();
        ui->btnCamera->setText("START");
        ui->txtLog->appendPlainText("Capture paused");
        ui->dataTab->setEnabled(true);
        ui->trainingTab->setEnabled(true);
    }
    else{
        ui->dataTab->setEnabled(false);
        ui->trainingTab->setEnabled(false);
        capture.open(0);
        if(!capture.isOpened()){
            ui->txtLog->appendPlainText("Error: capture not opened!");
            return;
        }
        if(!camFirstOpened){
            camFirstOpened=true;
            ui->txtLog->appendPlainText("Capture started!");
        }
        else{
            ui->txtLog->appendPlainText("Capture resumed");
        }
        timer->start(20);
        ui->btnCamera->setText("STOP");

        time(&startTime);
        counter=0;
    }
}

void Dialog::on_loadPicturesButton_clicked()
{
    string fileName = ui->pathText->toPlainText().toUtf8().constData();

    if(!pathIsCorrect(fileName)){
        ui->picturesLog->appendPlainText("Invalid path!");
        return;
    }

    trainer.trainingImagePath = fileName;

    int picturesFound=0;
    int facesFound=0;
    trainer.prepareData(picturesFound,facesFound,this, true);

    ui->picturesLog->appendPlainText(QString::fromStdString("Found "+to_string(picturesFound)+" pictures"));
    ui->picturesLog->appendPlainText(QString::fromStdString("Detected "+to_string(facesFound)+" faces"));
}


bool Dialog::pathIsCorrect(string fileName){
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
        string specificFolderPath = fileName + "/" + folderName;
        if(!QDir(QString::fromStdString(specificFolderPath)).exists()){
            return false;
        }
    }
    return true;
}

void Dialog::on_saveDataButton_clicked()
{
    string fileName = ui->pathSavedFile->toPlainText().toUtf8().constData();
    trainer.saveData(fileName);
    QMessageBox::information(this,tr("OK"),"Data saved!");
}

void Dialog::appendLine(string line, int i)
{
    if(i==1){
        ui->picturesLog->appendPlainText(QString::fromStdString(line));
    }
    else if(i==2){
        ui->picturesLog2->appendPlainText(QString::fromStdString(line));
    }
    else if(i==3){
        ui->evaluateLog->appendPlainText(QString::fromStdString(line));
    }
    qApp->processEvents();
}

void Dialog::getFPS()
{
    time(&endTime);
    ++counter;
    sec = difftime (endTime, startTime);
    fps = counter / sec;
    stringstream stream;
    stream << fixed << setprecision(0) << fps;
    string s = stream.str();
    string fpsString = s + " FPS";

    if(ui->FPSCheckBox->isChecked()){
        ui->fpsLabel->setText(QString::fromStdString(fpsString));
    }
    else{
        ui->fpsLabel->clear();
    }

}

void Dialog::on_loadDataButton_clicked()
{
    string fileName = ui->pathLoadFile->toPlainText().toUtf8().constData();
    trainer.loadData(fileName,this);
}

void Dialog::on_browseDataButton_clicked()
{
    QString pathString = QFileDialog::getOpenFileName(this,tr("Select data file"));
    ui->pathLoadFile->setText(pathString);
}

void Dialog::on_browsePicturesButton_clicked()
{
    QString pathString = QFileDialog::getExistingDirectory(this,tr("Select pictures directory"));
    ui->pathText->setText(pathString);
}

void Dialog::on_trainButton_clicked()
{
    ui->txtLog->clear();
    ui->analysePictureLog->clear();

    trainSVM(svm);
    trainKNN(knn);
    trainMLP(mlp);
    trainer.trainBayes(bayes);
    trainTree(dtree);

    ui->btnCamera->setEnabled(true);
    ui->analysePictureButton->setEnabled(true);

}

void Dialog::on_loadRadioButton_clicked()
{
    ui->analyseDataFrame->setEnabled(false);
    ui->loadDataFrame->setEnabled(true);
}

void Dialog::on_analyseRadioButton_clicked()
{
    ui->analyseDataFrame->setEnabled(true);
    ui->loadDataFrame->setEnabled(false);
}

void Dialog::trainSVM(CvSVM &svm)
{
    svm.clear();

    int svm_type;
    string svmTypeString = ui->SVMtype->currentText().toUtf8().constData();
    if(svmTypeString=="C_SVC"){
        svm_type=CvSVM::C_SVC;
    }
    else if(svmTypeString=="NU_SVC"){
        svm_type=CvSVM::NU_SVC;
    }
    else if(svmTypeString=="EPS_SVR"){
        svm_type=CvSVM::EPS_SVR;
    }
    else if(svmTypeString=="NU_SVR"){
        svm_type=CvSVM::NU_SVR;
    }

    int kernel_type;
    string svmKernelString = ui->SVMkernel->currentText().toUtf8().constData();
    if(svmKernelString=="LINEAR"){
        kernel_type=CvSVM::LINEAR;
    }
    else if(svmTypeString=="POLY"){
        kernel_type=CvSVM::POLY;
    }
    else if(svmTypeString=="RBF"){kernel_type=CvSVM::RBF;}
    else if(svmTypeString=="SIGMOID"){kernel_type=CvSVM::SIGMOID;}

    double degree = ui->SVMdegree->text().toDouble();
    double gamma = ui->SVMgamma->text().toDouble();
    double coef0 = ui->SVMcoef0->text().toDouble();
    double C = ui->SVMc->text().toDouble();
    double nu = ui->SVMnu->text().toDouble();
    double p = ui->SVMp->text().toDouble();
    int max_iter = ui->SVMiter->text().toInt();
    double epsilon = ui->SVMeps->text().toDouble();

    trainer.trainSVM(svm, svm_type, kernel_type, degree, gamma, coef0, C, nu, p, max_iter, epsilon);
}

void Dialog::trainKNN(CvKNearest &knn)
{
    knn.clear();
    K = ui->k->text().toInt();
    trainer.trainKNN(knn, K);
}

void Dialog::trainMLP(CvANN_MLP &mlp)
{
    mlp.clear();

    int train_method;
    string methodString = ui->MLPmethod->currentText().toUtf8().constData();
    if(methodString=="RPROP"){train_method=CvANN_MLP_TrainParams::RPROP;}
    else if(methodString=="BACKPROP"){train_method=CvANN_MLP_TrainParams::BACKPROP;}

    double dw_scale = ui->MLPdwscale->text().toDouble();
    double moment_scale = ui->MLPmomentscale->text().toDouble();
    double dw0 = ui->MLPdw0->text().toDouble();
    double dw_plus = ui->MLPdwplus->text().toDouble();
    double dw_minus = ui->MLPdwminus->text().toDouble();
    int layer1 = ui->MLPlayer1->text().toInt();
    int layer2 = ui->MLPlayer2->text().toInt();
    int max_iter = ui->MLPiter->text().toInt();
    double epsilon = ui->MLPeps->text().toDouble();
    trainer.trainMLP(mlp, train_method, dw_scale, moment_scale, dw0, dw_plus, dw_minus, layer1, layer2, max_iter, epsilon);

}

void Dialog::trainBayes(CvNormalBayesClassifier &bayes)
{
    bayes.clear();
    trainer.trainBayes(bayes);
}

void Dialog::trainTree(CvDTree &dtree)
{
    dtree.clear();

    int min_sample_count = ui->minsamples->text().toInt();
    double regression_accuracy = ui->accuracy->text().toDouble();
    bool use_surrogates = ui->surrogate->isChecked();
    int max_categories = ui->maxcategories->text().toInt();
    double cv_folds = ui->cvfolds->text().toDouble();
    bool use_1se_rule = ui->serule->isChecked();
    bool truncate_pruned_tree = ui->truncate->isChecked();

    trainer.trainDecisionTree(dtree, min_sample_count, regression_accuracy, use_surrogates, max_categories, cv_folds, use_1se_rule, truncate_pruned_tree);

}


void Dialog::on_scaleMinMax_clicked()
{
    if(ui->scaleFactor->isEnabled()){
        ui->scaleFactor->setEnabled(true);
    }
    else{
        ui->scaleFactor->setEnabled(false);
    }
}

void Dialog::on_browseButton2_clicked()
{
    QString pathString = QFileDialog::getExistingDirectory(this,tr("Select pictures directory"));
    ui->evaluatePath->setText(pathString);
}

void Dialog::on_evaluateButton_clicked()
{  
    float accuracy[5] = {0};

    for(int i=0;i<trainer.evaluateDataVector.size();i++){

        response.clear();
        float inputData[10];

        for(int j=0;j<10;j++){
            inputData[j] = trainer.evaluateDataVector[i][j];
            if(trainer.minmax){
                inputData[j] = (inputData[j]-trainer.smallestScaleVector[j])/(trainer.largestScaleVector[j]-trainer.smallestScaleVector[j]);
            }
            else {
                inputData[j]*=trainer.scaleFactor;
            }
        }

        Mat inputMat = Mat(1, 10, CV_32FC1, &inputData);

        response.push_back(static_cast<int>(svm.predict(inputMat)));
        response.push_back(static_cast<int>(knn.find_nearest(inputMat, K)));
        response.push_back(static_cast<int>(bayes.predict(inputMat)));

        cv::Mat mlpResponse(1, 1, CV_32FC1);
        mlp.predict(inputMat, mlpResponse);
        float MLPresponse;

        if(mlpResponse.at<float>(0,0)>=5){
            MLPresponse = 5;
        }
        else{
            MLPresponse = static_cast<int>(mlpResponse.at<float>(0,0));
        }

        response.push_back(MLPresponse);

        CvDTreeNode* prediction = dtree.predict(inputMat);
        response.push_back(static_cast<int>(prediction->value));

        for(int j=0;j<5;j++){
            if(response[j]==trainer.evaluateLabelsVector[i]){
                accuracy[j]++;
            }
        }

        string responseString = "Responses: "+to_string(response[0])+", "+to_string(response[1])+", "+to_string(response[2])+", "+to_string(response[3])+", "+to_string(response[4])+", REAL: "+to_string(trainer.evaluateLabelsVector[i]);
        appendLine(responseString,3);

    }

    classifierAccuracy.clear();
    for(int i=0;i<5;i++){
        accuracy[i]/=trainer.evaluateLabelsVector.size();
        classifierAccuracy.push_back(accuracy[i]);
    }

    string responseString2;
    responseString2 = "SVM accuracy: "+to_string(accuracy[0]);
    appendLine(responseString2,3);
    responseString2 = "KNN accuracy: "+to_string(accuracy[1]);
    appendLine(responseString2,3);
    responseString2 = "Bayes accuracy: "+to_string(accuracy[2]);
    appendLine(responseString2,3);
    responseString2 = "MLP accuracy: "+to_string(accuracy[3]);
    appendLine(responseString2,3);
    responseString2 = "Tree accuracy: "+to_string(accuracy[4]);
    appendLine(responseString2,3);

}

void Dialog::on_scaleButton_clicked()
{
    trainer.minmax = ui->scaleMinMax->isChecked();
    trainer.scaleFactor = ui->scaleFactor->text().toFloat();
    trainer.scaleData();
    ui->trainButton->setEnabled(true);
}

void Dialog::on_loadPicturesButton2_clicked()
{
    string fileName = ui->evaluatePath->toPlainText().toUtf8().constData();

    if(!pathIsCorrect(fileName)){
        ui->evaluateLog->appendPlainText("Invalid path!");
        return;
    }

    trainer.analysisImagePath = fileName;

    int picturesFound=0;
    int facesFound=0;
    trainer.prepareData(picturesFound,facesFound,this,false);
    ui->evaluateButton->setEnabled(true);

}

void Dialog::on_loadEvaluationRadioButton_clicked()
{
    ui->evaluateAnalyseFrame->setEnabled(false);
    ui->evaluateLoadFrame->setEnabled(true);
}

void Dialog::on_evaluationAnalyseRadioButton_clicked()
{
    ui->evaluateAnalyseFrame->setEnabled(true);
    ui->evaluateLoadFrame->setEnabled(false);
}

void Dialog::on_saveEvaluateDataButton_clicked()
{
    string fileName = ui->saveEvaluateDataPath->toPlainText().toUtf8().constData();
    trainer.saveEvaluationData(fileName);
    QMessageBox::information(this,tr("OK"),"Data saved!");
}

void Dialog::on_browseLoadEvaluationData_clicked()
{
    QString pathString = QFileDialog::getOpenFileName(this,tr("Select data file"));
    ui->loadEvaluationDataPath->setText(pathString);
}

void Dialog::on_loadEvaluationDataButton_clicked()
{
    string fileName = ui->loadEvaluationDataPath->toPlainText().toUtf8().constData();
    trainer.loadEvaluationData(fileName, this);
    ui->evaluateButton->setEnabled(true);

}

void Dialog::on_saveResultsButton_clicked()
{
    string fileName = ui->saveResultsPath->toPlainText().toUtf8().constData();
    ofstream myFile;
    myFile.open (fileName, std::ios_base::app);
    myFile<<classifierAccuracy[0]<<","<<classifierAccuracy[1]<<","<<classifierAccuracy[2]<<","<<classifierAccuracy[3]<<","<<classifierAccuracy[4]<<endl;
    myFile.close();
}

void Dialog::on_pictureBrowseButton_clicked()
{
    QString pathString = QFileDialog::getOpenFileName(this,tr("Select a picture"));
    ui->analysePicturePath->setText(pathString);
}

void Dialog::on_analysePictureButton_clicked()
{
    string fileName = ui->analysePicturePath->toPlainText().toUtf8().constData();
    Mat cameraImage;
    VideoCapture captureImage;

    if (fileName.find("http") != std::string::npos) {
        captureImage = VideoCapture(fileName);
        if(captureImage.isOpened()){
            cout<<"Loading image..."<<endl;
            captureImage.read(cameraImage);
        }
        else{
            cout<<"Invalid url!"<<endl;
            return;
        }
    }
    else{
        cameraImage = imread(fileName,1);
    }

    float inputData[10];
    std::vector<float> landmarks = faceAnalyzer.extractLandmarks(cameraImage, sp, detector, true,false);

    if(landmarks.size()>0){
        for(int i=0;i<10;i++){
            inputData[i] = (landmarks[i]-trainer.neutralImage[i]);

            if(trainer.minmax){
                inputData[i] = (inputData[i]-trainer.smallestScaleVector[i])/(trainer.largestScaleVector[i]-trainer.smallestScaleVector[i]);
            }
            else {
                inputData[i]*=trainer.scaleFactor;
            }

        }

        Mat inputMat = Mat(1, 10, CV_32FC1, &inputData);

        float response;

        ui->analysePictureLog->clear();

        response = svm.predict(inputMat);
        ui->analysePictureLog->appendPlainText(QString::fromStdString("SVM response: "+to_string(response)+"("+emotions[static_cast<int>(response)]+")"));

        response = knn.find_nearest(inputMat, K);
        ui->analysePictureLog->appendPlainText(QString::fromStdString("KNN response: "+to_string(response)+"("+emotions[static_cast<int>(response)]+")"));

        response = bayes.predict(inputMat);
        ui->analysePictureLog->appendPlainText(QString::fromStdString("Bayes response: "+to_string(response)+"("+emotions[static_cast<int>(response)]+")"));

        cv::Mat mlpResponse(1, 1, CV_32FC1);
        mlp.predict(inputMat, mlpResponse);

        if(mlpResponse.at<float>(0,0)>=5.5){
            response = 5;
        }
        else{
            response = mlpResponse.at<float>(0,0);
        }

        ui->analysePictureLog->appendPlainText(QString::fromStdString("MLP response: "+to_string(response)+"("+emotions[static_cast<int>(response+0.5)]+")"));

        CvDTreeNode* prediction = dtree.predict(inputMat);
        ui->analysePictureLog->appendPlainText(QString::fromStdString("Tree response: "+to_string(prediction->value)+"("+emotions[static_cast<int>((prediction->value)+0.5)]+")"));

    }

    else{
        ui->analysePictureLog->clear();
        ui->analysePictureLog->appendPlainText("Face not detected!");
    }


    cvtColor(cameraImage,cameraImage,CV_BGR2RGB);

    QImage qImgOriginal((uchar*)cameraImage.data,cameraImage.cols,cameraImage.rows,cameraImage.step,QImage::Format_RGB888);

    ui->camLabel_2->setPixmap(QPixmap::fromImage(qImgOriginal));

}

void Dialog::on_tabWidget_currentChanged(int index)
{
    if(index==1){
        if(trainer.trainingDataVector.size()>0 && !(ui->scaleButton->isEnabled())){
            ui->scaleButton->setEnabled(true);
        }
    }
}

void Dialog::on_resetToDefaults_clicked()
{
    ui->SVMtype->setCurrentIndex(0);
    ui->SVMkernel->setCurrentIndex(0);
    ui->SVMdegree->setText("0");
    ui->SVMgamma->setText("9");
    ui->SVMcoef0->setText("0");
    ui->SVMc->setText("8");
    ui->SVMnu->setText("0.1");
    ui->SVMp->setText("0.1");
    ui->SVMiter->setText("1000");
    ui->SVMeps->setText("0.000001");

    ui->MLPmethod->setCurrentIndex(0);
    ui->MLPdwscale->setText("0.1");
    ui->MLPmomentscale->setText("0.1");
    ui->MLPdw0->setText("0.1");
    ui->MLPdwplus->setText("1.2");
    ui->MLPdwminus->setText("0.5");
    ui->MLPlayer1->setText("3");
    ui->MLPlayer2->setText("14");
    ui->MLPiter->setText("1000");
    ui->MLPeps->setText("0.000001");

    ui->minsamples->setText("2");
    ui->accuracy->setText("0.01");
    ui->cvfolds->setText("10");
    ui->maxcategories->setText("10");
    ui->surrogate->setChecked(true);
    ui->serule->setChecked(true);
    ui->truncate->setChecked(true);

    ui->k->setText("10");

    ui->scaleMinMax->setChecked(false);
    ui->scaleFactor->setText("28");

}
