#-------------------------------------------------
#
# Project created by QtCreator 2016-05-19T19:16:34
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = qt_mgr
TEMPLATE = app


SOURCES += main.cpp\
        dialog.cpp \
    FaceAnalyzer.cpp \
    Trainer.cpp \
    ../dlib-18.18/dlib/all/source.cpp \


HEADERS  += dialog.h \
    FaceAnalyzer.h \
    Trainer.h

FORMS    += dialog.ui

INCLUDEPATH += /usr/local/include/opencv
LIBS += -L/usr/local/lib -lopencv_core -lopencv_highgui

INCLUDEPATH += /home/michal/dlib-18.18
LIBS += -pthread
CONFIG += link_pkgconfig
PKGCONFIG += x11

LIBS += `pkg-config opencv --libs`
