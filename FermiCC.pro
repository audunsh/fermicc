TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    basis/electrongas.cpp \
    solver/ccsolve.cpp

include(deployment.pri)
qtcAddDeployment()

HEADERS += \
    basis/electrongas.h \
    solver/ccsolve.h

LIBS += -larmadillo -lblas -llapack

