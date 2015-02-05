TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    basis/electrongas.cpp \
    solver/ccsolve.cpp \
    solver/amplitude_tensor.cpp

include(deployment.pri)
qtcAddDeployment()

HEADERS += \
    basis/electrongas.h \
    solver/ccsolve.h \
    solver/amplitude_tensor.h

LIBS += -larmadillo -lblas -llapack

