TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    basis/electrongas.cpp \
    solver/ccsolve.cpp \
    solver/ccd.cpp \
    solver/initializer.cpp \
    solver/flexmat.cpp \
    solver/unpack_sp_mat.cpp

include(deployment.pri)
qtcAddDeployment()

HEADERS += \
    basis/electrongas.h \
    solver/ccsolve.h \
    solver/ccd.h \
    solver/initializer.h \
    solver/flexmat.h \
    solver/unpack_sp_mat.h

LIBS += -larmadillo -lblas -llapack

