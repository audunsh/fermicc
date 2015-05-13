TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

release {
    DEFINES += ARMA_NO_DEBUG
    #QMAKE_CXXFLAGS_RELEASE -= -O2
    #QMAKE_CXXFLAGS_RELEASE += -O3
}

#if using openmp
QMAKE_CXXFLAGS+= -fopenmp
QMAKE_LFLAGS +=  -fopenmp


SOURCES += main.cpp \
    basis/electrongas.cpp \
    solver/ccsolve.cpp \
    solver/ccd.cpp \
    solver/initializer.cpp \
    solver/flexmat.cpp \
    solver/unpack_sp_mat.cpp \
    solver/blockmat.cpp \
    solver/flexmat6.cpp \
    solver/ccdt.cpp \
    solver/ccd_pt.cpp \
    solver/ccdt_mp.cpp \
    solver/ccd_mp.cpp

include(deployment.pri)
qtcAddDeployment()

HEADERS += \
    basis/electrongas.h \
    solver/ccsolve.h \
    solver/ccd.h \
    solver/initializer.h \
    solver/flexmat.h \
    solver/unpack_sp_mat.h \
    solver/blockmat.h \
    solver/flexmat6.h \
    solver/ccdt.h \
    solver/ccd_pt.h \
    solver/ccdt_mp.h \
    solver/ccd_mp.h

LIBS += -larmadillo -lblas -llapack

