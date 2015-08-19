TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt
#CONFIG += c++11

#INCLUDEPATH = ~/usr/include/armadillo_bits
#INCLUDEPATH += /usit/abel/u1/audunsh/usr/include

#INCLUDEPATH +=  ~/usr/include
#INCLUDEPATH += ~/usr/include/armadillo

#LIBS += -L ~/usr/include/armadillo_bits/ -larmadillo
#LIBS += -L/usit/abel/u1/audunsh/armadillo-5.200.3/ -larmadillo -lmkl_core -liomp5 -lpthread -lmkl_sequential


#For abel compilation
#LIBS += -L/cluster/software/VERSIONS/intel-2015.3/compiler/lib/intel64/ -liomp5 -lpthread
#LIBS += -L/cluster/software/VERSIONS/intel-2015.3/mkl/lib/intel64/ -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64 -lmkl_rt
#LIBS += -L/usit/abel/u1/audunsh/armadillo-5.000.0/ -larmadillo
#LIBS += -L/cluster/software/VERSIONS/gcc-5.1.0/lib64/ -lstdc++
#LIBS += -L/usit/abel/u1/audunsh/usr/lib64/
#LIBS += -L/usit/abel/u1/audunsh/usr/include
#LIBS += -L~/usr/include/armadillo
#LIBS += -L/usit/abel/u1/audunsh/usr/local/lib64/ -larmadillo -mkl
 
release {
    DEFINES += ARMA_NO_DEBUG
    #QMAKE_CXXFLAGS_RELEASE -= -O2
    QMAKE_CXXFLAGS_RELEASE += -O3
    #QMAKE_CXXFLAGS += -DMKL_LP64
}


#if using openmp
QMAKE_CXXFLAGS+= -fopenmp
QMAKE_LFLAGS +=  -fopenmp
QMAKE_CXXFLAGS += -std=c++11
#QMAKE_CXXFLAGS += -std=c++0x


#QMAKE_CXX = gcc
#QMAKE_LINK = gcc
#QMAKE_CXX_RELEASE = gcc

#CONFIG += c++11

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
    solver/ccd_mp.cpp \
    solver/blockmap.cpp \
    solver/amplitude.cpp \
    solver/bccd.cpp \
    solver/sccdt_mp.cpp

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
    solver/ccd_mp.h \
    solver/blockmap.h \
    solver/amplitude.h \
    solver/bccd.h \
    solver/sccdt_mp.h

LIBS += -larmadillo -lblas -llapack
#LIBS += -larmadillo -lopenblas

