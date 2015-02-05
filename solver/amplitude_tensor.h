#ifndef AMPLITUDE_TENSOR_H
#define AMPLITUDE_TENSOR_H

#include "boost/unordered_map.hpp"

class amplitude_tensor
{
public:
    amplitude_tensor();
    amplitude_tensor(int ndim);
    double at(int p, int q, int r, int s, int t, int u);
    int iDim;
};

#endif // AMPLITUDE_TENSOR_H
