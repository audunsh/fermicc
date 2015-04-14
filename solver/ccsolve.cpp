#include <iomanip>
//#define ARMA_64BIT_WORD
#include <armadillo>
#include "basis/electrongas.h"
#include "solver/ccsolve.h"

using namespace std;
using namespace arma;

ccsolve::ccsolve()
{
}

ccsolve::ccsolve(electrongas f)
{
    eBasis = f;
    iNs = eBasis.iNbstates;
    iNs2 = iNs*iNs;
    initialize_amplitudes();
}

double ccsolve::v(int a, int b, int i, int j){
    return eBasis.v2(a,b,i,j);
}

double ccsolve::h(int a, int i){
    return eBasis.h(a,i);
}

void ccsolve::initialize_amplitudes(){
    //initializing amplitude tensors
    t1a.set_size(iNs,iNs);
    t1a_new.set_size(iNs,iNs);
    t2a.set_size(iNs2,iNs2);
    t2a_new.set_size(iNs2,iNs2);
    //t3a.set_size(iNs2*iNs, iNs2*iNs);
}

void ccsolve::initialize_t3amplitudes(){
    t3a.set_size(iNs2*iNs, iNs2*iNs);
}

double ccsolve::CCSD_SG(int iNparticles){
    iNp = iNparticles;
    int a,b,i,j;
    //cout << "Initializing intermediates ...";
    initialize_SGIntermediates();
    //cout << "OK." << endl;
    //update_SGIntermediates();

    //test routines
    //cout << "Correlation energy prior to amplitude initialization:" << CCSD_SG_energy() << endl;

    //Initial guess for amplitudes
    //SpMat<double> t2a;
    //SpMat<double> t3a;
    //cout << "Initializing amplitudes ...";
    for(i = 0; i<iNp; i++){
        for(a = iNp; a<iNs; a++){
            t1a(a,i) = h(a,i)/(h(i,i) - h(a,a));
            //cout << h(a,a) << endl;
            //cout << h(i,i) << endl;
            for(j = 0; j<iNp; j++){
                for(b = iNp; b<iNs; b++){
                    t2a(a+b*iNs,i+j*iNs) = v(a,b,i,j)/(h(i,i) + h(j,j) - h(a,a) - h(b,b));
                }
            }
        }
    }
    //cout << "OK." << endl;
    cout << "Correlation energy after amplitude initialization (2.order PT:):" << CCSD_SG_energy() << endl;
    //cout << "Beginning iterations." << endl;
    for(int t = 0; t<3; t++){
        //cout << "Updating intermediates ... " << endl;
        update_SGIntermediates_optimized();
        //cout << "Done updating intermediates."<< endl;
        //cout << "Advancing solution one step...";

        CCSD_SG_advance();

        cout << "(ccsolve)Energy:" << CCSD_SG_energy() << endl;
        //cout << "OK" << endl;
    }
    return CCSD_SG_energy();
}

void ccsolve::CCSD_SG_advance(){
    int a,b,i,j;
    for(a=iNp; a<iNs; a++){
        for(i = 0; i<iNp; i++){
            t1a_new(a,i) = CCSD_SG_dt1(a,i)/(h(i,i) - h(a,a));
            //cout << a << " " << i << endl;

            for(b=iNp; b<iNs; b++){
                for(j = 0; j<iNp; j++){
                    t2a_new(a+b*iNs,i+j*iNs) = CCSD_SG_dt2(a,b,i,j)/(h(i,i) + h(j,j) - h(a,a) - h(b,b));
                }
            }

        }
    }

    //updating amplitudes
    t2a = t2a_new;
    t1a = t1a_new;
}


double ccsolve::CCSD_SG_energy(){
    //return correlation energy
    double val1, val2, val3, v_ijab;
    int a,b,i,j;
    /*
    val1 = 0.0;
    for(a=iNp; a<iNs; a++){
        for(i = 0; i<iNp; i++){
            val1 += h(a,i)*t1(a,i);
        }
    }

    val2 = 0.0;
    for(a=iNp; a<iNs; a++){
        for(i = 0; i<iNp; i++){
            for(b = iNp; b<iNs; b++){
                for(j = 0; j<iNp; j++){
                    val2 += v(i,j,a,b)*t1(a,i)*t1(b,j);
                }
            }
        }
    }
    val1 += .5*val2;
    */
    val1 = 0.0;
    val2 = 0.0;
    val3 = 0.0;
    for(a=iNp; a<iNs; a++){
        for(i = 0; i<iNp; i++){
            //val1 += h(a,i)*t1(a,i);
            for(b = iNp; b<iNs; b++){
                for(j = 0; j<iNp; j++){
                    v_ijab = v(i,j,a,b);
                    val2 += v_ijab*t2(a,b,i,j);
                    //val3 += v_ijab*t1(a,i)*t1(b,j);
                }
            }
        }
    }
    val1 += .25*val2 +.5*val3;
    return val1;
}

double ccsolve::CCSD_SG_dt2(int a, int b, int i, int j){
    //return Stanton-Gauss t2 amplitudes D(a,b,i,j)*t2a(a,b,i,j)
    int c,d,k,l;
    double val1, val2;
    val1 = v(i,j,a,b);
    val2 = 0;
    for(k=0; k<iNp; k++){
        for(l=0; l<iNp; l++){
            val2 += (t2(a,b,k,l) + t1(a,k)*t1(b,l) -t1(a,l)*t1(b,k))*w1a(k,l)(i,j);
        }
    }
    val1 += .5*val2;

    val2 = 0;
    for(k = 0; k<iNp; k++){
        val2 += t1(b,k)*w2a(a,k)(i,j);
        val2 -= t1(a,k)*w2a(b,k)(i,j);
    }
    val1 -= val2;

    val2 = 0.0;
    for(k = 0; k<iNp; k++){
        val2 += t2(a,b,j,k)*f2a(k,i);
        val2 -= t2(a,b,i,k)*f2a(k,j);
    }
    val1 += val2;

    val2 = 0.0;
    for(c = iNp; c<iNs; c++){
        for(d = iNp; d< iNs; d++){
            val2 += v(c,d,a,b)*t2(c,d,i,j);
        }
    }
    val1 += .5*val2;

    val2 = 0.0;
    for(c=iNp;c<iNs; c++){
        val2 += t2(b,c,i,j)*f3a(a,c);
        val2 -= t2(a,c,i,j)*f3a(b,c);
    }
    val1 += val2;

    val2 = 0.0;
    for(c=iNp;c<iNs; c++){
        val2 += v(c,j,a,b)*t1(c,i);
        val2 -= v(c,i,a,b)*t1(c,j);
    }
    val1 += val2;

    val2 = 0.0;
    for(c = iNp; c<iNs; c++){
        for(d = iNp; d< iNs; d++){
            val2 += v(c,d,a,b)*t1(c,i)*t1(d,j);
            val2 -= v(c,d,a,b)*t1(c,j)*t1(d,i);
        }
    }
    val1 += .5*val2;

    val2 = 0.0;
    for(k=0;k<iNp;k++){
        for(c=iNp; c< iNs; c++){
            val2 += t2(b,c,j,k)*w4a(a,k)(i,c);
            val2 -= t2(a,c,j,k)*w4a(b,k)(i,c);
            val2 -= t2(b,c,i,k)*w4a(a,k)(j,c);
            val2 += t2(a,c,i,k)*w4a(b,k)(j,c);
        }
    }
    val1 += val2;

    return val1;
}

double ccsolve::CCSD_SG_dt1(int a, int i){
    //return Stanton-Gauss t1 amplitudes
    int c,d,k,l;
    double val1, val2;
    val1 = h(a,i);

    val2 = 0.0;
    for(k = 0; k<iNp; k++){
        val2 += t1(a,k)*f2a(k,i);
    }
    val1 -= val2;

    val2 = 0.0;
    for(c=iNp; c<iNs; c++){
        if(c!= a){
            val2 += h(a,c)*t1(c,i);
        }
    }
    val1 +=val2;

    val2 = 0.0;
    for(k = 0; k<iNp; k++){
        for(c=iNp; c<iNs; c++){
            val2 += v(c,i,k,a)*t1(c,k);
        }
    }
    val1 += val2;

    val2 = 0.0;
    for(k = 0; k<iNp; k++){
        for(c=iNp; c<iNs; c++){
            for(l = 0; l<iNp; l++){
                val2 += t2(c,a,k,l)*w3a(i,c)(k,l);
            }
        }
    }
    val1 -= .5*val2;

    val2 = 0.0;
    for(k = 0; k<iNp; k++){
        for(c=iNp; c<iNs; c++){
            val2 += t2(a,c,i,k)*f1a(c,k);
        }
    }
    val1 += val2;

    val2 = 0.0;
    for(k = 0; k<iNp; k++){
        for(c=iNp; c<iNs; c++){
            for(d=iNp; d<iNs; d++){
                val2 += v(c,d,k,a)*t2(c,d,k,i);
            }
        }
    }
    val1 += .5*val2;

    val2 = 0.0;
    for(k = 0; k<iNp; k++){
        for(c=iNp; c<iNs; c++){
            for(d=iNp; d<iNs; d++){
                val2 += v(c,d,k,a)*t1(c,k)*t1(d,i);
            }
        }
    }
    val1 += val2;

    return val1;
}

void ccsolve::initialize_SGIntermediates(){
    //Initialize Stanton-Gauss intermediates
    f1a.set_size(iNs,iNs);
    f2a.set_size(iNs,iNs);
    f3a.set_size(iNs,iNs);

    w1a.set_size(iNs,iNs);
    w2a.set_size(iNs,iNs);
    w3a.set_size(iNs,iNs);
    w4a.set_size(iNs,iNs);
    for(int i= 0; i < iNs; i++){
        for(int j= 0; j < iNs; j++){
            w1a(i,j).set_size(iNs,iNs);
            w2a(i,j).set_size(iNs,iNs);
            w3a(i,j).set_size(iNs,iNs);
            w4a(i,j).set_size(iNs,iNs);
        }
    }
}

void ccsolve::update_SGIntermediates(){
    //Update Stanton-Gauss intermediates
    int a,b,c,d,i,j,k,l;
    double val1, val2;

    //update f1a
    for(c = iNp; c<iNs; c++){
        for(k = 0; k<iNp; k++){
            val1 = h(k,c);
            for(d = iNp; d<iNs; d++){
                for(l = 0; l < iNp; l++){
                    val1 += v(c,d,l,k)*t1a(d,l);
                }
            }
            f1a(c,k) = val1;
        }
    }

    //update f2a
    for(k = 0; k<iNp; k++){
        for(i = 0; i<iNp; i++){
            val1 = 0.0;
            if(k!=i){
                val1 += h(k,c); //this will be 0 anyways
            }
            for(c = iNp; c<iNs; c++){
                val1 += f1a(c,k)*t1a(c,i);
                for(l = 0; l < iNp; l++){
                    val1 += v(i,c,k,l)*t1(c,l);
                    val2 = 0.0;
                    for(d = iNp; d<iNs; d++){
                        val2 += v(c,d,k,l)*t2(c,d,i,l);
                    }
                    val1 += .5*val2;
                }
            }
            f2a(k,i) = val1;
        }
    }

    //update f3a
    for(a = iNp; a<iNs; a++){
        for(c = iNp; c<iNs; c++){
            val1 = 0.0;
            if(a!=c){
                val1 += h(a,c);
            }
            for(k = 0; k<iNp; k++){
                val1 += f1a(c,k)*t1(a,k);
                for(d = iNp; d<iNs; d++){
                    val1 += v(c,d,k,a)*t1(d,k);
                    val2 = 0.0;
                    for(l = 0; l<iNp; l++){
                        val2 += v(c,d,k,l)*t2(a,d,k,l);
                    }
                    val1 += .5*val2;
                }
            }
            f3a(a,c) = val1;
        }
    }

    //udpate w1a
    for(k = 0; k<iNp; k++){
        for(l = 0; l<iNp; l++){
            for(i = 0; i<iNp; i++){
                for(j = 0; j<iNp; j++){
                    val1 = v(i,j,k,l);
                    //w1a = v(i,j,k,l);
                    for(c = iNp; c<iNs; c++){
                        val1 += v(c,j,k,l)*t1(c,i) - v(c,i,k,l)*t1(c,j);
                        val2 = 0;
                        for(d = iNp; d<iNs; d++){
                            val2 += v(c,d,k,l)*(t1(c,i)*t1(d,j) - t1(c,j)*t1(d,i) + t2(c,d,i,j));
                        }
                        val1 += .5*val2;
                    }
                    w1a(k,l)(i,j) = val1;
                }
            }
        }
    }

    //update w2a
    for(a = iNp; a<iNs; a++){
        for(k = 0; k < iNp; k++){
            for(i = 0; i<iNp; i++){
                for(j = 0; j<iNp; j++){
                    val1 = v(i,j,a,k);
                    for(c = iNp; c<iNs; c++){
                        val1 += v(i,c,a,k)*t1(c,j) - v(j,c,a,k)*t1(c,i);
                        val2 = 0.0;
                        for(d = iNp; d<iNs; d++){
                            val2 += v(c,d,a,k)*(t2(c,d,i,j) + t1(c,i)*t1(d,j) - t1(c,j)*t1(d,i));

                        }
                        val1 += .5*val2;
                    }
                    w2a(a,k)(i,j) = val1;
                }
            }
        }
    }

    //update w3a
    for(k = 0; k<iNp; k++){
        for(l= 0; l<iNp; l++){
            for(c = iNp; c<iNs; c++){
                for(i = 0; i<iNp; i++){
                    val1 = v(c,i,k,l);
                    for(d = iNp; d<iNs; d++){
                        val1 += v(c,d,k,l)*t1(c,i);
                    }
                    w3a(k,l)(c,i) = val1;
                }
            }
        }
    }


    //update w4a
    for(a = iNp; a< iNs; a++){
        for(k=0; k<iNp; k++){
            for(i = 0; i<iNp; i++){
                for(c = iNp; c<iNs; c++){
                    val1 = v(i,c,a,k);
                    for(d = iNp; d<iNs; d++){
                        val1 += v(d,c,a,k)*t1(d,i);
                    }

                    for(l = 0; l<iNp; l++){
                        val1 += w3a(k,l)(c,i)*t1(a,l);
                    }

                    val2 = 0.0;
                    for(l = 0; l<iNp; l++){
                        for(d = iNp; d<iNs; d++){
                            val2 += v(c,d,k,l)*t2(a,d,i,l);
                        }
                    }
                    val1 += .5*val2;
                    w4a(a,k)(i,c) = val1;
                }
            }
        }
    }
}

void ccsolve::update_SGIntermediates_optimized(){
    //Update Stanton-Gauss intermediates
    int a,b,c,d,i,j,k,l;
    double val1, val2;

    //cout << "Updating f1a..." ;

    //update f1a
    for(c = iNp; c<iNs; c++){
        for(k = 0; k<iNp; k++){
            val1 = h(k,c);
            for(d = iNp; d<iNs; d++){
                for(l = 0; l < iNp; l++){
                    val1 += v(c,d,l,k)*t1a(d,l);
                }
            }
            f1a(c,k) = val1;
        }
    }
    //cout << "OK" << endl;

    //update f2a
    //cout << "Updating f2a..." ;
    for(k = 0; k<iNp; k++){
        for(i = 0; i<iNp; i++){
            val1 = 0.0;
            if(k!=i){
                val1 += h(k,i); //this will be 0 anyways
            }
            for(c = iNp; c<iNs; c++){
                val1 += f1a(c,k)*t1a(c,i);
                for(l = 0; l < iNp; l++){
                    val1 += v(i,c,k,l)*t1(c,l);
                    val2 = 0.0;
                    for(d = iNp; d<iNs; d++){
                        val2 += v(c,d,k,l)*t2(c,d,i,l);
                    }
                    val1 += .5*val2;
                }
            }
            f2a(k,i) = val1;
        }
    }
    //cout << "OK" << endl;

    //update f3a
    //cout << "Updating f3a...";
    for(a = iNp; a<iNs; a++){
        for(c = iNp; c<iNs; c++){
            val1 = 0.0;
            if(a!=c){
                val1 += h(a,c);
            }
            for(k = 0; k<iNp; k++){
                val1 += f1a(c,k)*t1(a,k);
                for(d = iNp; d<iNs; d++){
                    val1 += v(c,d,k,a)*t1(d,k);
                    val2 = 0.0;
                    for(l = 0; l<iNp; l++){
                        val2 += v(c,d,k,l)*t2(a,d,k,l);
                    }
                    val1 += .5*val2;
                }
            }
            f3a(a,c) = val1;
        }
    }
    //cout << "OK" << endl;

    //udpate w1a
    //cout << "Updating w1a...";
    for(k = 0; k<iNp; k++){
        for(l = 0; l<iNp; l++){
            for(i = 0; i<iNp; i++){
                for(j = 0; j<iNp; j++){
                    val1 = v(i,j,k,l);
                    //w1a = v(i,j,k,l);
                    for(c = iNp; c<iNs; c++){
                        val1 += v(c,j,k,l)*t1(c,i) - v(c,i,k,l)*t1(c,j);
                        val2 = 0;
                        for(d = iNp; d<iNs; d++){
                            val2 += v(c,d,k,l)*(t1(c,i)*t1(d,j) - t1(c,j)*t1(d,i) + t2(c,d,i,j));
                        }
                        val1 += .5*val2;
                    }
                    w1a(k,l)(i,j) = val1;
                }
            }
        }
    }
    //cout << "OK" << endl;

    //update w2a
    //cout << "Updating w2a...";
    for(a = iNp; a<iNs; a++){
        for(k = 0; k < iNp; k++){
            for(i = 0; i<iNp; i++){
                for(j = 0; j<iNp; j++){
                    val1 = v(i,j,a,k);
                    for(c = iNp; c<iNs; c++){
                        val1 += v(i,c,a,k)*t1(c,j) - v(j,c,a,k)*t1(c,i);
                        val2 = 0.0;
                        for(d = iNp; d<iNs; d++){
                            val2 += v(c,d,a,k)*(t2(c,d,i,j) + t1(c,i)*t1(d,j) - t1(c,j)*t1(d,i));

                        }
                        val1 += .5*val2;
                    }
                    w2a(a,k)(i,j) = val1;
                }
            }
        }
    }
    //cout << "OK" << endl;

    //update w3a
    //cout << "Updating w3a...";
    for(k = 0; k<iNp; k++){
        for(l= 0; l<iNp; l++){
            for(c = iNp; c<iNs; c++){
                for(i = 0; i<iNp; i++){
                    val1 = v(c,i,k,l);
                    for(d = iNp; d<iNs; d++){
                        val1 += v(c,d,k,l)*t1(c,i);
                    }
                    w3a(k,l)(c,i) = val1;
                }
            }
        }
    }
    //cout << "OK" << endl;


    //update w4a
    //cout << "Updating w4a...";
    for(a = iNp; a< iNs; a++){
        for(k=0; k<iNp; k++){
            for(i = 0; i<iNp; i++){
                for(c = iNp; c<iNs; c++){
                    val1 = v(i,c,a,k);
                    for(d = iNp; d<iNs; d++){
                        val1 += v(d,c,a,k)*t1(d,i);
                    }

                    for(l = 0; l<iNp; l++){
                        val1 += w3a(k,l)(c,i)*t1(a,l);
                    }

                    val2 = 0.0;
                    for(l = 0; l<iNp; l++){
                        for(d = iNp; d<iNs; d++){
                            val2 += v(c,d,k,l)*t2(a,d,i,l);
                        }
                    }
                    val1 += .5*val2;
                    w4a(a,k)(i,c) = val1;
                }
            }
        }
    }
    //cout << "OK" << endl;
}

void ccsolve::update_intermediates(){}

double ccsolve::t1(int a, int i){
    return t1a(a,i);
}

double ccsolve::t2(int a, int b, int i, int j){
    return t2a(a + iNs*b, i + iNs*j);
}

double ccsolve::t3(int a, int b, int c, int i, int j, int k){
    return t3a(a + iNs*b + iNs2*c,i + iNs*j + iNs2*k);
}

void ccsolve::scan_amplitudes(){
    //accessing all amplitudes
    double val = 0.0;
    for(int a = 0; a<iNs;a++){
        for(int i = 0; i<iNs; i++){
            val += t1(a,i);
            for(int b = 0; b< iNs; b++){
                for(int j =0; j < iNs; j++){
                    val += t2(a,b,i,j);
                    /*

                    for(int c= 0; c < iNs; c++){
                        for(int k= 0; k<iNs; k++){
                            val += t3(a,b,c,i,j,k);
                        }
                    }
                    */

                    //cout << a << " " << b << " " <<  i << " " << j << endl;
                }
            }
        }
    }
    cout << "Done summing all amplitudes:" << val << endl;
}
