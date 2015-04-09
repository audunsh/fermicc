#include "initializer.h"
#include "basis/electrongas.h"
#include "solver/flexmat.h"
#include <time.h>

#define ARMA_64BIT_WORD
#include <armadillo>

using namespace std;
using namespace arma;

initializer::initializer(){}

initializer::initializer(electrongas Bs)
{
    bs = Bs;
    iNp = bs.iNbstates-bs.iNparticles; //number of particle states
    iNh = bs.iNparticles; //number of hole states
    iNp2 = iNp*iNp;
    iNh2 = iNh*iNh;
    iNhp = iNh*iNp;
    iNmax = 2*bs.vKx.max()+2;
    iNmax2 = iNmax*iNmax;
}



vec initializer::V(uvec p, uvec q, uvec r, uvec s){
    //delta function of summation of momentum quantum numbers is assumed to have passed before entering here
    vec ret = zeros(p.size());

    //retrieve relevant quantum numbers (in vectors)
    vec Msa = bs.vMs.elem(p);
    vec Msb = bs.vMs.elem(q);
    vec Msc = bs.vMs.elem(r);
    vec Msd = bs.vMs.elem(s);

    vec Kax = bs.vKx.elem(p);
    vec Kay = bs.vKy.elem(p);
    vec Kaz = bs.vKz.elem(p);

    vec Kbx = bs.vKx.elem(q);
    vec Kby = bs.vKy.elem(q);
    vec Kbz = bs.vKz.elem(q);

    vec Kcx = bs.vKx.elem(r);
    vec Kcy = bs.vKy.elem(r);
    vec Kcz = bs.vKz.elem(r);

    vec Kdx = bs.vKx.elem(s);
    vec Kdy = bs.vKy.elem(s);
    vec Kdz = bs.vKz.elem(s);

    //set up interaction
    vec KDplus = ones(p.size()); //*4*pi/bs.dL3;
    KDplus = KDplus%(Kax+Kbx==Kcx+Kdx)%(Kay+Kby==Kcy+Kdy)%(Kaz+Kbz==Kcz+Kdz);

    vec diff_ca = absdiff2(Kcx, Kcy, Kcz, Kax, Kay, Kaz);
    vec diff_da = absdiff2(Kdx, Kdy, Kdz, Kax, Kay, Kaz);

    vec term1 = zeros(p.size()); //term 1
    term1.elem(conv_to<uvec>::from(find(Msa==Msc && Msb==Msd))) += 4*pi/bs.dL3; //By changing this i get comparable results
    term1.elem(conv_to<uvec>::from(find(Kax==Kcx && Kay==Kcy && Kaz==Kcz))) *= 0;
    term1.elem(find(term1!=0)) /= 4*pi*pi*diff_ca.elem(find(term1!=0))/bs.dL2;

    vec term2 = zeros(p.size()); //term 2
    term2.elem(conv_to<uvec>::from(find(Msa==Msd && Msb==Msc))) += 4*pi/bs.dL3;
    term2.elem(conv_to<uvec>::from(find(Kax==Kdx && Kay==Kdy && Kaz==Kdz))) *= 0;
    term2.elem(find(term2!=0)) /= 4*pi*pi*diff_da.elem(find(term2!=0))/bs.dL2;

    return KDplus%(term1 - term2);

}

vec initializer::V3(uvec p, uvec q, uvec r, uvec s){
    //Inefficient interaction calculation, not really vectorized but returns a vector
    vec vVals = zeros(p.size());
    for(int n = 0; n< p.size(); n++){
        vVals(n) = bs.v2(p(n), q(n), r(n), s(n));
    }
    return vVals;
}

vec initializer::V2(uvec t0, uvec t1){
    //delta function of summation of momentum quantum numbers is assumed to have passed before entering here
    uvec b = conv_to<uvec>::from(floor(t0/iNp)); //convert to unsigned integer indexing vector
    uvec a = conv_to<uvec>::from(t0) - b*iNp;
    uvec d = conv_to<uvec>::from(floor(t1/iNp)); //convert to unsigned integer indexing vector
    uvec c = conv_to<uvec>::from(t1) - d*iNp;

    vec ret = zeros(t0.size());

    vec Msa = bs.vMs.elem(a+iNh);
    vec Msb = bs.vMs.elem(b+iNh);
    vec Msc = bs.vMs.elem(c+iNh);
    vec Msd = bs.vMs.elem(d+iNh);

    vec Kax = bs.vKx.elem(a+iNh);
    vec Kay = bs.vKy.elem(a+iNh);
    vec Kaz = bs.vKz.elem(a+iNh);

    vec Kbx = bs.vKx.elem(b+iNh);
    vec Kby = bs.vKy.elem(b+iNh);
    vec Kbz = bs.vKz.elem(b+iNh);

    vec Kcx = bs.vKx.elem(c+iNh);
    vec Kcy = bs.vKy.elem(c+iNh);
    vec Kcz = bs.vKz.elem(c+iNh);

    vec Kdx = bs.vKx.elem(d+iNh);
    vec Kdy = bs.vKy.elem(d+iNh);
    vec Kdz = bs.vKz.elem(d+iNh);


    vec KDplus = ones(t0.size()); //*4*pi/bs.dL3;
    KDplus = KDplus%(Kax+Kbx==Kcx+Kdx)%(Kay+Kby==Kcy+Kdy)%(Kaz+Kbz==Kcz+Kdz);

    vec diff_ca = absdiff2(Kcx, Kcy, Kcz, Kax, Kay, Kaz);
    vec diff_da = absdiff2(Kdx, Kdy, Kdz, Kax, Kay, Kaz);

    vec term1 = zeros(t0.size()); //term 1
    term1.elem(conv_to<uvec>::from(find(Msa==Msc && Msb==Msd))) += 4*pi/bs.dL3;
    term1.elem(conv_to<uvec>::from(find(Kax==Kcx && Kay==Kcy && Kaz==Kcz))) *= 0;
    term1.elem(find(term1!=0)) /= 4*pi*pi*diff_ca.elem(find(term1!=0))/bs.dL2;

    vec term2 = zeros(t0.size()); //term 2
    term2.elem(conv_to<uvec>::from(find(Msa==Msd && Msb==Msc))) += 4*pi/bs.dL3;
    term2.elem(conv_to<uvec>::from(find(Kax==Kdx && Kay==Kdy && Kaz==Kdz))) *= 0;
    term2.elem(find(term2!=0)) /= 4*pi*pi*diff_da.elem(find(term2!=0))/bs.dL2;

    return KDplus%(term1 - term2);

}

vec initializer::absdiff2(vec kpx, vec kpy, vec kpz, vec kqx,vec kqy, vec kqz){
    //vectorized absdiff2 |kp - kq|^2
    vec ret1 = kpx-kqx;
    vec ret2 = kpy-kqy;
    vec ret3 = kpz-kqz;
    return ret1%ret1 + ret2%ret2 + ret3%ret3;

}

uvec initializer::append(uvec V1, uvec V2){
    int V1size = V1.size();
    V1.resize(V1size+V2.size());
    for(int i= 0; i<V2.size(); i++){
        V1(i+V1size) = V2(i);
    }
    return V1;
}


vec initializer::appendvec(vec V1, vec V2){
    //int V1size = V1.size();
    vec V3;
    V3.set_size(V1.size() + V2.size());
    for(int i= 0; i<V1.size(); i++){
        V3(i) = V1(i);
    }
    for(int i= V1.size(); i<V1.size()+V2.size(); i++){
        V3(i) = V2(i-V1.size());
    }

    return V3;
}

void initializer::sVpppp(){
    //cout << "Hello from initializer!" << endl;
    clock_t t;

    Vpppp.set_size(iNp2, iNp2);
    t = clock();

    vec AB = linspace(0,iNp2-1,iNp2);

    uvec B = conv_to<uvec>::from(floor(AB/iNp)); //convert to unsigned integer indexing vector
    uvec A = conv_to<uvec>::from(AB) - B*iNp;

    //vec KAx = bs.vKx.elem(A+iNh);
    //vec KAy = bs.vKy.elem(A+iNh);
    //vec KAz = bs.vKz.elem(A+iNh);

    //vec KBx = bs.vKx.elem(B+iNh);
    //vec KBy = bs.vKy.elem(B+iNh);
    //vec KBz = bs.vKz.elem(B+iNh);

    vec KABx = bs.vKx.elem(A+iNh)+bs.vKx.elem(B+iNh);
    vec KABy = iNmax*(bs.vKy.elem(A+iNh)+bs.vKy.elem(B+iNh));
    vec KABz = iNmax2*(bs.vKz.elem(A+iNh)+bs.vKz.elem(B+iNh));
    vec KABms = iNmax*iNmax2*(bs.vMs(A+iNh) + bs.vMs(B + iNh));

    vec KAB = KABx+KABy+KABz + KABms;
    vec KAB_unique = unique(KAB);
    //cout << "    Stage 1:" << (double)(clock() - t)/CLOCKS_PER_SEC << endl;
    //t = clock();

    uvec T0;// = conv_to<uvec>::from(zeros(0));
    uvec T1;// = conv_to<uvec>::from(zeros(0));
    T0.set_size(0);
    T1.set_size(0);

    //int iN;
    field<uvec> TT;
    TT.set_size(KAB_unique.size(), 2);
    int iN = 0;
    //cout << "    Stage 2:" << (double)(clock() - t)/CLOCKS_PER_SEC << endl;
    //t = clock();


    for(int i = 0; i < KAB_unique.size(); i++){
        //this is the most time-consuming process in initialization
        //vec T = AB.elem(find(KAB==KAB_unique(i)));
        vec T = conv_to<vec>::from(find(KAB==KAB_unique(i))); //Is it possible to exploit to make this vector should "shrink" ?
        vec O = ones(T.size());
        uvec t0 = conv_to<uvec>::from(kron(T, O));
        uvec t1 = conv_to<uvec>::from(kron(O, T));
        TT(i, 0) = t0;
        TT(i, 1) = t1;
        iN += t0.size();
    }

    //cout << "    Stage 3:" << (double)(clock() - t)/CLOCKS_PER_SEC << endl;
    //t = clock();
    T0.set_size(iN);
    T1.set_size(iN);
    iN = 0;
    //cout << "    Stage 4:" << (double)(clock() - t)/CLOCKS_PER_SEC << endl;
    //t = clock();
    for(int i = 0; i < KAB_unique.size(); i++){
        T0(span(iN, iN+TT(i,0).size()-1)) = TT(i,0);
        T1(span(iN, iN+TT(i,1).size()-1)) = TT(i,1);
        iN += TT(i,0).size();
    }


    bVpppp = conv_to<uvec>::from(floor(T0/iNp)); //convert to unsigned integer indexing vector
    aVpppp = conv_to<uvec>::from(T0) - bVpppp*iNp ;
    dVpppp = conv_to<uvec>::from(floor(T1/iNp)) ; //convert to unsigned integer indexing vector
    cVpppp = conv_to<uvec>::from(T1) - dVpppp*iNp;
    //cout << "    Stage 6:" << (double)(clock() - t)/CLOCKS_PER_SEC << endl;
    //t = clock();

    vValsVpppp = V(aVpppp+iNh,bVpppp+iNh,cVpppp+iNh,dVpppp+iNh); //this works, tested agains bs.v2, 9.8.2015

    double val = 0;
    double val2 = 0;
    int disccount = 0;
    int an,bn,cn,dn;
    for(int n =0; n< vValsVpppp.size(); n++){
        an = aVpppp(n) + iNh;
        bn = bVpppp(n) + iNh;
        cn = cVpppp(n) + iNh;
        dn = dVpppp(n) + iNh;

        val = bs.v2(an,bn,cn,dn);

        if((abs(val - vValsVpppp(n)))>0.00001){
            cout << val << " " << val2 << " " << vValsVpppp(n) << endl;
            disccount += 1;
        }
    }

    cout << "Discrepancies in Vpppp:" << disccount << endl;
    cout << "Size of Vpppp         :" << vValsVpppp.size() << endl;
}

void initializer::sVhhhh(){

    Vhhhh.set_size(iNh2, iNh2);

    vec IJ = linspace(0,iNh2-1,iNh2);

    uvec J = conv_to<uvec>::from(floor(IJ/iNh)); //convert to unsigned integer indexing vector
    uvec I = conv_to<uvec>::from(IJ) - J*iNh;

    vec KIx = bs.vKx.elem(I);
    vec KIy = bs.vKy.elem(I);
    vec KIz = bs.vKz.elem(I);

    vec KJx = bs.vKx.elem(J);
    vec KJy = bs.vKy.elem(J);
    vec KJz = bs.vKz.elem(J);

    vec KIJx =         bs.vKx.elem(I)+bs.vKx.elem(J);
    vec KIJy = iNmax* (bs.vKy.elem(I)+bs.vKy.elem(J));
    vec KIJz = iNmax2*(bs.vKz.elem(I)+bs.vKz.elem(J));

    vec KIJ = KIJx+KIJy+KIJz;
    vec KIJ_unique = unique(KIJ);

    uvec T0;// = conv_to<uvec>::from(zeros(0));
    uvec T1;// = conv_to<uvec>::from(zeros(0));
    //T0.set_size(0);
    //T1.set_size(0);

    field<uvec> TT;
    TT.set_size(KIJ_unique.size(), 2);
    int iN = 0;

    for(int i = 0; i < KIJ_unique.size(); i++){
        vec T = IJ.elem(find(KIJ==KIJ_unique(i)));
        vec O = ones(T.size());
        uvec t0 = conv_to<uvec>::from(kron(T, O));
        uvec t1 = conv_to<uvec>::from(kron(O, T));

        TT(i, 0) = t0;
        TT(i, 1) = t1;
        iN += t0.size();
    }


    T0.set_size(iN);
    T1.set_size(iN);
    iN = 0;
    int j;
    for(int i = 0; i < KIJ_unique.size(); i++){
        //this is the most time-consuming process in initialization
        //T0(span(iN, iN+TT(i,0).size()-1)) = TT(i,0);
        //T1(span(iN, iN+TT(i,1).size()-1)) = TT(i,1);
        //iN += TT(i,0).size();

        for(j = 0; j < TT(i,0).size(); j++){
            T0(iN) = TT(i,0)(j);
            T1(iN) = TT(i,1)(j);
            iN += 1;
        }
    }

    jVhhhh = conv_to<uvec>::from(floor(T0/iNh)); //convert to unsigned integer indexing vector
    iVhhhh = conv_to<uvec>::from(T0) - jVhhhh*iNh ;
    lVhhhh = conv_to<uvec>::from(floor(T1/iNh)) ; //convert to unsigned integer indexing vector
    kVhhhh = conv_to<uvec>::from(T1) - lVhhhh*iNh;

    vValsVhhhh = V(iVhhhh,jVhhhh,kVhhhh,lVhhhh);



    umat locations;
    locations.set_size(T0.size(),2);
    locations.col(0) = T0;
    locations.col(1) = T1;
    Vhhhh = sp_mat(locations.t(), vValsVhhhh, iNh2, iNh2);

    //test for consistency

    //int disccount = 0;
    for(int n= 0; n<vValsVhhhh.size();n++){
        if(vValsVhhhh(n) !=  bs.v2(iVhhhh(n), jVhhhh(n), kVhhhh(n), lVhhhh(n))){
            //disccount += 1;
            cout << iVhhhh(n) << " " << jVhhhh(n)<< " " << kVhhhh(n)<< " " << lVhhhh(n)<< " " << vValsVhhhh(n)<< " " <<bs.v(iVhhhh(n), jVhhhh(n), kVhhhh(n), lVhhhh(n)) << endl;
        }
    }
    //cout << "Number of discrepancies in Vhhhh:" <<  disccount << endl;

    double d1,d2;
    int discs = 0;
    for(int i = 0; i< iNh; i++){
        for(int j = i; j< iNh; j++){
            for(int k = j; k< iNh; k++){
                for(int l = k; l < iNh; l++){
                    d1 = Vhhhh(i + j*iNh, k + l*iNh);
                    d2 = bs.v2(i,j,k,l);
                    if(d1 != d2){
                        //cout << d1 << " " << d2 << endl;
                        discs += 1;

                    }
                }
            }
        }
    }

    cout << "Discrepancy found in Vhhhh:" << discs << endl;

}

void initializer::sVhhpp(){
    Vhhpp.set_size(iNh2, iNp2);

    //indexing rows

    vec IJ = linspace(0,iNh2-1,iNh2);

    uvec J = conv_to<uvec>::from(floor(IJ/iNh)); //convert to unsigned integer indexing vector
    uvec I = conv_to<uvec>::from(IJ) - J*iNh;

    vec KIx = bs.vKx.elem(I);
    vec KIy = bs.vKy.elem(I);
    vec KIz = bs.vKz.elem(I);

    vec KJx = bs.vKx.elem(J);
    vec KJy = bs.vKy.elem(J);
    vec KJz = bs.vKz.elem(J);

    vec KIJx = bs.vKx.elem(I)+bs.vKx.elem(J);
    vec KIJy = iNmax*(bs.vKy.elem(I)+bs.vKy.elem(J));
    vec KIJz = iNmax2*(bs.vKz.elem(I)+bs.vKz.elem(J));

    vec KIJ = KIJx+KIJy+KIJz;
    vec KIJ_unique = unique(KIJ);

    //indexing columns

    vec AB = linspace(0,iNp2-1,iNp2);

    uvec B = conv_to<uvec>::from(floor(AB/iNp)); //convert to unsigned integer indexing vector
    uvec A = conv_to<uvec>::from(AB) - B*iNp;

    vec KAx = bs.vKx.elem(A+iNh);
    vec KAy = bs.vKy.elem(A+iNh);
    vec KAz = bs.vKz.elem(A+iNh);

    vec KBx = bs.vKx.elem(B+iNh);
    vec KBy = bs.vKy.elem(B+iNh);
    vec KBz = bs.vKz.elem(B+iNh);

    vec KABx = bs.vKx.elem(A+iNh)+bs.vKx.elem(B+iNh);
    vec KABy = iNmax*(bs.vKy.elem(A+iNh)+bs.vKy.elem(B+iNh));
    vec KABz = iNmax2*(bs.vKz.elem(A+iNh)+bs.vKz.elem(B+iNh));

    vec KAB = KABx+KABy+KABz;
    vec KAB_unique = unique(KAB);

    //consolidating rows and columns

    vec K_joined = join_cols<mat>(KIJ_unique, KAB_unique);
    //K_joined.set_size(KAB_unique.size() + KIJ_unique.size());
    //K_joined(span(0,KAB_unique.size()-1)) = KAB_unique;
    //K_joined(span(KAB_unique.size(),KIJ_unique.size())) = KIJ_unique;

    vec K_unique = unique(K_joined);




    uvec T0;// = conv_to<uvec>::from(zeros(0));
    uvec T1;// = conv_to<uvec>::from(zeros(0));
    T0.set_size(0);
    T1.set_size(0);

    field<uvec> TT;
    TT.set_size(K_unique.size(), 2);
    int iN = 0;

    for(int i = 0; i < K_unique.size(); i++){
        vec Tij = IJ.elem(find(KIJ==K_unique(i)));
        vec ONh = ones(Tij.size());

        vec Tab = AB.elem(find(KAB==K_unique(i)));
        vec ONp = ones(Tab.size());


        if(Tij.size() != 0 && Tab.size() != 0){
            uvec t0 = conv_to<uvec>::from(kron(Tij, ONp));
            uvec t1 = conv_to<uvec>::from(kron(ONh, Tab));
            //T0 = append(T0,t0);
            //T1 = append(T1,t1);
            TT(i, 0) = t0;
            TT(i, 1) = t1;
            iN += t0.size();

        }
    }

    T0.set_size(iN);
    T1.set_size(iN);
    iN = 0;
    int j;
    for(int i = 0; i < K_unique.size(); i++){
        //this is the most time-consuming process in initialization of Vhhpp
        if(TT(i,0).size() != 0){
            T0(span(iN, iN+TT(i,0).size()-1)) = TT(i,0);
            T1(span(iN, iN+TT(i,1).size()-1)) = TT(i,1);
            iN += TT(i,0).size();}
        //for(j = 0; j < TT(i,0).size(); j++){
        //    T0(iN) = TT(i,0)(j);
        //    T1(iN) = TT(i,1)(j);
        //    iN += 1;
        //}
    }

    jVhhpp = conv_to<uvec>::from(floor(T0/iNh)); //convert to unsigned integer indexing vector
    iVhhpp = conv_to<uvec>::from(T0) - jVhhpp*iNh ;
    bVhhpp = conv_to<uvec>::from(floor(T1/iNp)) ; //convert to unsigned integer indexing vector
    aVhhpp = conv_to<uvec>::from(T1) - bVhhpp*iNp;
    //cout << "Maximum a:" << aVhhpp.max() << endl;

    iVpphh = iVhhpp;
    jVpphh = jVhhpp;
    aVpphh = aVhhpp;
    bVpphh = bVhhpp;

    vValsVhhpp = V(iVhhpp,jVhhpp,aVhhpp+iNh,bVhhpp+iNh);
    vValsVpphh = V(aVhhpp+iNh,bVhhpp+iNh,iVhhpp,jVhhpp); //Symmetric? Do a test

    umat locations;
    locations.set_size(T0.size(),2);
    locations.col(0) = T0;
    locations.col(1) = T1;
    Vhhpp = sp_mat(locations.t(), vValsVhhpp, iNh2, iNp2);

    //locations.col(0) = T1;
    //locations.col(1) = T0;
    //fmVpphh = flexmat(a,b,i,j,iNp,iNh);

    //cout << V(conv_to<uvec>::from(ones(2)*13),conv_to<uvec>::from(ones(2)*12),conv_to<uvec>::from(ones(2)*46),conv_to<uvec>::from(ones(2)*42)) << endl; //this element is not inlcuded in the testing above!!!
    //cout << Vhhpp(181, 1152) << endl;

    /*
    double val = 0;
    double val2 = 0;
    int disccount = 0;
    int an,bn,in,jn;

    for(int n =0; n< vValsVhhpp.size(); n++){
        //cout << aVhhpp(n) + iNh << endl;
        an = aVhhpp(n) + iNh;
        bn = bVhhpp(n) + iNh;
        in = iVhhpp(n);
        jn = jVhhpp(n);
        //cout << an << endl;

        val = bs.v2(an,bn,in,jn);

        if((abs(val - vValsVpphh(n)))>0.00001){
            cout << val << " " << vValsVpphh(n) << endl;
            disccount += 1;
        }
    }
    cout << "Discrepancies in Vpphh:" << disccount << endl;


    val = 0;
    for(int i = 0; i<iNh; i++){
        for(int j = 0; j<iNh; j++){
            for(int a = 0; a<iNp; a++){
                for(int b = 0; b<iNp; b++){
                    val = bs.v2(a+iNh,b+iNh,i,j);
                    //cout << Vhhpp(i + j*iNh, a + b*iNp) << "       " << val << endl;
                    //cout << Vhhpp(i + j*iNh, a + b*iNp) << endl;
                    //cout << i + j*iNh << " " << a + b*iNp << endl;
                    if(abs(Vhhpp(i + j*iNh, a + b*iNp) - val)>0.00000001){
                        //cout << i + j*iNh << " " << a + b*iNp << endl;
                        cout << i << " " << j << " " << a  << " " << b  << endl;
                        cout << Vhhpp(i + j*iNh, a + b*iNp) - val << endl;
                    }
                }
            }
        }
    }
    */


}

void initializer::sVpphh(){}

void initializer::sVhpph(){
    Vhpph.set_size(iNhp, iNhp);

    //indexing rows

    vec IA = linspace(0,iNhp-1,iNhp);

    uvec A = conv_to<uvec>::from(floor(IA/iNh)); //convert to unsigned integer indexing vector
    uvec I = conv_to<uvec>::from(IA) - A*iNh;

    vec KIx = bs.vKx.elem(I);
    vec KIy = bs.vKy.elem(I);
    vec KIz = bs.vKz.elem(I);

    vec KAx = bs.vKx.elem(A+iNh);
    vec KAy = bs.vKy.elem(A+iNh);
    vec KAz = bs.vKz.elem(A+iNh);

    vec KIAx = bs.vKx.elem(I)+bs.vKx.elem(A+iNh);
    vec KIAy = iNmax*(bs.vKy.elem(I)+bs.vKy.elem(A+iNh));
    vec KIAz = iNmax2*(bs.vKz.elem(I)+bs.vKz.elem(A+iNh));

    vec KIA = KIAx+KIAy+KIAz;
    vec KIA_unique = unique(KIA);

    //indexing columns

    vec BJ = linspace(0,iNhp-1,iNhp); //AB

    uvec J = conv_to<uvec>::from(floor(BJ/iNp)); //convert to unsigned integer indexing vector
    uvec B = conv_to<uvec>::from(BJ) - J*iNp;

    vec KJx = bs.vKx.elem(J);
    vec KJy = bs.vKy.elem(J);
    vec KJz = bs.vKz.elem(J);

    vec KBx = bs.vKx.elem(B+iNh);
    vec KBy = bs.vKy.elem(B+iNh);
    vec KBz = bs.vKz.elem(B+iNh);

    vec KBJx = bs.vKx.elem(B+iNh)+bs.vKx.elem(J);
    vec KBJy = iNmax*(bs.vKy.elem(B+iNh)+bs.vKy.elem(J));
    vec KBJz = iNmax2*(bs.vKz.elem(B+iNh)+bs.vKz.elem(J));

    vec KBJ = KBJx+KBJy+KBJz;
    vec KBJ_unique = unique(KBJ);

    //consolidating rows and columns

    vec K_joined = join_cols<mat>(KIA_unique, KBJ_unique);
    //K_joined.set_size(KAB_unique.size() + KIJ_unique.size());
    //K_joined(span(0,KAB_unique.size()-1)) = KAB_unique;
    //K_joined(span(KAB_unique.size(),KIJ_unique.size())) = KIJ_unique;

    vec K_unique = unique(K_joined);



    uvec T0;// = conv_to<uvec>::from(zeros(0));
    uvec T1;// = conv_to<uvec>::from(zeros(0));
    T0.set_size(0);
    T1.set_size(0);


    field<uvec> TT;
    TT.set_size(K_unique.size(), 2);
    int iN = 0;
    //int iN;
    for(int i = 0; i < K_unique.size(); i++){
        vec Tia = IA.elem(find(KIA==K_unique(i)));
        vec ONh = ones(Tia.size());

        vec Tbj = BJ.elem(find(KBJ==K_unique(i)));
        vec ONp = ones(Tbj.size());

        //153 1586
        //uvec t0 = conv_to<uvec>::from(kron(Tij, ONp));
        //uvec t1 = conv_to<uvec>::from(kron(ONh, Tab));
        //T0 = append(T0,t0);
        //T1 = append(T1,t1);

        if(Tia.size() != 0 && Tbj.size() != 0){
            uvec t0 = conv_to<uvec>::from(kron(Tia, ONp));
            uvec t1 = conv_to<uvec>::from(kron(ONh, Tbj));
            //T0 = append(T0,t0);
            //T1 = append(T1,t1);
            TT(i, 0) = t0;
            TT(i, 1) = t1;
            iN += t0.size();
        }
    }

    T0.set_size(iN);
    T1.set_size(iN);
    iN = 0;
    for(int i = 0; i < K_unique.size(); i++){
        //this is the most time-consuming process in initialization
        if(TT(i,0).size() != 0){
            T0(span(iN, iN+TT(i,0).size()-1)) = TT(i,0);
            T1(span(iN, iN+TT(i,1).size()-1)) = TT(i,1);
            iN += TT(i,0).size();}
    }


    aVhpph = conv_to<uvec>::from(floor(T0/iNh)); //convert to unsigned integer indexing vector
    iVhpph = conv_to<uvec>::from(T0) - aVhpph*iNh ;
    jVhpph = conv_to<uvec>::from(floor(T1/iNp)) ; //convert to unsigned integer indexing vector
    bVhpph = conv_to<uvec>::from(T1) - jVhpph*iNp;

    vValsVhpph = V(iVhpph,aVhpph+iNh,bVhpph+iNh,jVhpph);
    umat locations;
    locations.set_size(T0.size(),2);
    locations.col(0) = T0;
    locations.col(1) = T1;
    Vhpph = sp_mat(locations.t(), vValsVhpph, iNhp, iNhp);

    //cout << V(conv_to<uvec>::from(ones(2)*13),conv_to<uvec>::from(ones(2)*12),conv_to<uvec>::from(ones(2)*46),conv_to<uvec>::from(ones(2)*42)) << endl; //this element is not inlcuded in the testing above!!!
    //cout << Vhhpp(181, 1152) << endl;



    double val = 0;
    for(int i = 0; i<iNh; i++){
        for(int j = 0; j<iNh; j++){
            for(int a = 0; a<iNp; a++){
                for(int b = 0; b<iNp; b++){
                    val = bs.v2(i,a+iNh,b+iNh,j);
                    //cout << Vhhpp(i + j*iNh, a + b*iNp) << "       " << val << endl;
                    //cout << Vhhpp(i + j*iNh, a + b*iNp) << endl;
                    //cout << i + j*iNh << " " << a + b*iNp << endl;
                    if(abs(Vhpph(i + a*iNh, b + j*iNp) - val)>0.000001){
                        //cout << i + j*iNh << " " << a + b*iNp << endl;
                        cout << i << " " << a << " " << b  << " " << j  << endl;
                        cout << Vhpph(i + a*iNh, b + j*iNp) << " " << val << endl;
                    }
                }
            }
        }
    }


}


