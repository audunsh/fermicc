//#define ARMA_64BIT_WORD
#include <armadillo>

#include "solver/ccsolve.h"
#include "basis/electrongas.h"


using namespace std;
using namespace arma;


electrongas::electrongas()
{
}

void electrongas::generate_state_list(int Ne, double rs, int Np){
    iN = Ne;

    //Volum = nokkuperte*4.d0*pi*r_s**3/3.d0
    dr_s = rs;


    //prefactor1 = 4*pi/(L*L*L); //These are not necessarily correct

    //prefactor1 = 3/(14.0*r_s*r_s*r_s);
    //prefactor2 = .5;

    double Nmax = iN + 1; //sqrt(iN) + 1;
    int energy = 0;
    int nStates = 0;
    iNparticles = Np;

    //Counting the number of states needed up to energy level N
    for(int x = -Nmax; x<Nmax+1; x++){
        for(int y = -Nmax; y<Nmax+1; y++){
            for(int z = -Nmax; z<Nmax+1; z++){
                energy = x*x + y*y + z*z;
                if(energy < iN + 1){
                    //cout << "Energy:" << energy << " State: " << x << " " << y << " " <<  z << endl;
                    nStates += 2; //Due to spin degeneracy
                }
            }
        }
    }
    //cout << "Up to energy level " << N << " there will be " << nStates << " states." << endl;


    dPrefactor1 = 3.0/(iNparticles*dr_s*dr_s*dr_s); //this is correct

    //dL3 = nStates*4.0*pi*dr_s*dr_s*dr_s/3.0;
    dL3 = iNparticles*4.0*pi*dr_s*dr_s*dr_s/3.0;
    dL2 = pow(dL3, 2.0/3.0);
    dPrefactor1 = 2*pi/dL2;
    mu =0;
    //Setting up all all states
    mat k_combinations; // = zeros(nStates, 5);
    k_combinations.set_size(nStates, 5);
    mSortedEnergy.set_size(nStates, 4);
    int index_count = 0;
    double e2;

    for(int x = -Nmax; x<Nmax+1; x++){
        for(int y = -Nmax; y<Nmax+1; y++){
            for(int z = -Nmax; z<Nmax+1; z++){
                energy = (x*x + y*y + z*z);
                e2 = 2*energy*(pi*pi)/(dL2);
                if(energy < iN + 1){
                    k_combinations(index_count, 0) = e2; //energy*prefactor2*(53.63609*pi*pi/L3);
                    k_combinations(index_count, 1) = x;
                    k_combinations(index_count, 2) = y;
                    k_combinations(index_count, 3) = z;
                    k_combinations(index_count, 4) = 1; //Changed this 21.3.15
                    index_count += 1;

                    k_combinations(index_count, 0) = e2; //energy*prefactor2*(53.63609*pi*pi/L3);
                    k_combinations(index_count, 1) = x;
                    k_combinations(index_count, 2) = y;
                    k_combinations(index_count, 3) = z;
                    k_combinations(index_count, 4) = -1;
                    index_count += 1;
                }
            }
        }
    }
    //k_combinations.print();

    vec temp_vec = k_combinations.col(0);
    vEnergy.set_size(index_count);
    iNbstates = index_count;
    uvec sorted_vector = sort_index(temp_vec);
    vKx.set_size(index_count);
    vKy.set_size(index_count);
    vKz.set_size(index_count);
    vMs.set_size(index_count);


    for(int i = 0; i< index_count; i++){
        for(int j = 0; j< 4; j++){
            mSortedEnergy(i,j) = k_combinations(sorted_vector(i), j+1);
        }
        vKx(i) = k_combinations(sorted_vector(i), 1);
        vKy(i) = k_combinations(sorted_vector(i), 2);
        vKz(i) = k_combinations(sorted_vector(i), 3);
        vMs(i) = k_combinations(sorted_vector(i), 4);

        vEnergy(i) = k_combinations(sorted_vector(i), 0);
    }
    //sorted_energy.print();
    //Energy.print();
    cout << "#Electrongas: number of states:" << nStates << endl;
}

double electrongas::absdiff2(rowvec A, rowvec B){
    double D = 0;
    for (int i =0; i < 3; i++){
        D += (A(i) - B(i))*(A(i) - B(i));
    }
    return D;
}

int electrongas::kd(int A, int B){
    return 1*(A==B);
}

int electrongas::kd_vec(rowvec A, rowvec B){
    int D = 1;
    for(int i = 0; i < A.n_elem; i++){
        D*=(A(i)==B(i));
    }
    return D;
}

double electrongas::h(int P, int Q){
    return vEnergy(P)*(P==Q);
}

double electrongas::v2(int p, int q, int r, int s){
    //alternate implementation, written for pedagogical reasons
    double kpx = vKx(p);
    double kpy = vKy(p);
    double kpz = vKz(p);

    double kqx = vKx(q);
    double kqy = vKy(q);
    double kqz = vKz(q);

    double krx = vKx(r);
    double kry = vKy(r);
    double krz = vKz(r);

    double ksx = vKx(s);
    double ksy = vKy(s);
    double ksz = vKz(s);

    double val = 0;
    double term1 = 0.0;
    double term2 = 0.0;

    if(kpx+kqx==krx+ksx && kpy+kqy==kry+ksy && kpz+kqz==krz+ksz){

        val = 4*pi/dL3;

        double mps = vMs(p);
        double mqs = vMs(q);
        double mrs = vMs(r);
        double mss = vMs(s);

        double kdiff1, kdiff2, kdiff3;

        if(mps==mrs && mqs==mss){

            if(kpx != krx || kpy != kry || kpz != krz){

                kdiff1 = krx-kpx;
                kdiff2 = kry-kpy;
                kdiff3 = krz-kpz;
                term1 = dL2/((kdiff1*kdiff1 + kdiff2*kdiff2 + kdiff3*kdiff3)*4*pi*pi);
            }
        }

        if(mps==mss && mqs==mrs){

            if(kpx != ksx || kpy != ksy || kpz != ksz){

                kdiff1 = ksx-kpx;
                kdiff2 = ksy-kpy;
                kdiff3 = ksz-kpz;
                term2 = dL2/((kdiff1*kdiff1 + kdiff2*kdiff2 + kdiff3*kdiff3)*4*pi*pi);
            }
        }
    }
    return val*(term1 - term2);
}

double electrongas::v(int P, int Q, int R, int S){
    //Two body interaction
    rowvec KP = mSortedEnergy.row(P);
    rowvec KQ = mSortedEnergy.row(Q);
    rowvec KR = mSortedEnergy.row(R);
    rowvec KS = mSortedEnergy.row(S);
    //Two electron interaction
    double value;
    double term1= 0;
    double term2= 0;
    double kd1, kd2;
    int spinP = KP(3);
    int spinQ = KQ(3);
    int spinR = KR(3);
    int spinS = KS(3);

    rowvec kp, kq, kr, ks;
    kp << KP(0) << KP(1) << KP(2);
    kq << KQ(0) << KQ(1) << KQ(2);
    kr << KR(0) << KR(1) << KR(2);
    ks << KS(0) << KS(1) << KS(2);

    value = kd_vec((kp+kq), (kr+ks));

    double spin1, spin2;


    if(value ==0 ){
        return 0;
    }
    else{
        value = 0;
        //term1 = kd(spinP, spinQ)*kd(spinR,spinS);
        spin1 = kd(spinP, spinR)*kd(spinQ,spinS);
        kd1 = kd_vec(kp, kr);
        if(kd1!= 1.0){
            //term1 = term1 / (mu*mu + 4*pi*pi*absdiff2(kr, kp));
            //cout << "Direct" << endl;
            term1 = (mu*mu + 4*pi*pi*absdiff2(kr, kp))/dL2;
            value += spin1/term1;
        }

        term2 = kd(spinP, spinS)*kd(spinQ,spinR);
        kd2 = kd_vec(kp, ks);
        if(kd2!= 1.0){
            term2 = (mu*mu + 4*pi*pi*absdiff2(ks, kp))/dL2;
            value -= 1.0/term2;
        }

        return 4.0*pi*value/dL3; //prefactor used to be 2*pi*...
    }
}

double electrongas::f(int P, int Q){
    //Fock operator matrix elements
    double val = dPrefactor2;
    rowvec KP = mSortedEnergy.row(P);
    rowvec KQ = mSortedEnergy.row(Q);
    vec kp;
    kp.zeros(3);
    kp(0) = KP(0);
    kp(1) = KP(1);
    kp(2) = KP(2);
    val *= dot(kp, kp);
    val *= kd_vec(KP,KQ);
    double val2 = 0;
    for(int i = 0; i < iNbstates; i++){
        if((i != P) && (i != Q)){
            val2 += v(P, i, Q, i);
        }
    }
    val += dPrefactor1*val2;
    return val;
}

double electrongas::eref(int nParticles){
    //returns the reference energy in the current basis
    double sp_energy =0.0;
    double in_energy = 0.0;
    double reference_energy = 0.0;
    for(int i =0; i <nParticles; i++){
        sp_energy += h(i,i);
        reference_energy += h(i,i);
        for(int j=0; j<nParticles; j++){
            if(i!=j){

                //if(v(i,j,i,j) != 0){
                //    cout << i << " " << j << " " << v(i,j,i,j) << endl;
                //}
                reference_energy += .5*v2(i,j, i,j);
                in_energy += .5*v2(i,j,i,j);

            }
        }
    }
    //cout << "Single particle energy:" << sp_energy << endl;
    //cout << "Interaction energy:" << in_energy << endl;
    return reference_energy;
}

double electrongas::analytic_energy(int nParticles){
    //Returns the analytic energy
    return .5*(2.21/(dr_s * dr_s) - 0.916/dr_s)*nParticles;
}

