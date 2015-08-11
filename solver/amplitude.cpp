#include "amplitude.h"
#define ARMA_64BIT_WORD
#include <armadillo>
#include "basis/electrongas.h"

using namespace std;
using namespace arma;

amplitude::amplitude(){}

amplitude::amplitude(electrongas bs, int n_configs, uvec size)
{
    nthreads = 1;
    eBs = bs;
    k_step = 2*eBs.vKx.max()+3; //stepsize for identifying unique regions
    iNconfigs = n_configs;
    fmBlocks.set_size(iNconfigs); //block of indices
    fvConfigs.set_size(iNconfigs); //configuration in quantum numbers of each block
    blocklengths.set_size(iNconfigs);  //number of blocks in each configuration
    fmOrdering.set_size(iNconfigs); //the ordering of each configuration
    permutative_ordering.set_size(iNconfigs);
    uiCurrent_block = 0;

    Np = eBs.iNbstates-eBs.iNparticles; //conflicting naming here
    Nh = eBs.iNparticles;

    uvSize = size; //true state configurations (Np, Np, Nh, Nh) or (Np, Np, Np, Nh, Nh, Nh) (or similar)

    memsize = 0; //bookkeeping for allocated memory (presently only for t3 amplitudes)
    uiStatAlloc = 10000000;

}

void amplitude::init(electrongas bs, int n_configs, uvec size){
    nthreads = 1;
    eBs = bs;
    k_step = 2*eBs.vKx.max()+3; //stepsize for identifying unique regions
    iNconfigs = n_configs;
    fmBlocks.set_size(iNconfigs); //block of indices
    fvConfigs.set_size(iNconfigs); //configuration in quantum numbers of each block
    blocklengths.set_size(iNconfigs);  //number of blocks in each configuration
    fmOrdering.set_size(iNconfigs); //the ordering of each configuration
    uiCurrent_block = 0;

    permutative_ordering.set_size(iNconfigs);

    Np = eBs.iNbstates-eBs.iNparticles; //conflicting naming here
    Nh = eBs.iNparticles;

    uvSize = size; //true state configurations (Np, Np, Nh, Nh) or (Np, Np, Np, Nh, Nh, Nh) (or similar)

    ivBconfigs.set_size(0);
    uiStatAlloc = 100000;
    memsize = 0;
}

// ##################################################
// ##                                              ##
// ## Element related functions                    ##
// ##                                              ##
// ##################################################

void amplitude::zeros(){
    vElements *= 0;
} //zero out all elements

void amplitude::tempZeros(){
    vElements_temp *= 0;
} //zero out all elements

void amplitude::insert_zeros(){
    vec z(1);
    z(0) = 0;
    uiLastind = vElements.n_rows;
    vElements = join_cols(vElements, z);
    vEnergies = join_cols(vEnergies, z);
    vEnergies(uiLastind) = 1;
}

void amplitude::scan_uvElements(){
    for(uint i= 0; i<uvElements.n_rows; ++i){
        uvec p = from6(uvElements(i));
        double d = eBs.vHFEnergy(p(3)) + eBs.vHFEnergy(p(4))+eBs.vHFEnergy(p(5))-eBs.vHFEnergy(p(0)+Nh)-eBs.vHFEnergy(p(1)+Nh)-eBs.vHFEnergy(p(2)+Nh);
        if(d != d){
            cout << "Warning2" << endl;
            p.print();
            cout << eBs.vHFEnergy(p(3)) + eBs.vHFEnergy(p(4))+eBs.vHFEnergy(p(5)) << endl;
            cout << -eBs.vHFEnergy(p(0)+Nh)-eBs.vHFEnergy(p(1)+Nh)-eBs.vHFEnergy(p(2)+Nh) << endl;
            cout << endl;
        }

    }
}

void amplitude::init_t3_amplitudes(){
    vElements.set_size(uvElements.n_rows);
    vEnergies.set_size(uvElements.n_rows);
    vElements_temp.set_size(uvElements.n_rows);
    //memsize += uvElements.n_rows*2;
    //uvElements.print(); //could we maybe retrieve these "on the fly" ? (would mean to locate blocks "on the fly")
    #pragma omp parallel for num_threads(nthreads)
    for(uint i= 0; i<uvElements.n_rows; ++i){
        uvec p = from6(uvElements(i));

        vElements(i) = 0;
        vElements_temp(i) = 0;

        vEnergies(i) = eBs.vHFEnergy(p(3)) + eBs.vHFEnergy(p(4))+eBs.vHFEnergy(p(5))-eBs.vHFEnergy(p(0)+Nh)-eBs.vHFEnergy(p(1)+Nh)-eBs.vHFEnergy(p(2)+Nh);

    }
    //uvElements.set_size(0);
    //uvElemtemp1.set_size(0);
    uvElements.reset();
    uvElemtemp1.reset();
    uvNsort1.reset();
    uvBsort1.reset();

} //initialize as t3 amplitude


void amplitude::init_amplitudes(){
    vElements.set_size(uvElements.n_rows);
    vEnergies.set_size(uvElements.n_rows);
    //uvElements.print(); //could we maybe retrieve these "on the fly" ? (would mean to locate blocks "on the fly")
    //#pragma omp parallel for num_threads(nthreads)
    for(uint i= 0; i<uvElements.n_rows; ++i){
        uvec p = from(uvElements(i));
        //cout << p(0) <<  " " << p(1) << " " << p(2) << " " << p(3) << " "<< endl;
        vElements(i) = eBs.v2(p(0)+Nh,p(1)+Nh,p(2),p(3));
        //double v = eBs.vEnergy(p(2))+ eBs.vEnergy(p(3));
        //vEnergies(i) = eBs.vEnergy(p(2)) + eBs.vEnergy(p(3))-eBs.vEnergy(p(0)+Nh)-eBs.vEnergy(p(1)+Nh);

        vEnergies(i) = eBs.F(p(2)) + eBs.F(p(3))-eBs.F(p(0)+Nh)-eBs.F(p(1)+Nh);
    }
} //initialize as amplitude


void amplitude::enroll_block(umat umBlock, uint tempElementsSize, uvec tempElements, uvec tempBlockmap1,uvec tempBlockmap2,uvec tempBlockmap3){

    // ####################################################################
    // ## Consolidate block  with existing configuration                 ##
    // ####################################################################

    umat n = sort_index(tempElements);
    Col<u64> flatElements = tempElements.elem(n);
    uvec flatBlockmap1 = tempBlockmap1.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,
    uvec flatBlockmap2 = tempBlockmap2.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,
    uvec flatBlockmap3 = tempBlockmap3.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,

    uint tempN = 0;
    uint trueN = 0;
    uint tempL = flatElements.n_rows;
    uint trueL = uvElements.n_rows;

    //uvec uvNsort = sort_index(uvElements);
    //uvec uvElemtemp = uvElements.elem(uvNsort);
    //uvec uvBsort = sort_index(uvNsort); //backsort

    uvec remains(flatElements.n_rows);
    uint remN = 0;



    fspDims(flatBlockmap1(0), 0) = umBlock.n_rows; //nx
    fspDims(flatBlockmap1(0), 1) = umBlock.n_cols; //ny


    umat sparseblock(3, tempL);
    uint count = 0;
    //for(tempN = 0; tempN < tempL; tempN++){
    while(tempN < tempL){
        if(trueN<trueL){
            if(uvElemtemp1(trueN) == flatElements(tempN)){
                //multiple occurances, correct elem index in block
                //fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = uvNsort1(trueN);
                //block(flatBlockmap2(tempN),flatBlockmap3(tempN)) = uvNsort1(trueN);
                //block2(flatBlockmap2(tempN),flatBlockmap3(tempN)) = uvNsort1(trueN);
                sparseblock(0, count) = uvNsort1(trueN); //element
                sparseblock(1, count) = flatBlockmap2(tempN); //x
                sparseblock(2, count) = flatBlockmap3(tempN); //y
                count += 1;


                trueN += 1;
                tempN += 1;
                debug_enroll += 1;

            }
            else{
                if(uvElemtemp1(trueN) < flatElements(tempN)){
                    //advance trueN until multiple occurance
                    //what about this one?
                    trueN += 1;

                }
                else{
                    //flat < evElem, append elem to remains
                    remains(remN) = flatElements(tempN);
                    //fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueL + remN;
                    remN += 1;
                    tempN += 1;

                }
            }
        }
        else{
            //block outside domain
            remains(remN) = flatElements(tempN);
            //fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueL + remN;
            remN += 1;
            tempN += 1;
        }
    }

    //cout << "remains:" << remN << endl;


    umat b(3,count+1);
    b(0,0) =0;
    b(1,0) = 0;
    b(2,0) = 0;
    for(uint i = 0; i < count; ++i){
        b(0,i+1) = sparseblock(0,i);
        b(1,i+1) = sparseblock(1,i);
        b(2,i+1) = sparseblock(2,i);
    }

    fspBlocks(flatBlockmap1(0)) = b;

    /*
    if(count != 0){
        fspBlocks(flatBlockmap1(0)) = sparseblock.cols(span(0,count-1));
    }
    else{
        umat b(3,1);
        b*=0;
        fspBlocks(flatBlockmap1(0)) = b;
    }*/
    memsize += count*3;
}


void amplitude::consolidate_blocks(uint uiN, uint tempElementsSize, field<Col<u64> > tempElements, field<uvec> tempBlockmap1,field<uvec> tempBlockmap2,field<uvec> tempBlockmap3){

    //Flatten fields into 1d uvecs

    uvec flatElements(tempElementsSize);
    uvec flatBlockmap1(tempElementsSize);
    uvec flatBlockmap2(tempElementsSize);
    uvec flatBlockmap3(tempElementsSize);

    ivec flatBconfig(tempElementsSize);

    uint counter = 0;
    for(uint ni = 0; ni<uiN; ++ni){
        for(uint nj = 0; nj < tempElements(ni).n_rows; ++nj){
            flatElements(counter) = tempElements(ni)(nj);
            flatBlockmap1(counter) = tempBlockmap1(ni)(nj);
            flatBlockmap2(counter) = tempBlockmap2(ni)(nj);
            flatBlockmap3(counter) = tempBlockmap3(ni)(nj);

            //flatBconfig(counter) = bconfigs(ni)(nj);

            counter += 1;
        }
        //tempElements(ni).set_size(0);
        //tempBlockmap1(ni).set_size(0);
        //tempBlockmap2(ni).set_size(0);
        //tempBlockmap3(ni).set_size(0);

        //tempElements(ni).clear();
        //tempBlockmap1(ni).clear();
        //tempBlockmap2(ni).clear();
        //tempBlockmap3(ni).clear();

    }
    // ####################################################################
    // ## Consolidate blocks with existing configurations                ##
    // ####################################################################

    //sort elements so their indices appear in increasing order
    Mat<u64> n = sort_index(flatElements);
    flatElements = flatElements.elem(n);
    flatBlockmap1 = flatBlockmap1.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,
    flatBlockmap2 = flatBlockmap2.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,
    flatBlockmap3 = flatBlockmap3.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,

    uint tempN = 0;
    uint trueN = 0;
    uint tempL = flatElements.n_rows;
    uint trueL = uvElements.n_rows;

    uvec uvNsort = sort_index(uvElements);
    Col<u64> uvElemtemp = uvElements.elem(uvNsort);
    uvec uvBsort = sort_index(uvNsort); //backsort

    uvec remains(flatElements.n_rows);
    uint remN = 0;


    //for(tempN = 0; tempN < tempL; tempN++){
    while(tempN < tempL){
        if(trueN<trueL){
            if(uvElemtemp(trueN) == flatElements(tempN)){
                //multiple occurances, correct elem index in block
                fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = uvNsort(trueN);
                trueN += 1;
                tempN += 1;

            }
            else{
                if(uvElemtemp(trueN) < flatElements(tempN)){
                    //advance trueN until multiple occurance
                    //what about this one?
                    trueN += 1;

                }
                else{
                    //flat < evElem, append elem to remains
                    remains(remN) = flatElements(tempN);
                    fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueL + remN;
                    remN += 1;
                    tempN += 1;

                }
            }
        }
        else{
            //block outside domain
            remains(remN) = flatElements(tempN);
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueL + remN;
            remN += 1;
            tempN += 1;
        }
    }

    //join remains to uvElements
    if(remN!=0){
        uvElements = join_cols<umat>(uvElements, remains(span(0,remN-1)));
    }


}

void amplitude::init_interaction(ivec shift){
    //imat L(2,3);
    //imat R(2,3);

    ivec l0 = {0,1,shift(0)};
    ivec l1 = {1,1,shift(1)};
    ivec r0 = {2,1,shift(2)};
    ivec r1 = {3,1,shift(3)};

    imat L = join_rows(l0,l1);
    imat R = join_rows(r0,r1);
    //L.print();
    map_regions(L.t(), R.t());


    vElements.set_size(uvElements.n_rows);
    #pragma omp parallel for num_threads(nthreads)
    for(uint i= 0; i<uvElements.n_rows; ++i){
        uvec p = from(uvElements(i));
        //cout << p(0) <<  " " << p(1) << " " << p(2) << " " << p(3) << " "<< endl;
        vElements(i) = eBs.v2(p(0),p(1),p(2)+Nh,p(3)+Nh);
    }
}

void amplitude::divide_energy(){
    #pragma omp parallel for num_threads(nthreads)
    for(uint i= 0; i<vElements.n_rows; ++i){
        vElements(i) /= vEnergies(i);
    }
} //divide all elements by corresponding energy (for amplitudes)

void amplitude::print_block_maximum(){
    //debugging function
    for(uint i = 0; i < blocklengths(0); ++i){
        getblock(0, i).print();
        cout << endl;
    }
}

// ##################################################
// ##                                              ##
// ## Index related functions                      ##
// ##                                              ##
// ##################################################

ivec amplitude::intersect1d(ivec A, ivec B){
    // ###################################################
    // ## Returns the intersection of two unique arrays ##
    // ###################################################
    ivec ret(A.n_rows);
    uint counter = 0;
    for(uint i = 0; i < A.n_rows; ++i){
        int a = A(i);

        for(uint j = 0; j < B.n_rows; ++j){
            if(B(j) == a){
                ret(counter) = a;
                counter += 1;
                break;
            }
        }

    }
    ivec ret2(counter);
    for(uint i = 0; i < counter; ++i){
        ret2(i) = ret(i);
    }
    return ret2;
}

field<uvec> amplitude::unpack(uvec vStream, imat imOrder){
    // ###############################################
    // ## unpack a disorganized sequence of indices ##
    // ###############################################
    int iMsize = imOrder.n_rows+1;
    ivec M(iMsize);
    M(iMsize-1) = 1;
    int mn = 1;
    for(int i = 0; i < iMsize-1; ++i){
        mn *= uvSize(imOrder(i,0));
        M(iMsize - i-2) = mn;
    }
    field<uvec> indices(imOrder.n_rows);
    for(uint i = 0; i < imOrder.n_rows; ++i){
        uvec P = vStream;
        for(uint e = 0; e<i; ++e){
            P -= indices(imOrder.n_rows-e-1)*M(e+1);
        }

        uvec elem = floor(P/M(i+1));
        indices(imOrder.n_rows-i-1) =  elem;
    }
    return indices;
}

uvec amplitude::unpack_uvec(uint vStream, imat imOrder){
    // ###############################################
    // ## unpack a disorganized sequence of indices ##
    // ###############################################
    uint iMsize = imOrder.n_rows+1;
    ivec M(iMsize);
    M(iMsize-1) = 1;
    uint mn = 1;
    for(int i = 0; i < iMsize-1; ++i){
        mn *= uvSize(imOrder(i,0));
        M(iMsize - i-2) = mn;
    }
    uvec indices(imOrder.n_rows);
    for(uint i = 0; i < imOrder.n_rows; ++i){
        uint P = vStream;
        for(uint e = 0; e<i; ++e){
            P -= indices(imOrder.n_rows - e - 1)*M(e+1);
        }
        indices(imOrder.n_rows - i - 1) = floor(P/M(i+1));
    }
    return indices;
}



u64 amplitude::to(uint p, uint q, uint r, uint s){
    return p + q*uvSize(0) + r*uvSize(0)*uvSize(1) + s * uvSize(0)*uvSize(1)*uvSize(2);
}  //compressed index

Col<u64> amplitude::from(uint i){
    uvec ret(4);
    ret(3) = floor(i/(uvSize(0)*uvSize(1)*uvSize(2)));
    ret(2) = floor((i-ret(3)*uvSize(2)*uvSize(1)*uvSize(0))/(uvSize(0)*uvSize(1)));
    ret(1) = floor((i-ret(3)*uvSize(2)*uvSize(1)*uvSize(0) - ret(2)*uvSize(1)*uvSize(0))/uvSize(0));
    ret(0) = i-ret(3)*uvSize(2)*uvSize(1)*uvSize(0) - ret(2)*uvSize(1)*uvSize(0) - ret(1)*uvSize(0);
    return ret;
} //expanded index



u64 amplitude::to6(u64 p, u64 q, u64 r, u64 s, u64 t, u64 u){
    return p + q*n1 + r*n2 + s*n3 + t*n4 + u*n5;
}  //compressed index, t3 amplitude

void amplitude::make_t3(){
    // ###########################################################
    // ## Make the tensor a t3 amplitude                        ##
    // ###########################################################
    uvSize = {Np, Np, Np, Nh, Nh, Nh};
    n5 = uvSize(0)*uvSize(1)*uvSize(2)*uvSize(3)*uvSize(4); //quantities needed for quick access
    n4 = uvSize(0)*uvSize(1)*uvSize(2)*uvSize(3);
    n3 = uvSize(0)*uvSize(1)*uvSize(2);
    n2 = uvSize(0)*uvSize(1);
    n1 = uvSize(0);
}


Col<u64> amplitude::from6(u64 i){
    Col<u64> ret(6);
    ret(5) = floor(i/n5);
    ret(4) = floor((i-ret(5)*n5)/n4);
    ret(3) = floor((i-ret(5)*n5 - ret(4)*n4)/n3);
    ret(2) = floor((i-ret(5)*n5 - ret(4)*n4 - ret(3)*n3)/n2);
    ret(1) = floor((i-ret(5)*n5 - ret(4)*n4 - ret(3)*n3 - ret(2)*n2)/n1);
    ret(0) = i - ret(5)*n5 - ret(4)*n4 - ret(3)*n3 - ret(2)*n2 - ret(1)*n1;
    return ret;
} //expanded index, t3 amplitude

// ##################################################
// ##                                              ##
// ## Specialized amplitudes                       ##
// ##                                              ##
// ##################################################

void amplitude::map_t3_236_145(ivec Kk_unique){
    field<ivec> bck = pph();
    field<ivec> aij = phh();

    //Kk_unique.print();
    ivec k_unique = intersect1d(unique(bck(3)), unique(aij(3)));

    ivec K_unique = intersect1d(k_unique, Kk_unique);

    //field<uvec> tempRows = partition_hhp(bck, K_unique);
    //field<uvec> tempCols = partition_hpp(aij, K_unique);
    //cout << "Number of blocks:" << K_unique.n_rows << endl;
    field<uvec> tempRows = partition_pph(bck, K_unique);
    field<uvec> tempCols = partition_phh(aij, K_unique);


    //Kk_unique.print();
    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    fmBlocks(uiCurrent_block).set_size(uiN);

    field<Col<u64> > tempElements(uiN);
    field<uvec> tempBlockmap1(uiN);
    field<uvec> tempBlockmap2(uiN);
    field<uvec> tempBlockmap3(uiN);

    uint tempElementsSize = 0;
    //uvec a,b,c,i,j,k;
    //uint systemsize = 0;
    uint Np2 = Np*Np;
    uint Nph = Np*Nh;

    field<ivec> bconfigs(uiN); //normally aligned configurations for each block
    uint bconfig_len = 0;
    #pragma omp parallel for num_threads(nthreads)
    for(uint n = 0; n <K_unique.n_rows; ++n){
        uvec dim(6);
        uvec row = tempRows(n);
        uvec col = tempCols(n);
        //systemsize += row.n_rows*col.n_rows;


        int Nx = row.n_rows;
        int Ny = col.n_rows;
        //memsize += Nx*Ny;
        //cout << Nx << " " << Ny << " " << " " << K_unique(n) << endl;
        Mat<u64> block(Nx,Ny);
        uvec pqrs(6);
        uvec tElements(Nx*Ny);
        uvec tBlockmap1(Nx*Ny);
        uvec tBlockmap2(Nx*Ny);
        uvec tBlockmap3(Nx*Ny);

        uvec k = floor(row/Np2); //k
        uvec c = floor((row  - k*Np2)/Np);
        uvec b = row - k*Np2 - c*Np;

        uvec j = floor(col/Nph); //k
        uvec i = floor((col - j*Nph)/Np);
        uvec a = col - j*Nph - i*Np;

        bconfigs(n).set_size(Nx*Ny); //collect unique config "unaligned"
        bconfig_len += Ny;

        u64 index;
        tempElementsSize += Nx*Ny;
        for(int nx = 0; nx < Nx; nx++){
            for(int ny = 0; ny < Ny; ny++){

                index = to6(a(ny), b(nx), c(nx), i(ny), j(ny), k(nx));

                bconfigs(n)(nx*Ny + ny) = eBs.unique_int(i(ny)) + eBs.unique_int(j(ny)) +eBs.unique_int(k(nx));


                tElements(nx*Ny + ny) = index;
                tBlockmap1(nx*Ny + ny) = n;
                tBlockmap2(nx*Ny + ny) = nx;
                tBlockmap3(nx*Ny + ny) = ny;
                block(nx, ny) = index;
            }
        }
        fmBlocks(uiCurrent_block)(n) = block;
        tempElements(n) = tElements;
        tempBlockmap1(n) = tBlockmap1; //block that element belongs to
        tempBlockmap2(n) = tBlockmap2; //row of element
        tempBlockmap3(n) = tBlockmap3; //column of element


    }

    consolidate_blocks(uiN, tempElementsSize, tempElements, tempBlockmap1, tempBlockmap2, tempBlockmap3);

    ivec flatBconfig(tempElementsSize);

    uint counter = 0;
    for(uint ni = 0; ni<uiN; ++ni){
        for(uint nj = 0; nj < tempElements(ni).n_rows; ++nj){


            flatBconfig(counter) = bconfigs(ni)(nj);

            counter += 1;
        }



    }

    ivBconfigs = unique(join_cols(ivBconfigs, flatBconfig)); //consolidate unique configs "unaligned"
    ivBconfigs = sort(ivBconfigs);

    /*
    // ####################################################################
    // ## Flatten tempElements and tempBlockmap                          ##
    // ####################################################################
    uvec flatElements(tempElementsSize);
    uvec flatBlockmap1(tempElementsSize);
    uvec flatBlockmap2(tempElementsSize);
    uvec flatBlockmap3(tempElementsSize);

    ivec flatBconfig(tempElementsSize);

    uint counter = 0;
    for(uint ni = 0; ni<uiN; ++ni){
        for(uint nj = 0; nj < tempElements(ni).n_rows; ++nj){
            flatElements(counter) = tempElements(ni)(nj);
            flatBlockmap1(counter) = tempBlockmap1(ni)(nj);
            flatBlockmap2(counter) = tempBlockmap2(ni)(nj);
            flatBlockmap3(counter) = tempBlockmap3(ni)(nj);

            flatBconfig(counter) = bconfigs(ni)(nj);

            counter += 1;
        }
        tempElements(ni).set_size(0);
        tempBlockmap1(ni).set_size(0);
        tempBlockmap2(ni).set_size(0);
        tempBlockmap3(ni).set_size(0);


    }

    ivBconfigs = unique(join_cols(ivBconfigs, flatBconfig)); //consolidate unique configs "unaligned"
    ivBconfigs = sort(ivBconfigs);

    // ####################################################################
    // ## Consolidate blocks with existing configurations                ##
    // ####################################################################


    umat n = sort_index(flatElements);
    flatElements = flatElements.elem(n);
    flatBlockmap1 = flatBlockmap1.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,
    flatBlockmap2 = flatBlockmap2.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,
    flatBlockmap3 = flatBlockmap3.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,

    uint tempN = 0;
    uint trueN = 0;
    uint tempL = flatElements.n_rows;
    uint trueL = uvElements.n_rows;
    bool all_resolved = false;
    while(trueN < trueL){
        if(uvElements(trueN) == flatElements(tempN)){
            //identical indexes occuring in
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN;
            trueN += 1;
            tempN += 1;
        }
        else{
            if(uvElements(trueN) < flatElements(tempN)){
                trueN += 1;
            }
            else{
                tempN += 1;
                if(tempN>=tempL){
                    all_resolved = true;
                    break;
                }
            }

        }
    }


    //if(all_resolved != true){
    if(tempN<tempL){
        uvec remaining(tempL-tempN);
        uint tN = 0;
        while(tempN<tempL){
            remaining(tN) = flatElements(tempN);
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN + tN;
            tempN += 1;
            tN += 1;
        }

        //cout << tempL << " "<< tempN << " " << remaining.n_elem << " " << uvElements.n_elem << endl;
        uvElements = join_cols<umat>(uvElements, remaining);

    }*/


    //lock and load
    uiCurrent_block += 1;
}

void amplitude::map_t3_623_451(ivec Kk_unique){
    field<ivec> kbc = hpp();
    field<ivec> ija = hhp();

    //Kk_unique.print();
    ivec k_unique = intersect1d(unique(kbc(3)), unique(ija(3)));

    ivec K_unique = intersect1d(k_unique, Kk_unique);

    //K_unique.print();

    field<uvec> tempRows = partition_hpp(kbc, K_unique);
    field<uvec> tempCols = partition_hhp(ija, K_unique);
    //Kk_unique.print();
    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    fmBlocks(uiCurrent_block).set_size(uiN);

    field<Col<u64> > tempElements(uiN);
    field<uvec> tempBlockmap1(uiN);
    field<uvec> tempBlockmap2(uiN);
    field<uvec> tempBlockmap3(uiN);

    uint tempElementsSize = 0;
    //uvec a,b,c,i,j,k;
    //uint systemsize = 0;
    uint Np2 = Np*Np;
    uint Nph = Np*Nh;
    uint Nh2 = Nh*Nh;

    field<ivec> bconfigs(uiN); //normally aligned configurations for each block
    uint bconfig_len = 0;

    #pragma omp parallel for num_threads(nthreads)
    for(uint n = 0; n <K_unique.n_rows; ++n){
        uvec dim(6);
        uvec row = tempRows(n);
        uvec col = tempCols(n);
        //row.print();
        //cout << endl;
        //systemsize += row.n_rows*col.n_rows;


        int Nx = row.n_rows;
        int Ny = col.n_rows;
        //memsize += Nx*Ny;
        //cout << Nx << " " << Ny << " " << " " << K_unique(n) << endl;
        Mat<u64> block(Nx,Ny);
        uvec pqrs(6);
        Col<u64> tElements(Nx*Ny);
        uvec tBlockmap1(Nx*Ny);
        uvec tBlockmap2(Nx*Ny);
        uvec tBlockmap3(Nx*Ny);

        uvec c = floor(row/Nph); //k
        uvec b = floor((row  - c*Nph)/Nh);
        uvec k = row - c*Nph - b*Nh;

        uvec a = floor(col/Nh2); //k
        uvec j = floor((col - a*Nh2)/Nh);
        uvec i = col - a*Nh2 - j*Nh;

        bconfigs(n).set_size(Nx*Ny); //collect unique config "unaligned"
        bconfig_len += Ny;


        u64 index;
        tempElementsSize += Nx*Ny;
        for(int nx = 0; nx < Nx; nx++){
            for(int ny = 0; ny < Ny; ny++){

                index = to6(a(ny), b(nx), c(nx), i(ny), j(ny), k(nx));

                /*
                if(from6(index).max()>Np){
                    cout << "Warning in Np" << endl;
                    from6(index).print();
                    cout << index << " " << a(ny) << " " << b(nx) << " " << c(nx) << " " << i(ny) << " " << j(ny) << " " << k(nx) << endl;
                    cout << endl;
                    uvSize.print();
                    cout << endl;
                    cout << n1 << " " << n2 << " " << n3 << " " << n4 << " " << n5 << " " << endl;
                    cout << endl;
                }*/

                bconfigs(n)(nx*Ny + ny) = eBs.unique_int(i(ny)) + eBs.unique_int(j(ny)) +eBs.unique_int(k(nx));


                tElements(nx*Ny + ny) = index;
                tBlockmap1(nx*Ny + ny) = n;
                tBlockmap2(nx*Ny + ny) = nx;
                tBlockmap3(nx*Ny + ny) = ny;
                block(nx, ny) = index;
            }
        }
        fmBlocks(uiCurrent_block)(n) = block;
        tempElements(n) = tElements;
        tempBlockmap1(n) = tBlockmap1; //block that element belongs to
        tempBlockmap2(n) = tBlockmap2; //row of element
        tempBlockmap3(n) = tBlockmap3; //column of element


    }

    // ####################################################################
    // ## Flatten bconfigs                                               ##
    // ####################################################################





    // ####################################################################
    // ## Flatten tempElements and tempBlockmap                          ##
    // ####################################################################
    uvec flatElements(tempElementsSize);
    uvec flatBlockmap1(tempElementsSize);
    uvec flatBlockmap2(tempElementsSize);
    uvec flatBlockmap3(tempElementsSize);

    ivec flatBconfig(tempElementsSize);

    uint counter = 0;
    for(uint ni = 0; ni<uiN; ++ni){
        for(uint nj = 0; nj < tempElements(ni).n_rows; ++nj){
            flatElements(counter) = tempElements(ni)(nj);
            flatBlockmap1(counter) = tempBlockmap1(ni)(nj);
            flatBlockmap2(counter) = tempBlockmap2(ni)(nj);
            flatBlockmap3(counter) = tempBlockmap3(ni)(nj);

            flatBconfig(counter) = bconfigs(ni)(nj);

            counter += 1;
        }
        tempElements(ni).set_size(0); //release memory
        tempBlockmap1(ni).set_size(0);
        tempBlockmap2(ni).set_size(0);
        tempBlockmap3(ni).set_size(0);

    }

    ivBconfigs = unique(join_cols(ivBconfigs, flatBconfig)); //consolidate unique configs "unaligned"
    ivBconfigs = sort(ivBconfigs);

    // ####################################################################
    // ## Consolidate blocks with existing configurations                ##
    // ####################################################################


    umat n = sort_index(flatElements);  //traverse flatElements in increasing order (existing elements already sorted)
    flatElements = flatElements.elem(n);
    flatBlockmap1 = flatBlockmap1.elem(n);
    flatBlockmap2 = flatBlockmap2.elem(n);
    flatBlockmap3 = flatBlockmap3.elem(n);

    uint tempN = 0;
    uint trueN = 0;
    uint tempL = flatElements.n_rows; //number of elements in this projection
    uint trueL = uvElements.n_rows; //number of existing elem
    bool all_resolved = false;
    while(trueN < trueL){
        if(uvElements(trueN) == flatElements(tempN)){
            //identical indexes occuring in flat and uvElements
            //element already present in uvElements
            //cout << uvElements(trueN) << " " << flatElements(tempN) << endl;
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN;
            trueN += 1;
            tempN += 1;
        }
        else{
            //elements unequal
            //cout << uvElements(trueN) << " " << flatElements(tempN) << endl;
            //if(uvElements(trueN) == flatElements(tempN)){
            if(uvElements(trueN) < flatElements(tempN)){
                trueN += 1;
            }
            else{
                tempN += 1;
                if(tempN>=tempL){
                    all_resolved = true;
                    break;
                }
            }
        }
    }

    //if(tempN<tempL){
    if(all_resolved != true){
        //cout << "remaining  blocks" << endl;
        uvec remaining(tempL-tempN);
        uint tN = 0;
        while(tempN<tempL){
            //cout << trueN + tN << endl;
            remaining(tN) = flatElements(tempN);
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN + tN;
            tempN += 1;
            tN += 1;
        }

        //cout << tempL << " "<< tempN << " " << remaining.n_elem << " " << uvElements.n_elem << endl;
        uvElements = join_cols<umat>(uvElements, remaining);

    }


    //lock and load
    uiCurrent_block += 1;
}

void amplitude::map_t3_124_356(ivec Kk_unique){
    field<ivec> abi = pph();
    field<ivec> cjk = phh();

    //Kk_unique.print();
    ivec k_unique = intersect1d(unique(abi(3)), unique(cjk(3)));

    ivec K_unique = intersect1d(k_unique, Kk_unique);

    field<uvec> tempRows = partition_pph(abi, K_unique);
    field<uvec> tempCols = partition_phh(cjk, K_unique);
    //Kk_unique.print();
    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    fmBlocks(uiCurrent_block).set_size(uiN);

    field<Col<u64> > tempElements(uiN);
    field<uvec> tempBlockmap1(uiN);
    field<uvec> tempBlockmap2(uiN);
    field<uvec> tempBlockmap3(uiN);

    uint tempElementsSize = 0;
    //uvec a,b,c,i,j,k;
    //uint systemsize = 0;
    uint Np2 = Np*Np;
    uint Nph = Np*Nh;

    field<ivec> bconfigs(uiN); //normally aligned configurations for each block
    uint bconfig_len = 0;
    #pragma omp parallel for num_threads(nthreads)
    for(uint n = 0; n <K_unique.n_rows; ++n){
        uvec dim(6);
        uvec row = tempRows(n);
        uvec col = tempCols(n);
        //systemsize += row.n_rows*col.n_rows;


        int Nx = row.n_rows;
        int Ny = col.n_rows;
        //cout << Nx << " " << Ny << " " << " " << K_unique(n) << endl;
        Mat<u64> block(Nx,Ny);
        uvec pqrs(6);
        uvec tElements(Nx*Ny);
        uvec tBlockmap1(Nx*Ny);
        uvec tBlockmap2(Nx*Ny);
        uvec tBlockmap3(Nx*Ny);

        uvec i = floor(row/Np2); //k
        uvec b = floor((row  - i*Np2)/Np);
        uvec a = row - i*Np2 - b*Np;

        uvec k = floor(col/Nph); //k
        uvec j = floor((col - k*Nph)/Np);
        uvec c = col - k*Nph - j*Np;

        bconfigs(n).set_size(Nx*Ny); //collect unique config "unaligned"
        bconfig_len += Ny;

        u64 index;
        tempElementsSize += Nx*Ny;
        for(int nx = 0; nx < Nx; nx++){
            for(int ny = 0; ny < Ny; ny++){

                index = to6(a(nx), b(nx), c(ny), i(nx), j(ny), k(ny));


                bconfigs(n)(nx*Ny + ny) = eBs.unique_int(i(nx)) + eBs.unique_int(j(ny)) +eBs.unique_int(k(ny));

                tElements(nx*Ny + ny) = index;
                tBlockmap1(nx*Ny + ny) = n;
                tBlockmap2(nx*Ny + ny) = nx;
                tBlockmap3(nx*Ny + ny) = ny;
                block(nx, ny) = index;
            }
        }
        //memsize += Nx*Ny;



        fmBlocks(uiCurrent_block)(n) = block;
        tempElements(n) = tElements;
        tempBlockmap1(n) = tBlockmap1; //block that element belongs to
        tempBlockmap2(n) = tBlockmap2; //row of element
        tempBlockmap3(n) = tBlockmap3; //column of element


    }

    consolidate_blocks(uiN, tempElementsSize, tempElements, tempBlockmap1, tempBlockmap2, tempBlockmap3);

    // ####################################################################
    // ## Flatten bconfigs                                               ##
    // ####################################################################

    ivec flatBconfig(tempElementsSize);

    uint counter = 0;
    for(uint ni = 0; ni<uiN; ++ni){
        for(uint nj = 0; nj < tempElements(ni).n_rows; ++nj){

            flatBconfig(counter) = bconfigs(ni)(nj);

            counter += 1;
        }

    }

    ivBconfigs = unique(join_cols(ivBconfigs, flatBconfig)); //consolidate unique configs "unaligned"
    ivBconfigs = sort(ivBconfigs);




    //lock and load
    uiCurrent_block += 1;
}

void amplitude::map_t3_124_356_new(ivec Kk_unique){
    field<ivec> abi = pph();
    field<ivec> cjk = phh();

    //Kk_unique.print();
    ivec k_unique = intersect1d(unique(abi(3)), unique(cjk(3)));

    ivec K_unique = intersect1d(k_unique, Kk_unique);

    field<uvec> tempRows = partition_pph(abi, K_unique);
    field<uvec> tempCols = partition_phh(cjk, K_unique);
    //Kk_unique.print();
    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    fmBlocks(uiCurrent_block).set_size(uiN);

    field<uvec> tempElements(uiN);
    field<uvec> tempBlockmap1(uiN);
    field<uvec> tempBlockmap2(uiN);
    field<uvec> tempBlockmap3(uiN);

    uint tempElementsSize = 0;
    //uvec a,b,c,i,j,k;
    //uint systemsize = 0;
    uint Np2 = Np*Np;
    uint Nph = Np*Nh;

    field<ivec> bconfigs(uiN); //normally aligned configurations for each block
    uint bconfig_len = 0;

    uvec uvNsort = sort_index(uvElements); //tosort
    uvec uvElemtemp = uvElements.elem(uvNsort); //sorted
    uvec uvBsort = sort_index(uvElemtemp); //backsort
    #pragma omp parallel for num_threads(nthreads)
    for(uint n = 0; n <K_unique.n_rows; ++n){
        uvec dim(6);
        uvec row = tempRows(n);
        uvec col = tempCols(n);
        //systemsize += row.n_rows*col.n_rows;


        int Nx = row.n_rows;
        int Ny = col.n_rows;
        //cout << Nx << " " << Ny << " " << " " << K_unique(n) << endl;
        umat block(Nx,Ny);
        uvec pqrs(6);
        uvec tElements(Nx*Ny);
        uvec tBlockmap1(Nx*Ny);
        uvec tBlockmap2(Nx*Ny);
        uvec tBlockmap3(Nx*Ny);

        uvec i = floor(row/Np2); //k
        uvec b = floor((row  - i*Np2)/Np);
        uvec a = row - i*Np2 - b*Np;

        uvec k = floor(col/Nph); //k
        uvec j = floor((col - k*Nph)/Np);
        uvec c = col - k*Nph - j*Np;

        bconfigs(n).set_size(Nx*Ny); //collect unique config "unaligned"
        bconfig_len += Ny;

        uint index;
        tempElementsSize += Nx*Ny;
        for(int nx = 0; nx < Nx; nx++){
            for(int ny = 0; ny < Ny; ny++){

                index = to6(a(nx), b(nx), c(ny), i(nx), j(ny), k(ny));


                bconfigs(n)(nx*Ny + ny) = eBs.unique_int(i(nx)) + eBs.unique_int(j(ny)) +eBs.unique_int(k(ny));

                tElements(nx*Ny + ny) = index;
                tBlockmap1(nx*Ny + ny) = n;
                tBlockmap2(nx*Ny + ny) = nx;
                tBlockmap3(nx*Ny + ny) = ny;
                block(nx, ny) = index;
            }
        }
        fmBlocks(uiCurrent_block)(n) = block;
        //tempElements(n) = tElements;
        //tempBlockmap1(n) = tBlockmap1; //block that element belongs to
        //tempBlockmap2(n) = tBlockmap2; //row of element
        //tempBlockmap3(n) = tBlockmap3; //column of element

        //consolidate each block as they are traversed

        umat ns = sort_index(tElements);
        uvec flatElements = tElements.elem(ns);
        uvec flatBlockmap1 = tBlockmap1.elem(ns); //DOES THIS SORT PROPERLY? UNKNOWN,
        uvec flatBlockmap2 = tBlockmap2.elem(ns); //DOES THIS SORT PROPERLY? UNKNOWN,
        uvec flatBlockmap3 = tBlockmap3.elem(ns); //DOES THIS SORT PROPERLY? UNKNOWN,

        uint tempN = 0;
        uint trueN = 0;
        uint tempL = flatElements.n_rows;
        uint trueL = uvElements.n_rows;




        bool all_resolved = false;
        while(trueN < trueL){
            //if(uvElements(trueN) == flatElements(tempN)){
            if(uvElemtemp(trueN) == flatElements(tempN)){

                //identical indexes occuring in
                //cout << trueN << endl;
                //fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN;
                fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = uvNsort(trueN);
                trueN += 1;
                tempN += 1;
                if(tempN>=tempL){
                    all_resolved = true;
                    break;
                }
            }
            else{

                //if(uvElements(trueN) < flatElements(tempN)){
                if(uvElemtemp(trueN) < flatElements(tempN)){
                    trueN += 1;
                }
                else{
                    //flatElements(tempN) does not occur in uvElements
                    cout << fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) << endl;
                    tempN += 1;
                    if(tempN>=tempL){
                        all_resolved = true;
                        break;
                    }
                }

            }
        }


        //if(all_resolved != true){
        if(tempN<tempL){
            uvec remaining(tempL-tempN);
            uint tN = 0;
            while(tempN<tempL){
                //cout << "new block" << trueN + tN << endl;
                remaining(tN) = flatElements(tempN);
                fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN + tN;
                tempN += 1;
                tN += 1;
            }

            //cout << tempL << " "<< tempN << " " << remaining.n_elem << " " << uvElements.n_elem << endl;
            uvElements = join_cols<umat>(uvElements, remaining);
        }









    }


    ivec flatBconfig(tempElementsSize);

    uint counter = 0;
    for(uint ni = 0; ni<uiN; ++ni){
        for(uint nj = 0; nj < tempElements(ni).n_rows; ++nj){
            flatBconfig(counter) = bconfigs(ni)(nj);

            counter += 1;
        }
    }
    ivBconfigs = unique(join_cols(ivBconfigs, flatBconfig)); //consolidate unique configs "unaligned"
    ivBconfigs = sort(ivBconfigs);

    // ####################################################################
    // ## Consolidate blocks with existing configurations                ##
    // ####################################################################





    //lock and load
    uiCurrent_block += 1;
}

// ##################################################
// ##                                              ##
// ## External functions                           ##
// ##                                              ##
// ##################################################

void amplitude::map6c(ivec left, ivec right, ivec preconf){
    // ###########################################
    // ##  Map t3 amplitudes to a given configuration (preconf)
    // ##############################################
    imat L(left.n_rows,3);
    //left.print();
    for(uint i = 0; i<left.n_rows; ++i){
        L(i,0) =abs(left(i)) - 1;
        if(left(i)<0){
            L(i,1) = -1;
        }
        else{
            L(i,1) = 1;
        }
        if(abs(left(i))<=3){
            L(i,2) = Nh;
        }
        else{
            L(i,2) = 0;
        }
    }
    //L.print();

    imat R(right.n_rows,3);
    for(uint i = 0; i<right.n_rows; ++i){
        R(i,0) =abs(right(i)) - 1;
        if(right(i)<0){
            R(i,1) = -1;
        }
        else{
            R(i,1) = 1;
        }
        if(abs(right(i))<=3){
            R(i,2) = Nh;
        }
        else{
            R(i,2) = 0;
        }
    }
    //R.print();
    map_regions6c(L,R, preconf);

}

void amplitude::map6(ivec left, ivec right){
    imat L(left.n_rows,3);
    //left.print();
    for(uint i = 0; i<left.n_rows; ++i){
        L(i,0) =abs(left(i)) - 1;
        if(left(i)<0){
            L(i,1) = -1;
        }
        else{
            L(i,1) = 1;
        }
        if(abs(left(i))<=3){
            L(i,2) = Nh;
        }
        else{
            L(i,2) = 0;
        }
    }
    //L.print();

    imat R(right.n_rows,3);
    for(uint i = 0; i<right.n_rows; ++i){
        R(i,0) =abs(right(i)) - 1;
        if(right(i)<0){
            R(i,1) = -1;
        }
        else{
            R(i,1) = 1;
        }
        if(abs(right(i))<=3){
            R(i,2) = Nh;
        }
        else{
            R(i,2) = 0;
        }
    }
    //R.print();
    map_regions6(L,R);

}

void amplitude::map(ivec left, ivec right){
    imat L(left.n_rows,3);
    //left.print();
    for(uint i = 0; i<left.n_rows; ++i){
        L(i,0) =abs(left(i)) - 1;
        if(left(i)<0){
            L(i,1) = -1;
        }
        else{
            L(i,1) = 1;
        }
        if(abs(left(i))<=2){
            L(i,2) = Nh;
        }
        else{
            L(i,2) = 0;
        }
    }
    //L.print();

    imat R(right.n_rows,3);
    for(uint i = 0; i<right.n_rows; ++i){
        R(i,0) =abs(right(i)) - 1;
        if(right(i)<0){
            R(i,1) = -1;
        }
        else{
            R(i,1) = 1;
        }
        if(abs(right(i))<=2){
            R(i,2) = Nh;
        }
        else{
            R(i,2) = 0;
        }
    }
    //R.print();
    map_regions(L,R);

}

field<uvec> amplitude::blocksort(ivec LHS, ivec K_unique){
    // #####################################################
    // ## Sort indices of LHS into corresponding blocks   ##
    // #####################################################


    uvec l_sorted = sort_index(LHS); //most time is spent doing this -- maybe we should reorganize the items so they are sorted all along?
    bool adv = false;

    //LHS.elem(l_sorted).print();
    //cout << endl;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);
    uvec row(uiStatAlloc);
    uint nx = 0;
    int l_c= LHS(l_sorted(lc));
    field<uvec> tempRows(uiN);



    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    bool br = false;
    bool row_collect = false;
    while(lc < uiS){
        l_c = LHS(l_sorted(lc));

        if(l_c == C){
            row(nx) = l_sorted(lc);
            nx += 1;
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                tempRows(i) = sort(row(span(0,nx-1)));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }


        lc += 1;
        //cout << l_c - C << endl;
    }
    if(nx>0){
        //collect final row
        tempRows(i) = sort(row(span(0,nx-1)));
    }

    return tempRows;

}

void amplitude::map_regions6c(imat L, imat R, ivec preconf){


    // ###########################################################
    // ## Counting number of rows and columns in representation ##
    // ###########################################################
    uint iNrows = 1;
    uint iNcols = 1;

    for(uint i = 0; i<L.n_rows; ++i){
        iNrows *= uvSize(L(i,0));
    }
    for(uint i = 0; i<R.n_rows; ++i){
        iNcols *= uvSize(R(i,0));
    }

    // #########################################
    // ## Extract and organize actual indices ##
    // #########################################
    //uvec rows = conv_to<uvec>::from(linspace(0,iNrows-1, iNrows));

    uvec rows = linspace<uvec>(0,iNrows-1, iNrows);
    uvec cols = linspace<uvec>(0,iNcols-1, iNcols);

    //uvec cols = conv_to<uvec>::from(linspace(0,iNcols-1, iNcols));

    field<uvec> left = unpack(rows, L);
    field<uvec> right = unpack(cols, R);


    field<uvec> PQRS(6);
    for(uint i = 0; i< left.n_rows; ++i){
        PQRS(L(i,0)) = left(i);
    }
    for(uint i = 0; i< right.n_rows; ++i){
        PQRS(R(i,0)) = right(i);
    }
    //PQRS.print();

    // ############################################################
    // ## assign nonambiguous integer to each bra and ket config ##
    // ############################################################
    ivec LHS(iNrows); // = conv_to<ivec>::from(zeros(iNrows));
    for(uint i = 0; i<iNrows;++i){
        LHS(i) = 0;
    }
    //ivec LHS = conv_to<ivec>::from(zeros(iNrows));

    ivec RHS(iNcols); // = conv_to<ivec>::from(zeros(iNcols));
    for(uint i = 0; i<iNcols;++i){
        RHS(i) = 0;
    }

    //LHS*=0;
    //RHS*=0;
    //ivec LHS = zeros<int> (iNrows);
    //cout << iNrows << " " << iNcols << endl;


    //cout << L.n_rows << " " << L.n_cols << " " << L.n_elem << endl;
    for(uint i = 0; i<L.n_rows; ++i){
        LHS += eBs.unique(PQRS(L(i,0))+L(i,2))*L(i,1);
        //LHS += eBs.unique(PQRS(L(i,0)));
    }
    for(uint i = 0; i<R.n_rows; ++i){
        RHS += eBs.unique(PQRS(R(i,0))+R(i,2))*R(i,1);
    }
    //cout << LHS.n_rows << endl;
    //RHS.print();

    // ####################################################################
    // ## Iterate over unique combinations, retain blocks where RHS==LHS ##
    // ####################################################################
    //LHS.print();
    //cout << LHS.n_elem << endl << endl;
    ivec unique_L = unique(LHS);
    ivec unique_R = unique(RHS);
    ivec K_unique = intersect1d(unique_L, unique_R);
    K_unique = intersect1d(K_unique, preconf); //map against configuration

    //K_unique.print();
    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    fmBlocks(uiCurrent_block).set_size(uiN);

    field<uvec> tempElements(uiN);
    field<uvec> tempBlockmap1(uiN);
    field<uvec> tempBlockmap2(uiN);
    field<uvec> tempBlockmap3(uiN);

    uint tempElementsSize = 0;

    clock_t t0;

    t0 = clock();
    //experiment, trying to speed up initialization
    field<uvec> tempRows = blocksort(LHS, K_unique);
    field<uvec> tempCols = blocksort(RHS, K_unique);
    cout << " First method:" << (float)(clock()-t0)/CLOCKS_PER_SEC << endl;
    t0 = clock();


    for(uint i = 0; i<uiN; ++i){
        //uvec indx = find(LHS==K_unique(i));
        //LHS.elem(indx).print();
        //uvec row = rows.elem(find(LHS==K_unique(i)));
        //uvec col = cols.elem(find(RHS==K_unique(i)));
        uvec row = tempRows(i);
        uvec col = tempCols(i);
        //srow.print();
        int Nx = row.n_rows;
        int Ny = col.n_rows;
        //cout << Nx << " " << Ny << " " << " " << K_unique(i) << endl;
        umat block(Nx,Ny);
        uvec pqrs(6);
        uvec tElements(Nx*Ny);
        uvec tBlockmap1(Nx*Ny);
        uvec tBlockmap2(Nx*Ny);
        uvec tBlockmap3(Nx*Ny);

        uint index;
        tempElementsSize += Nx*Ny;
        for(int nx = 0; nx < Nx; nx++){
            for(int ny = 0; ny < Ny; ny++){
                uvec lhs = unpack_uvec(row(nx), L);
                uvec rhs = unpack_uvec(col(ny), R);
                for(uint j = 0; j<lhs.n_elem; ++j){
                    pqrs(L(j,0)) = lhs(j);
                }
                for(uint j = 0; j<rhs.n_elem; ++j){
                    pqrs(R(j,0)) = rhs(j);
                }
                index = to6(pqrs(0), pqrs(1), pqrs(2), pqrs(3), pqrs(4), pqrs(5));

                tElements(nx*Ny + ny) = index;
                tBlockmap1(nx*Ny + ny) = i;
                tBlockmap2(nx*Ny + ny) = nx;
                tBlockmap3(nx*Ny + ny) = ny;
                block(nx, ny) = index;
            }
        }
        fmBlocks(uiCurrent_block)(i) = block;
        tempElements(i) = tElements;
        tempBlockmap1(i) = tBlockmap1;
        tempBlockmap2(i) = tBlockmap2;
        tempBlockmap3(i) = tBlockmap3;

        //block.print();
        //cout << i << endl;
    }
    //fmBlocks(uiCurrent_block)(3).print();
    cout << "Second method:" << (float)(clock()-t0)/CLOCKS_PER_SEC << endl;
    t0 = clock();

    // ####################################################################
    // ## Flatten tempElements and tempBlockmap                          ##
    // ####################################################################
    uvec flatElements(tempElementsSize);
    uvec flatBlockmap1(tempElementsSize);
    uvec flatBlockmap2(tempElementsSize);
    uvec flatBlockmap3(tempElementsSize);

    uint counter = 0;
    for(uint i = 0; i<uiN; ++i){
        for(uint j = 0; j < tempElements(i).n_rows; ++j){
            flatElements(counter) = tempElements(i)(j);
            flatBlockmap1(counter) = tempBlockmap1(i)(j);
            flatBlockmap2(counter) = tempBlockmap2(i)(j);
            flatBlockmap3(counter) = tempBlockmap3(i)(j);
            counter += 1;
        }
        tempElements(i).set_size(0);
        tempBlockmap1(i).set_size(0);
        tempBlockmap2(i).set_size(0);
        tempBlockmap3(i).set_size(0);

    }

    // ####################################################################
    // ## Consolidate blocks with existing configurations                ##
    // ####################################################################


    umat n = sort_index(flatElements);
    flatElements = flatElements.elem(n);
    flatBlockmap1 = flatBlockmap1.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,
    flatBlockmap2 = flatBlockmap2.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,
    flatBlockmap3 = flatBlockmap3.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,

    uint tempN = 0;
    uint trueN = 0;
    uint tempL = flatElements.n_rows;
    uint trueL = uvElements.n_rows;
    bool all_resolved = false;
    while(trueN < trueL){
        if(uvElements(trueN) == flatElements(tempN)){
            //identical indexes occuring in
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN;
            trueN += 1;
            tempN += 1;
        }
        else{
            if(uvElements(trueN) < flatElements(tempN)){
                trueN += 1;
            }
            else{
                tempN += 1;
                if(tempN>=tempL){
                    all_resolved = true;
                    break;
                }
            }

        }
    }


    //if(all_resolved != true){
    if(tempN<tempL){
        uvec remaining(tempL-tempN);
        uint tN = 0;
        while(tempN<tempL){
            remaining(tN) = flatElements(tempN);
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN + tN;
            tempN += 1;
            tN += 1;
        }

        //cout << tempL << " "<< tempN << " " << remaining.n_elem << " " << uvElements.n_elem << endl;
        uvElements = join_cols<umat>(uvElements, remaining);
    }


    //lock and load
    uiCurrent_block += 1;
}


void amplitude::map_regions6(imat L, imat R){


    // ###########################################################
    // ## Counting number of rows and columns in representation ##
    // ###########################################################
    uint iNrows = 1;
    uint iNcols = 1;

    for(uint i = 0; i<L.n_rows; ++i){
        iNrows *= uvSize(L(i,0));
    }
    for(uint i = 0; i<R.n_rows; ++i){
        iNcols *= uvSize(R(i,0));
    }

    // #########################################
    // ## Extract and organize actual indices ##
    // #########################################
    //uvec rows = conv_to<uvec>::from(linspace(0,iNrows-1, iNrows));

    uvec rows = linspace<uvec>(0,iNrows-1, iNrows);
    uvec cols = linspace<uvec>(0,iNcols-1, iNcols);

    //uvec cols = conv_to<uvec>::from(linspace(0,iNcols-1, iNcols));

    field<uvec> left = unpack(rows, L);
    field<uvec> right = unpack(cols, R);


    field<uvec> PQRS(6);
    for(uint i = 0; i< left.n_rows; ++i){
        PQRS(L(i,0)) = left(i);
    }
    for(uint i = 0; i< right.n_rows; ++i){
        PQRS(R(i,0)) = right(i);
    }
    //PQRS.print();

    // ############################################################
    // ## assign nonambiguous integer to each bra and ket config ##
    // ############################################################
    ivec LHS(iNrows); // = conv_to<ivec>::from(zeros(iNrows));
    for(uint i = 0; i<iNrows;++i){
        LHS(i) = 0;
    }
    //ivec LHS = conv_to<ivec>::from(zeros(iNrows));

    ivec RHS(iNcols); // = conv_to<ivec>::from(zeros(iNcols));
    for(uint i = 0; i<iNcols;++i){
        RHS(i) = 0;
    }

    //LHS*=0;
    //RHS*=0;
    //ivec LHS = zeros<int> (iNrows);
    //cout << iNrows << " " << iNcols << endl;


    //cout << L.n_rows << " " << L.n_cols << " " << L.n_elem << endl;
    for(uint i = 0; i<L.n_rows; ++i){
        LHS += eBs.unique(PQRS(L(i,0))+L(i,2))*L(i,1);
        //LHS += eBs.unique(PQRS(L(i,0)));
    }
    for(uint i = 0; i<R.n_rows; ++i){
        RHS += eBs.unique(PQRS(R(i,0))+R(i,2))*R(i,1);
    }
    //cout << LHS.n_rows << endl;
    //RHS.print();

    // ####################################################################
    // ## Iterate over unique combinations, retain blocks where RHS==LHS ##
    // ####################################################################
    //LHS.print();
    //cout << LHS.n_elem << endl << endl;
    ivec unique_L = unique(LHS);
    ivec unique_R = unique(RHS);
    ivec K_unique = intersect1d(unique_L, unique_R);

    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    fmBlocks(uiCurrent_block).set_size(uiN);

    field<uvec> tempElements(uiN);
    field<uvec> tempBlockmap1(uiN);
    field<uvec> tempBlockmap2(uiN);
    field<uvec> tempBlockmap3(uiN);

    uint tempElementsSize = 0;

    clock_t t0;

    t0 = clock();
    //experiment, trying to speed up initialization
    field<uvec> tempRows = blocksort(LHS, K_unique);
    field<uvec> tempCols = blocksort(RHS, K_unique);
    //cout << " First method:" << (float)(clock()-t0)/CLOCKS_PER_SEC << endl;
    //t0 = clock();


    for(uint i = 0; i<uiN; ++i){
        //uvec indx = find(LHS==K_unique(i));
        //LHS.elem(indx).print();
        //uvec row = rows.elem(find(LHS==K_unique(i)));
        //uvec col = cols.elem(find(RHS==K_unique(i)));
        uvec row = tempRows(i);
        uvec col = tempCols(i);
        //row.print();
        //cout << endl;
        //srow.print();
        int Nx = row.n_rows;
        int Ny = col.n_rows;
        //cout << Nx << " " << Ny << " " << " " << K_unique(i) << endl;
        umat block(Nx,Ny);
        uvec pqrs(6);
        uvec tElements(Nx*Ny);
        uvec tBlockmap1(Nx*Ny);
        uvec tBlockmap2(Nx*Ny);
        uvec tBlockmap3(Nx*Ny);

        uint index;
        tempElementsSize += Nx*Ny;
        for(int nx = 0; nx < Nx; nx++){
            for(int ny = 0; ny < Ny; ny++){
                uvec lhs = unpack_uvec(row(nx), L);
                uvec rhs = unpack_uvec(col(ny), R);
                for(uint j = 0; j<lhs.n_elem; ++j){
                    pqrs(L(j,0)) = lhs(j);
                }
                for(uint j = 0; j<rhs.n_elem; ++j){
                    pqrs(R(j,0)) = rhs(j);
                }
                index = to6(pqrs(0), pqrs(1), pqrs(2), pqrs(3), pqrs(4), pqrs(5));

                tElements(nx*Ny + ny) = index;
                tBlockmap1(nx*Ny + ny) = i;
                tBlockmap2(nx*Ny + ny) = nx;
                tBlockmap3(nx*Ny + ny) = ny;
                block(nx, ny) = index;
            }
        }
        //cout << block.max() << endl;
        fmBlocks(uiCurrent_block)(i) = block;
        tempElements(i) = tElements;
        tempBlockmap1(i) = tBlockmap1;
        tempBlockmap2(i) = tBlockmap2;
        tempBlockmap3(i) = tBlockmap3;

        //block.print();
        //cout << i << endl;
    }
    //fmBlocks(uiCurrent_block)(3).print();
    //cout << "Second method:" << (float)(clock()-t0)/CLOCKS_PER_SEC << endl;
    //t0 = clock();

    // ####################################################################
    // ## Flatten tempElements and tempBlockmap                          ##
    // ####################################################################
    uvec flatElements(tempElementsSize);
    uvec flatBlockmap1(tempElementsSize);
    uvec flatBlockmap2(tempElementsSize);
    uvec flatBlockmap3(tempElementsSize);

    uint counter = 0;
    for(uint i = 0; i<uiN; ++i){
        for(uint j = 0; j < tempElements(i).n_rows; ++j){
            flatElements(counter) = tempElements(i)(j);
            flatBlockmap1(counter) = tempBlockmap1(i)(j);
            flatBlockmap2(counter) = tempBlockmap2(i)(j);
            flatBlockmap3(counter) = tempBlockmap3(i)(j);
            counter += 1;
        }
        tempElements(i).set_size(0);
        tempBlockmap1(i).set_size(0);
        tempBlockmap2(i).set_size(0);
        tempBlockmap3(i).set_size(0);

    }

    // ####################################################################
    // ## Consolidate blocks with existing configurations                ##
    // ####################################################################


    umat n = sort_index(flatElements);
    flatElements = flatElements.elem(n);
    flatBlockmap1 = flatBlockmap1.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,
    flatBlockmap2 = flatBlockmap2.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,
    flatBlockmap3 = flatBlockmap3.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,

    uint tempN = 0;
    uint trueN = 0;
    uint tempL = flatElements.n_rows;
    uint trueL = uvElements.n_rows;
    bool all_resolved = false;
    while(trueN < trueL){
        //cout << "Consol rows 0" << endl;
        if(uvElements(trueN) == flatElements(tempN)){
            //identical indexes occuring in
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN;
            trueN += 1;
            tempN += 1;
        }
        else{
            if(uvElements(trueN) < flatElements(tempN)){
                trueN += 1;
            }
            else{
                tempN += 1;
                if(tempN>=tempL){
                    all_resolved = true;
                    break;
                }
            }

        }
    }
    //cout << "TrueN:" << trueN << " " << tempN << endl;


    //if(all_resolved != true){
    if(tempN<tempL){
        //cout << "consolidating blocks" << endl;
        uvec remaining(tempL-tempN);
        uint tN = 0;
        while(tempN<tempL){
            remaining(tN) = flatElements(tempN);
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN + tN;
            tempN += 1;
            tN += 1;
        }
        //cout << tempL << " "<< tempN << " " << remaining.n_elem << " " << uvElements.n_elem << endl;
        uvElements = join_cols<umat>(uvElements, remaining);
    }
    //lock and load
    //cout << "TrueN:" << trueN << " " << tempN << " " << endl;

    uiCurrent_block += 1;
} //map all regions defined by L == R


void amplitude::map_t2_permutations(){
    // ###################################################################
    // ## Set up amplitude as t3temp with index permutations in blocks  ##
    // ###################################################################

    //Basically, we set up the standard amplitude sorting as abc-ijk, but store dimensions of each block so we ,may easily permute them later
    field<ivec> ab = pp();
    field<ivec> ij = hh_compact();

    ivec K_unique = intersect1d(unique(ab(2)), unique(ij(2)));
    field<uvec> tempRows = partition_pp_permutations(ab, K_unique);
    //field<uvec> tempCols = partition(ijk(3), K_unique);
    field<uvec> tempCols = partition_hh_permutations(ij, K_unique);
    //uvec row;
    //uvec col;
    //uvec a,b,c;
    //uvec i,j,k;


    //for use in actual amplitude mapping
    permutative_ordering.set_size(K_unique.n_rows);
    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    fmBlocks(uiCurrent_block).set_size(uiN);

    field<uvec> tempElements(uiN);
    field<uvec> tempBlockmap1(uiN);
    field<uvec> tempBlockmap2(uiN);
    field<uvec> tempBlockmap3(uiN);

    uint tempElementsSize = 0;
    //uvec a,b,c,i,j,k;
    uint systemsize = 0;

    //#pragma omp parallel for num_threads(nthreads)
    for(uint n = 0; n <K_unique.n_rows; ++n){
        uvec dim(6);
        uvec row = tempRows(n);
        uvec col = tempCols(n);
        //systemsize += row.n_rows*col.n_rows;


        int Nx = row.n_rows;
        int Ny = col.n_rows;
        //cout << Nx << " " << Ny << " " << " " << K_unique(i) << endl;
        umat block(Nx,Ny);
        uvec pqrs(4);
        uvec tElements(Nx*Ny);
        uvec tBlockmap1(Nx*Ny);
        uvec tBlockmap2(Nx*Ny);
        uvec tBlockmap3(Nx*Ny);

        uvec b = floor(row/Np); //k
        uvec a = row  - b*Np;

        uvec j = floor(col/Nh); //k

        uvec i = col  - j*Nh;


        uint index;
        tempElementsSize += Nx*Ny;
        for(int nx = 0; nx < Nx; nx++){
            for(int ny = 0; ny < Ny; ny++){

                index = to(a(nx), b(nx), i(ny), j(ny));

                tElements(nx*Ny + ny) = index;
                tBlockmap1(nx*Ny + ny) = n;
                tBlockmap2(nx*Ny + ny) = nx;
                tBlockmap3(nx*Ny + ny) = ny;
                block(nx, ny) = index;
            }
        }
        fmBlocks(0)(n) = block;
        tempElements(n) = tElements;
        tempBlockmap1(n) = tBlockmap1; //block that element belongs to
        tempBlockmap2(n) = tBlockmap2; //row of element
        tempBlockmap3(n) = tBlockmap3; //column of element








    }



    // ####################################################################
    // ## Flatten tempElements and tempBlockmap                          ##
    // ####################################################################
    uvec flatElements(tempElementsSize);
    uvec flatBlockmap1(tempElementsSize);
    uvec flatBlockmap2(tempElementsSize);
    uvec flatBlockmap3(tempElementsSize);

    uint counter = 0;
    for(uint ni = 0; ni<uiN; ++ni){
        for(uint nj = 0; nj < tempElements(ni).n_rows; ++nj){
            flatElements(counter) = tempElements(ni)(nj);
            flatBlockmap1(counter) = tempBlockmap1(ni)(nj);
            flatBlockmap2(counter) = tempBlockmap2(ni)(nj);
            flatBlockmap3(counter) = tempBlockmap3(ni)(nj);
            counter += 1;
        }
        tempElements(ni).set_size(0);
        tempBlockmap1(ni).set_size(0);
        tempBlockmap2(ni).set_size(0);
        tempBlockmap3(ni).set_size(0);

    }

    // ####################################################################
    // ## Consolidate blocks with existing configurations                ##
    // ####################################################################


    umat n = sort_index(flatElements);
    flatElements = flatElements.elem(n);
    flatBlockmap1 = flatBlockmap1.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,
    flatBlockmap2 = flatBlockmap2.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,
    flatBlockmap3 = flatBlockmap3.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,

    uint tempN = 0;
    uint trueN = 0;
    uint tempL = flatElements.n_rows;
    uint trueL = uvElements.n_rows;
    bool all_resolved = false;
    while(trueN < trueL){
        if(uvElements(trueN) == flatElements(tempN)){
            //identical indexes occuring in
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN;
            trueN += 1;
            tempN += 1;
        }
        else{
            if(uvElements(trueN) < flatElements(tempN)){
                trueN += 1;
            }
            else{
                tempN += 1;
                if(tempN>=tempL){
                    all_resolved = true;
                    break;
                }
            }

        }
    }


    //if(all_resolved != true){
    if(tempN<tempL){
        uvec remaining(tempL-tempN);
        uint tN = 0;
        while(tempN<tempL){
            remaining(tN) = flatElements(tempN);
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN + tN;
            tempN += 1;
            tN += 1;
        }

        //cout << tempL << " "<< tempN << " " << remaining.n_elem << " " << uvElements.n_elem << endl;
        uvElements = join_cols<umat>(uvElements, remaining);
    }


    //lock and load
    uiCurrent_block += 1;

}


void amplitude::map_t3_permutations(){
    // ###################################################################
    // ## Set up amplitude as t3temp with index permutations in blocks  ##
    // ###################################################################

    //Basically, we set up the standard amplitude sorting as abc-ijk, but store dimensions of each block so we ,may easily permute them later
    field<ivec> abc = ppp({1,1,1});
    field<ivec> ijk = hhh();

    ivec K_unique = intersect1d(unique(abc(3)), unique(ijk(3)));
    field<uvec> tempRows = partition_ppp_permutations(abc, K_unique);
    //field<uvec> tempCols = partition(ijk(3), K_unique);
    field<uvec> tempCols = partition_hhh_permutations(ijk, K_unique);
    uvec row;
    uvec col;
    uvec a,b,c;
    uvec i,j,k;


    //for use in actual amplitude mapping
    permutative_ordering.set_size(K_unique.n_rows);
    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    fmBlocks(uiCurrent_block).set_size(uiN);

    field<uvec> tempElements(uiN);
    field<uvec> tempBlockmap1(uiN);
    field<uvec> tempBlockmap2(uiN);
    field<uvec> tempBlockmap3(uiN);

    uint tempElementsSize = 0;
    //uvec a,b,c,i,j,k;
    uint systemsize = 0;
    for(uint n = 0; n <K_unique.n_rows; ++n){
        uvec dim(6);
        row = tempRows(n);
        col = tempCols(n);
        //systemsize += row.n_rows*col.n_rows;


        int Nx = row.n_rows;
        int Ny = col.n_rows;
        //cout << Nx << " " << Ny << " " << " " << K_unique(i) << endl;
        umat block(Nx,Ny);
        uvec pqrs(4);
        uvec tElements(Nx*Ny);
        uvec tBlockmap1(Nx*Ny);
        uvec tBlockmap2(Nx*Ny);
        uvec tBlockmap3(Nx*Ny);

        c = floor(row/(Np*Np)); //k
        b = floor(((row - c*Np*Np))/Np);
        a = row - c*Np*Np - b*Np;

        k = floor(col/(Nh*Nh)); //k
        j = floor(((col - k*Nh*Nh))/Nh);
        i = col - k*Nh*Nh - j*Nh;


        uint index;
        tempElementsSize += Nx*Ny;
        for(int nx = 0; nx < Nx; nx++){
            for(int ny = 0; ny < Ny; ny++){

                index = to6(a(nx), b(nx), c(nx), i(ny), j(ny), k(ny));

                tElements(nx*Ny + ny) = index;
                tBlockmap1(nx*Ny + ny) = n;
                tBlockmap2(nx*Ny + ny) = nx;
                tBlockmap3(nx*Ny + ny) = ny;
                block(nx, ny) = index;
            }
        }
        fmBlocks(0)(n) = block;
        tempElements(n) = tElements;
        tempBlockmap1(n) = tBlockmap1; //block that element belongs to
        tempBlockmap2(n) = tBlockmap2; //row of element
        tempBlockmap3(n) = tBlockmap3; //column of element








    }
    //cout << "Blocks:" << tempRows.n_rows << endl;
    //cout << "Size:" << systemsize << endl;
    //cout << "init_size:" << Np*(Np+1)*(Np+2)/6 << endl;


    // ####################################################################
    // ## Flatten tempElements and tempBlockmap                          ##
    // ####################################################################
    uvec flatElements(tempElementsSize);
    uvec flatBlockmap1(tempElementsSize);
    uvec flatBlockmap2(tempElementsSize);
    uvec flatBlockmap3(tempElementsSize);

    uint counter = 0;
    for(uint ni = 0; ni<uiN; ++ni){
        for(uint nj = 0; nj < tempElements(ni).n_rows; ++nj){
            flatElements(counter) = tempElements(ni)(nj);
            flatBlockmap1(counter) = tempBlockmap1(ni)(nj);
            flatBlockmap2(counter) = tempBlockmap2(ni)(nj);
            flatBlockmap3(counter) = tempBlockmap3(ni)(nj);
            counter += 1;
        }
        tempElements(ni).set_size(0);
        tempBlockmap1(ni).set_size(0);
        tempBlockmap2(ni).set_size(0);
        tempBlockmap3(ni).set_size(0);

    }

    // ####################################################################
    // ## Consolidate blocks with existing configurations                ##
    // ####################################################################


    umat n = sort_index(flatElements);
    flatElements = flatElements.elem(n);
    flatBlockmap1 = flatBlockmap1.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,
    flatBlockmap2 = flatBlockmap2.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,
    flatBlockmap3 = flatBlockmap3.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,

    uint tempN = 0;
    uint trueN = 0;
    uint tempL = flatElements.n_rows;
    uint trueL = uvElements.n_rows;
    bool all_resolved = false;
    while(trueN < trueL){
        if(uvElements(trueN) == flatElements(tempN)){
            //identical indexes occuring in
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN;
            trueN += 1;
            tempN += 1;
        }
        else{
            if(uvElements(trueN) < flatElements(tempN)){
                trueN += 1;
            }
            else{
                tempN += 1;
                if(tempN>=tempL){
                    all_resolved = true;
                    break;
                }
            }

        }
    }


    //if(all_resolved != true){
    if(tempN<tempL){
        uvec remaining(tempL-tempN);
        uint tN = 0;
        while(tempN<tempL){
            remaining(tN) = flatElements(tempN);
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN + tN;
            tempN += 1;
            tN += 1;
        }

        //cout << tempL << " "<< tempN << " " << remaining.n_elem << " " << uvElements.n_elem << endl;
        uvElements = join_cols<umat>(uvElements, remaining);
    }


    //lock and load
    uiCurrent_block += 1;




}

void amplitude::map_t3_permutations_bconfig(){
    // ###################################################################
    // ## Set up amplitude as t3temp with index permutations in blocks  ##
    // ###################################################################

    //Basically, we set up the standard amplitude sorting as abc-ijk, but store dimensions of each block so we ,may easily permute them later
    field<ivec> abc = ppp({1,1,1});
    field<ivec> ijk = hhh();

    ivec K_unique = intersect1d(unique(abc(3)), unique(ijk(3)));

    K_unique = intersect1d(K_unique, ivBconfigs); //map against preexisting configs

    field<uvec> tempRows = partition_ppp_permutations(abc, K_unique);
    //field<uvec> tempCols = partition(ijk(3), K_unique);
    field<uvec> tempCols = partition_hhh_permutations(ijk, K_unique);
    uvec row;
    uvec col;
    uvec a,b,c;
    uvec i,j,k;


    //for use in actual amplitude mapping
    permutative_ordering.set_size(K_unique.n_rows);
    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    fmBlocks(uiCurrent_block).set_size(uiN);

    field<uvec> tempElements(uiN);
    field<uvec> tempBlockmap1(uiN);
    field<uvec> tempBlockmap2(uiN);
    field<uvec> tempBlockmap3(uiN);

    uint tempElementsSize = 0;
    //uvec a,b,c,i,j,k;
    uint systemsize = 0;
    for(uint n = 0; n <K_unique.n_rows; ++n){
        uvec dim(6);
        row = tempRows(n);
        col = tempCols(n);
        //systemsize += row.n_rows*col.n_rows;


        int Nx = row.n_rows;
        int Ny = col.n_rows;
        //cout << Nx << " " << Ny << " " << " " << K_unique(i) << endl;
        umat block(Nx,Ny);
        uvec pqrs(4);
        uvec tElements(Nx*Ny);
        uvec tBlockmap1(Nx*Ny);
        uvec tBlockmap2(Nx*Ny);
        uvec tBlockmap3(Nx*Ny);

        c = floor(row/(Np*Np)); //k
        b = floor(((row - c*Np*Np))/Np);
        a = row - c*Np*Np - b*Np;

        k = floor(col/(Nh*Nh)); //k
        j = floor(((col - k*Nh*Nh))/Nh);
        i = col - k*Nh*Nh - j*Nh;


        uint index;
        tempElementsSize += Nx*Ny;
        for(int nx = 0; nx < Nx; nx++){
            for(int ny = 0; ny < Ny; ny++){

                index = to6(a(nx), b(nx), c(nx), i(ny), j(ny), k(ny));

                tElements(nx*Ny + ny) = index;
                tBlockmap1(nx*Ny + ny) = n;
                tBlockmap2(nx*Ny + ny) = nx;
                tBlockmap3(nx*Ny + ny) = ny;
                block(nx, ny) = index;
            }
        }
        fmBlocks(0)(n) = block;
        tempElements(n) = tElements;
        tempBlockmap1(n) = tBlockmap1; //block that element belongs to
        tempBlockmap2(n) = tBlockmap2; //row of element
        tempBlockmap3(n) = tBlockmap3; //column of element








    }
    //cout << "Blocks:" << tempRows.n_rows << endl;
    //cout << "Size:" << systemsize << endl;
    //cout << "init_size:" << Np*(Np+1)*(Np+2)/6 << endl;


    // ####################################################################
    // ## Flatten tempElements and tempBlockmap                          ##
    // ####################################################################
    uvec flatElements(tempElementsSize);
    uvec flatBlockmap1(tempElementsSize);
    uvec flatBlockmap2(tempElementsSize);
    uvec flatBlockmap3(tempElementsSize);

    uint counter = 0;
    for(uint ni = 0; ni<uiN; ++ni){
        for(uint nj = 0; nj < tempElements(ni).n_rows; ++nj){
            flatElements(counter) = tempElements(ni)(nj);
            flatBlockmap1(counter) = tempBlockmap1(ni)(nj);
            flatBlockmap2(counter) = tempBlockmap2(ni)(nj);
            flatBlockmap3(counter) = tempBlockmap3(ni)(nj);
            counter += 1;
        }
        tempElements(ni).set_size(0);
        tempBlockmap1(ni).set_size(0);
        tempBlockmap2(ni).set_size(0);
        tempBlockmap3(ni).set_size(0);

    }

    // ####################################################################
    // ## Consolidate blocks with existing configurations                ##
    // ####################################################################


    umat n = sort_index(flatElements);
    flatElements = flatElements.elem(n);
    flatBlockmap1 = flatBlockmap1.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,
    flatBlockmap2 = flatBlockmap2.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,
    flatBlockmap3 = flatBlockmap3.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,

    uint tempN = 0;
    uint trueN = 0;
    uint tempL = flatElements.n_rows;
    uint trueL = uvElements.n_rows;
    bool all_resolved = false;
    while(trueN < trueL){
        if(uvElements(trueN) == flatElements(tempN)){
            //identical indexes occuring in
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN;
            trueN += 1;
            tempN += 1;
        }
        else{
            if(uvElements(trueN) < flatElements(tempN)){
                trueN += 1;
            }
            else{
                tempN += 1;
                if(tempN>=tempL){
                    all_resolved = true;
                    break;
                }
            }

        }
    }


    //if(all_resolved != true){
    if(tempN<tempL){
        uvec remaining(tempL-tempN);
        uint tN = 0;
        while(tempN<tempL){
            remaining(tN) = flatElements(tempN);
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN + tN;
            tempN += 1;
            tN += 1;
        }

        //cout << tempL << " "<< tempN << " " << remaining.n_elem << " " << uvElements.n_elem << endl;
        uvElements = join_cols<umat>(uvElements, remaining);
    }


    //lock and load
    uiCurrent_block += 1;




}

void amplitude::map_t3_permutations_bconfig_sparse(){
    // ###################################################################
    // ## Set up amplitude as t3temp with index permutations in blocks  ##
    // ###################################################################

    //Basically, we set up the standard amplitude sorting as abc-ijk, but store dimensions of each block so we ,may easily permute them later
    field<ivec> abc = ppp({1,1,1});
    field<ivec> ijk = hhh();

    ivec K_unique = intersect1d(unique(abc(3)), unique(ijk(3)));

    K_unique = intersect1d(K_unique, ivBconfigs); //map against preexisting configs

    field<uvec> tempRows = partition_ppp_permutations(abc, K_unique);
    //field<uvec> tempCols = partition(ijk(3), K_unique);
    field<uvec> tempCols = partition_hhh_permutations(ijk, K_unique);
    //uvec row;
    //uvec col;
    //uvec a,b,c;
    //uvec i,j,k;


    //for use in actual amplitude mapping
    permutative_ordering.set_size(K_unique.n_rows);
    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    fmBlocks(uiCurrent_block).set_size(uiN);
    fspBlocks.set_size(uiN);
    fspDims.set_size(uiN,2);

    field<Col<u64> > tempElements(uiN);
    field<uvec> tempBlockmap1(uiN);
    field<uvec> tempBlockmap2(uiN);
    field<uvec> tempBlockmap3(uiN);

    uint tempElementsSize = 0;
    //uvec a,b,c,i,j,k;
    uint systemsize = 0;

    uvNsort1 = sort_index(uvElements);
    uvElemtemp1 = uvElements.elem(uvNsort1);
    //uvBsort1 = sort_index(uvNsort1); //backsort

    #pragma omp parallel for num_threads(nthreads)
    for(uint n = 0; n <K_unique.n_rows; ++n){
        uvec dim(6);
        uvec row = tempRows(n);
        uvec col = tempCols(n);
        //systemsize += row.n_rows*col.n_rows;


        int Nx = row.n_rows;
        int Ny = col.n_rows;
        //cout << Nx << " " << Ny << " " << " " << K_unique(i) << endl;
        umat block(Nx,Ny);
        uvec pqrs(4);
        uvec tElements(Nx*Ny);
        uvec tBlockmap1(Nx*Ny);
        uvec tBlockmap2(Nx*Ny);
        uvec tBlockmap3(Nx*Ny);

        uvec c = floor(row/(Np*Np)); //k
        uvec b = floor(((row - c*Np*Np))/Np);
        uvec a = row - c*Np*Np - b*Np;

        uvec k = floor(col/(Nh*Nh)); //k
        uvec j = floor(((col - k*Nh*Nh))/Nh);
        uvec i = col - k*Nh*Nh - j*Nh;


        u64 index;
        tempElementsSize += Nx*Ny;
        for(int nx = 0; nx < Nx; nx++){
            for(int ny = 0; ny < Ny; ny++){

                index = to6(a(nx), b(nx), c(nx), i(ny), j(ny), k(ny));

                tElements(nx*Ny + ny) = index;
                tBlockmap1(nx*Ny + ny) = n;
                tBlockmap2(nx*Ny + ny) = nx;
                tBlockmap3(nx*Ny + ny) = ny;
                block(nx, ny) = index;
            }
        }

        enroll_block(block, Nx*Ny, tElements, tBlockmap1, tBlockmap2, tBlockmap3);

    }

    //lock and load
    uiCurrent_block += 1;


}

void amplitude::map_regions(imat L, imat R){


    // ###########################################################
    // ## Counting number of rows and columns in representation ##
    // ###########################################################
    uint iNrows = 1;
    uint iNcols = 1;

    for(uint i = 0; i<L.n_rows; ++i){
        iNrows *= uvSize(L(i,0));
    }
    for(uint i = 0; i<R.n_rows; ++i){
        iNcols *= uvSize(R(i,0));
    }

    // #########################################
    // ## Extract and organize actual indices ##
    // #########################################
    //uvec rows = conv_to<uvec>::from(linspace(0,iNrows-1, iNrows));

    uvec rows = linspace<uvec>(0,iNrows-1, iNrows);
    uvec cols = linspace<uvec>(0,iNcols-1, iNcols);

    //uvec cols = conv_to<uvec>::from(linspace(0,iNcols-1, iNcols));

    field<uvec> left = unpack(rows, L);
    field<uvec> right = unpack(cols, R);


    field<uvec> PQRS(4);
    for(uint i = 0; i< left.n_rows; ++i){
        PQRS(L(i,0)) = left(i);
    }
    for(uint i = 0; i< right.n_rows; ++i){
        PQRS(R(i,0)) = right(i);
    }
    //PQRS.print();

    // ############################################################
    // ## assign nonambiguous integer to each bra and ket config ##
    // ############################################################
    ivec LHS(iNrows); // = conv_to<ivec>::from(zeros(iNrows));
    for(uint i = 0; i<iNrows;++i){
        LHS(i) = 0;
    }
    //ivec LHS = conv_to<ivec>::from(zeros(iNrows));

    ivec RHS(iNcols); // = conv_to<ivec>::from(zeros(iNcols));
    for(uint i = 0; i<iNcols;++i){
        RHS(i) = 0;
    }

    //LHS*=0;
    //RHS*=0;
    //ivec LHS = zeros<int> (iNrows);
    //cout << iNrows << " " << iNcols << endl;


    //cout << L.n_rows << " " << L.n_cols << " " << L.n_elem << endl;
    for(uint i = 0; i<L.n_rows; ++i){
        LHS += eBs.unique(PQRS(L(i,0))+L(i,2))*L(i,1);
        //LHS += eBs.unique(PQRS(L(i,0)));
    }
    for(uint i = 0; i<R.n_rows; ++i){
        RHS += eBs.unique(PQRS(R(i,0))+R(i,2))*R(i,1);
    }
    //cout << LHS.n_rows << endl;
    //RHS.print();

    // ####################################################################
    // ## Iterate over unique combinations, retain blocks where RHS==LHS ##
    // ####################################################################
    //LHS.print();
    //cout << LHS.n_elem << endl << endl;
    ivec unique_L = unique(LHS);
    ivec unique_R = unique(RHS);
    ivec K_unique = intersect1d(unique_L, unique_R);

    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    fmBlocks(uiCurrent_block).set_size(uiN);

    field<uvec> tempElements(uiN);
    field<uvec> tempBlockmap1(uiN);
    field<uvec> tempBlockmap2(uiN);
    field<uvec> tempBlockmap3(uiN);

    uint tempElementsSize = 0;

    //field<uvec> tempRows = blocksort(LHS, K_unique);
    //field<uvec> tempCols = blocksort(RHS, K_unique);
    for(uint i = 0; i<uiN; ++i){
        uvec indx = find(LHS==K_unique(i));
        //LHS.elem(indx).print();
        uvec row = rows.elem(find(LHS==K_unique(i)));
        uvec col = cols.elem(find(RHS==K_unique(i)));





        //cout << "row:" << endl;
        //row.print();
        //cout << "r1:" << endl;
        //row1.print();

        //srow.print();
        int Nx = row.n_rows;
        int Ny = col.n_rows;
        //cout << Nx << " " << Ny << " " << " " << K_unique(i) << endl;
        umat block(Nx,Ny);
        uvec pqrs(4);
        uvec tElements(Nx*Ny);
        uvec tBlockmap1(Nx*Ny);
        uvec tBlockmap2(Nx*Ny);
        uvec tBlockmap3(Nx*Ny);

        uint index;
        tempElementsSize += Nx*Ny;
        for(int nx = 0; nx < Nx; nx++){
            for(int ny = 0; ny < Ny; ny++){
                uvec lhs = unpack_uvec(row(nx), L);
                uvec rhs = unpack_uvec(col(ny), R);
                for(uint j = 0; j<lhs.n_elem; ++j){
                    pqrs(L(j,0)) = lhs(j);
                }
                for(uint j = 0; j<rhs.n_elem; ++j){
                    pqrs(R(j,0)) = rhs(j);
                }
                index = to(pqrs(0), pqrs(1), pqrs(2), pqrs(3));

                tElements(nx*Ny + ny) = index;
                tBlockmap1(nx*Ny + ny) = i;
                tBlockmap2(nx*Ny + ny) = nx;
                tBlockmap3(nx*Ny + ny) = ny;
                block(nx, ny) = index;
            }
        }
        fmBlocks(uiCurrent_block)(i) = block;
        tempElements(i) = tElements;
        tempBlockmap1(i) = tBlockmap1; //block that element belongs to
        tempBlockmap2(i) = tBlockmap2; //row of element
        tempBlockmap3(i) = tBlockmap3; //column of element

        //block.print();
        //cout << i << endl;
    }
    //fmBlocks(uiCurrent_block)(3).print();
    //cout << tempElementsSize << endl;

    // ####################################################################
    // ## Flatten tempElements and tempBlockmap                          ##
    // ####################################################################
    uvec flatElements(tempElementsSize);
    uvec flatBlockmap1(tempElementsSize);
    uvec flatBlockmap2(tempElementsSize);
    uvec flatBlockmap3(tempElementsSize);

    uint counter = 0;
    for(uint i = 0; i<uiN; ++i){
        for(uint j = 0; j < tempElements(i).n_rows; ++j){
            flatElements(counter) = tempElements(i)(j);
            flatBlockmap1(counter) = tempBlockmap1(i)(j);
            flatBlockmap2(counter) = tempBlockmap2(i)(j);
            flatBlockmap3(counter) = tempBlockmap3(i)(j);
            counter += 1;
        }
        tempElements(i).set_size(0);
        tempBlockmap1(i).set_size(0);
        tempBlockmap2(i).set_size(0);
        tempBlockmap3(i).set_size(0);

    }

    // ####################################################################
    // ## Consolidate blocks with existing configurations                ##
    // ####################################################################


    umat n = sort_index(flatElements);
    flatElements = flatElements.elem(n);
    flatBlockmap1 = flatBlockmap1.elem(n);
    flatBlockmap2 = flatBlockmap2.elem(n);
    flatBlockmap3 = flatBlockmap3.elem(n);

    //cout <<counter << " " << flatElements.n_rows << endl;

    uint tempN = 0;
    uint trueN = 0;
    uint tempL = flatElements.n_rows;
    uint trueL = uvElements.n_rows;
    bool all_resolved = false;
    while(trueN < trueL){
        if(uvElements(trueN) == flatElements(tempN)){
            //identical indexes occuring in
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN;
            trueN += 1;
            tempN += 1;
            if(tempN>=tempL){
                all_resolved = true;
                break;
            }
        }
        else{
            if(uvElements(trueN) < flatElements(tempN)){
                trueN += 1;
            }
            else{
                tempN += 1;
                if(tempN>=tempL){
                    all_resolved = true;
                    break;
                }
            }

        }
    }


    //if(all_resolved != true){
    if(tempN<tempL){
        uvec remaining(tempL-tempN);
        uint tN = 0;
        while(tempN<tempL){
            remaining(tN) = flatElements(tempN);
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN + tN;
            tempN += 1;
            tN += 1;
        }

        //cout << tempL << " "<< tempN << " " << remaining.n_elem << " " << uvElements.n_elem << endl;
        uvElements = join_cols<umat>(uvElements, remaining);
    }


    //lock and load
    uiCurrent_block += 1;
} //map all regions defined by L == R


mat amplitude::getblock(int u, int i){
    //umat block = fmBlocks(u)(i);
    mat block = vElements.elem(fmBlocks(u)(i));
    block.reshape(fmBlocks(u)(i).n_rows, fmBlocks(u)(i).n_cols);
    return block;
}

mat amplitude::getsblock(int u, int i){
    umat indblock = getfspBlock(i);

    //sp_mat<uint> fs = fspBlocks(i);
    //umat indblock(fs);

    mat block = vElements.elem(indblock);
    block.reshape(fspDims(i,0), fspDims(i,1));
    return block;
}

mat amplitude::getsblock_temp(int u, int i){
    umat indblock = getfspBlock(i);

    //sp_mat<uint> fs = fspBlocks(i);
    //umat indblock(fs);

    mat block = vElements_temp.elem(indblock);
    block.reshape(fspDims(i,0), fspDims(i,1));
    return block;
}

umat amplitude::getraw(int u, int i){
    //return index block, used for debugging and optimization
    return fmBlocks(u)(i);
}

umat amplitude::getraw_permuted(int u, int i, int n){
    //return index block, used for debugging and optimization
    umat aligned = fmBlocks(u)(i);
    if(n==0){
        aligned = aligned.rows(Pab(i));
    }
    if(n==1){
        aligned = aligned.rows(Pac(i));
    }
    if(n==2){
        aligned = aligned.rows(Pbc(i));
    }
    if(n==3){
        aligned = aligned.cols(Pij(i));
    }
    if(n==4){
        aligned = aligned.cols(Pik(i));
    }
    if(n==5){
        aligned = aligned.cols(Pjk(i));
    }
    return aligned;
}

mat amplitude::getblock_permuted(int u, int i, int n){
    //return index block, used for debugging and optimization

    umat aligned = fmBlocks(u)(i);
    if(n==0){
        aligned = aligned.rows(Pab(i));
    }
    if(n==1){
        aligned = aligned.rows(Pac(i));
    }
    if(n==2){
        aligned = aligned.rows(Pbc(i));
    }
    if(n==3){
        aligned = aligned.cols(Pij(i));
    }
    if(n==4){
        aligned = aligned.cols(Pik(i));
    }
    if(n==5){
        aligned = aligned.cols(Pjk(i));
    }
    if(n==6){
        aligned = aligned.rows(Pab(i));
        aligned = aligned.cols(Pij(i));
    }
    if(n==7){
        aligned = aligned.rows(Pab(i));
        aligned = aligned.cols(Pik(i));
    }
    if(n==8){
        aligned = aligned.rows(Pab(i));
        aligned = aligned.cols(Pjk(i));
    }
    if(n==9){
        aligned = aligned.rows(Pac(i));
        aligned = aligned.cols(Pij(i));
    }
    if(n==10){
        aligned = aligned.rows(Pac(i));
        aligned = aligned.cols(Pik(i));
    }
    if(n==11){
        aligned = aligned.rows(Pac(i));
        aligned = aligned.cols(Pjk(i));
    }
    if(n==12){
        aligned = aligned.rows(Pbc(i));
        aligned = aligned.cols(Pij(i));
    }
    if(n==13){
        aligned = aligned.rows(Pbc(i));
        aligned = aligned.cols(Pik(i));
    }
    if(n==14){
        aligned = aligned.rows(Pbc(i));
        aligned = aligned.cols(Pjk(i));
    }

    mat block = vElements.elem(aligned);
    block.reshape(fmBlocks(u)(i).n_rows, fmBlocks(u)(i).n_cols);

    return block;
}

umat amplitude::getfspBlock(uint i){
    uint Nx = fspDims(i,0);
    uint Ny = fspDims(i,1);
    umat block(Nx, Ny);
    //block*=0;
    block.fill(uiLastind);

    umat indices = fspBlocks(i);
    //indices.print();
    for(uint ni = 0; ni < indices.n_cols; ++ni){

        block(indices(1,ni), indices(2,ni)) = indices(0,ni);
    }

    return block;
}

mat amplitude::getsblock_permuted(int u, int i, int n){
    //return index block, used for debugging and optimization

    umat aligned = getfspBlock(i);
    // umat aligned = fmBlocks(u)(i);

    if(n==0){
        aligned = aligned.rows(Pab(i));
    }
    if(n==1){
        aligned = aligned.rows(Pac(i));
    }
    if(n==2){
        aligned = aligned.rows(Pbc(i));
    }
    if(n==3){
        aligned = aligned.cols(Pij(i));
    }
    if(n==4){
        aligned = aligned.cols(Pik(i));
    }
    if(n==5){
        aligned = aligned.cols(Pjk(i));
    }
    if(n==6){
        aligned = aligned.rows(Pab(i));
        aligned = aligned.cols(Pij(i));
    }
    if(n==7){
        aligned = aligned.rows(Pab(i));
        aligned = aligned.cols(Pik(i));
    }
    if(n==8){
        aligned = aligned.rows(Pab(i));
        aligned = aligned.cols(Pjk(i));
    }
    if(n==9){
        aligned = aligned.rows(Pac(i));
        aligned = aligned.cols(Pij(i));
    }
    if(n==10){
        aligned = aligned.rows(Pac(i));
        aligned = aligned.cols(Pik(i));
    }
    if(n==11){
        aligned = aligned.rows(Pac(i));
        aligned = aligned.cols(Pjk(i));
    }
    if(n==12){
        aligned = aligned.rows(Pbc(i));
        aligned = aligned.cols(Pij(i));
    }
    if(n==13){
        aligned = aligned.rows(Pbc(i));
        aligned = aligned.cols(Pik(i));
    }
    if(n==14){
        aligned = aligned.rows(Pbc(i));
        aligned = aligned.cols(Pjk(i));
    }

    mat block = vElements.elem(aligned);
    block.reshape(fspDims(i,0), fspDims(i,1));

    return block;
}

mat amplitude::getsblock_permuted_temp(int u, int i, int n){
    //return index block, used for debugging and optimization

    umat aligned = getfspBlock(i);
    // umat aligned = fmBlocks(u)(i);

    if(n==0){
        aligned = aligned.rows(Pab(i));
    }
    if(n==1){
        aligned = aligned.rows(Pac(i));
    }
    if(n==2){
        aligned = aligned.rows(Pbc(i));
    }
    if(n==3){
        aligned = aligned.cols(Pij(i));
    }
    if(n==4){
        aligned = aligned.cols(Pik(i));
    }
    if(n==5){
        aligned = aligned.cols(Pjk(i));
    }
    if(n==6){
        aligned = aligned.rows(Pab(i));
        aligned = aligned.cols(Pij(i));
    }
    if(n==7){
        aligned = aligned.rows(Pab(i));
        aligned = aligned.cols(Pik(i));
    }
    if(n==8){
        aligned = aligned.rows(Pab(i));
        aligned = aligned.cols(Pjk(i));
    }
    if(n==9){
        aligned = aligned.rows(Pac(i));
        aligned = aligned.cols(Pij(i));
    }
    if(n==10){
        aligned = aligned.rows(Pac(i));
        aligned = aligned.cols(Pik(i));
    }
    if(n==11){
        aligned = aligned.rows(Pac(i));
        aligned = aligned.cols(Pjk(i));
    }
    if(n==12){
        aligned = aligned.rows(Pbc(i));
        aligned = aligned.cols(Pij(i));
    }
    if(n==13){
        aligned = aligned.rows(Pbc(i));
        aligned = aligned.cols(Pik(i));
    }
    if(n==14){
        aligned = aligned.rows(Pbc(i));
        aligned = aligned.cols(Pjk(i));
    }

    mat block = vElements_temp.elem(aligned);
    block.reshape(fspDims(i,0), fspDims(i,1));

    return block;
}

mat amplitude::getspblock_permuted(umat aligned, uint i, int n){
    //return index block, used for debugging and optimization

    //umat aligned = getfspBlock(i);
    // umat aligned = fmBlocks(u)(i);

    if(n==0){
        aligned = aligned.rows(Pab(i));
    }
    if(n==1){
        aligned = aligned.rows(Pac(i));
    }
    if(n==2){
        aligned = aligned.rows(Pbc(i));
    }
    if(n==3){
        aligned = aligned.cols(Pij(i));
    }
    if(n==4){
        aligned = aligned.cols(Pik(i));
    }
    if(n==5){
        aligned = aligned.cols(Pjk(i));
    }
    if(n==6){
        aligned = aligned.rows(Pab(i));
        aligned = aligned.cols(Pij(i));
    }
    if(n==7){
        aligned = aligned.rows(Pab(i));
        aligned = aligned.cols(Pik(i));
    }
    if(n==8){
        aligned = aligned.rows(Pab(i));
        aligned = aligned.cols(Pjk(i));
    }
    if(n==9){
        aligned = aligned.rows(Pac(i));
        aligned = aligned.cols(Pij(i));
    }
    if(n==10){
        aligned = aligned.rows(Pac(i));
        aligned = aligned.cols(Pik(i));
    }
    if(n==11){
        aligned = aligned.rows(Pac(i));
        aligned = aligned.cols(Pjk(i));
    }
    if(n==12){
        aligned = aligned.rows(Pbc(i));
        aligned = aligned.cols(Pij(i));
    }
    if(n==13){
        aligned = aligned.rows(Pbc(i));
        aligned = aligned.cols(Pik(i));
    }
    if(n==14){
        aligned = aligned.rows(Pbc(i));
        aligned = aligned.cols(Pjk(i));
    }

    mat block = vElements.elem(aligned);
    block.reshape(fspDims(i,0), fspDims(i,1));

    return block;
}


void amplitude::setblock(int u, int i, mat mBlock){
    vElements.elem(fmBlocks(u)(i)) = vectorise(mBlock);
}

void amplitude::addblock_temp(int u, int i, mat mBlock){

    vElements_temp.elem(fmBlocks(u)(i)) += vectorise(mBlock);
}

void amplitude::addblock(int u, int i, mat mBlock){

    vElements.elem(fmBlocks(u)(i)) += vectorise(mBlock);
}

void amplitude::addsblock(int u, int i, mat mBlock){
    umat block = getfspBlock(i);

    vElements.elem(block) += vectorise(mBlock);
}


void amplitude::compress(){
    //store only unique amplitude values
    //experimental functionality

    //this will probably not work (why?)
    //do instead: - treat each set of blocks distinctly, especially first set is easily compressible (store only first element in row)
    vec uniqueElements = unique(vElements);
    uvec tempind;
    for(uint i = 0; i < uniqueElements.n_rows; ++i){
        tempind = find(vElements==uniqueElements(i));

    }
}


// ###########################################
// ##
// ##  SEQUENCES OF ROWS/COLUMNS
// ##
// ############################################

field<ivec> amplitude::pp(){
    //return a "compacted" particle particle unique indexvector

    //length of "compacted" vector
    uint N = Np*(Np+1)/2;

    //indices
    ivec a(N);
    ivec b(N);
    uint count = 0;
    //#pragma omp parallel for num_threads(4)
    for(int na = 0; na<Np; ++na){
        for(int nb = 0; nb<na+1; ++nb){
            a(count) = na;
            b(count) = nb;
            count += 1;
        }
    }

    ivec Kab = eBs.unique(conv_to<uvec>::from(a) +Nh) + eBs.unique(conv_to<uvec>::from(b)+Nh);

    field<ivec> ppmap(3);
    ppmap(0) = a;
    ppmap(1) = b;
    ppmap(2) = Kab;
    return ppmap;
    }

field<ivec> amplitude::hh_compact(){
    //return a "compacted" hole-hole unique indexvector

    //length of "compacted" vector
    uint N = Nh*(Nh+1)/2;

    //indices
    ivec a(N);
    ivec b(N);
    uint count = 0;
    for(int na = 0; na<Nh; ++na){
        for(int nb = 0; nb<na+1; ++nb){
            a(count) = na;
            b(count) = nb;
            count += 1;
        }
    }

    ivec Kab = eBs.unique(conv_to<uvec>::from(a) ) + eBs.unique(conv_to<uvec>::from(b));

    field<ivec> ppmap(3);
    ppmap(0) = a;
    ppmap(1) = b;
    ppmap(2) = Kab;
    return ppmap;
}

field<ivec> amplitude::ppp(ivec signs){
    //return a "compacted" particle particle particle unique indexvector
    //length of "compacted" vector
    uint N = Np*(Np+1)*(Np+2)/6;
    //indices
    ivec a(N);
    ivec b(N);
    ivec c(N);
    uint count = 0;
    for(int na = 0; na<Np; ++na){
        for(int nb = 0; nb<na+1; ++nb){
            for(int nc = 0; nc<nb+1; ++nc){
                a(count) = na;
                b(count) = nb;
                c(count) = nc;
                count += 1;
            }
        }
    }
    ivec Kabc = signs(0)*eBs.unique(conv_to<uvec>::from(a) +Nh) + signs(1)*eBs.unique(conv_to<uvec>::from(b)+Nh) + signs(2)*eBs.unique(conv_to<uvec>::from(c) +Nh);

    field<ivec> pppmap(4);
    pppmap(0) = a;
    pppmap(1) = b;
    pppmap(2) = c;
    pppmap(3) = Kabc;
    return pppmap;
}

field<ivec> amplitude::hpp(){
    //return a "compacted" particle particle particle unique indexvector
    //length of "compacted" vector
    uint N = Nh*Np*(Np+1)/2;
    //indices
    ivec i(N);
    ivec a(N);
    ivec b(N);
    uint count = 0;
    for(int ni = 0; ni<Nh; ++ni){
        for(int na = 0; na<Np; ++na){
            for(int nb = 0; nb<na+1; ++nb){
                i(count) = ni;
                a(count) = na;
                b(count) = nb;
                count += 1;
            }
        }
    }
    ivec Kiab = -eBs.unique(conv_to<uvec>::from(i)) + eBs.unique(conv_to<uvec>::from(a)+Nh) + eBs.unique(conv_to<uvec>::from(b) +Nh);
    field<ivec> pppmap(4);
    pppmap(0) = i;
    pppmap(1) = a;
    pppmap(2) = b;
    pppmap(3) = Kiab;
    return pppmap;
}

field<ivec> amplitude::php(){
    //return a "compacted" particle particle particle unique indexvector
    //length of "compacted" vector
    uint N = Nh*Np*(Np+1)/2;
    //indices
    ivec a(N);
    ivec i(N);
    ivec b(N);
    uint count = 0;
    for(int na = 0; na<Np; ++na){
        for(int ni = 0; ni<Nh; ++ni){
            for(int nb = 0; nb<na+1; ++nb){
                i(count) = ni;
                a(count) = na;
                b(count) = nb;
                count += 1;
            }
        }
    }
    ivec Kaib = -eBs.unique(conv_to<uvec>::from(i)) + eBs.unique(conv_to<uvec>::from(a)+Nh) + eBs.unique(conv_to<uvec>::from(b) +Nh);

    field<ivec> pppmap(4);
    pppmap(0) = a;
    pppmap(1) = i;
    pppmap(2) = b;
    pppmap(3) = Kaib;
    return pppmap;
}

field<ivec> amplitude::pph(){
    //return a "compacted" particle particle particle unique indexvector
    //length of "compacted" vector
    uint N = Nh*Np*(Np+1)/2;
    //indices
    ivec i(N);
    ivec a(N);
    ivec b(N);
    uint count = 0;
    for(int na = 0; na<Np; ++na){
        for(int nb = 0; nb<na+1; ++nb){
            for(int ni = 0; ni<Nh; ++ni){
                i(count) = ni;
                a(count) = na;
                b(count) = nb;
                count += 1;
            }
        }
    }
    ivec Kabi = - eBs.unique(conv_to<uvec>::from(i)) + eBs.unique(conv_to<uvec>::from(a)+Nh) + eBs.unique(conv_to<uvec>::from(b) +Nh);

    field<ivec> pppmap(4);
    pppmap(0) = a;
    pppmap(1) = b;
    pppmap(2) = i;
    pppmap(3) = Kabi;
    return pppmap;
}





field<ivec> amplitude::phh(){
    //return a "compacted" particle particle particle unique indexvector
    //length of "compacted" vector
    uint N = Np*Nh*(Nh+1)/2;
    //indices
    ivec i(N);
    ivec a(N);
    ivec b(N);
    uint count = 0;
    for(int ni = 0; ni<Np; ++ni){
        for(int na = 0; na<Nh; ++na){
            for(int nb = 0; nb<na+1; ++nb){
                i(count) = ni;
                a(count) = na;
                b(count) = nb;
                count += 1;
            }
        }
    }
    ivec Kiab = -eBs.unique(conv_to<uvec>::from(i)+Nh) + eBs.unique(conv_to<uvec>::from(a)) + eBs.unique(conv_to<uvec>::from(b));
    field<ivec> pppmap(4);
    pppmap(0) = i;
    pppmap(1) = a;
    pppmap(2) = b;
    pppmap(3) = Kiab;
    return pppmap;
}

field<ivec> amplitude::hph(){
    //return a "compacted" particle particle particle unique indexvector
    //length of "compacted" vector
    uint N = Np*Nh*(Nh+1)/2;
    //indices
    ivec a(N);
    ivec i(N);
    ivec b(N);
    uint count = 0;
    for(int na = 0; na<Nh; ++na){
        for(int ni = 0; ni<Np; ++ni){
            for(int nb = 0; nb<na+1; ++nb){
                i(count) = ni;
                a(count) = na;
                b(count) = nb;
                count += 1;
            }
        }
    }
    ivec Kaib = -eBs.unique(conv_to<uvec>::from(i)+Nh) + eBs.unique(conv_to<uvec>::from(a)) + eBs.unique(conv_to<uvec>::from(b));

    field<ivec> pppmap(4);
    pppmap(0) = a;
    pppmap(1) = i;
    pppmap(2) = b;
    pppmap(3) = Kaib;
    return pppmap;
}

field<ivec> amplitude::hhp(){
    //return a "compacted" particle particle particle unique indexvector
    //length of "compacted" vector
    uint N = Np*Nh*(Nh+1)/2;
    //indices
    ivec i(N);
    ivec a(N);
    ivec b(N);
    uint count = 0;
    for(int na = 0; na<Nh; ++na){
        for(int nb = 0; nb<na+1; ++nb){
            for(int ni = 0; ni<Np; ++ni){
                i(count) = ni;
                a(count) = na;
                b(count) = nb;
                count += 1;
            }
        }
    }
    ivec Kabi = - eBs.unique(conv_to<uvec>::from(i)+Nh) + eBs.unique(conv_to<uvec>::from(a)) + eBs.unique(conv_to<uvec>::from(b));

    field<ivec> pppmap(4);
    pppmap(0) = a;
    pppmap(1) = b;
    pppmap(2) = i;
    pppmap(3) = Kabi;
    return pppmap;
}

field<ivec> amplitude::hh(){
    //noncompacted
    ivec h = linspace<ivec>(0,Nh-1,Nh);
    field<ivec> hhmap(3);
    hhmap(0) = kron(ones<ivec>(Nh), h);
    hhmap(1) = kron(h, ones<ivec>(Nh));
    hhmap(2) = eBs.unique(conv_to<uvec>::from(hhmap(0))) + eBs.unique(conv_to<uvec>::from(hhmap(1)));
    return hhmap;
}

/*
field<ivec> amplitude::hhh(){
    //noncompacted
    ivec h = linspace<ivec>(0,Nh*Nh*Nh-1,Nh*Nh*Nh);
    field<ivec> hhhmap(4);

    hhhmap(2) = conv_to<ivec>::from(floor(h/(Nh*Nh))); //k
    hhhmap(1) = conv_to<ivec>::from(floor(((h - hhhmap(2)*Nh*Nh))/Nh));
    hhhmap(0) = conv_to<ivec>::from(h - hhhmap(2)*Nh*Nh - hhhmap(1)*Nh);

    hhhmap(3) = eBs.unique(conv_to<uvec>::from(hhhmap(0))) + eBs.unique(conv_to<uvec>::from(hhhmap(1)))+eBs.unique(conv_to<uvec>::from(hhhmap(2)));
    return hhhmap;
}*/

field<ivec> amplitude::hhh(){
    //return a "compacted" hole-hole-hole particle unique indexvector
    //length of "compacted" vector
    uint N = Nh*(Nh+1)*(Nh+2)/6;
    //indices
    ivec a(N);
    ivec b(N);
    ivec c(N);
    uint count = 0;
    for(int na = 0; na<Nh; ++na){
        for(int nb = 0; nb<na+1; ++nb){
            for(int nc = 0; nc<nb+1; ++nc){
                a(count) = na;
                b(count) = nb;
                c(count) = nc;
                count += 1;
            }
        }
    }
    ivec Kabc = eBs.unique(conv_to<uvec>::from(a)) + eBs.unique(conv_to<uvec>::from(b)) + eBs.unique(conv_to<uvec>::from(c));

    field<ivec> pppmap(4);
    pppmap(0) = a;
    pppmap(1) = b;
    pppmap(2) = c;
    pppmap(3) = Kabc;
    return pppmap;
}

field<ivec> amplitude::hp(){
    ivec h = linspace<ivec>(0,Nh-1,Nh);
    ivec p = linspace<ivec>(0,Np-1,Np);
    field<ivec> hhmap(3);
    hhmap(0) = kron(h, ones<ivec>(Np));
    hhmap(1) = kron(ones<ivec>(Nh), p);
    hhmap(2) = eBs.unique(conv_to<uvec>::from(hhmap(0))) + eBs.unique(conv_to<uvec>::from(hhmap(1))+Nh);
    return hhmap;
}

field<ivec> amplitude::ph(){
    ivec h = linspace<ivec>(0,Nh-1,Nh);
    ivec p = linspace<ivec>(0,Np-1,Np);
    field<ivec> hhmap(3);
    hhmap(0) = kron(p, ones<ivec>(Nh));
    hhmap(1) = kron(ones<ivec>(Np), h);
    hhmap(2) = eBs.unique(conv_to<uvec>::from(hhmap(0))+Nh) + eBs.unique(conv_to<uvec>::from(hhmap(1)));
    return hhmap;
}

// ##################################################
// ##                                              ##
// ## Block partition functions                           ##
// ##                                              ##
// ##################################################

field<uvec> amplitude::partition(ivec LHS, ivec K_unique){
    //identify i blocks of LHS where LHS==K(i)
    //Note that by "blocks", we actually means indices corresponding to preserved quantum numbers
    uvec l_sorted = sort_index(LHS);
    //bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);
    uvec row(uiStatAlloc); //arbitrarily large number (biggest possible block)
    uint nx = 0;
    int l_c= LHS(l_sorted(lc));
    field<uvec> tempRows(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    bool row_collect = false;
    //int ll_c = l_c;

    while(lc < uiS){
        l_c = LHS(l_sorted(lc));

        //if element belongs to block
        if(l_c == C){
            //collect element
            row(nx) = l_sorted(lc);
            nx += 1;
            row_collect = true;
        }
        //if row is complete
        else{
            //if row contains elements
            if(row_collect){
                //collect block
                tempRows(i) = sort(row(span(0,nx-1)));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }


        lc += 1;
        //cout << l_c - C << endl;
    }


    return tempRows;

}


field<uvec> amplitude::partition_pp(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(2));
    //bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(2)(l_sorted(lc));
    field<uvec> tempRows(uiN);
    Pab.set_size(uiN);
    //tempRows(uiN) = K_unique;
    //tempRows(0).set_size(uiN);
    //tempRows(1).set_size(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(2)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint a;
    uint b;
    //uint c;
    uint l_sorted_lc;
    uvec row(uiStatAlloc);


    uvec Na = conv_to<uvec>::from(LHS(0));
    uvec Nb = conv_to<uvec>::from(LHS(1));
    uvec pab(uiStatAlloc);

    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(2)(l_sorted_lc);

        if(l_c == C){
            a = Na(l_sorted_lc);
            b = Nb(l_sorted_lc);
            //c = Cn(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block


            //row_b(nx) = b;
            if(a!=b){
                row(nx) = a + b*Np;
                row(nx+1) = b + a*Np;
                pab(nx) = nx+1;
                pab(nx+1) = nx;


                nx += 2;

                //row_b(nx) = a;
            }
            else{
                row(nx) = a + b*Np;
                pab(nx) = nx;
                nx += 1;
            }
            //row(nx) = l_sorted(lc);
            //nx += 1;
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                uvec rr = sort(row(span(0,nx-1)));
                //uvec rr = sort_index(row);
                uvec row_rr = row(span(0,nx-1));
                uvec ab_index = pab(span(0,nx-1));
                tempRows(i) = row_rr.elem(rr);
                //tempRows(i) = row.elem(rr);
                Pab(i) = ab_index.elem(rr.elem(ab_index));
                //tempRows(1)(i) = row_b(span(0,nx-1));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    tempRows(i) = sort(row(span(0,nx-1)));
    return tempRows;
}

field<uvec> amplitude::partition_ppp(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(3));
    //bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(3)(l_sorted(lc));
    field<uvec> tempRows(uiN);
    //tempRows(uiN) = K_unique;
    //tempRows(0).set_size(uiN);
    //tempRows(1).set_size(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(3)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint p,q,r;

    uint l_sorted_lc;
    uvec row(uiStatAlloc);


    uvec Na = conv_to<uvec>::from(LHS(0));
    uvec Nb = conv_to<uvec>::from(LHS(1));
    uvec Nc = conv_to<uvec>::from(LHS(2));

    uint Np2 =Np*Np;
    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(3)(l_sorted_lc);

        if(l_c == C){
            p = Na(l_sorted_lc);
            q = Nb(l_sorted_lc);
            r = Nc(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block
            //a b c
            row(nx) = p + q*Np + r*Np2;
            nx += 1;
            if(r!=q){
                //b!=c
                row(nx) = p + r*Np + q*Np2;
                nx += 1;
                if(p!=q){
                    //a!=b (!=c)
                    row(nx) = q + p*Np + r*Np2;
                    nx += 1;
                    row(nx) = q + r*Np + p*Np2;
                    nx += 1;
                    //it follows that p!=r, so
                    row(nx) = r + p*Np + q*Np2;
                    nx += 1;
                    row(nx) = r + q*Np + p*Np2;
                    nx += 1;
                }
            }
            else{
                if(p!=q){
                    row(nx) = q + p*Np + r*Np2;
                    nx += 1;
                }
                if(p!=r){
                    row(nx) = r + q*Np + p*Np2;
                    nx += 1;
                }
            }
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                tempRows(i) = sort(row(span(0,nx-1)));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    //tempRows(i) = sort(row(span(0,nx-1))); //is this needed?
    return tempRows;
}

field<uvec> amplitude::partition_hh_permutations(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(2));
    //bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(2)(l_sorted(lc));
    field<uvec> tempRows(uiN);
    Pij.set_size(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(2)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    bool row_collect = false;
    uint a;
    uint b;
    uint l_sorted_lc;
    uvec row(uiStatAlloc);


    uvec Na = conv_to<uvec>::from(LHS(0));
    uvec Nb = conv_to<uvec>::from(LHS(1));
    uvec pab(uiStatAlloc);

    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(2)(l_sorted_lc);

        if(l_c == C){
            a = Na(l_sorted_lc);
            b = Nb(l_sorted_lc);
            //c = Cn(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block


            //row_b(nx) = b;
            if(a!=b){
                row(nx) = a + b*Nh;
                row(nx+1) = b + a*Nh;
                pab(nx) = nx+1;
                pab(nx+1) = nx;


                nx += 2;

                //row_b(nx) = a;
            }
            else{
                row(nx) = a + b*Nh;
                pab(nx) = nx;
                nx += 1;
            }
            //row(nx) = l_sorted(lc);
            //nx += 1;
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                uvec rr = sort_index(row(span(0,nx-1)));
                uvec r_inv = sort_index(rr);
                uvec row_rr = row(span(0,nx-1));
                uvec ab_index = pab(span(0,nx-1));
                tempRows(i) = row_rr.elem(rr);
                uvec ab_perm(ab_index.n_rows);
                for(uint h =0; h < ab_index.n_rows; ++h){
                    uint a_new = r_inv(h);
                    uint b_new = r_inv(ab_index(h));
                    ab_perm(a_new) = b_new;
                    ab_perm(b_new) = a_new;
                }
                Pij(i) = ab_perm;
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    //tempRows(i) = sort(row(span(0,nx-1)));
    if(nx>0){
        uvec rr = sort_index(row(span(0,nx-1)));
        uvec r_inv = sort_index(rr);
        uvec row_rr = row(span(0,nx-1));
        uvec ab_index = pab(span(0,nx-1));
        tempRows(i) = row_rr.elem(rr);
        uvec ab_perm(ab_index.n_rows);
        for(uint h =0; h < ab_index.n_rows; ++h){
            uint a_new = r_inv(h);
            uint b_new = r_inv(ab_index(h));
            ab_perm(a_new) = b_new;
            ab_perm(b_new) = a_new;
        }
        Pij(i) = ab_perm;}
    return tempRows;

}

field<uvec> amplitude::partition_pp_permutations(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(2));
    //bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(2)(l_sorted(lc));
    field<uvec> tempRows(uiN);
    Pab.set_size(uiN);
    //tempRows(uiN) = K_unique;
    //tempRows(0).set_size(uiN);
    //tempRows(1).set_size(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(2)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint a;
    uint b;
    //uint c;
    uint l_sorted_lc;
    uvec row(uiStatAlloc);


    uvec Na = conv_to<uvec>::from(LHS(0));
    uvec Nb = conv_to<uvec>::from(LHS(1));
    uvec pab(uiStatAlloc);

    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(2)(l_sorted_lc);

        if(l_c == C){
            a = Na(l_sorted_lc);
            b = Nb(l_sorted_lc);
            //c = Cn(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block


            //row_b(nx) = b;
            if(a!=b){
                row(nx) = a + b*Np;
                row(nx+1) = b + a*Np;
                pab(nx) = nx+1;
                pab(nx+1) = nx;


                nx += 2;

                //row_b(nx) = a;
            }
            else{
                row(nx) = a + b*Np;
                pab(nx) = nx;
                nx += 1;
            }
            //row(nx) = l_sorted(lc);
            //nx += 1;
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                uvec rr = sort_index(row(span(0,nx-1)));
                uvec r_inv = sort_index(rr);
                uvec row_rr = row(span(0,nx-1));
                uvec ab_index = pab(span(0,nx-1));
                tempRows(i) = row_rr.elem(rr);

                uvec ab_perm(ab_index.n_rows);
                for(uint h =0; h < ab_index.n_rows; ++h){
                    uint a_new = r_inv(h);
                    uint b_new = r_inv(ab_index(h));
                    ab_perm(a_new) = b_new;
                    ab_perm(b_new) = a_new;
                }

                Pab(i) = ab_perm;
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    //tempRows(i) = sort(row(span(0,nx-1)));
    if(nx>0){
        uvec rr = sort_index(row(span(0,nx-1)));
        uvec r_inv = sort_index(rr);
        uvec row_rr = row(span(0,nx-1));
        uvec ab_index = pab(span(0,nx-1));
        tempRows(i) = row_rr.elem(rr);

        uvec ab_perm(ab_index.n_rows);
        for(uint h =0; h < ab_index.n_rows; ++h){
            uint a_new = r_inv(h);
            uint b_new = r_inv(ab_index(h));
            ab_perm(a_new) = b_new;
            ab_perm(b_new) = a_new;
        }

        Pab(i) = ab_perm;}
    return tempRows;

}


field<uvec> amplitude::partition_ppp_permutations(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(3));
    //bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    Pab.set_size(uiN);
    Pac.set_size(uiN);
    Pbc.set_size(uiN);

    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(3)(l_sorted(lc));
    field<uvec> tempRows(uiN);
    //tempRows(uiN) = K_unique;
    //tempRows(0).set_size(uiN);
    //tempRows(1).set_size(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(3)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint a,b,c;

    uint l_sorted_lc;
    uvec row(uiStatAlloc);


    uvec Na = conv_to<uvec>::from(LHS(0));
    uvec Nb = conv_to<uvec>::from(LHS(1));
    uvec Nc = conv_to<uvec>::from(LHS(2));

    uvec permute_ab(uiStatAlloc);
    uvec permute_ac(uiStatAlloc);
    uvec permute_bc(uiStatAlloc);



    uint nx0 = 0;
    uint Np2 =Np*Np;
    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(3)(l_sorted_lc);

        if(l_c == C){
            a = Na(l_sorted_lc);
            b = Nb(l_sorted_lc);
            c = Nc(l_sorted_lc);
            if(a==b){
                if(a==c){
                    //(2)
                    row(nx) = a + b*Np + c*Np2;

                    permute_ab(nx) = nx;

                    permute_ac(nx) = nx;

                    permute_bc(nx) = nx;

                    nx += 1;

                }
                else{
                    //(3)
                    row(nx) = a + b*Np + c*Np2;
                    row(nx+1) = a + c*Np + b*Np2;
                    row(nx+2) = c + b*Np + a*Np2;

                    permute_ab(nx) = nx;
                    permute_ab(nx+1) = nx +2;
                    permute_ab(nx+2) = nx+1;

                    permute_ac(nx) = nx+2;
                    permute_ac(nx+1) = nx+1;
                    permute_ac(nx+2) = nx;

                    permute_bc(nx) = nx+1;
                    permute_bc(nx+1) = nx;
                    permute_bc(nx+2) = nx+2;



                    nx += 3;

                }
            }
            else{
                //a!=b
                if(a==c){
                    //(1)
                    row(nx) = a + b*Np + c*Np2;
                    row(nx+1) = b + a*Np + c*Np2;
                    row(nx+2) = a + c*Np + b*Np2;

                    permute_ab(nx) = nx+1;
                    permute_ab(nx+1) = nx;
                    permute_ab(nx+2) = nx+2;

                    permute_ac(nx) = nx;
                    permute_ac(nx+1) = nx+2;
                    permute_ac(nx+2) = nx+1;

                    permute_bc(nx) = nx+2;
                    permute_bc(nx+1) = nx+1;
                    permute_bc(nx+2) = nx;

                    nx += 3;



                }
                else{
                    if(b==c){
                        //(4)
                        row(nx) = a + b*Np + c*Np2;
                        row(nx+1) = b + a*Np + c*Np2;
                        row(nx+2) = c + b*Np + a*Np2;

                        permute_ab(nx) = nx+1;
                        permute_ab(nx+1) = nx;
                        permute_ab(nx+2) = nx+2;

                        permute_ac(nx) = nx+2;
                        permute_ac(nx+1) = nx+1;
                        permute_ac(nx+2) = nx;

                        permute_bc(nx) = nx;
                        permute_bc(nx+1) = nx+2;
                        permute_bc(nx+2) = nx+1;

                        nx += 3;

                    }
                    else{
                        //(5)
                        row(nx) =   a + b*Np + c*Np2;
                        row(nx+1) = b + a*Np + c*Np2;
                        row(nx+2) = a + c*Np + b*Np2;
                        row(nx+3) = c + a*Np + b*Np2;
                        row(nx+4) = b + c*Np + a*Np2;
                        row(nx+5) = c + b*Np + a*Np2;

                        permute_ab(nx) = nx+1;
                        permute_ab(nx+1) = nx;
                        permute_ab(nx+2) = nx+3;
                        permute_ab(nx+3) = nx+2;
                        permute_ab(nx+4) = nx+5;
                        permute_ab(nx+5) = nx+4;

                        permute_ac(nx) = nx+5;
                        permute_ac(nx+1) = nx+3;
                        permute_ac(nx+2) = nx+4;
                        permute_ac(nx+3) = nx+1;
                        permute_ac(nx+4) = nx+2;
                        permute_ac(nx+5) = nx;

                        permute_bc(nx) = nx+2;
                        permute_bc(nx+1) = nx+4;
                        permute_bc(nx+2) = nx;
                        permute_bc(nx+3) = nx+5;
                        permute_bc(nx+4) = nx+1;
                        permute_bc(nx+5) = nx+3;


                        nx += 6;

                    }
                }
            }


            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                uvec sorted = sort_index(row(span(0,nx-1)));
                uvec rr = row(span(0,nx-1));
                uvec pab = permute_ab(span(0,nx-1));
                uvec pac = permute_ac(span(0,nx-1));
                uvec pbc = permute_bc(span(0,nx-1));

                tempRows(i) = rr.elem(sorted);




                uvec r_inv = sort_index(sorted);
                uvec ab_perm(pab.n_rows);
                uvec ac_perm(pab.n_rows);
                uvec bc_perm(pab.n_rows);


                for(uint h =0; h < ab_perm.n_rows; ++h){
                    uint current = r_inv(h);

                    uint ab_new = r_inv(pab(h));
                    ab_perm(current) = ab_new;
                    ab_perm(ab_new) = current;

                    uint ac_new = r_inv(pac(h));
                    ac_perm(current) = ac_new;
                    ac_perm(ac_new) = current;

                    uint bc_new = r_inv(pbc(h));
                    bc_perm(current) = bc_new;
                    bc_perm(bc_new) = current;


                }

                Pab(i) = ab_perm; //pab.elem(sorted);
                Pac(i) = ac_perm; //pac.elem(sorted);
                Pbc(i) = bc_perm; //pbc.elem(sorted);

                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    //tempRows(i) = sort(row(span(0,nx-1))); //is this needed?
    if(nx>0){
        uvec sorted = sort_index(row(span(0,nx-1)));
        uvec rr = row(span(0,nx-1));
        uvec pab = permute_ab(span(0,nx-1));
        uvec pac = permute_ac(span(0,nx-1));
        uvec pbc = permute_bc(span(0,nx-1));

        tempRows(i) = rr.elem(sorted);

        uvec r_inv = sort_index(sorted);
        uvec ab_perm(pab.n_rows);
        uvec ac_perm(pab.n_rows);
        uvec bc_perm(pab.n_rows);


        for(uint h =0; h < ab_perm.n_rows; ++h){
            uint current = r_inv(h);

            uint ab_new = r_inv(pab(h));
            ab_perm(current) = ab_new;
            ab_perm(ab_new) = current;

            uint ac_new = r_inv(pac(h));
            ac_perm(current) = ac_new;
            ac_perm(ac_new) = current;

            uint bc_new = r_inv(pbc(h));
            bc_perm(current) = bc_new;
            bc_perm(bc_new) = current;


        }

        Pab(i) = ab_perm; //pab.elem(sorted);
        Pac(i) = ac_perm; //pac.elem(sorted);
        Pbc(i) = bc_perm; //pbc.elem(sorted);
    }


    return tempRows;
}

field<uvec> amplitude::partition_hhh_permutations(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(3));
    //bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    Pij.set_size(uiN);
    Pik.set_size(uiN);
    Pjk.set_size(uiN);

    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(3)(l_sorted(lc));
    field<uvec> tempRows(uiN);
    //tempRows(uiN) = K_unique;
    //tempRows(0).set_size(uiN);
    //tempRows(1).set_size(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(3)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint a,b,c;

    uint l_sorted_lc;
    uvec row(uiStatAlloc);



    uvec Ni = conv_to<uvec>::from(LHS(0));
    uvec Nj = conv_to<uvec>::from(LHS(1));
    uvec Nk = conv_to<uvec>::from(LHS(2));

    uvec permute_ij(uiStatAlloc);
    uvec permute_ik(uiStatAlloc);
    uvec permute_jk(uiStatAlloc);



    uint nx0 = 0;
    uint Nh2 =Nh*Nh;
    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(3)(l_sorted_lc);

        if(l_c == C){
            a = Ni(l_sorted_lc);
            b = Nj(l_sorted_lc);
            c = Nk(l_sorted_lc);
            if(a==b){
                if(a==c){
                    //(2)
                    row(nx) = a + b*Nh + c*Nh2;

                    permute_ij(nx) = nx;

                    permute_ik(nx) = nx;

                    permute_jk(nx) = nx;

                    nx += 1;

                }
                else{
                    //(3)
                    row(nx) = a + b*Nh + c*Nh2;
                    row(nx+1) = a + c*Nh + b*Nh2;
                    row(nx+2) = c + b*Nh + a*Nh2;

                    permute_ij(nx) = nx;
                    permute_ij(nx+1) = nx +2;
                    permute_ij(nx+2) = nx+1;

                    permute_ik(nx) = nx+2;
                    permute_ik(nx+1) = nx+1;
                    permute_ik(nx+2) = nx;

                    permute_jk(nx) = nx+1;
                    permute_jk(nx+1) = nx;
                    permute_jk(nx+2) = nx+2;



                    nx += 3;

                }
            }
            else{
                //a!=b
                if(a==c){
                    //(1)
                    row(nx) = a + b*Nh + c*Nh2;
                    row(nx+1) = b + a*Nh + c*Nh2;
                    row(nx+2) = a + c*Nh + b*Nh2;

                    permute_ij(nx) = nx+1;
                    permute_ij(nx+1) = nx;
                    permute_ij(nx+2) = nx+2;

                    permute_ik(nx) = nx;
                    permute_ik(nx+1) = nx+2;
                    permute_ik(nx+2) = nx+1;

                    permute_jk(nx) = nx+2;
                    permute_jk(nx+1) = nx+1;
                    permute_jk(nx+2) = nx;

                    nx += 3;



                }
                else{
                    if(b==c){
                        //(4)
                        row(nx) = a + b*Nh + c*Nh2;
                        row(nx+1) = b + a*Nh + c*Nh2;
                        row(nx+2) = c + b*Nh + a*Nh2;

                        permute_ij(nx) = nx+1;
                        permute_ij(nx+1) = nx;
                        permute_ij(nx+2) = nx+2;

                        permute_ik(nx) = nx+2;
                        permute_ik(nx+1) = nx+1;
                        permute_ik(nx+2) = nx;

                        permute_jk(nx) = nx;
                        permute_jk(nx+1) = nx+2;
                        permute_jk(nx+2) = nx+1;

                        nx += 3;

                    }
                    else{
                        //(5)
                        row(nx) =   a + b*Nh + c*Nh2;
                        row(nx+1) = b + a*Nh + c*Nh2;
                        row(nx+2) = a + c*Nh + b*Nh2;
                        row(nx+3) = c + a*Nh + b*Nh2;
                        row(nx+4) = b + c*Nh + a*Nh2;
                        row(nx+5) = c + b*Nh + a*Nh2;

                        permute_ij(nx) = nx+1;
                        permute_ij(nx+1) = nx;
                        permute_ij(nx+2) = nx+3;
                        permute_ij(nx+3) = nx+2;
                        permute_ij(nx+4) = nx+5;
                        permute_ij(nx+5) = nx+4;

                        permute_ik(nx) = nx+5;
                        permute_ik(nx+1) = nx+3;
                        permute_ik(nx+2) = nx+4;
                        permute_ik(nx+3) = nx+1;
                        permute_ik(nx+4) = nx+2;
                        permute_ik(nx+5) = nx;

                        permute_jk(nx) = nx+2;
                        permute_jk(nx+1) = nx+4;
                        permute_jk(nx+2) = nx;
                        permute_jk(nx+3) = nx+5;
                        permute_jk(nx+4) = nx+1;
                        permute_jk(nx+5) = nx+3;


                        nx += 6;

                    }
                }
            }


            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                uvec sorted = sort_index(row(span(0,nx-1)));
                uvec rr = row(span(0,nx-1));
                uvec pij = permute_ij(span(0,nx-1));
                uvec pik = permute_ik(span(0,nx-1));
                uvec pjk = permute_jk(span(0,nx-1));

                tempRows(i) = rr.elem(sorted);



                uvec r_inv = sort_index(sorted);
                uvec ab_perm(pij.n_rows);
                uvec ac_perm(pik.n_rows);
                uvec bc_perm(pjk.n_rows);


                for(uint h =0; h < ab_perm.n_rows; ++h){
                    uint current = r_inv(h);

                    uint ab_new = r_inv(pij(h));
                    ab_perm(current) = ab_new;
                    ab_perm(ab_new) = current;

                    uint ac_new = r_inv(pik(h));
                    ac_perm(current) = ac_new;
                    ac_perm(ac_new) = current;

                    uint bc_new = r_inv(pjk(h));
                    bc_perm(current) = bc_new;
                    bc_perm(bc_new) = current;


                }

                Pij(i) = ab_perm; //pab.elem(sorted);
                Pik(i) = ac_perm; //pac.elem(sorted);
                Pjk(i) = bc_perm; //pbc.elem(sorted);


                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    //tempRows(i) = sort(row(span(0,nx-1))); //is this needed?
    if(nx>0){
        uvec sorted = sort_index(row(span(0,nx-1)));
        uvec rr = row(span(0,nx-1));
        uvec pij = permute_ij(span(0,nx-1));
        uvec pik = permute_ik(span(0,nx-1));
        uvec pjk = permute_jk(span(0,nx-1));

        tempRows(i) = rr.elem(sorted);



        uvec r_inv = sort_index(sorted);
        uvec ab_perm(pij.n_rows);
        uvec ac_perm(pik.n_rows);
        uvec bc_perm(pjk.n_rows);


        for(uint h =0; h < ab_perm.n_rows; ++h){
            uint current = r_inv(h);

            uint ab_new = r_inv(pij(h));
            ab_perm(current) = ab_new;
            ab_perm(ab_new) = current;

            uint ac_new = r_inv(pik(h));
            ac_perm(current) = ac_new;
            ac_perm(ac_new) = current;

            uint bc_new = r_inv(pjk(h));
            bc_perm(current) = bc_new;
            bc_perm(bc_new) = current;


        }

        Pij(i) = ab_perm; //pab.elem(sorted);
        Pik(i) = ac_perm; //pac.elem(sorted);
        Pjk(i) = bc_perm; //pbc.elem(sorted);
    }
    return tempRows;
}

field<uvec> amplitude::partition_hpp(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(3));
    int Nhp = Np*Nh;
    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;
    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(3)(l_sorted(lc));
    field<uvec> tempRows(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(3)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint p,q,r;

    uint l_sorted_lc;
    uvec row(uiStatAlloc);



    uvec Ni = conv_to<uvec>::from(LHS(0));
    uvec Na = conv_to<uvec>::from(LHS(1));
    uvec Nb = conv_to<uvec>::from(LHS(2));

    uint Np2 =Np*Np;
    while(i < uiN){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(3)(l_sorted_lc);

        if(l_c == C){
            p = Ni(l_sorted_lc);
            q = Na(l_sorted_lc);
            r = Nb(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block
            row(nx) = p + q*Nh + r*Nhp;
            nx += 1;
            if(r!=q){
                row(nx) = p + r*Nh + q*Nhp;
                nx += 1;
            }
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                tempRows(i) = sort(row(span(0,nx-1)));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    if(nx>0){
        tempRows(i) = sort(row(span(0,nx-1)));
    }
    return tempRows;
}

field<uvec> amplitude::partition_php(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(3));
    int Nhp = Np*Nh;
    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;
    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(3)(l_sorted(lc));
    field<uvec> tempRows(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(3)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint p,q,r;

    uint l_sorted_lc;
    uvec row(uiStatAlloc);



    uvec Ni = conv_to<uvec>::from(LHS(0));
    uvec Na = conv_to<uvec>::from(LHS(1));
    uvec Nb = conv_to<uvec>::from(LHS(2));

    uint Np2 =Np*Np;
    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(3)(l_sorted_lc);

        if(l_c == C){
            p = Ni(l_sorted_lc);
            q = Na(l_sorted_lc);
            r = Nb(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block
            row(nx) = p + q*Np + r*Nhp;
            nx += 1;
            if(r!=p){
                row(nx) = r + q*Np + p*Nhp;
                nx += 1;
            }
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                tempRows(i) = sort(row(span(0,nx-1)));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    if(nx>0){
        tempRows(i) = sort(row(span(0,nx-1)));
    }
    return tempRows;
}

field<uvec> amplitude::partition_pph(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(3));
    int Nhp = Np*Nh;
    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;
    int C = K_unique(i);
    //K_unique.print();

    uint nx = 0;
    int l_c= LHS(3)(l_sorted(lc));
    field<uvec> tempRows(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(3)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint p,q,r;

    uint l_sorted_lc;
    uvec row(uiStatAlloc);



    uvec Ni = conv_to<uvec>::from(LHS(0));
    uvec Na = conv_to<uvec>::from(LHS(1));
    uvec Nb = conv_to<uvec>::from(LHS(2));

    uint Np2 =Np*Np;
    //while(lc < uiS){
    while(i < uiN){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(3)(l_sorted_lc);

        if(l_c == C){
            p = Ni(l_sorted_lc);
            q = Na(l_sorted_lc);
            r = Nb(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block
            row(nx) = p + q*Np + r*Np2;
            nx += 1;
            if(p!=q){
                row(nx) = q + p*Np + r*Np2;
                nx += 1;
            }
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                //cout << C;
                //cout << "     " << i << " : " << nx << " " << row(0) << " " << endl;
                tempRows(i) = sort(row(span(0,nx-1)));

                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    if(nx>0){
        tempRows(i) = sort(row(span(0,nx-1)));
    }
    return tempRows;
}


field<uvec> amplitude::partition_phh(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(3));
    int Nhp = Np*Nh;
    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;
    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(3)(l_sorted(lc));
    field<uvec> tempRows(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(3)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint p,q,r;

    uint l_sorted_lc;
    uvec row(uiStatAlloc);



    uvec Ni = conv_to<uvec>::from(LHS(0));
    uvec Na = conv_to<uvec>::from(LHS(1));
    uvec Nb = conv_to<uvec>::from(LHS(2));

    uint Nh2 =Nh*Nh;
    //while(lc < uiS){
    while(i < uiN){

        l_sorted_lc = l_sorted(lc);
        l_c = LHS(3)(l_sorted_lc);

        if(l_c == C){
            p = Ni(l_sorted_lc);
            q = Na(l_sorted_lc);
            r = Nb(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block
            row(nx) = p + q*Np + r*Nhp;
            nx += 1;
            if(r!=q){
                row(nx) = p + r*Np + q*Nhp;
                nx += 1;
            }
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                tempRows(i) = sort(row(span(0,nx-1)));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    if(nx>0){
        tempRows(i) = sort(row(span(0,nx-1)));
    }
    return tempRows;
}

field<uvec> amplitude::partition_hph(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(3));
    int Nhp = Np*Nh;
    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;
    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(3)(l_sorted(lc));
    field<uvec> tempRows(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(3)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint p,q,r;

    uint l_sorted_lc;
    uvec row(uiStatAlloc);



    uvec Ni = conv_to<uvec>::from(LHS(0));
    uvec Na = conv_to<uvec>::from(LHS(1));
    uvec Nb = conv_to<uvec>::from(LHS(2));

    uint Nh2 =Nh*Nh;
    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(3)(l_sorted_lc);

        if(l_c == C){
            p = Ni(l_sorted_lc);
            q = Na(l_sorted_lc);
            r = Nb(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block
            row(nx) = p + q*Np + r*Nhp;
            nx += 1;
            if(r!=p){
                row(nx) = r + q*Np + p*Nhp;
                nx += 1;
            }
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                tempRows(i) = sort(row(span(0,nx-1)));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    if(nx>0){
        tempRows(i) = sort(row(span(0,nx-1)));
    }
    return tempRows;
}

field<uvec> amplitude::partition_hhp(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(3));
    int Nhp = Np*Nh;
    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;
    int C = K_unique(i);

    //K_unique.print();

    uint nx = 0;
    int l_c= LHS(3)(l_sorted(lc));
    field<uvec> tempRows(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(3)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint p,q,r;

    uint l_sorted_lc;
    uvec row(uiStatAlloc);



    uvec Ni = conv_to<uvec>::from(LHS(0));
    uvec Na = conv_to<uvec>::from(LHS(1));
    uvec Nb = conv_to<uvec>::from(LHS(2));

    uint Nh2 =Nh*Nh;
    while(i < uiN){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(3)(l_sorted_lc);

        if(l_c == C){
            p = Ni(l_sorted_lc);
            q = Na(l_sorted_lc);
            r = Nb(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block
            row(nx) = p + q*Nh + r*Nh2;
            nx += 1;
            if(p!=q){
                row(nx) = q + p*Nh + r*Nh2;
                nx += 1;
            }
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                //cout << C << endl;
                tempRows(i) = sort(row(span(0,nx-1)));
                lc -= 1;
                i += 1;
                nx = 0;

                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    if(nx>0){
        tempRows(i) = sort(row(span(0,nx-1)));
    }
    return tempRows;
}
