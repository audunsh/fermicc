#include "flexmat.h"
#define ARMA_64BIT_WORD
#include <armadillo>

using namespace std;
using namespace arma;


flexmat::flexmat(vec values, uvec p, uvec q, uvec r, uvec s, int Np, int Nq, int Nr, int Ns)
{
    iNp = Np;
    iNq = Nq;
    iNr = Nr;
    iNs = Ns;
    /*iNp2 = Np*Np;
    iNh2 = Nh*Nh;
    iNhp = Nh*Np;
    iNp2h = iNp2*iNh;
    iNh2p = iNh2*iNp;

    */
    vValues = values;
    vp = p;
    vq = q;
    vr = r;
    vs = s;

}

void flexmat::update(){
    //update object with a new sparse matrix
}


//The following flex-routines is maintaned from a separate python-script.

sp_mat flexmat::pq_rs(){
    if(Npq_rs == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vp + vq*iNp;
        locations.col(1) = vr + vs*iNr;
        Vpq_rs = sp_mat(locations.t(), vValues, iNp*iNq, iNr*iNs);
        Npq_rs = 1;
        return Vpq_rs;
    }
    else{
        return Vpq_rs;
    }
}
sp_mat flexmat::pq_sr(){
    if(Npq_sr == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vp + vq*iNp;
        locations.col(1) = vs + vr*iNs;
        Vpq_sr = sp_mat(locations.t(), vValues, iNp*iNq, iNs*iNr);
        Npq_sr = 1;
        return Vpq_sr;
    }
    else{
        return Vpq_sr;
    }
}
sp_mat flexmat::pr_qs(){
    if(Npr_qs == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vp + vr*iNp;
        locations.col(1) = vq + vs*iNq;
        Vpr_qs = sp_mat(locations.t(), vValues, iNp*iNr, iNq*iNs);
        Npr_qs = 1;
        return Vpr_qs;
    }
    else{
        return Vpr_qs;
    }
}
sp_mat flexmat::pr_sq(){
    if(Npr_sq == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vp + vr*iNp;
        locations.col(1) = vs + vq*iNs;
        Vpr_sq = sp_mat(locations.t(), vValues, iNp*iNr, iNs*iNq);
        Npr_sq = 1;
        return Vpr_sq;
    }
    else{
        return Vpr_sq;
    }
}
sp_mat flexmat::ps_qr(){
    if(Nps_qr == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vp + vs*iNp;
        locations.col(1) = vq + vr*iNq;
        Vps_qr = sp_mat(locations.t(), vValues, iNp*iNs, iNq*iNr);
        Nps_qr = 1;
        return Vps_qr;
    }
    else{
        return Vps_qr;
    }
}
sp_mat flexmat::ps_rq(){
    if(Nps_rq == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vp + vs*iNp;
        locations.col(1) = vr + vq*iNr;
        Vps_rq = sp_mat(locations.t(), vValues, iNp*iNs, iNr*iNq);
        Nps_rq = 1;
        return Vps_rq;
    }
    else{
        return Vps_rq;
    }
}
sp_mat flexmat::qp_rs(){
    if(Nqp_rs == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vq + vp*iNq;
        locations.col(1) = vr + vs*iNr;
        Vqp_rs = sp_mat(locations.t(), vValues, iNq*iNp, iNr*iNs);
        Nqp_rs = 1;
        return Vqp_rs;
    }
    else{
        return Vqp_rs;
    }
}
sp_mat flexmat::qp_sr(){
    if(Nqp_sr == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vq + vp*iNq;
        locations.col(1) = vs + vr*iNs;
        Vqp_sr = sp_mat(locations.t(), vValues, iNq*iNp, iNs*iNr);
        Nqp_sr = 1;
        return Vqp_sr;
    }
    else{
        return Vqp_sr;
    }
}
sp_mat flexmat::qr_ps(){
    if(Nqr_ps == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vq + vr*iNq;
        locations.col(1) = vp + vs*iNp;
        Vqr_ps = sp_mat(locations.t(), vValues, iNq*iNr, iNp*iNs);
        Nqr_ps = 1;
        return Vqr_ps;
    }
    else{
        return Vqr_ps;
    }
}
sp_mat flexmat::qr_sp(){
    if(Nqr_sp == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vq + vr*iNq;
        locations.col(1) = vs + vp*iNs;
        Vqr_sp = sp_mat(locations.t(), vValues, iNq*iNr, iNs*iNp);
        Nqr_sp = 1;
        return Vqr_sp;
    }
    else{
        return Vqr_sp;
    }
}
sp_mat flexmat::qs_pr(){
    if(Nqs_pr == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vq + vs*iNq;
        locations.col(1) = vp + vr*iNp;
        Vqs_pr = sp_mat(locations.t(), vValues, iNq*iNs, iNp*iNr);
        Nqs_pr = 1;
        return Vqs_pr;
    }
    else{
        return Vqs_pr;
    }
}
sp_mat flexmat::qs_rp(){
    if(Nqs_rp == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vq + vs*iNq;
        locations.col(1) = vr + vp*iNr;
        Vqs_rp = sp_mat(locations.t(), vValues, iNq*iNs, iNr*iNp);
        Nqs_rp = 1;
        return Vqs_rp;
    }
    else{
        return Vqs_rp;
    }
}
sp_mat flexmat::rp_qs(){
    if(Nrp_qs == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vr + vp*iNr;
        locations.col(1) = vq + vs*iNq;
        Vrp_qs = sp_mat(locations.t(), vValues, iNr*iNp, iNq*iNs);
        Nrp_qs = 1;
        return Vrp_qs;
    }
    else{
        return Vrp_qs;
    }
}
sp_mat flexmat::rp_sq(){
    if(Nrp_sq == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vr + vp*iNr;
        locations.col(1) = vs + vq*iNs;
        Vrp_sq = sp_mat(locations.t(), vValues, iNr*iNp, iNs*iNq);
        Nrp_sq = 1;
        return Vrp_sq;
    }
    else{
        return Vrp_sq;
    }
}
sp_mat flexmat::rq_ps(){
    if(Nrq_ps == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vr + vq*iNr;
        locations.col(1) = vp + vs*iNp;
        Vrq_ps = sp_mat(locations.t(), vValues, iNr*iNq, iNp*iNs);
        Nrq_ps = 1;
        return Vrq_ps;
    }
    else{
        return Vrq_ps;
    }
}
sp_mat flexmat::rq_sp(){
    if(Nrq_sp == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vr + vq*iNr;
        locations.col(1) = vs + vp*iNs;
        Vrq_sp = sp_mat(locations.t(), vValues, iNr*iNq, iNs*iNp);
        Nrq_sp = 1;
        return Vrq_sp;
    }
    else{
        return Vrq_sp;
    }
}
sp_mat flexmat::rs_pq(){
    if(Nrs_pq == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vr + vs*iNr;
        locations.col(1) = vp + vq*iNp;
        Vrs_pq = sp_mat(locations.t(), vValues, iNr*iNs, iNp*iNq);
        Nrs_pq = 1;
        return Vrs_pq;
    }
    else{
        return Vrs_pq;
    }
}
sp_mat flexmat::rs_qp(){
    if(Nrs_qp == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vr + vs*iNr;
        locations.col(1) = vq + vp*iNq;
        Vrs_qp = sp_mat(locations.t(), vValues, iNr*iNs, iNq*iNp);
        Nrs_qp = 1;
        return Vrs_qp;
    }
    else{
        return Vrs_qp;
    }
}
sp_mat flexmat::sp_qr(){
    if(Nsp_qr == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vs + vp*iNs;
        locations.col(1) = vq + vr*iNq;
        Vsp_qr = sp_mat(locations.t(), vValues, iNs*iNp, iNq*iNr);
        Nsp_qr = 1;
        return Vsp_qr;
    }
    else{
        return Vsp_qr;
    }
}
sp_mat flexmat::sp_rq(){
    if(Nsp_rq == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vs + vp*iNs;
        locations.col(1) = vr + vq*iNr;
        Vsp_rq = sp_mat(locations.t(), vValues, iNs*iNp, iNr*iNq);
        Nsp_rq = 1;
        return Vsp_rq;
    }
    else{
        return Vsp_rq;
    }
}
sp_mat flexmat::sq_pr(){
    if(Nsq_pr == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vs + vq*iNs;
        locations.col(1) = vp + vr*iNp;
        Vsq_pr = sp_mat(locations.t(), vValues, iNs*iNq, iNp*iNr);
        Nsq_pr = 1;
        return Vsq_pr;
    }
    else{
        return Vsq_pr;
    }
}
sp_mat flexmat::sq_rp(){
    if(Nsq_rp == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vs + vq*iNs;
        locations.col(1) = vr + vp*iNr;
        Vsq_rp = sp_mat(locations.t(), vValues, iNs*iNq, iNr*iNp);
        Nsq_rp = 1;
        return Vsq_rp;
    }
    else{
        return Vsq_rp;
    }
}
sp_mat flexmat::sr_pq(){
    if(Nsr_pq == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vs + vr*iNs;
        locations.col(1) = vp + vq*iNp;
        Vsr_pq = sp_mat(locations.t(), vValues, iNs*iNr, iNp*iNq);
        Nsr_pq = 1;
        return Vsr_pq;
    }
    else{
        return Vsr_pq;
    }
}
sp_mat flexmat::sr_qp(){
    if(Nsr_qp == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vs + vr*iNs;
        locations.col(1) = vq + vp*iNq;
        Vsr_qp = sp_mat(locations.t(), vValues, iNs*iNr, iNq*iNp);
        Nsr_qp = 1;
        return Vsr_qp;
    }
    else{
        return Vsr_qp;
    }
}
sp_mat flexmat::p_qrs(){
    if(Np_qrs == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vp;
        locations.col(1) = vq + vr*iNq + vs*iNq*iNr;
        Vp_qrs = sp_mat(locations.t(), vValues, iNp, iNq*iNr*iNs);
        Np_qrs = 1;
        return Vp_qrs;
    }
    else{
        return Vp_qrs;
    }
}
sp_mat flexmat::p_qsr(){
    if(Np_qsr == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vp;
        locations.col(1) = vq + vs*iNq + vr*iNq*iNs;
        Vp_qsr = sp_mat(locations.t(), vValues, iNp, iNq*iNs*iNr);
        Np_qsr = 1;
        return Vp_qsr;
    }
    else{
        return Vp_qsr;
    }
}
sp_mat flexmat::p_rqs(){
    if(Np_rqs == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vp;
        locations.col(1) = vr + vq*iNr + vs*iNr*iNq;
        Vp_rqs = sp_mat(locations.t(), vValues, iNp, iNr*iNq*iNs);
        Np_rqs = 1;
        return Vp_rqs;
    }
    else{
        return Vp_rqs;
    }
}
sp_mat flexmat::p_rsq(){
    if(Np_rsq == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vp;
        locations.col(1) = vr + vs*iNr + vq*iNr*iNs;
        Vp_rsq = sp_mat(locations.t(), vValues, iNp, iNr*iNs*iNq);
        Np_rsq = 1;
        return Vp_rsq;
    }
    else{
        return Vp_rsq;
    }
}
sp_mat flexmat::p_sqr(){
    if(Np_sqr == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vp;
        locations.col(1) = vs + vq*iNs + vr*iNs*iNq;
        Vp_sqr = sp_mat(locations.t(), vValues, iNp, iNs*iNq*iNr);
        Np_sqr = 1;
        return Vp_sqr;
    }
    else{
        return Vp_sqr;
    }
}
sp_mat flexmat::p_srq(){
    if(Np_srq == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vp;
        locations.col(1) = vs + vr*iNs + vq*iNs*iNr;
        Vp_srq = sp_mat(locations.t(), vValues, iNp, iNs*iNr*iNq);
        Np_srq = 1;
        return Vp_srq;
    }
    else{
        return Vp_srq;
    }
}
sp_mat flexmat::q_prs(){
    if(Nq_prs == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vq;
        locations.col(1) = vp + vr*iNp + vs*iNp*iNr;
        Vq_prs = sp_mat(locations.t(), vValues, iNq, iNp*iNr*iNs);
        Nq_prs = 1;
        return Vq_prs;
    }
    else{
        return Vq_prs;
    }
}
sp_mat flexmat::q_psr(){
    if(Nq_psr == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vq;
        locations.col(1) = vp + vs*iNp + vr*iNp*iNs;
        Vq_psr = sp_mat(locations.t(), vValues, iNq, iNp*iNs*iNr);
        Nq_psr = 1;
        return Vq_psr;
    }
    else{
        return Vq_psr;
    }
}
sp_mat flexmat::q_rps(){
    if(Nq_rps == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vq;
        locations.col(1) = vr + vp*iNr + vs*iNr*iNp;
        Vq_rps = sp_mat(locations.t(), vValues, iNq, iNr*iNp*iNs);
        Nq_rps = 1;
        return Vq_rps;
    }
    else{
        return Vq_rps;
    }
}
sp_mat flexmat::q_rsp(){
    if(Nq_rsp == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vq;
        locations.col(1) = vr + vs*iNr + vp*iNr*iNs;
        Vq_rsp = sp_mat(locations.t(), vValues, iNq, iNr*iNs*iNp);
        Nq_rsp = 1;
        return Vq_rsp;
    }
    else{
        return Vq_rsp;
    }
}
sp_mat flexmat::q_spr(){
    if(Nq_spr == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vq;
        locations.col(1) = vs + vp*iNs + vr*iNs*iNp;
        Vq_spr = sp_mat(locations.t(), vValues, iNq, iNs*iNp*iNr);
        Nq_spr = 1;
        return Vq_spr;
    }
    else{
        return Vq_spr;
    }
}
sp_mat flexmat::q_srp(){
    if(Nq_srp == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vq;
        locations.col(1) = vs + vr*iNs + vp*iNs*iNr;
        Vq_srp = sp_mat(locations.t(), vValues, iNq, iNs*iNr*iNp);
        Nq_srp = 1;
        return Vq_srp;
    }
    else{
        return Vq_srp;
    }
}
sp_mat flexmat::r_pqs(){
    if(Nr_pqs == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vr;
        locations.col(1) = vp + vq*iNp + vs*iNp*iNq;
        Vr_pqs = sp_mat(locations.t(), vValues, iNr, iNp*iNq*iNs);
        Nr_pqs = 1;
        return Vr_pqs;
    }
    else{
        return Vr_pqs;
    }
}
sp_mat flexmat::r_psq(){
    if(Nr_psq == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vr;
        locations.col(1) = vp + vs*iNp + vq*iNp*iNs;
        Vr_psq = sp_mat(locations.t(), vValues, iNr, iNp*iNs*iNq);
        Nr_psq = 1;
        return Vr_psq;
    }
    else{
        return Vr_psq;
    }
}
sp_mat flexmat::r_qps(){
    if(Nr_qps == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vr;
        locations.col(1) = vq + vp*iNq + vs*iNq*iNp;
        Vr_qps = sp_mat(locations.t(), vValues, iNr, iNq*iNp*iNs);
        Nr_qps = 1;
        return Vr_qps;
    }
    else{
        return Vr_qps;
    }
}
sp_mat flexmat::r_qsp(){
    if(Nr_qsp == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vr;
        locations.col(1) = vq + vs*iNq + vp*iNq*iNs;
        Vr_qsp = sp_mat(locations.t(), vValues, iNr, iNq*iNs*iNp);
        Nr_qsp = 1;
        return Vr_qsp;
    }
    else{
        return Vr_qsp;
    }
}
sp_mat flexmat::r_spq(){
    if(Nr_spq == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vr;
        locations.col(1) = vs + vp*iNs + vq*iNs*iNp;
        Vr_spq = sp_mat(locations.t(), vValues, iNr, iNs*iNp*iNq);
        Nr_spq = 1;
        return Vr_spq;
    }
    else{
        return Vr_spq;
    }
}
sp_mat flexmat::r_sqp(){
    if(Nr_sqp == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vr;
        locations.col(1) = vs + vq*iNs + vp*iNs*iNq;
        Vr_sqp = sp_mat(locations.t(), vValues, iNr, iNs*iNq*iNp);
        Nr_sqp = 1;
        return Vr_sqp;
    }
    else{
        return Vr_sqp;
    }
}
sp_mat flexmat::s_pqr(){
    if(Ns_pqr == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vs;
        locations.col(1) = vp + vq*iNp + vr*iNp*iNq;
        Vs_pqr = sp_mat(locations.t(), vValues, iNs, iNp*iNq*iNr);
        Ns_pqr = 1;
        return Vs_pqr;
    }
    else{
        return Vs_pqr;
    }
}
sp_mat flexmat::s_prq(){
    if(Ns_prq == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vs;
        locations.col(1) = vp + vr*iNp + vq*iNp*iNr;
        Vs_prq = sp_mat(locations.t(), vValues, iNs, iNp*iNr*iNq);
        Ns_prq = 1;
        return Vs_prq;
    }
    else{
        return Vs_prq;
    }
}
sp_mat flexmat::s_qpr(){
    if(Ns_qpr == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vs;
        locations.col(1) = vq + vp*iNq + vr*iNq*iNp;
        Vs_qpr = sp_mat(locations.t(), vValues, iNs, iNq*iNp*iNr);
        Ns_qpr = 1;
        return Vs_qpr;
    }
    else{
        return Vs_qpr;
    }
}
sp_mat flexmat::s_qrp(){
    if(Ns_qrp == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vs;
        locations.col(1) = vq + vr*iNq + vp*iNq*iNr;
        Vs_qrp = sp_mat(locations.t(), vValues, iNs, iNq*iNr*iNp);
        Ns_qrp = 1;
        return Vs_qrp;
    }
    else{
        return Vs_qrp;
    }
}
sp_mat flexmat::s_rpq(){
    if(Ns_rpq == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vs;
        locations.col(1) = vr + vp*iNr + vq*iNr*iNp;
        Vs_rpq = sp_mat(locations.t(), vValues, iNs, iNr*iNp*iNq);
        Ns_rpq = 1;
        return Vs_rpq;
    }
    else{
        return Vs_rpq;
    }
}
sp_mat flexmat::s_rqp(){
    if(Ns_rqp == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vs;
        locations.col(1) = vr + vq*iNr + vp*iNr*iNq;
        Vs_rqp = sp_mat(locations.t(), vValues, iNs, iNr*iNq*iNp);
        Ns_rqp = 1;
        return Vs_rqp;
    }
    else{
        return Vs_rqp;
    }
}

sp_mat flexmat::pqr_s(){
    if(Npqr_s == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vp + vq*iNp + vr*iNp*iNq;
        locations.col(1) = vs;
        Vpqr_s = sp_mat(locations.t(), vValues, iNp*iNq*iNr,iNs);
        Npqr_s = 1;
        return Vpqr_s;
    }
    else{
        return Vpqr_s;
    }
}

sp_mat flexmat::pqs_r(){
    if(Npqs_r == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vp + vq*iNp + vs*iNp*iNq;
        locations.col(1) = vr;
        Vpqs_r = sp_mat(locations.t(), vValues, iNp*iNq*iNs,iNr);
        Npqs_r = 1;
        return Vpqs_r;
    }
    else{
        return Vpqs_r;
    }
}

sp_mat flexmat::prq_s(){
    if(Nprq_s == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vp + vr*iNp + vq*iNp*iNr;
        locations.col(1) = vs;
        Vprq_s = sp_mat(locations.t(), vValues, iNp*iNr*iNq,iNs);
        Nprq_s = 1;
        return Vprq_s;
    }
    else{
        return Vprq_s;
    }
}

sp_mat flexmat::prs_q(){
    if(Nprs_q == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vp + vr*iNp + vs*iNp*iNr;
        locations.col(1) = vq;
        Vprs_q = sp_mat(locations.t(), vValues, iNp*iNr*iNs,iNq);
        Nprs_q = 1;
        return Vprs_q;
    }
    else{
        return Vprs_q;
    }
}

sp_mat flexmat::psq_r(){
    if(Npsq_r == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vp + vs*iNp + vq*iNp*iNs;
        locations.col(1) = vr;
        Vpsq_r = sp_mat(locations.t(), vValues, iNp*iNs*iNq,iNr);
        Npsq_r = 1;
        return Vpsq_r;
    }
    else{
        return Vpsq_r;
    }
}

sp_mat flexmat::psr_q(){
    if(Npsr_q == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vp + vs*iNp + vr*iNp*iNs;
        locations.col(1) = vq;
        Vpsr_q = sp_mat(locations.t(), vValues, iNp*iNs*iNr,iNq);
        Npsr_q = 1;
        return Vpsr_q;
    }
    else{
        return Vpsr_q;
    }
}

sp_mat flexmat::qpr_s(){
    if(Nqpr_s == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vq + vp*iNq + vr*iNq*iNp;
        locations.col(1) = vs;
        Vqpr_s = sp_mat(locations.t(), vValues, iNq*iNp*iNr,iNs);
        Nqpr_s = 1;
        return Vqpr_s;
    }
    else{
        return Vqpr_s;
    }
}

sp_mat flexmat::qps_r(){
    if(Nqps_r == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vq + vp*iNq + vs*iNq*iNp;
        locations.col(1) = vr;
        Vqps_r = sp_mat(locations.t(), vValues, iNq*iNp*iNs,iNr);
        Nqps_r = 1;
        return Vqps_r;
    }
    else{
        return Vqps_r;
    }
}

sp_mat flexmat::qrp_s(){
    if(Nqrp_s == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vq + vr*iNq + vp*iNq*iNr;
        locations.col(1) = vs;
        Vqrp_s = sp_mat(locations.t(), vValues, iNq*iNr*iNp,iNs);
        Nqrp_s = 1;
        return Vqrp_s;
    }
    else{
        return Vqrp_s;
    }
}

sp_mat flexmat::qrs_p(){
    if(Nqrs_p == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vq + vr*iNq + vs*iNq*iNr;
        locations.col(1) = vp;
        Vqrs_p = sp_mat(locations.t(), vValues, iNq*iNr*iNs,iNp);
        Nqrs_p = 1;
        return Vqrs_p;
    }
    else{
        return Vqrs_p;
    }
}

sp_mat flexmat::qsp_r(){
    if(Nqsp_r == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vq + vs*iNq + vp*iNq*iNs;
        locations.col(1) = vr;
        Vqsp_r = sp_mat(locations.t(), vValues, iNq*iNs*iNp,iNr);
        Nqsp_r = 1;
        return Vqsp_r;
    }
    else{
        return Vqsp_r;
    }
}

sp_mat flexmat::qsr_p(){
    if(Nqsr_p == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vq + vs*iNq + vr*iNq*iNs;
        locations.col(1) = vp;
        Vqsr_p = sp_mat(locations.t(), vValues, iNq*iNs*iNr,iNp);
        Nqsr_p = 1;
        return Vqsr_p;
    }
    else{
        return Vqsr_p;
    }
}

sp_mat flexmat::rpq_s(){
    if(Nrpq_s == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vr + vp*iNr + vq*iNr*iNp;
        locations.col(1) = vs;
        Vrpq_s = sp_mat(locations.t(), vValues, iNr*iNp*iNq,iNs);
        Nrpq_s = 1;
        return Vrpq_s;
    }
    else{
        return Vrpq_s;
    }
}

sp_mat flexmat::rps_q(){
    if(Nrps_q == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vr + vp*iNr + vs*iNr*iNp;
        locations.col(1) = vq;
        Vrps_q = sp_mat(locations.t(), vValues, iNr*iNp*iNs,iNq);
        Nrps_q = 1;
        return Vrps_q;
    }
    else{
        return Vrps_q;
    }
}

sp_mat flexmat::rqp_s(){
    if(Nrqp_s == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vr + vq*iNr + vp*iNr*iNq;
        locations.col(1) = vs;
        Vrqp_s = sp_mat(locations.t(), vValues, iNr*iNq*iNp,iNs);
        Nrqp_s = 1;
        return Vrqp_s;
    }
    else{
        return Vrqp_s;
    }
}

sp_mat flexmat::rqs_p(){
    if(Nrqs_p == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vr + vq*iNr + vs*iNr*iNq;
        locations.col(1) = vp;
        Vrqs_p = sp_mat(locations.t(), vValues, iNr*iNq*iNs,iNp);
        Nrqs_p = 1;
        return Vrqs_p;
    }
    else{
        return Vrqs_p;
    }
}

sp_mat flexmat::rsp_q(){
    if(Nrsp_q == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vr + vs*iNr + vp*iNr*iNs;
        locations.col(1) = vq;
        Vrsp_q = sp_mat(locations.t(), vValues, iNr*iNs*iNp,iNq);
        Nrsp_q = 1;
        return Vrsp_q;
    }
    else{
        return Vrsp_q;
    }
}

sp_mat flexmat::rsq_p(){
    if(Nrsq_p == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vr + vs*iNr + vq*iNr*iNs;
        locations.col(1) = vp;
        Vrsq_p = sp_mat(locations.t(), vValues, iNr*iNs*iNq,iNp);
        Nrsq_p = 1;
        return Vrsq_p;
    }
    else{
        return Vrsq_p;
    }
}

sp_mat flexmat::spq_r(){
    if(Nspq_r == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vs + vp*iNs + vq*iNs*iNp;
        locations.col(1) = vr;
        Vspq_r = sp_mat(locations.t(), vValues, iNs*iNp*iNq,iNr);
        Nspq_r = 1;
        return Vspq_r;
    }
    else{
        return Vspq_r;
    }
}

sp_mat flexmat::spr_q(){
    if(Nspr_q == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vs + vp*iNs + vr*iNs*iNp;
        locations.col(1) = vq;
        Vspr_q = sp_mat(locations.t(), vValues, iNs*iNp*iNr,iNq);
        Nspr_q = 1;
        return Vspr_q;
    }
    else{
        return Vspr_q;
    }
}

sp_mat flexmat::sqp_r(){
    if(Nsqp_r == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vs + vq*iNs + vp*iNs*iNq;
        locations.col(1) = vr;
        Vsqp_r = sp_mat(locations.t(), vValues, iNs*iNq*iNp,iNr);
        Nsqp_r = 1;
        return Vsqp_r;
    }
    else{
        return Vsqp_r;
    }
}

sp_mat flexmat::sqr_p(){
    if(Nsqr_p == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vs + vq*iNs + vr*iNs*iNq;
        locations.col(1) = vp;
        Vsqr_p = sp_mat(locations.t(), vValues, iNs*iNq*iNr,iNp);
        Nsqr_p = 1;
        return Vsqr_p;
    }
    else{
        return Vsqr_p;
    }
}

sp_mat flexmat::srp_q(){
    if(Nsrp_q == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vs + vr*iNs + vp*iNs*iNr;
        locations.col(1) = vq;
        Vsrp_q = sp_mat(locations.t(), vValues, iNs*iNr*iNp,iNq);
        Nsrp_q = 1;
        return Vsrp_q;
    }
    else{
        return Vsrp_q;
    }
}

sp_mat flexmat::srq_p(){
    if(Nsrq_p == 0){
        locations.set_size(vp.size(), 2);
        locations.col(0) = vs + vr*iNs + vq*iNs*iNr;
        locations.col(1) = vp;
        Vsrq_p = sp_mat(locations.t(), vValues, iNs*iNr*iNq,iNp);
        Nsrq_p = 1;
        return Vsrq_p;
    }
    else{
        return Vsrq_p;
    }
}
