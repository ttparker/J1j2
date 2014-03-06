#include "Hamiltonian.h"

#define j1 couplingConstants[0]
#define j2 couplingConstants[1]
#define sigmaplus h2[0]
#define sigmaz h2[1]
#define sigmaminus h2[2]
#define rhoBasisSigmaplus rhoBasisH2[0]
#define rhoBasisSigmaz rhoBasisH2[1]
#define off0RhoBasisSigmaplus off0RhoBasisH2[0]
#define off0RhoBasisSigmaz off0RhoBasisH2[1]
#define off1RhoBasisSigmaplus off1RhoBasisH2[0]
#define off1RhoBasisSigmaz off1RhoBasisH2[1]

using namespace Eigen;

Hamiltonian::Hamiltonian() : oneSiteQNums({1, -1})
{
    h2.resize(3);
    sigmaplus << 0., 1.,
                 0., 0.;
    sigmaminus << 0., 0.,
                  1., 0.;
    sigmaz << 1., 0.,
              0., -1.;                               // define Pauli matrices
};

void Hamiltonian::setParams(const std::vector<double>& couplingConstantsIn,
                            int targetQNumIn, int lSysIn)
{
    couplingConstants = couplingConstantsIn;
    targetQNum = targetQNumIn;
    lSys = lSysIn;
};

MatrixXd Hamiltonian::blockAdjacentSiteJoin(int j,
                                            const std::vector<MatrixXd>& rhoBasisH2)
                                            const
{
    MatrixXd plusMinus = kp(rhoBasisSigmaplus, sigmaminus);
    return couplingConstants[j - 1] *
        (kp(rhoBasisSigmaz, sigmaz) + 2 * (plusMinus + plusMinus.adjoint()));
};

MatrixXd Hamiltonian::lBlockrSiteJoin(const std::vector<Eigen::MatrixXd>&
                                      off0RhoBasisH2, int mlE) const
{
    MatrixXd plusMinus = kp(kp(off0RhoBasisSigmaplus, Id(d * mlE)), sigmaminus);
    return j2 * (kp(kp(off0RhoBasisSigmaz, Id(d * mlE)), sigmaz)
                 + 2 * (plusMinus + plusMinus.adjoint()));
};

MatrixXd Hamiltonian::lSiterBlockJoin(int ml,
                                      const std::vector<Eigen::MatrixXd>&
                                      off0RhoBasisH2) const
{
    MatrixXd plusMinus = kp(sigmaplus, off0RhoBasisSigmaplus.adjoint());
    return j2 *
        kp(kp(Id(ml), kp(sigmaz, off0RhoBasisSigmaz)
                      + 2 * (plusMinus + plusMinus.adjoint())),
           Id_d);
};

MatrixXd Hamiltonian::siteSiteJoin(int ml, int mlE) const
{
    MatrixXd plusMinus = kp(kp(sigmaplus, Id(mlE)), sigmaminus);
    return j1 * kp(Id(ml), kp(kp(sigmaz, Id(mlE)), sigmaz)
                           + 2 * (plusMinus + plusMinus.adjoint()));
};
