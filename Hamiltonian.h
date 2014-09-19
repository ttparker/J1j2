#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "main.h"

#define kp kroneckerProduct
#define Id(size) MatrixXd::Identity(size, size)
#define Id_d Matrix<double, d, d>::Identity()       // one-site identity matrix

typedef std::vector<MatrixD_t, Eigen::aligned_allocator<MatrixD_t>> vecMatD_t;

class Hamiltonian
{
    public:
        Hamiltonian();
        void setParams(const std::vector<double>& couplingConstants,
                       int targetQNumIn, int lSysIn);
        
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    private:
        std::vector<double> couplingConstants;
        std::vector<int> oneSiteQNums;              // one-site quantum numbers
        int targetQNum,                              // targeted quantum number
            lSys;                                      // current system length
        vecMatD_t siteBasisH2;                 // site-basis coupling operators
        
        MatrixX_t blockAdjacentSiteJoin(int j,
                                        const std::vector<MatrixX_t>& rhoBasisH2)
                                        const,
                                         // j gives the j-1th coupling constant
                  lBlockrSiteJoin(const std::vector<MatrixX_t>& off0RhoBasisH2,
                                  int compm) const,
                  lSiterBlockJoin(int m,
                                  const std::vector<MatrixX_t>& off0RhoBasisH2)
                                  const,
                  siteSiteJoin(int m, int compm) const;
                                           // joins the two free sites together
    
    friend class TheBlock;
};

#endif
