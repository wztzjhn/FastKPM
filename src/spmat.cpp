#include "fastkpm.h"
#include <armadillo>

namespace fkpm {
    
    // Sparse matrix in Coordinate list format
    template <typename T>
    int SpMatElems<T>::size() const {
        return row_idx.size();
    }

    template <typename T>
    void SpMatElems<T>::clear() {
        row_idx.resize(0);
        col_idx.resize(0);
        val.resize(0);
    }
    
    template <typename T>
    void SpMatElems<T>::add(int i, int j, T v) {
        row_idx.push_back(i);
        col_idx.push_back(j);
        val.push_back(v);
    }
    
    // Sparse matrix in Compressed Sparse Row format
    template <typename T>
    SpMatCsr<T>::SpMatCsr() {}
    
    template <typename T>
    SpMatCsr<T>::SpMatCsr(int n_rows, int n_cols, SpMatElems<T> const& that) { build(n_rows, n_cols, that); }

    template <typename T>
    int SpMatCsr<T>::size() const {
        return row_idx.size();
    }

    template <typename T>
    void SpMatCsr<T>::clear() {
        row_idx.resize(0);
        col_idx.resize(0);
        row_ptr.resize(0);
        val.resize(0);
    }

    template <typename T>
    int SpMatCsr<T>::find_index(int i, int j) const {
        for (int p = row_ptr[i]; p < row_ptr[i+1]; p++) {
            if (col_idx[p] == j) {
                return p;
            }
        }
        std::cerr << "Could not find index (" << i << "," << j << ") in sparse matrix.\n";
        abort();
    }

    template <typename T>
    T& SpMatCsr<T>::operator()(int i, int j) {
        return val[find_index(i, j)];
    }
    
    template <typename T>
    T const& SpMatCsr<T>::operator()(int i, int j) const {
        return val[find_index(i, j)];
    }
    
    template <typename T>
    void SpMatCsr<T>::build(int n_rows, int n_cols, SpMatElems<T> const& elems) {
        this->n_rows = n_rows;
        this->n_cols = n_cols;
        row_idx.resize(elems.size());
        col_idx.resize(elems.size());
        row_ptr.resize(n_rows+1);
        val.resize(elems.size());
        sorted_ptrs.resize(elems.size());
        for (int p = 0; p < size(); p++) {
            sorted_ptrs[p] = p;
        }
        std::sort(sorted_ptrs.begin(), sorted_ptrs.end(), [&](int p1, int p2) {
            int i1 = elems.row_idx[p1];
            int j1 = elems.col_idx[p1];
            int i2 = elems.row_idx[p2];
            int j2 = elems.col_idx[p2];
            return (i1 < i2 || (i1 == i2 && j1 < j2));
        });
        int max_row = -1; // largest row observed
        int k = 0;        // number of unique elements observed
        for (int p : sorted_ptrs) {
            int i = elems.row_idx[p];
            int j = elems.col_idx[p];
            // if element already exists, accumulate previous value
            if (k > 0 && row_idx[k-1] == i && col_idx[k-1] == j) {
                val[k-1] += elems.val[p];
            }
            // otherwise add new element and update row_ptr
            else {
                row_idx[k] = i;
                col_idx[k] = j;
                val[k] = elems.val[p];
                while (max_row < i) {
                    row_ptr[++max_row] = k;
                }
                k++;
            }
        }
        while (max_row < n_rows) {
            row_ptr[++max_row] = k;
        }
        row_idx.resize(k);
        col_idx.resize(k);
        val.resize(k);
    }
    
    template <typename T>
    void SpMatCsr<T>::zeros() {
        for (T& v: val) {
            v = 0;
        }
    }
    
    template <typename T>
    void SpMatCsr<T>::symmetrize() {
        for (int k = 0; k < size(); k++) {
            int i = row_idx[k];
            int j = col_idx[k];
            if (i >= j) {
                T v1 = (*this)(i, j);
                T v2 = (*this)(j, i);
                (*this)(i, j) = 0.5 * (v1 + conj(v2));
                (*this)(j, i) = 0.5 * (conj(v1) + v2);
            }
        }
    }
    
    template <typename T>
    arma::SpMat<T> SpMatCsr<T>::to_arma() const {
        arma::umat locations = arma::umat(2, row_idx.size());
        for (int j = 0; j < row_idx.size(); j++) {
            locations(0, j) = row_idx[j];
            locations(1, j) = col_idx[j];
        }
        auto values = arma::Col<T>(val);
        return arma::SpMat<T>(true, locations, values, n_rows, n_cols);
    }
    
    template <typename T>
    arma::Mat<T> SpMatCsr<T>::to_arma_dense() const {
        return arma::eye<arma::Mat<T>>(n_rows, n_rows) * to_arma();
    }
    
    
    // instantiations
    template class SpMatElems<double>;
    template class SpMatElems<cx_double>;
    template class SpMatCsr<double>;
    template class SpMatCsr<cx_double>;
}

