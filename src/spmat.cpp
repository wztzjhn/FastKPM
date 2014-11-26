#include <armadillo>
#include "fastkpm.h"

namespace fkpm {
    
    // --- SpMatElems ----------------------------------------------------------------
 
    template <typename T>
    SpMatElems<T>::SpMatElems(int n_rows, int n_cols, int b_len)
        : n_rows(n_rows), n_cols(n_cols), b_len(b_len)
    {}
    
    template <typename T>
    int SpMatElems<T>::n_blocks() const {
        return row_idx.size();
    }
    
    template <typename T>
    void SpMatElems<T>::clear() {
        row_idx.resize(0);
        col_idx.resize(0);
        val.resize(0);
    }
    
    template <typename T>
    void SpMatElems<T>::add(int i, int j, T const* v) {
        row_idx.push_back(i);
        col_idx.push_back(j);
        val.insert(val.end(), v, v+b_len*b_len);
    }
    
    // --- SpMatBsr ----------------------------------------------------------------
    
    template <typename T>
    SpMatBsr<T>::SpMatBsr() {}
    
    template <typename T>
    SpMatBsr<T>::SpMatBsr(SpMatElems<T> const& elems) {
        build(elems);
    }
    
    template <typename T>
    void SpMatBsr<T>::build(SpMatElems<T> const& elems) {
        n_rows = elems.n_rows;
        n_cols = elems.n_cols;
        b_len = elems.b_len;
        row_idx.resize(elems.n_blocks());
        col_idx.resize(elems.n_blocks());
        row_ptr.resize(n_rows+1);
        val.resize(b_len*b_len*elems.n_blocks());
        
        // check if sorted
        bool sorted = true;
        for (int k = 0; k < elems.n_blocks()-1; k++) {
            int i1 = elems.row_idx[k], i2 = elems.row_idx[k+1];
            int j1 = elems.col_idx[k], j2 = elems.col_idx[k+1];
            if (i1 > i2 || (i1 == i2 && j1 > j2)) {
                sorted = false;
                break;
            }
        }
        
        // sort if necessary
        if (!sorted) {
            std::cout << "sorting!\n";
            sorted_ptr_bin.resize(n_rows);
            for (int i = 0; i < n_rows; i++) {
                sorted_ptr_bin[i].resize(0);
            }
            for (int p = 0; p < elems.n_blocks(); p++) {
                sorted_ptr_bin[elems.row_idx[p]].push_back(p);
            }
            auto sort_row = [&](std::size_t i) {
                std::sort(sorted_ptr_bin[i].begin(), sorted_ptr_bin[i].end(), [&](int p1, int p2) {
                    return elems.col_idx[p1] < elems.col_idx[p2];
                });
            };
            parallel_for(0, n_rows, sort_row);
            
            sorted_ptr.clear();
            for (int i = 0; i < n_rows; i++) {
                sorted_ptr.insert(sorted_ptr.end(), sorted_ptr_bin[i].begin(), sorted_ptr_bin[i].end());
            }
        }
        else {
            std::cout << "no sort!\n";
            sorted_ptr.resize(elems.n_blocks());
            for (int p = 0; p < elems.n_blocks(); p++) {
                sorted_ptr[p] = p;
            }
        }
        
        // set row_idx, col_idx, row_ptr, and accumulated val
        int max_row = -1; // largest row observed
        int cnt = 0;      // number of unique (i, j) elements observed
        for (int p: sorted_ptr) {
            int i = elems.row_idx[p];
            int j = elems.col_idx[p];
            // if element already exists, accumulate previous value
            if (cnt > 0 && row_idx[cnt-1] == i && col_idx[cnt-1] == j) {
                T const* src = &elems.val[b_len*b_len*p];
                T*       dst = &this->val[b_len*b_len*(cnt-1)];
                for (int bk = 0; bk < b_len*b_len; bk++) {
                    dst[bk] += src[bk];
                }
            }
            // otherwise add new element and update row_ptr
            else {
                row_idx[cnt] = i;
                col_idx[cnt] = j;
                T const* src = &elems.val[b_len*b_len*p];
                T*       dst = &this->val[b_len*b_len*cnt];
                std::copy_n(src, b_len*b_len, dst);
                while (max_row < i) {
                    row_ptr[++max_row] = cnt;
                }
                cnt++;
            }
        }
        while (max_row < n_rows) {
            row_ptr[++max_row] = cnt;
        }
        
        // trim sizes of vectors in case there were duplicates
        row_idx.resize(cnt);
        col_idx.resize(cnt);
        val.resize(b_len*b_len*cnt);
    }
    
    template <typename T>
    int SpMatBsr<T>::n_blocks() const {
        return row_idx.size();
    }
    
    template <typename T>
    void SpMatBsr<T>::clear() {
        row_idx.resize(0);
        col_idx.resize(0);
        row_ptr.resize(0);
        val.resize(0);
    }
    
    template <typename T>
    int SpMatBsr<T>::find_index(int i, int j) const {
        for (int p = row_ptr[i]; p < row_ptr[i+1]; p++) {
            if (col_idx[p] == j) {
                return p;
            }
        }
        std::cerr << "Could not find block index (" << i << "," << j << ") in BSR matrix!\n";
        abort();
    }
    
    template <typename T>
    T* SpMatBsr<T>::operator()(int i, int j) {
        return &val[b_len*b_len*find_index(i, j)];
    }
    
    template <typename T>
    T const* SpMatBsr<T>::operator()(int i, int j) const {
        return &val[b_len*b_len*find_index(i, j)];
    }
    
    template <typename T>
    void SpMatBsr<T>::zeros() {
        std::fill(val.begin(), val.end(), 0);
    }
    
    template <typename T>
    void SpMatBsr<T>::symmetrize() {
        for (int k = 0; k < n_blocks(); k++) {
            int i = row_idx[k];
            int j = col_idx[k];
            if (i >= j) {
                T* v1_base = &val[b_len*b_len*k];
                T* v2_base = (*this)(j, i);
                for (int bj = 0; bj < b_len; bj++) {
                    for (int bi = 0; bi < b_len; bi++) {
                        T* v1 = &v1_base[bj*b_len+bi];
                        T* v2 = &v2_base[bi*b_len+bj];
                        T sym_val = (*v1 + conj(*v2)) / T(2);
                        *v1 = sym_val;
                        *v2 = conj(sym_val);
                    }
                }
            }
        }
    }
    
    template <typename T>
    void SpMatBsr<T>::scale(T alpha) {
        for (int k = 0; k < val.size(); k++) {
            val[k] *= alpha;
        }
    }
    
    template <typename T>
    arma::SpMat<T> SpMatBsr<T>::to_arma() const {
        arma::umat locations = arma::umat(2, b_len*b_len*n_blocks());
        int cnt = 0;
        for (int k = 0; k < n_blocks(); k++) {
            int i = row_idx[k];
            int j = col_idx[k];
            for (int bj = 0; bj < b_len; bj++) {
                for (int bi = 0; bi < b_len; bi++) {
                    locations(0, cnt) = b_len*i + bi;
                    locations(1, cnt) = b_len*j + bj;
                    cnt++;
                }
            }
        }
        auto values = arma::Col<T>(val);
        return arma::SpMat<T>(true, locations, values, b_len*n_rows, b_len*n_cols);
    }
    
    template <typename T>
    arma::Mat<T> SpMatBsr<T>::to_arma_dense() const {
        return arma::eye<arma::Mat<T>>(b_len*n_rows, b_len*n_rows) * to_arma();
    }
    
    
    // instantiations
    template class SpMatElems<float>;
    template class SpMatElems<double>;
    template class SpMatElems<cx_float>;
    template class SpMatElems<cx_double>;
    template class SpMatBsr<float>;
    template class SpMatBsr<double>;
    template class SpMatBsr<cx_float>;
    template class SpMatBsr<cx_double>;
}

