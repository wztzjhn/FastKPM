#ifndef __spmat__
#define __spmat__

#include <armadillo>

namespace fkpm {
    template <typename T>
    using Vec = std::vector<T>;
    
    template <typename T>
    arma::SpMat<T> arma_sp_mat(int n_rows, int n_cols, Vec<int> row_idx, Vec<int> col_idx, Vec<T> val) {
        arma::umat locations = arma::umat(2, row_idx.size());
        for (int j = 0; j < row_idx.size(); j++) {
            locations(0, j) = row_idx[j];
            locations(1, j) = col_idx[j];
        }
        auto values = arma::Col<T>(val);
        return arma::SpMat<T>(true, locations, values, n_rows, n_cols);
    }
    
    // Sparse matrix in Coordinate list format
    template <typename T>
    class SpMatCoo {
    public:
        int n_rows, n_cols;
        Vec<int> row_idx;
        Vec<int> col_idx;
        Vec<T> val;
        SpMatCoo(int n_rows, int n_cols): n_rows(n_rows), n_cols(n_cols) {}
        SpMatCoo(): SpMatCoo(0, 0) {}
        int size() const {
            return row_idx.size();
        }
        void clear() {
            row_idx.resize(0);
            col_idx.resize(0);
            val.resize(0);
        }
        void add(int i, int j, T v) {
            row_idx.push_back(i);
            col_idx.push_back(j);
            val.push_back(v);
        }
        arma::SpMat<T> to_arma() const {
            return arma_sp_mat(n_rows, n_cols, row_idx, col_idx, val);
        }
        arma::Mat<T> to_arma_dense() const {
            return arma::eye<arma::Mat<T>>(n_rows, n_rows) * to_arma();
        }
    };
    
    // Sparse matrix in Compressed Sparse Row format
    template <typename T>
    class SpMatCsr {
    public:
        int n_rows = 0, n_cols = 0;
        Vec<int> row_idx;
        Vec<int> col_idx;
        Vec<int> row_ptr;
        Vec<T> val;
        Vec<int> sorted_ptrs;
        SpMatCsr() {}
        SpMatCsr(SpMatCoo<T> const& that) { build(that); }
        int size() const {
            return row_idx.size();
        }
        void clear() {
            row_idx.resize(0);
            col_idx.resize(0);
            row_ptr.resize(0);
            val.resize(0);
        }
        int find_index(int i, int j) const {
            for (int p = row_ptr[i]; p < row_ptr[i+1]; p++) {
                if (col_idx[p] == j) {
                    return p;
                }
            }
            std::cerr << "Could not find index (" << i << "," << j << ") in sparse matrix.\n";
            abort();
        }
        T& operator()(int i, int j) {
            return val[find_index(i, j)];
        }
        T const& operator()(int i, int j) const {
            return val[find_index(i, j)];
        }
        void build(SpMatCoo<T> const& that) {
            n_rows = that.n_rows;
            n_cols = that.n_cols;
            row_idx.resize(that.size());
            col_idx.resize(that.size());
            row_ptr.resize(n_rows+1);
            val.resize(that.size());
            sorted_ptrs.resize(that.size());
            for (int p = 0; p < size(); p++) {
                sorted_ptrs[p] = p;
            }
            std::sort(sorted_ptrs.begin(), sorted_ptrs.end(), [&](int p1, int p2) {
                int i1 = that.row_idx[p1];
                int j1 = that.col_idx[p1];
                int i2 = that.row_idx[p2];
                int j2 = that.col_idx[p2];
                return (i1 < i2 || (i1 == i2 && j1 < j2));
            });
            int max_row = -1; // largest row observed
            int k = 0;        // number of unique elements observed
            for (int p : sorted_ptrs) {
                int i = that.row_idx[p];
                int j = that.col_idx[p];
                // if element already exists, accumulate previous value
                if (k > 0 && row_idx[k-1] == i && col_idx[k-1] == j) {
                    val[k-1] += that.val[p];
                }
                // otherwise add new element and update row_ptr
                else {
                    row_idx[k] = i;
                    col_idx[k] = j;
                    val[k] = that.val[p];
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
        arma::SpMat<T> to_arma() const {
            return arma_sp_mat(n_rows, n_cols, row_idx, col_idx, val);
        }
        arma::Mat<T> to_arma_dense() const {
            return arma::eye<arma::Mat<T>>(n_rows, n_rows) * to_arma();
        }
    };
}


#endif // defined(__spmat__)
