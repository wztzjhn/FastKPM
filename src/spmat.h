#ifndef __spmat__
#define __spmat__

#include <armadillo>

namespace fkpm {
    template <typename T>
    using Vec = std::vector<T>;
    
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
        virtual void clear() {
            row_idx.resize(0);
            col_idx.resize(0);
            val.resize(0);
        }
        virtual void add(int i, int j, T v) {
            row_idx.push_back(i);
            col_idx.push_back(j);
            val.push_back(v);
        }
        virtual SpMatCoo<T>& operator=(SpMatCoo<T> const& that) {
            n_rows = that.n_rows;
            n_cols = that.n_cols;
            row_idx = that.row_idx;
            col_idx = that.col_idx;
            val = that.val;
            return *this;
        }
        arma::SpMat<T> to_arma() const {
            arma::umat locations = arma::umat(2, size());
            for (int j = 0; j < size(); j++) {
                locations(0, j) = row_idx[j];
                locations(1, j) = col_idx[j];
            }
            auto values = arma::Col<T>(val);
            return arma::SpMat<T>(true, locations, values, n_rows, n_cols);
        }
        arma::Mat<T> to_arma_dense() const {
            return arma::eye<arma::Mat<T>>(n_rows, n_rows) * to_arma();
        }
    };
    
    // Sparse matrix in Compressed Sparse Row format
    template <typename T>
    class SpMatCsr : public SpMatCoo<T> {
    public:
        Vec<int> row_ptr;
        Vec<int> sorted_ptrs;
        SpMatCsr(): SpMatCoo<T>(0, 0) {}
        SpMatCsr(SpMatCoo<T> const& that) { build(that); }
        void clear() {
            SpMatCoo<T>::clear();
            row_ptr.resize(0);
        }
        void add(int i, int j, T v) {
            std::cerr << "Cannot add individual elements to CSR matrix. Use SpMatCsr::build() instead.\n";
            abort();
        }
        T& operator()(int i, int j) {
            for (int p = row_ptr[i]; p < row_ptr[i+1]; p++) {
                if (this->col_idx[p] == j) {
                    return this->val[p];
                }
            }
            std::cerr << "Could not find index (" << i << "," << j << ") in sparse matrix.\n";
            abort();
        }
        void build(SpMatCoo<T> const& that) {
            if (this == &that) return;
            this->n_rows = that.n_rows;
            this->n_cols = that.n_cols;
            this->row_idx.resize(that.size());
            this->col_idx.resize(that.size());
            this->val.resize(that.size());
            row_ptr.resize(this->n_rows+1);
            sorted_ptrs.resize(that.size());
            for (int p = 0; p < this->size(); p++) {
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
                if (k > 0 && this->row_idx[k-1] == i && this->col_idx[k-1] == j) {
                    this->val[k-1] += that.val[p];
                }
                // otherwise update and increment all fields
                else {
                    this->row_idx[k] = i;
                    this->col_idx[k] = j;
                    this->val[k] = that.val[p];
                    while (max_row < i) {
                        this->row_ptr[++max_row] = k;
                    }
                    k++;
                }
            }
            while (max_row < this->n_rows) {
                this->row_ptr[++max_row] = k;
            }
            this->row_idx.resize(k);
            this->col_idx.resize(k);
            this->val.resize(k);
        }
    };
}


#endif // defined(__spmat__)
