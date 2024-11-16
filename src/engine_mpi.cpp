#include "fastkpm.h"

#ifndef WITH_MPI

namespace fkpm {
    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine_mpi() {
        std::cerr << "MPI unavailable. Falling back to single-node KPM engine.\n";
        return mk_engine<T>();
    }
    template std::shared_ptr<Engine<float>> mk_engine_mpi();
    template std::shared_ptr<Engine<double>> mk_engine_mpi();
    template std::shared_ptr<Engine<cx_float>> mk_engine_mpi();
    template std::shared_ptr<Engine<cx_double>> mk_engine_mpi();
}

#else

// workaround missing "is_trivially_copyable" in g++ < 5.0
#if __GNUG__ && __GNUC__ < 5
#define IS_TRIVIALLY_COPYABLE(T) __has_trivial_copy(T)
#else
#define IS_TRIVIALLY_COPYABLE(T) std::is_trivially_copyable<T>::value
#endif

#include <mpi.h>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace fkpm {

    template<typename T> MPI_Datatype mpi_datatype();
    template<> inline MPI_Datatype mpi_datatype<float>()      { return MPI_FLOAT; }
    template<> inline MPI_Datatype mpi_datatype<double>()     { return MPI_DOUBLE; }
    template<> inline MPI_Datatype mpi_datatype<cx_float>()   { return MPI_COMPLEX; }
    template<> inline MPI_Datatype mpi_datatype<cx_double>()  { return MPI_DOUBLE_COMPLEX; }

    class Serializer {
    public:
        Vec<uint8_t> buffer;

        void reset() {
            buffer.clear();
        }

        void write_bytes(uint8_t const* data, size_t size) {
            buffer.insert(buffer.end(), data, data+size);
        }

        template <typename T, class=typename std::enable_if<IS_TRIVIALLY_COPYABLE(T)>::type>
        Serializer& operator<< (T const& x) {
            write_bytes((uint8_t *)&x, sizeof(T)*CHAR_BIT/8);
            return *this;
        }

        template <typename T>
        Serializer& operator<< (Vec<T> const& x) {
            *this << x.size();
            if (IS_TRIVIALLY_COPYABLE(T)) {
                write_bytes((uint8_t *)x.data(), x.size()*sizeof(T)*CHAR_BIT/8);
            }
            else {
                for (int i = 0; i < x.size(); i++) {
                    *this << x[i];
                }
            }
            return *this;
        }

        template <typename T>
        Serializer& operator<< (SpMatBsr<T> const& x) {
            return *this << x.n_rows << x.n_cols << x.b_len << x.row_idx << x.col_idx << x.val;
        }
    };

    class Deserializer {
    public:
        Vec<uint8_t> buffer;
        int pos = 0;

        void init_empty(size_t size) {
            buffer.resize(size, 0);
            pos = 0;
        }

        void init(Serializer const& ser) {
            buffer = ser.buffer;
            pos = 0;
        }

        void read_bytes(uint8_t* data, size_t size) {
            assert(pos+size <= buffer.size());
            memcpy(data, buffer.data()+pos, size);
            pos += size;
        }

        template <typename T, class=typename std::enable_if<IS_TRIVIALLY_COPYABLE(T)>::type>
        Deserializer& operator>> (T& x) {
            read_bytes((uint8_t *)&x, sizeof(T)*CHAR_BIT/8);
            return *this;
        }

        template <typename T>
        Deserializer& operator>> (Vec<T>& x) {
            size_t size; *this >> size;
            x.resize(size);
            if (IS_TRIVIALLY_COPYABLE(T)) {
                read_bytes((uint8_t *)x.data(), x.size()*sizeof(T)*CHAR_BIT/8);
            }
            else {
                for (int i = 0; i < size; i++) {
                    *this >> x[i];
                }
            }
            return *this;
        }

        template <typename T>
        Deserializer& operator>> (SpMatBsr<T>& x) {
            int n_rows, n_cols, b_len;
            *this >> n_rows >> n_cols >> b_len;
            SpMatElems<T> elems(n_rows, n_cols, b_len);
            *this >> elems.row_idx >> elems.col_idx >> elems.val;
            x = SpMatBsr<T>(elems);
            return *this;
        }

        template <typename T>
        T take() {
            T ret;
            *this >> ret;
            return ret;
        }
    };

    template <typename T>
    class Engine_MPI: public Engine<T> {
    public:
        std::shared_ptr<Engine<T>> worker;

        int n_ranks = 0, rank = -1;
        const int root_rank = 0;

        Vec<std::function<void(Engine_MPI*)>> exec_cmds {
            &Engine_MPI::exec_set_R_identity,
            &Engine_MPI::exec_set_R_uncorrelated,
            &Engine_MPI::exec_set_R_correlated,
            &Engine_MPI::exec_set_H,
            &Engine_MPI::exec_moments,
            &Engine_MPI::exec_moments2_v1,
            &Engine_MPI::exec_moments2_v2,
            &Engine_MPI::exec_stoch_matrix,
            &Engine_MPI::exec_autodiff_matrix,
        };

        enum Tag: int { // must correspond to ordering of exec_cmds
            tag_set_R_identity,
            tag_set_R_uncorrelated,
            tag_set_R_correlated,
            tag_set_H,
            tag_moments,
            tag_moments2_v1,
            tag_moments2_v2,
            tag_stoch_matrix,
            tag_autodiff_matrix,
            tag_quit,
        };

        struct Cmd {
            Tag tag;
            int size; // in bytes
        };

        Serializer ser;
        Deserializer des;

        Engine_MPI() {
            MPI_Init(nullptr, nullptr);
            worker = mk_engine<T>();
            MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank != root_rank) {
                // worker loop
                bool running = true;
                while (running) {
                    Cmd cmd;
                    MPI_Bcast(&cmd, 2, MPI_INT, root_rank, MPI_COMM_WORLD);
                    if (cmd.size > 0) {
                        des.init_empty(cmd.size);
                        MPI_Bcast(des.buffer.data(), cmd.size, MPI_BYTE, root_rank, MPI_COMM_WORLD);
                    }
                    if (cmd.tag == tag_quit)
                        running = false;
                    else
                        exec_cmds[cmd.tag](this);
                }
                MPI_Finalize();
                std::exit(0);
            }
        }

        ~Engine_MPI() {
            Cmd cmd = { tag_quit, 0 };
            MPI_Bcast(&cmd, 2, MPI_INT, root_rank, MPI_COMM_WORLD);
            MPI_Finalize();
        }

        // Lanczos approximaton to eigenvalue span
        EnergyScale energy_scale(SpMatBsr<T> const& H, double extend, int iters) {
            return worker->energy_scale(H, extend, iters);
        }

        void broadcast_cmd(Tag tag) {
            Cmd cmd = { tag, static_cast<int>(ser.buffer.size()) };
            MPI_Bcast(&cmd, 2, MPI_INT, root_rank, MPI_COMM_WORLD);
            MPI_Bcast(ser.buffer.data(), ser.buffer.size(), MPI_BYTE, root_rank, MPI_COMM_WORLD);
            des.init(ser);
            ser.reset();
            exec_cmds[tag](this);
            des.init(ser);
        }

        void exec_set_R_identity() {
            int n, j_start, j_end;
            des >> n >> j_start >> j_end;
            int sz = j_end - j_start;
            int j1 = j_start + rank * sz / n_ranks;
            int j2 = j_start + (rank + 1) * sz / n_ranks;
            worker->set_R_identity(n, j1, j2);
        }
        void set_R_identity(int n, int j_start, int j_end) {
            ser.reset();
            ser << n << j_start << j_end;
            broadcast_cmd(tag_set_R_identity);
        }

        void exec_set_R_uncorrelated() {
            int n, s, j_start, j_end;
            RNG rng;
            des >> n >> s >> rng >> j_start >> j_end;
            int sz = j_end - j_start;
            int j1 = j_start + rank * sz / n_ranks;
            int j2 = j_start + (rank + 1) * sz / n_ranks;
            rng.discard(j1 - j_start);
            worker->set_R_uncorrelated(n, s, rng, j1, j2);
            rng.discard(j_end - j2);
            ser << rng;
        }
        void set_R_uncorrelated(int n, int s, RNG& rng, int j_start, int j_end) {
            ser.reset();
            ser << n << s << rng << j_start << j_end;
            broadcast_cmd(tag_set_R_uncorrelated);
            des >> rng;
        }

        void exec_set_R_correlated() {
            Vec<int> groups;
            RNG rng;
            int j_start, j_end;
            des >> groups >> rng >> j_start >> j_end;
            int sz = j_end - j_start;
            int j1 = j_start + rank * sz / n_ranks;
            int j2 = j_start + (rank + 1) * sz / n_ranks;
            rng.discard(j1 - j_start);
            worker->set_R_correlated(groups, rng, j1, j2);
            rng.discard(j_end - j2);
            ser << rng;
        }
        void set_R_correlated(Vec<int> const& groups, RNG& rng, int j_start, int j_end) {
            ser.reset();
            ser << groups << rng << j_start << j_end;
            broadcast_cmd(tag_set_R_correlated);
            des >> rng;
        }

        void exec_set_H() {
            SpMatBsr<T> H;
            EnergyScale es;
            des >> H >> es;
            worker->set_H(H, es);
        }
        void set_H(SpMatBsr<T> const& H, EnergyScale const& es) {
            ser.reset();
            ser << H << es;
            broadcast_cmd(tag_set_H);
        }

        void exec_moments() {
            int M;
            des >> M;
            Vec<double> mu = worker->moments(M), reduced(M);
            MPI_Reduce(mu.data(), reduced.data(), mu.size(), MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);
            ser << reduced;
        }
        Vec<double> moments(int M) {
            ser.reset();
            ser << M;
            broadcast_cmd(tag_moments);
            return des.take<Vec<double>>();
        }

        void exec_moments2_v1() {
            int M, a_chunk_ncols, R_chunk_ncols;
            SpMatBsr<T> j1op, j2op;
            des >> M >> j1op >> j2op >> a_chunk_ncols >> R_chunk_ncols;
            Vec<Vec<cx_double>> mu = worker->moments2_v1(M, j1op, j2op, a_chunk_ncols, R_chunk_ncols);
            Vec<Vec<cx_double>> reduced(M);
            assert(M == mu.size());

            // convert to real numbers, to use MPI_Reduce
            Vec<double> mu_real(M*M), mu_imag(M*M);
            Vec<double> reduced_real(M*M), reduced_imag(M*M);
            for (int m1 = 0; m1 < M; m1++) {
                for (int m2 = 0; m2 < M; m2++) {
                    mu_real[m2+m1*M] = std::real(mu[m1][m2]);
                    mu_imag[m2+m1*M] = std::imag(mu[m1][m2]);
                }
            }
            MPI_Reduce(mu_real.data(), reduced_real.data(), mu_real.size(), MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);
            MPI_Reduce(mu_imag.data(), reduced_imag.data(), mu_imag.size(), MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);
            for (int m1 = 0; m1 < M; m1++) {
                reduced[m1].resize(M);
                for (int m2 = 0; m2 < M; m2++) {
                    reduced[m1][m2] = cx_double(reduced_real[m2+m1*M], reduced_imag[m2+m1*M]);
                }
            }
            ser << reduced;
        }
        Vec<Vec<cx_double>> moments2_v1(int M, SpMatBsr<T> const& j1op, SpMatBsr<T> const& j2op,
                                        int a_chunk_ncols=0, int R_chunk_ncols=0) {
            ser.reset();
            ser << M << j1op << j2op << a_chunk_ncols << R_chunk_ncols;
            broadcast_cmd(tag_moments2_v1);
            return des.take<Vec<Vec<cx_double>>>();
        }

        void exec_moments2_v2() {
            int M, a_chunk_ncols, R_chunk_ncols;
            SpMatBsr<T> j1op, j2op;
            des >> M >> j1op >> j2op >> a_chunk_ncols >> R_chunk_ncols;
            Vec<Vec<cx_double>> mu = worker->moments2_v2(M, j1op, j2op, a_chunk_ncols, R_chunk_ncols);
            assert(false && "Convert mu to matrix and implement reduce-sum");
            ser << mu;
        }
        Vec<Vec<cx_double>> moments2_v2(int M, SpMatBsr<T> const& j1op, SpMatBsr<T> const& j2op,
                                        int a_chunk_ncols=0, int R_chunk_ncols=0) {
            ser.reset();
            ser << M << j1op << j2op << a_chunk_ncols << R_chunk_ncols;
            broadcast_cmd(tag_moments2_v2);
            return des.take<Vec<Vec<cx_double>>>();
        }

        void exec_stoch_matrix() {
            Vec<double> c;
            SpMatBsr<T> D;
            des >> c >> D;
            worker->stoch_matrix(c, D);
            Vec<T> reduced(D.val.size());
            MPI_Reduce(D.val.data(), reduced.data(), D.val.size(), mpi_datatype<T>(), MPI_SUM, root_rank, MPI_COMM_WORLD);
            ser << reduced;
        }
        void stoch_matrix(Vec<double> const& c, SpMatBsr<T>& D) {
            ser.reset();
            ser << c << D;
            broadcast_cmd(tag_stoch_matrix);
            des >> D.val;
        }

        void exec_autodiff_matrix() {
            Vec<double> c;
            SpMatBsr<T> D;
            des >> c >> D;
            worker->autodiff_matrix(c, D);
            Vec<T> reduced(D.val.size());
            MPI_Reduce(D.val.data(), reduced.data(), D.val.size(), mpi_datatype<T>(), MPI_SUM, root_rank, MPI_COMM_WORLD);
            ser << reduced;
        }
        void autodiff_matrix(Vec<double> const& c, SpMatBsr<T>& D) {
            ser.reset();
            ser << c << D;
            broadcast_cmd(tag_autodiff_matrix);
            des >> D.val;
        }

    };

    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine_mpi() {
        return std::make_shared<Engine_MPI<T>>();
    }
    template std::shared_ptr<Engine<float>> mk_engine_mpi();
    template std::shared_ptr<Engine<double>> mk_engine_mpi();
    template std::shared_ptr<Engine<cx_float>> mk_engine_mpi();
    template std::shared_ptr<Engine<cx_double>> mk_engine_mpi();
}

#endif
