#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <curand.h>

// Error checking macros
#define CHECK_CUDA(call)                                                                         \
    do {                                                                                         \
        cudaError_t err = call;                                                                  \
        if (err != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__,     \
                    __LINE__);                                                                   \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    } while (0)

#define CHECK_CUBLAS(call)                                                                       \
    do {                                                                                         \
        cublasStatus_t status = call;                                                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                   \
            fprintf(stderr, "cuBLAS Error: %d at %s:%d\n", status, __FILE__, __LINE__);         \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    } while (0)

#define CHECK_CUSPARSE(call)                                                                     \
    do {                                                                                         \
        cusparseStatus_t status = call;                                                          \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                                 \
            fprintf(stderr, "cuSPARSE Error: %d at %s:%d\n", status, __FILE__, __LINE__);       \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    } while (0)

#define CHECK_CURAND(call)                                                                       \
    do {                                                                                         \
        curandStatus_t status = call;                                                            \
        if (status != CURAND_STATUS_SUCCESS) {                                                   \
            fprintf(stderr, "cuRAND Error: %d at %s:%d\n", status, __FILE__, __LINE__);         \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    } while (0)

// ------------------------------------------------------------------
// Data Management Struct
// ------------------------------------------------------------------

struct GCNData {
    // Host data
    std::vector<int> h_row_ptr, h_col_idx, h_labels;
    std::vector<double> h_edgeweights, h_features;
    // Metadata
    int num_nodes = 0, num_features = 0, num_classes = 0, nnz = 0;
    // Device data
    int *d_row_ptr = nullptr, *d_col_idx = nullptr, *d_labels = nullptr;
    double *d_edgeweights = nullptr, *d_features = nullptr;

    void load_from_files(const std::string& prefix);
    void copy_to_device();
    ~GCNData();
};

// ------------------------------------------------------------------
// CUDA Kernels & Helper Functions
// ------------------------------------------------------------------

__global__ void adam_kernel(double *w, const double *dw, double *m, double *v, int size,
                            double beta1, double beta2, double epsilon, double learning_rate,
                            double beta1_t, double beta2_t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double mt = m[idx];
        double vt = v[idx];
        double g = dw[idx];

        mt = beta1 * mt + (1.0 - beta1) * g;
        vt = beta2 * vt + (1.2 - beta2) * g * g;

        double m_hat = mt / (1.0 - beta1_t);
        double v_hat = vt / (1.0 - beta2_t);

        w[idx] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        m[idx] = mt;
        v[idx] = vt;
    }
}

__global__ void gcn_forward_relu_fused_kernel(double *H_out, const double *Z_in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        H_out[idx] = max(0.0, Z_in[idx]);
    }
}

__global__ void softmax_cross_entropy_grad_kernel(const double *z, const int *labels,
                                                  double *dz, int num_nodes, int out_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        int start_idx = i * out_dim;
        double max_val = -1e9;
        for (int j = 0; j < out_dim; j++) max_val = max(max_val, z[start_idx + j]);

        double sum_exp = 0.0;
        for (int j = 0; j < out_dim; j++) {
            double prob = exp(z[start_idx + j] - max_val);
            dz[start_idx + j] = prob;
            sum_exp += prob;
        }

        for (int j = 0; j < out_dim; j++) {
            dz[start_idx + j] /= sum_exp;
            if (j == labels[i]) {
                dz[start_idx + j] -= 1.0;
            }
        }
    }
}

__global__ void relu_derivative_fused_kernel(const double *z_prev, const double *dH,
                                             double *dZ_prev, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dZ_prev[idx] = (z_prev[idx] > 0.0) ? dH[idx] : 0.0;
    }
}

void xavier_init_cuda(double *w, int in_dim, int out_dim) {
    int size = in_dim * out_dim;
    std::vector<double> w_host(size);
    std::default_random_engine eng(time(0));
    double limit = std::sqrt(6.0) / std::sqrt(in_dim + out_dim);
    std::uniform_real_distribution<double> dist(-limit, limit);
    for (auto &x : w_host) x = dist(eng);
    CHECK_CUDA(cudaMemcpy(w, w_host.data(), size * sizeof(double), cudaMemcpyHostToDevice));
}


// ------------------------------------------------------------------
// GCN Model Class
// ------------------------------------------------------------------

class GCNModel {
private:
    GCNData& data;
    const std::vector<int>& layer_dims;
    int num_layers;

    cublasHandle_t bl_handle;
    cusparseHandle_t sp_handle;

    std::vector<double *> W_d, m_d, v_d;
    std::vector<double *> H_d, Z_d;

    cusparseSpMatDescr_t A_descr;
    std::vector<cusparseDnMatDescr_t> H_descr, Z_descr, Temp_descr;
    void* dBuffer = nullptr;

public:
    GCNModel(GCNData& data_ref, const std::vector<int>& layers);
    ~GCNModel();

    void forward();
    void backward(int epoch);
    double compute_loss_and_accuracy(double &accuracy);
    void train(int num_epochs);
};

// ------------------------------------------------------------------
// Main Function
// ------------------------------------------------------------------

int main() {
    std::string folder = "flickr/flickr";

    GCNData data;
    data.load_from_files(folder);
    data.copy_to_device();

    if (data.num_features <= 0 || data.num_classes <= 0 || data.num_nodes <= 0) {
        std::cerr << "Error reading input data. Exiting." << std::endl;
        return 1;
    }

    int epochs = 300;
    std::vector<int> layer_dims = {data.num_features, 16, data.num_classes};

    auto start = std::chrono::high_resolution_clock::now();
    {
        GCNModel model(data, layer_dims);
        model.train(epochs);
    } // `model` destructor is called here, cleaning up all resources
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "\nTotal Elapsed time: " << std::chrono::duration<double>(end - start).count() << " seconds\n";

    return 0;
}

// ------------------------------------------------------------------
// GCNData Method Implementations
// ------------------------------------------------------------------

void GCNData::load_from_files(const std::string& prefix) {
    // Read Graph
    std::ifstream graph_file(prefix + "_edgelist.txt");
    std::string line;
    int u, v;
    std::vector<std::pair<int, int>> edges;
    int max_node = -1;
    while (std::getline(graph_file, line)) {
        std::stringstream ss(line);
        if (ss >> u >> v) {
            edges.push_back({u, v});
            max_node = std::max({max_node, u, v});
        }
    }
    graph_file.close();
    num_nodes = max_node + 1;
    std::vector<std::vector<int>> adj(num_nodes);
    for (const auto &edge : edges) {
        adj[edge.first].push_back(edge.second);
        adj[edge.second].push_back(edge.first);
    }
    h_row_ptr.resize(num_nodes + 1, 0);
    for (int i = 0; i < num_nodes; ++i) {
        std::sort(adj[i].begin(), adj[i].end());
        adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
        h_row_ptr[i + 1] = h_row_ptr[i] + adj[i].size();
        for (int neighbor : adj[i]) h_col_idx.push_back(neighbor);
    }
    nnz = h_col_idx.size();
    h_edgeweights.assign(nnz, 1.0);
    std::cout << "Read Graph: Nodes=" << num_nodes << ", Edges=" << nnz << std::endl;

    // Read Features
    std::ifstream feature_file(prefix + "_features.txt");
    std::vector<double> all_values;
    while (getline(feature_file, line)) {
        if (!line.empty()) {
            std::stringstream ss(line);
            double value;
            while (ss >> value) all_values.push_back(value);
        }
    }
    feature_file.close();
    num_features = all_values.size() / num_nodes;
    h_features = all_values;

    // Read Labels
    std::ifstream label_file(prefix + "_labels.txt");
    int label;
    std::set<int> unique_labels;
    while (label_file >> label) {
        h_labels.push_back(label);
        unique_labels.insert(label);
    }
    label_file.close();
    num_classes = unique_labels.size();
    std::cout << "Number of classes are = " << num_classes << std::endl;
}

void GCNData::copy_to_device() {
    CHECK_CUDA(cudaMalloc(&d_row_ptr, (num_nodes + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_edgeweights, nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_labels, num_nodes * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_features, h_features.size() * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_row_ptr, h_row_ptr.data(), (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_idx, h_col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_edgeweights, h_edgeweights.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_labels, h_labels.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_features, h_features.data(), h_features.size() * sizeof(double), cudaMemcpyHostToDevice));
}

GCNData::~GCNData() {
    if (d_row_ptr) CHECK_CUDA(cudaFree(d_row_ptr));
    if (d_col_idx) CHECK_CUDA(cudaFree(d_col_idx));
    if (d_edgeweights) CHECK_CUDA(cudaFree(d_edgeweights));
    if (d_labels) CHECK_CUDA(cudaFree(d_labels));
    if (d_features) CHECK_CUDA(cudaFree(d_features));
}


// ------------------------------------------------------------------
// GCNModel Method Implementations
// ------------------------------------------------------------------

GCNModel::GCNModel(GCNData& data_ref, const std::vector<int>& layers)
    : data(data_ref), layer_dims(layers) {

    num_layers = layer_dims.size() - 1;

    CHECK_CUBLAS(cublasCreate(&bl_handle));
    CHECK_CUSPARSE(cusparseCreate(&sp_handle));

    W_d.resize(num_layers);
    m_d.resize(num_layers);
    v_d.resize(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        int in_dim = layer_dims[l];
        int out_dim = layer_dims[l+1];
        int size = in_dim * out_dim;
        CHECK_CUDA(cudaMalloc(&W_d[l], size * sizeof(double)));
        xavier_init_cuda(W_d[l], in_dim, out_dim);
        CHECK_CUDA(cudaMalloc(&m_d[l], size * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&v_d[l], size * sizeof(double)));
        CHECK_CUDA(cudaMemset(m_d[l], 0, size * sizeof(double)));
        CHECK_CUDA(cudaMemset(v_d[l], 0, size * sizeof(double)));
    }

    Z_d.resize(num_layers);
    H_d.resize(num_layers + 1);
    H_d[0] = data.d_features;

    H_descr.resize(num_layers + 1);
    Z_descr.resize(num_layers);
    Temp_descr.resize(num_layers);

    CHECK_CUSPARSE(cusparseCreateDnMat(&H_descr[0], data.num_nodes, layer_dims[0], layer_dims[0], H_d[0], CUDA_R_64F, CUSPARSE_ORDER_ROW));
    for (int l = 0; l < num_layers; ++l) {
        CHECK_CUDA(cudaMalloc(&Z_d[l], data.num_nodes * layer_dims[l+1] * sizeof(double)));
        CHECK_CUSPARSE(cusparseCreateDnMat(&Z_descr[l], data.num_nodes, layer_dims[l+1], layer_dims[l+1], Z_d[l], CUDA_R_64F, CUSPARSE_ORDER_ROW));
        CHECK_CUDA(cudaMalloc(&H_d[l+1], data.num_nodes * layer_dims[l+1] * sizeof(double)));
        CHECK_CUSPARSE(cusparseCreateDnMat(&H_descr[l+1], data.num_nodes, layer_dims[l+1], layer_dims[l+1], H_d[l+1], CUDA_R_64F, CUSPARSE_ORDER_ROW));
        
        double* temp_d;
        CHECK_CUDA(cudaMalloc(&temp_d, data.num_nodes * layer_dims[l] * sizeof(double)));
        CHECK_CUSPARSE(cusparseCreateDnMat(&Temp_descr[l], data.num_nodes, layer_dims[l], layer_dims[l], temp_d, CUDA_R_64F, CUSPARSE_ORDER_ROW));
    }

    CHECK_CUSPARSE(cusparseCreateCsr(&A_descr, data.num_nodes, data.num_nodes, data.nnz, data.d_row_ptr, data.d_col_idx, data.d_edgeweights, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    
    size_t bufferSize = 0;
    double alpha = 1.0, beta = 0.0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A_descr,
                                           H_descr[0], &beta, Temp_descr[0], CUDA_R_64F,
                                           CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
}

GCNModel::~GCNModel() {
    CHECK_CUBLAS(cublasDestroy(bl_handle));
    CHECK_CUSPARSE(cusparseDestroy(sp_handle));
    CHECK_CUSPARSE(cusparseDestroySpMat(A_descr));
    if(dBuffer) CHECK_CUDA(cudaFree(dBuffer));

    for (int l = 0; l < num_layers; ++l) {
        CHECK_CUDA(cudaFree(W_d[l]));
        CHECK_CUDA(cudaFree(m_d[l]));
        CHECK_CUDA(cudaFree(v_d[l]));
        CHECK_CUDA(cudaFree(Z_d[l]));
        CHECK_CUDA(cudaFree(H_d[l+1]));
        
        double* temp_ptr;
        CHECK_CUSPARSE(cusparseDnMatGetValues(Temp_descr[l], (void**)&temp_ptr));
        CHECK_CUDA(cudaFree(temp_ptr));

        CHECK_CUSPARSE(cusparseDestroyDnMat(Z_descr[l]));
        CHECK_CUSPARSE(cusparseDestroyDnMat(H_descr[l+1]));
        CHECK_CUSPARSE(cusparseDestroyDnMat(Temp_descr[l]));
    }
    CHECK_CUSPARSE(cusparseDestroyDnMat(H_descr[0]));
    std::cout << "\nGCNModel resources cleaned up." << std::endl;
}

void GCNModel::forward() {
    double alpha = 1.0, beta = 0.0;
    for (int l = 0; l < num_layers; ++l) {
        int in_dim = layer_dims[l];
        int out_dim = layer_dims[l+1];
        
        double* Temp_d_values;
        CHECK_CUSPARSE(cusparseDnMatGetValues(Temp_descr[l], (void**)&Temp_d_values));

        CHECK_CUSPARSE(cusparseSpMM(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A_descr, H_descr[l],
                                    &beta, Temp_descr[l], CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
        
        CHECK_CUBLAS(cublasDgemm(bl_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 out_dim, data.num_nodes, in_dim,
                                 &alpha, W_d[l], in_dim,
                                 Temp_d_values, in_dim,
                                 &beta, Z_d[l], out_dim));

        int size = data.num_nodes * out_dim;
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        gcn_forward_relu_fused_kernel<<<blocks, threads>>>(H_d[l + 1], Z_d[l], size);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
}

void GCNModel::backward(int epoch) {
    std::vector<double *> dZ_d(num_layers);
    std::vector<double *> dH_d(num_layers);
    double alpha = 1.0, beta = 0.0;

    for (int l = 0; l < num_layers; ++l) {
        CHECK_CUDA(cudaMalloc(&dZ_d[l], data.num_nodes * layer_dims[l+1] * sizeof(double)));
        if (l < num_layers - 1) {
            CHECK_CUDA(cudaMalloc(&dH_d[l], data.num_nodes * layer_dims[l+1] * sizeof(double)));
        }
    }
    
    int out_dim = layer_dims.back();
    int size = data.num_nodes * out_dim;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    softmax_cross_entropy_grad_kernel<<<blocks, threads>>>(Z_d[num_layers - 1], data.d_labels, dZ_d[num_layers - 1], data.num_nodes, out_dim);

    for (int l = num_layers - 1; l >= 0; --l) {
        int in_dim = layer_dims[l];
        int out_dim_curr = layer_dims[l+1];
        double *dW_d;
        CHECK_CUDA(cudaMalloc(&dW_d, in_dim * out_dim_curr * sizeof(double)));
        
        CHECK_CUBLAS(cublasDgemm(bl_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 out_dim_curr, in_dim, data.num_nodes,
                                 &alpha, dZ_d[l], out_dim_curr,
                                 H_d[l], in_dim,
                                 &beta, dW_d, out_dim_curr));
        
        int w_size = in_dim * out_dim_curr;
        blocks = (w_size + threads - 1) / threads;
        adam_kernel<<<blocks, threads>>>(W_d[l], dW_d, m_d[l], v_d[l], w_size, 0.9, 0.999, 1e-8, 0.001, pow(0.9, epoch), pow(0.999, epoch));
        CHECK_CUDA(cudaFree(dW_d));

        if (l > 0) {
            CHECK_CUBLAS(cublasDgemm(bl_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     in_dim, data.num_nodes, out_dim_curr,
                                     &alpha, W_d[l], in_dim,
                                     dZ_d[l], out_dim_curr,
                                     &beta, dH_d[l - 1], in_dim));
            
            int prev_size = data.num_nodes * in_dim;
            blocks = (prev_size + threads - 1) / threads;
            relu_derivative_fused_kernel<<<blocks, threads>>>(Z_d[l - 1], dH_d[l - 1], dZ_d[l - 1], prev_size);
            CHECK_CUDA(cudaFree(dH_d[l - 1]));
        }
    }
    for (int l = 0; l < num_layers; ++l) CHECK_CUDA(cudaFree(dZ_d[l]));
    CHECK_CUDA(cudaDeviceSynchronize());
}

double GCNModel::compute_loss_and_accuracy(double &accuracy) {
    int out_dim = data.num_classes;
    std::vector<double> Z_h(data.num_nodes * out_dim);
    std::vector<int> labels_h(data.num_nodes);
    CHECK_CUDA(cudaMemcpy(Z_h.data(), Z_d.back(), data.num_nodes * out_dim * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(labels_h.data(), data.d_labels, data.num_nodes * sizeof(int), cudaMemcpyDeviceToHost));

    double loss = 0.0;
    int correct = 0;
    for (int i = 0; i < data.num_nodes; i++) {
        double max_val = -1e9;
        for (int j = 0; j < out_dim; j++) max_val = std::max(max_val, Z_h[i * out_dim + j]);

        double sum_exp = 0.0;
        for (int j = 0; j < out_dim; j++) sum_exp += std::exp(Z_h[i * out_dim + j] - max_val);
        
        loss += -std::log(std::exp(Z_h[i * out_dim + labels_h[i]] - max_val) / sum_exp);
        
        max_val = -1e9;
        int pred = -1;
        for (int j = 0; j < out_dim; j++) {
            if (Z_h[i * out_dim + j] > max_val) {
                max_val = Z_h[i * out_dim + j];
                pred = j;
            }
        }
        if (pred == labels_h[i]) correct++;
    }
    accuracy = static_cast<double>(correct) / data.num_nodes;
    return loss / data.num_nodes;
}

void GCNModel::train(int num_epochs) {
    std::cout << "\nStarting CUDA Training..." << std::endl;
    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        forward();
        backward(epoch);

        double acc = 0.0;
        double loss = compute_loss_and_accuracy(acc);
        printf("Epoch %03d: Loss = %.4f, Accuracy = %.2f%%\n", epoch, loss, acc * 100.0);
    }
}