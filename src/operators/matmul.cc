#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================

        const Tensor& A = inputs[0];
        const Tensor& B = inputs[1];
        size_t max_rank = std::max(A->getDims().size(), B->getDims().size());
        Shape a_dims = A->getDims();
        Shape b_dims = B->getDims();
        Shape result(max_rank, 1);

        if (transA) std::swap(a_dims[a_dims.size()-1], a_dims[a_dims.size()-2]); 
        if (transB) std::swap(b_dims[b_dims.size()-1], b_dims[b_dims.size()-2]);

        m = a_dims[a_dims.size()-2];
        n = b_dims[b_dims.size()-1];
        k = a_dims[a_dims.size()-1];

        for (size_t i = 0; i < max_rank; i++) {
            if (i == 0) {
                result[max_rank - 1 - i] = b_dims[b_dims.size() - 1];
                continue;
            }
            
            if (i == 1) {
                result[max_rank - 1 - i] = a_dims[a_dims.size() - 2];
                continue;
            }

            size_t a_dim = (i < a_dims.size()) ? a_dims[a_dims.size() - 1 - i] : 1;
            size_t b_dim = (i < b_dims.size()) ? b_dims[b_dims.size() - 1 - i] : 1;
            
            result[max_rank - 1 - i] = std::max(a_dim, b_dim);
        } 

        return {{result}}; 
    }

} // namespace infini