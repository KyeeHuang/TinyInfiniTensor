#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <unordered_set>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================

        std::unordered_set<UidBaseType> deleted_ops;
        std::unordered_set<UidBaseType> deleted_tensors;
        OpVec optimized_ops;
        TensorVec optimized_tensors;

        for (auto &op : ops) {
            if (deleted_ops.count(op->getGuid())) continue;
            if (op->getOpType() == OpType::Transpose) {
                auto fops = op->getPredecessors();
                for (auto &fop : fops) {
                    if (fop->getOpType() == OpType::Transpose) {
                        auto transop = std::dynamic_pointer_cast<TransposeObj>(op);
                        Shape dims = transop->getPermute();
                        auto transfop = std::dynamic_pointer_cast<TransposeObj>(fop);
                        Shape fdims = transfop->getPermute();
                        
                        if (dims.size() != fdims.size()) continue;

                        bool IsDiff = false;
                        size_t tranCnt = 0;
                        for (size_t i = 0; i < dims.size(); i++) {
                            if (dims[i] != fdims[i]) {
                                IsDiff = true;
                                break;
                            } else {
                                tranCnt += (i != dims[i]);
                            }
                        }

                        if (tranCnt != 2 || IsDiff) continue;

                        auto ffops = fop->getPredecessors();
                        auto bops = op->getSuccessors();
                        deleted_ops.insert(fop->getGuid());
                        deleted_ops.insert(op->getGuid());

                        deleted_tensors.insert(fop->getOutput()->getFuid());
                        deleted_tensors.insert(op->getOutput()->getFuid());
                        auto SavedTensor = fop->getInputs(0);
                        SavedTensor->removeTarget(fop);

                        for (auto &ffop : ffops) {
                            ffop->removeSuccessors(fop);
                            for (auto &bop : bops) ffop->addSuccessors(bop);
                        }
                        for (auto &bop : bops) {
                            bop->removePredecessors(op);
                            for (auto &ffop : ffops) bop->addPredecessors(ffop);
                            SavedTensor->addTarget(bop);  
                        }

                        for (auto &bop : bops) {
                            auto input = bop->getInputs(0);
                            bop->replaceInput(input, SavedTensor);
                        }

                        // T1 -> OP -> T2 -> OP -> T3 -> OP
                    }
                } 
            }
            
            if (op->getOpType() == OpType::MatMul) {
                const Tensor& A = op->getInputs()[0];
                const Tensor& B = op->getInputs()[1];
                auto matop = std::dynamic_pointer_cast<MatmulObj>(op);

                // T1 -> TransOP -> T2 -> MatOP -> T3

                auto fop = A->getSource();
                if (fop != nullptr && !deleted_ops.count(fop->getGuid()) && fop->getOpType() == OpType::Transpose) {
                    auto transfop = std::dynamic_pointer_cast<TransposeObj>(fop);
                    if (!transfop) continue;
                    
                    Shape dims = transfop->getPermute();
                    if (dims.size() >= 2 &&
                        dims[dims.size()-1] == dims.size()-2 &&
                        dims[dims.size()-2] == dims.size()-1) {

                        matop->setTransA(!matop->getTransA());
                        auto ffops = fop->getPredecessors();

                        deleted_ops.insert(fop->getGuid());
                        deleted_tensors.insert(fop->getOutput()->getFuid()); 
                        auto SavedTensor = fop->getInputs(0);

                        matop->removePredecessors(fop);
                        for (auto &ffop : ffops) {
                            ffop->removeSuccessors(fop);
                            ffop->addSuccessors(matop);
                            matop->addPredecessors(ffop);
                        }

                        SavedTensor->removeTarget(fop);
                        SavedTensor->addTarget(matop);
                        auto input = matop->getInputs(0);
                        matop->replaceInput(input, SavedTensor);
                    }
                }

                fop = B->getSource();
                if (fop != nullptr && fop->getOpType() == OpType::Transpose) {
                    auto transfop = std::dynamic_pointer_cast<TransposeObj>(fop);
                    if (!transfop) continue;
                    
                    Shape dims = transfop->getPermute();
                    if (dims.size() >= 2 &&
                        dims[dims.size()-1] == dims.size()-2 &&
                        dims[dims.size()-2] == dims.size()-1) {

                        matop->setTransB(!matop->getTransB());
                        auto ffops = fop->getPredecessors();

                        deleted_ops.insert(fop->getGuid());
                        deleted_tensors.insert(fop->getOutput()->getFuid()); 
                        auto SavedTensor = fop->getInputs(0);

                        matop->removePredecessors(fop);
                        for (auto &ffop : ffops) {
                            ffop->removeSuccessors(fop);
                            ffop->addSuccessors(matop);
                            matop->addPredecessors(ffop);
                        }

                        SavedTensor->removeTarget(fop);
                        SavedTensor->addTarget(matop);
                        auto input = matop->getInputs(1);
                        matop->replaceInput(input, SavedTensor);
                    }
                }
            }
        }

        for (auto &op : ops) {
            if (deleted_ops.find(op->getGuid()) == deleted_ops.end()) {
                optimized_ops.push_back(op);
            }
        }

        for (auto &tensor : tensors) {
            if (deleted_tensors.find(tensor->getFuid()) == deleted_tensors.end()) {
                optimized_tensors.push_back(tensor);
            }
        }

        ops = std::move(optimized_ops);
        tensors = std::move(optimized_tensors);
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================

        auto n = tensors.size();
        vector<size_t> offsets(n);

        for (size_t i = 0; i < n; i++) {
            offsets[i] = allocator.alloc(tensors[i]->getBytes());
        }
        auto hptr = allocator.getPtr();

        for (size_t i = 0; i < n; i++) {
            tensors[i]->setDataBlob(make_ref<BlobObj>(runtime, static_cast<char*>(hptr) + offsets[i]));
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini