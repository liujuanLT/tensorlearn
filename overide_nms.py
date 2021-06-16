import unittest
import torch
import torch.utils.cpp_extension
from torch.onnx.symbolic_helper import parse_args

def register_custom_nms():
        op_source = """
        #include <torch/script.h>
        #include <vector>
        #include <tuple>

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> custom_nms(torch::Tensor boxes, 
        torch::Tensor scores, 
        torch::Tensor max_output_boxes_per_class, 
        torch::Tensor iou_threshold,
        torch::Tensor score_threshold) {
            return std::make_tuple(boxes, scores, boxes, scores);
        }

        static auto registry =
        torch::RegisterOperators("custom_namespace::custom_nms", &custom_nms);
        """
        torch.utils.cpp_extension.load_inline(
            name="custom_nms",
            cpp_sources=op_source,
            is_python_module=False,
            verbose=True,
        )

        @parse_args('v', 'v', 'v', 'v', 'v')
        def symbolic_custom_nms(g, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
            outs = g.op("mydomain::BatchedNMS_TRT", boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, outputs=4)
            return outs

        from torch.onnx import register_custom_op_symbolic
        register_custom_op_symbolic("custom_namespace::custom_nms", symbolic_custom_nms, 11)
        print("register pytorch cumstom operator: custom_namespace::custom_nms -> (symbolic) NonMaxSuppression")        
        
class TestCustomOps(unittest.TestCase):
    def test_custom_nms(self):
        class CustomNmsModel(torch.nn.Module):
            def forward(self, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
                nmsed_num_detections, nmsed_boxes, nmsed_scores, nmsed_labels = \
                    torch.ops.custom_namespace.custom_nms(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
                return nmsed_num_detections, nmsed_boxes, nmsed_scores, nmsed_labels
        
        # fake input
        batch_size = 1
        num_class = 80
        num_detections = 3652
        after_top_k = 200
        boxes = torch.rand(batch_size, num_detections, 1, 4)
        scores = torch.rand(batch_size, num_detections, num_class)
        # boxes = torch.randn(batch_size, num_detections, 4, requires_grad=False)
        # scores = torch.randn(batch_size, num_class, num_detections, requires_grad=False)        
        # max_output_boxes_per_class = torch.tensor([after_top_k], dtype=torch.int64)
        # max_output_boxes_per_class = torch.tensor([after_top_k], dtype=torch.long)
        max_output_boxes_per_class = torch.LongTensor([after_top_k])   # TODO, or int64?
        iou_threshold = torch.tensor([0.5], dtype=torch.float32)
        score_threshold = torch.tensor([0.02], dtype=torch.float32)
        print("boxes={}".format(boxes))
        print("scores={}".format(scores))
        print("max_output_boxes_per_class".format(max_output_boxes_per_class))
        print("iou_threshold".format(iou_threshold))
        print("score_threshold".format(score_threshold))

        model = CustomNmsModel()
        nmsed_num_detections, nmsed_boxes, nmsed_scores, nmsed_labels  = \
         model.forward(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)

        print("nmsed_num_detections={}".format(nmsed_num_detections))
        print("nmsed_boxes={}".format(nmsed_boxes))
        print("nmsed_scores={}".format(nmsed_scores))
        print("nmsed_labels={}".format(nmsed_labels))
        torch.onnx.export(model, 
        (boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold), 
         "/home/jliu/data/models/cumstom_op_nms.onnx", 
         input_names=["boxes", "scores", "max_output_boxes_per_class","iou_threshold", "score_threshold"],
         output_names=['nmsed_num_detections', 'nmsed_boxes', 'nmsed_scores', 'nmsed_labels'], 
         verbose=True, 
         opset_version=11,
         custom_opsets={"mydomain": 1})
        print("done")

if __name__ == "__main__":
    register_custom_nms()
    unittest.main()
