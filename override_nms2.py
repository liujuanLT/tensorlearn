import unittest
import torch
import torch.utils.cpp_extension

class TestCustomAutogradFunction(unittest.TestCase):

    def test1(self):

        class DummyONNXNMSop(torch.autograd.Function):
            @staticmethod
            def forward(ctx, boxes, scores, max_output_boxes_per_class, iou_threshold,
                        score_threshold):
                return DummyONNXNMSop.output

            @staticmethod
            def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold,
                        score_threshold):
                return g.op(
                    'NonMaxSuppression', 
                    boxes,
                    scores,
                    max_output_boxes_per_class,
                    iou_threshold,
                    score_threshold,
                    outputs=1)

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # self.nms = DummyONNXNMSop.apply

            def forward(self, boxes, scores, max_output_boxes_per_class, iou_threshold,
                        score_threshold, labels=None):

                boxes = boxes + 0.1
                state = torch._C._get_tracing_state()
                num_fake_det = 2
                num_box = 3652
                batch_inds = torch.randint(batch_size, (num_fake_det, 1))
                cls_inds = torch.randint(num_class, (num_fake_det, 1))
                box_inds = torch.randint(num_box, (num_fake_det, 1))
                indices = torch.cat([batch_inds, cls_inds, box_inds], dim=1)
                output = indices
                setattr(DummyONNXNMSop, 'output', output)
                torch._C._set_tracing_state(state)

                # selected_indices = self.nms(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
                selected_indices = DummyONNXNMSop.apply(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
                        
                return 4*selected_indices

        batch_size = 1
        num_class = 80
        num_detections = 3652
        after_top_k = 200
        boxes = torch.rand(batch_size, num_detections, 4)
        scores = torch.rand(batch_size, num_class, num_detections)
        max_output_boxes_per_class = torch.LongTensor([after_top_k])   # TODO, or int64?
        iou_threshold = torch.tensor([0.5], dtype=torch.float32)
        score_threshold = torch.tensor([0.02], dtype=torch.float32)
        model = MyModule()
        nmsed_ret  = \
         model.forward(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)        
        torch.onnx.export(model, 
        (boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold), 
         "/home/jliu/data/models/cumstom_op_nms.onnx", 
         input_names=["boxes", "scores", "max_output_boxes_per_class","iou_threshold", "score_threshold"],
         output_names=['nmsed_ret'], 
         verbose=True, 
         opset_version=11)
        print("done")

if __name__ == "__main__":
    unittest.main()