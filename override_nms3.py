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
                    'mydomain::BatchedNMS_TRT', 
                    boxes,
                    scores,
                    max_output_boxes_per_class,
                    iou_threshold,
                    score_threshold,
                    outputs=4)

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # self.nms = DummyONNXNMSop.apply

            def forward(self, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):

                boxes = boxes + 0.1
                state = torch._C._get_tracing_state()

                num_fake_det = 2
                dummy_num_detections = torch.tensor([[num_fake_det]]).expand(batch_size, 1)  # [1,1]
                dummy_boxes = torch.rand(batch_size, num_fake_det, 4)
                dummy_scores = torch.rand(batch_size, num_fake_det)
                dummy_labels = torch.randint(num_class, (batch_size, num_fake_det)) 
                setattr(DummyONNXNMSop, 'output', (dummy_num_detections, dummy_boxes, dummy_scores, dummy_labels))

                torch._C._set_tracing_state(state)

                # selected_indices = self.nms(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
                out1, out2, out3, out4 = DummyONNXNMSop.apply(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
                        
                return out1, 2*out2, out3, out4

        batch_size = 1
        num_class = 80
        num_detections = 3652
        after_top_k = 200
        boxes = torch.rand(batch_size, num_detections, 1, 4)
        scores = torch.rand(batch_size, num_detections, num_class)
        max_output_boxes_per_class = torch.LongTensor([after_top_k])   # TODO, or int64?
        iou_threshold = torch.tensor([0.5], dtype=torch.float32)
        score_threshold = torch.tensor([0.02], dtype=torch.float32)
        model = MyModule()
        nmsed_num_detections, nmsed_boxes, nmsed_scores, nmsed_labels  = \
         model.forward(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)        
        torch.onnx.export(model, 
        (boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold), 
         "/home/jliu/data/models/cumstom_op_nms.onnx", 
         input_names=["boxes", "scores", "max_output_boxes_per_class","iou_threshold", "score_threshold"],
         output_names=['nmsed_num_detections', 'nmsed_boxes', 'nmsed_scores', 'nmsed_labels'], 
         verbose=True, 
         opset_version=11)
        print("done")

if __name__ == "__main__":
    unittest.main()