import unittest
import torch
import torch.utils.cpp_extension

def register_custom_add():
        op_source = """
        #include <torch/script.h>

        torch::Tensor custom_add(torch::Tensor self, torch::Tensor other) {
        return self + other;
        }

        static auto registry =
        torch::RegisterOperators("custom_namespace::custom_add", &custom_add);
        """

        torch.utils.cpp_extension.load_inline(
            name="custom_add",
            cpp_sources=op_source,
            is_python_module=False,
            verbose=True,
        )

        def symbolic_custom_add(g, self, other):
            return g.op("Add", self, other)

        from torch.onnx import register_custom_op_symbolic
        register_custom_op_symbolic("custom_namespace::custom_add", symbolic_custom_add, 11)
        print("register pytorch cumstom operator: custom_namespace::custom_add -> (symbolic) Add")
       
        
class TestCustomOps(unittest.TestCase):
    def test_custom_add(self):
        class CustomAddModel(torch.nn.Module):
            self.output = None
            def forward(self, a, b):
                return torch.ops.custom_namespace.custom_add(a, b)
        
        x = torch.randn(2, 3, 4, requires_grad=False)
        y = torch.randn(2, 3, 4, requires_grad=False)
        model = CustomAddModel()
        z = model.forward(x, y)
        print("x={}".format(x))
        print("y={}".format(y))
        print("z={}".format(z))
        torch.onnx.export(model, (x,y), "/home/jliu/data/models/cumstom_op_Add.onnx", verbose=True, opset_version=11)
        print("done")
  
if __name__ == "__main__":
    register_custom_add()
    unittest.main()
