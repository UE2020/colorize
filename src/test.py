import torch
import unittest

def leaky_relu(x, alpha=0.01):
    return torch.maximum(alpha * x, x)

class TestLeakyReLU(unittest.TestCase):
    def test_leaky_relu_positive(self):
        input_tensor = torch.tensor([0.0, 1.0, 2.0])
        output_tensor = leaky_relu(input_tensor)
        expected_output = torch.tensor([0.0, 1.0, 2.0])
        self.assertTrue(torch.all(torch.eq(output_tensor, expected_output)))

    def test_leaky_relu_negative(self):
        input_tensor = torch.tensor([-2.0, -1.0, -0.5])
        output_tensor = leaky_relu(input_tensor)
        expected_output = torch.tensor([-0.02, -0.01, -0.005])
        self.assertTrue(torch.all(torch.eq(output_tensor, expected_output)))

    def test_leaky_relu_custom_alpha(self):
        input_tensor = torch.tensor([-2.0, -1.0, -0.5])
        alpha = 0.1
        output_tensor = leaky_relu(input_tensor, alpha)
        expected_output = torch.tensor([-0.2, -0.1, -0.05])
        self.assertTrue(torch.all(torch.eq(output_tensor, expected_output)))

if __name__ == '__main__':
    unittest.main()