import unittest
from autograd_py import variable


class VariableBackward(unittest.TestCase):

    def test_add(self):
        x = variable(5.0)
        y = variable(4.0)
        z = x + y
        self.assertAlmostEqual(z.value(), 9.0)
        z.backward()
        self.assertAlmostEqual(x.grad(), 1.0)
        self.assertAlmostEqual(y.grad(), 1.0)

    def test_order2(self):
        x = variable(5.0)
        y = variable(3.0)
        z = y * x * x + x * y * y
        self.assertAlmostEqual(z.value(), 120)
        z.backward()
        self.assertAlmostEqual(y.grad(), 55)
        self.assertAlmostEqual(x.grad(), 39)


if __name__ == "__main__":
    unittest.main()
