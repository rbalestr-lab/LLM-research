"""Unit test for loraexp_lib.
"""

import unittest

import loraexp_lib
import torch


class LoraexpLibTest(unittest.TestCase):

  def test_apply_x_square(self):
    x = torch.tensor([1, -2, 3, -4, 5])
    self.assertTrue(
        torch.equal(loraexp_lib._apply_x_square(x),
                    torch.tensor([1, -4, 9, -16, 25])))
    self.assertTrue(
        torch.equal(loraexp_lib._apply_x_square(x, sign=False),
                    torch.tensor([1, 4, 9, 16, 25])))

  def test_apply_x_square_root(self):
    x = torch.tensor([1, -4, 9, -16, 25])
    self.assertLessEqual(
        torch.norm(loraexp_lib._apply_x_square_root(x)
                   - torch.tensor([1, -2, 3, -4, 5]), p=2), 1e-5)
    self.assertLessEqual(
        torch.norm(loraexp_lib._apply_x_square_root(x, sign=False)
                   - torch.tensor([1, 2, 3, 4, 5]), p=2), 1e-5)

  def test_apply_baabba(self):
    lora_A = torch.nn.Linear(5, 2, dtype=torch.float32, bias=False)
    torch.nn.init.ones_(lora_A.weight)
    lora_B = torch.nn.Linear(2, 3, dtype=torch.float32, bias=False)
    torch.nn.init.ones_(lora_B.weight)
    dropout = torch.nn.Dropout(p=0.)
    x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
    with torch.no_grad():
      z2, z1 = loraexp_lib._apply_baabba(lora_A, lora_B, dropout, x)
      self.assertLessEqual(
          torch.norm(z1 - torch.tensor([3., 3., 3.]), p=2), 1e-5 * 3.)
      self.assertLessEqual(
          torch.norm(z2 - torch.tensor([180., 180., 180.]), p=2), 1e-5 * 180.)

  def test_apply_mask(self):
    x = torch.arange(10) / 10.
    self.assertTrue(
        torch.equal(
            loraexp_lib._apply_mask(3, 10, x),
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.8, 0.9])
        )
    )

  def test_apply_norm_scaling(self):
    z1 = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5]]])
    z2 = torch.tensor([[[0.2, 0.4, 0.6, 0.8, 1.0]]])
    self.assertLessEqual(
        torch.norm(
            loraexp_lib._apply_norm_scaling(z1, z2)
            - torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5]]]), p=2), 1e-5)

  def test_apply_sum(self):
    z1 = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5]]])
    z2 = torch.tensor([[[0.2, 0.4, 0.6, 0.8, 1.0]]])
    self.assertLessEqual(
        torch.norm(
            loraexp_lib._apply_sum(z1, z2, scale=False)
            - torch.tensor([[[0.3, 0.6, 0.9, 1.2, 1.5]]]), p=2), 1e-5)
    self.assertLessEqual(
        torch.norm(
            loraexp_lib._apply_sum(z1, z2)
            - torch.tensor([[[0.2, 0.4, 0.6, 0.8, 1.0]]]), p=2), 1e-5)
    self.assertLessEqual(
        torch.norm(
            loraexp_lib._apply_sum(z1, z2, mos=True)
            - torch.tensor([[[0.1512, 0.3050, 0.4612, 0.6199, 0.7809]]]), p=2),
        1e-4)

  def test_apply_layers(self):
    lora_A = torch.nn.Linear(5, 2, dtype=torch.float32, bias=False)
    torch.nn.init.ones_(lora_A.weight)
    lora_B = torch.nn.Linear(2, 3, dtype=torch.float32, bias=False)
    torch.nn.init.ones_(lora_B.weight)
    dropout = torch.nn.Dropout(p=0.)
    x = torch.tensor([[[0.1, -0.2, 0.3, -0.4, 0.5]]], dtype=torch.float32)

    results = {
        "x": torch.tensor([[[0.6000, 0.6000, 0.6000]]]),
        "x^2": torch.tensor([[[0.3000, 0.3000, 0.3000]]]),
        "ns-x^2": torch.tensor([[[1.1000, 1.1000, 1.1000]]]),
        "sqrt(x)": torch.tensor([[[0.9828, 0.9828, 0.9828]]]),
        "ns-sqrt(x)": torch.tensor([[[5.3015, 5.3015, 5.3015]]]),
        "mask": torch.tensor([[[0.2000, 0.2000, 0.2000]]]),
        "mask-scale": torch.tensor([[[0.8000, 0.8000, 0.8000]]]),
        "ba+baabba": torch.tensor([[[1.2, 1.2, 1.2]]]),
        "ba+baabba(x^2)": torch.tensor([[[0.6, 0.6, 0.6]]]),
        "ba+baabba(mask(x))": torch.tensor([[[0.4, 0.4, 0.4]]]),
        "ba+baabba(mask-scale(x))":
            torch.tensor([[[1.6, 1.6, 1.6]]]),
    }
    for superlinear, expected_result in results.items():
      z = loraexp_lib._apply_layers(
          superlinear=superlinear, r=2, in_features=5, lora_A=lora_A,
          lora_B=lora_B, dropout=dropout, x=x)
      self.assertLessEqual(torch.norm(z - expected_result, p=2), 1e-4)

    mos_results = {
        "ba+baabba": loraexp_lib._apply_sum(
            torch.tensor([[[36.0000, 36.0000, 36.0000]]]),
            torch.tensor([[[0.6000, 0.6000, 0.6000]]]), mos=True),
        "ba+baabba(x^2)": loraexp_lib._apply_sum(
            torch.tensor([[[18.0000, 18.0000, 18.0000]]]),
            torch.tensor([[[0.3000, 0.3000, 0.3000]]]), mos=True),
        "ba+baabba(mask(x))": loraexp_lib._apply_sum(
            torch.tensor([[[12.0000, 12.0000, 12.0000]]]),
            torch.tensor([[[0.2000, 0.2000, 0.2000]]]), mos=True),
        "ba+baabba(mask-scale(x))": loraexp_lib._apply_sum(
            torch.tensor([[[48.0000, 48.0000, 48.0000]]]),
            torch.tensor([[[0.8000, 0.8000, 0.8000]]]), mos=True),
    }
    for superlinear, expected_result in mos_results.items():
      z = loraexp_lib._apply_layers(
          superlinear=superlinear, r=2, in_features=5, lora_A=lora_A,
          lora_B=lora_B, dropout=dropout, x=x, mos=True)
      self.assertLessEqual(torch.norm(z - expected_result, p=2), 1e-4)

    noscale_results = {
        "ba+baabba": torch.tensor([[[36.6000, 36.6000, 36.6000]]]),
        "ba+baabba(x^2)": torch.tensor([[[18.3000, 18.3000, 18.3000]]]),
        "ba+baabba(mask(x))": torch.tensor([[[12.2000, 12.2000, 12.2000]]]),
        "ba+baabba(mask-scale(x))":
            torch.tensor([[[48.8000, 48.8000, 48.8000]]]),
    }
    for superlinear, expected_result in noscale_results.items():
      z = loraexp_lib._apply_layers(
          superlinear=superlinear, r=2, in_features=5, lora_A=lora_A,
          lora_B=lora_B, dropout=dropout, x=x, scale=False)
      self.assertLessEqual(torch.norm(z - expected_result, p=2), 1e-4)

    z1, z2 = loraexp_lib._apply_layers(
        superlinear="baabba", r=2, in_features=5, lora_A=lora_A,
        lora_B=lora_B, dropout=dropout, x=x)
    self.assertLessEqual(
        torch.norm(z1 - torch.tensor([[[36., 36., 36.]]]), p=2), 1e-4)
    self.assertLessEqual(
        torch.norm(z2 - torch.tensor([[[0.6, 0.6, 0.6]]]), p=2), 1e-4)

  @unittest.mock.patch("loraexp_lib._apply_layers")
  def test_inner_apply_layers_regular(self, mocked_apply_layers):
    layer = loraexp_lib.LinearExp(
        torch.nn.Linear(5, 3),
        adapter_name="default",
        r=2,
        lora_alpha=1,
        lora_dropout=0.0,
        fan_in_fan_out=False,
        is_target_conv_1d_layer=False,
        init_lora_weights=True,
        use_rslora=False,
        use_dora=False,
        m=None,
        use_lora0=False,
        superlinear=None,
        use_scaling_gamma=False,
    )
    layer.superlinear["default"] = "test"
    layer._apply_layers("default", None, None, None, None)
    mocked_apply_layers.assert_called_once_with(
        "test", 2, 5, None, None, None, None, mos=False, scale=True,
    )

  @unittest.mock.patch("loraexp_lib._apply_layers")
  def test_inner_apply_layers_mos(self, mocked_apply_layers):
    layer = loraexp_lib.LinearExp(
        torch.nn.Linear(5, 3),
        adapter_name="default",
        r=2,
        lora_alpha=1,
        lora_dropout=0.0,
        fan_in_fan_out=False,
        is_target_conv_1d_layer=False,
        init_lora_weights=True,
        use_rslora=False,
        use_dora=False,
        m=None,
        use_lora0=False,
        superlinear=None,
        use_scaling_gamma=False,
    )
    layer.superlinear["default"] = "mos:test"
    layer._apply_layers("default", None, None, None, None)
    mocked_apply_layers.assert_called_once_with(
        "test", 2, 5, None, None, None, None, mos=True, scale=True,
    )

  @unittest.mock.patch("loraexp_lib._apply_layers")
  def test_inner_apply_layers_noscale(self, mocked_apply_layers):
    layer = loraexp_lib.LinearExp(
        torch.nn.Linear(5, 3),
        adapter_name="default",
        r=2,
        lora_alpha=1,
        lora_dropout=0.0,
        fan_in_fan_out=False,
        is_target_conv_1d_layer=False,
        init_lora_weights=True,
        use_rslora=False,
        use_dora=False,
        m=None,
        use_lora0=False,
        superlinear=None,
        use_scaling_gamma=False,
    )
    layer.superlinear["default"] = "noscale:test"
    layer._apply_layers("default", None, None, None, None)
    mocked_apply_layers.assert_called_once_with(
        "test", 2, 5, None, None, None, None, mos=False, scale=False,
    )

  @unittest.mock.patch("loraexp_lib._apply_layers")
  def test_inner_apply_layers_none(self, mocked_apply_layers):
    layer = loraexp_lib.LinearExp(
        torch.nn.Linear(5, 3),
        adapter_name="default",
        r=2,
        lora_alpha=1,
        lora_dropout=0.0,
        fan_in_fan_out=False,
        is_target_conv_1d_layer=False,
        init_lora_weights=True,
        use_rslora=False,
        use_dora=False,
        m=None,
        use_lora0=False,
        superlinear=None,
        use_scaling_gamma=False,
    )
    layer.superlinear["default"] = None
    layer._apply_layers("default", None, None, None, None)
    mocked_apply_layers.assert_called_once_with(
        "x", 2, 5, None, None, None, None, mos=False, scale=True,
    )

  def test_accelerated_gamma(self):
    gamma = torch.tensor([0.002, 0.004, 0.0, -0.001])
    self.assertLessEqual(
        torch.norm(
            loraexp_lib._accelerated_gamma(gamma, 100.)
            - torch.tensor([0.3, 0.5, 0.1, 0.1]), p=2), 1e-4)

    layer = loraexp_lib.LinearExp(
        torch.nn.Linear(5, 3),
        adapter_name="default",
        r=2,
        lora_alpha=1,
        lora_dropout=0.0,
        fan_in_fan_out=False,
        is_target_conv_1d_layer=False,
        init_lora_weights=True,
        use_rslora=False,
        use_dora=False,
        m=None,
        use_lora0=False,
        superlinear=None,
        use_scaling_gamma=False,
    )
    x = torch.arange(10) / 10.
    self.assertLessEqual(
        torch.norm(
            layer._apply_scale("default", x)
            - torch.tensor(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        * 10. / 3., 1e-4)

    layer = loraexp_lib.LinearExp(
        torch.nn.Linear(5, 3),
        adapter_name="default",
        r=2,
        lora_alpha=1,
        lora_dropout=0.0,
        fan_in_fan_out=False,
        is_target_conv_1d_layer=False,
        init_lora_weights=True,
        use_rslora=False,
        use_dora=False,
        m=None,
        use_lora0=False,
        superlinear=None,
        use_scaling_gamma=False,
        scaling_gamma_acceleration=10.,
    )
    self.assertEqual(layer.scaling_gamma_acceleration, 10.)
    self.assertLessEqual(
        torch.norm(
            layer._apply_scale("default", x)
            - torch.tensor(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        * 10. / 3., 1e-4)


if __name__ == "__main__":
  unittest.main()
