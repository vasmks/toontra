from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from toontra.contracts import BubbleMasker, validate_gray_mask
from toontra.errors import ModelContractError, OptionalDependencyError
from toontra.modules import ManetBubbleMasker
from toontra.modules.manet_bubble_masker import (
    _IMAGE_MEAN,
    _IMAGE_STD,
    _checkpoint_provenance,
    _preprocess,
    _remove_letterbox,
    _validate_checkpoint,
)

_MANET_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("segmentation_models_pytorch") is not None
)
_SKIP_REASON = "requires the optional 'manet' dependencies (pip install -e \".[manet]\")"


def valid_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "architecture": "MAnet",
        "encoder": "resnet34",
        "in_channels": 3,
        "classes": 1,
        "image_size": 512,
        "crop_padding": 0.15,
        "threshold": 0.45,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "training_dataset": "Roboflow manga-segment_v2",
        "training_dataset_version": 5,
        "best_epoch": 18,
        "validation_dice": 0.98,
        "test_dice": 0.98,
        "test_iou": 0.96,
        "state_dict": {"weight": object()},
    }
    payload.update(overrides)
    return payload


class PreprocessingTests(unittest.TestCase):
    """These never touch torch -- preprocessing is plain NumPy/OpenCV."""

    def test_output_shape_is_512_batch(self) -> None:
        crop = np.full((80, 160, 3), 128, dtype=np.uint8)
        batch, _ = _preprocess(crop)
        self.assertEqual(batch.shape, (1, 3, 512, 512))
        self.assertEqual(batch.dtype, np.float32)

    def test_aspect_ratio_is_preserved_with_centered_letterbox(self) -> None:
        # A 100x400 (h, w) crop must scale to fit within 512 on its longer
        # side (width) without distortion, then be centered top/bottom.
        crop = np.zeros((100, 400, 3), dtype=np.uint8)
        _, (x0, y0, content_width, content_height) = _preprocess(crop)
        self.assertEqual(content_width, 512)
        self.assertEqual(content_height, 128)  # 100 * (512 / 400)
        self.assertEqual(x0, 0)
        self.assertEqual(y0, (512 - 128) // 2)

    def test_letterbox_padding_is_white(self) -> None:
        crop = np.zeros((100, 400, 3), dtype=np.uint8)  # black content
        batch, (x0, y0, content_width, content_height) = _preprocess(crop)
        # Un-normalize a corner pixel known to be outside the content box.
        pixel = batch[0, :, 0, 0] * _IMAGE_STD + _IMAGE_MEAN
        np.testing.assert_allclose(pixel, [1.0, 1.0, 1.0], atol=1e-3)
        self.assertGreater(y0, 0)

    def test_imagenet_normalization_is_applied(self) -> None:
        crop = np.full((512, 512, 3), 255, dtype=np.uint8)  # no padding, full white
        batch, _ = _preprocess(crop)
        expected = (1.0 - _IMAGE_MEAN) / _IMAGE_STD
        np.testing.assert_allclose(batch[0, :, 256, 256], expected, atol=1e-5)

    def test_remove_letterbox_crops_back_to_content_box(self) -> None:
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[10:20, 30:50] = 255
        restored = _remove_letterbox(mask, (30, 10, 20, 10))
        self.assertEqual(restored.shape, (10, 20))
        self.assertTrue(np.all(restored == 255))


class CheckpointValidationTests(unittest.TestCase):
    """Metadata validation is plain dict inspection -- no torch required."""

    def test_accepts_well_formed_payload(self) -> None:
        state_dict = _validate_checkpoint(valid_payload())
        self.assertEqual(list(state_dict), ["weight"])

    def test_rejects_wrong_architecture(self) -> None:
        with self.assertRaisesRegex(ModelContractError, "architecture"):
            _validate_checkpoint(valid_payload(architecture="Unet"))

    def test_rejects_wrong_encoder(self) -> None:
        with self.assertRaisesRegex(ModelContractError, "encoder"):
            _validate_checkpoint(valid_payload(encoder="efficientnet-b1"))

    def test_rejects_wrong_class_count(self) -> None:
        with self.assertRaisesRegex(ModelContractError, "classes"):
            _validate_checkpoint(valid_payload(classes=2))

    def test_rejects_wrong_image_size(self) -> None:
        with self.assertRaisesRegex(ModelContractError, "image_size"):
            _validate_checkpoint(valid_payload(image_size=256))

    def test_rejects_wrong_normalization(self) -> None:
        normalization = {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.229, 0.224, 0.225],
        }
        with self.assertRaisesRegex(ModelContractError, "normalization"):
            _validate_checkpoint(valid_payload(normalization=normalization))

    def test_rejects_out_of_range_threshold(self) -> None:
        with self.assertRaisesRegex(ModelContractError, "threshold"):
            _validate_checkpoint(valid_payload(threshold=1.5))

    def test_rejects_empty_state_dict(self) -> None:
        with self.assertRaisesRegex(ModelContractError, "state_dict"):
            _validate_checkpoint(valid_payload(state_dict={}))

    def test_public_checkpoint_provenance_is_recognized(self) -> None:
        source_url, license_spdx = _checkpoint_provenance(
            "6350601676c9e2b5448cf4cf109fb459a1f805d4101cfe34d4db47661f28df21"
        )
        self.assertIn("huggingface.co/toontra-research", source_url)
        self.assertEqual(license_spdx, "Apache-2.0")

    def test_unknown_checkpoint_has_no_public_provenance(self) -> None:
        source_url, license_spdx = _checkpoint_provenance("0" * 64)
        self.assertIsNone(source_url)
        self.assertIsNone(license_spdx)

class ConstructorErrorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="toontra-manet-")
        self.addCleanup(self.temporary_directory.cleanup)

    def test_missing_checkpoint_gives_a_clear_error(self) -> None:
        missing = Path(self.temporary_directory.name) / "does-not-exist.pth"
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            ManetBubbleMasker(missing)

    def test_missing_optional_dependency_gives_a_clear_error(self) -> None:
        checkpoint = Path(self.temporary_directory.name) / "checkpoint.pth"
        checkpoint.write_bytes(b"not a real checkpoint")
        with patch(
            "toontra.modules.manet_bubble_masker.importlib.import_module",
            side_effect=ImportError("no torch"),
        ):
            with self.assertRaises(OptionalDependencyError) as caught:
                ManetBubbleMasker(checkpoint)
        self.assertIn(".[manet]", str(caught.exception))

    def test_rejects_invalid_threshold_override(self) -> None:
        checkpoint = Path(self.temporary_directory.name) / "checkpoint.pth"
        checkpoint.write_bytes(b"not a real checkpoint")
        with self.assertRaises(ValueError):
            ManetBubbleMasker(checkpoint, threshold=1.5)


@unittest.skipUnless(_MANET_AVAILABLE, _SKIP_REASON)
class RealArchitectureTests(unittest.TestCase):
    """Build a real, randomly-initialized ResNet34 MA-Net so these tests
    exercise the actual construction and forward-pass code paths without
    depending on the large trained checkpoint or any network access.
    """

    @classmethod
    def setUpClass(cls) -> None:
        import segmentation_models_pytorch as smp
        import torch

        cls._temporary_directory = tempfile.TemporaryDirectory(prefix="toontra-manet-real-")
        model = smp.MAnet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        cls.checkpoint_path = Path(cls._temporary_directory.name) / "checkpoint.pth"
        torch.save(valid_payload(state_dict=model.state_dict()), cls.checkpoint_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def test_constructs_known_resnet34_manet_architecture(self) -> None:
        import segmentation_models_pytorch as smp

        masker = ManetBubbleMasker(self.checkpoint_path)
        self.assertIsInstance(masker._model, smp.MAnet)

    def test_no_network_download_during_construction_or_inference(self) -> None:
        import torch

        def _fail_download(*args: object, **kwargs: object) -> None:
            raise AssertionError("attempted to download weights over the network")

        with patch.object(torch.hub, "load_state_dict_from_url", side_effect=_fail_download):
            masker = ManetBubbleMasker(self.checkpoint_path)
            crop = np.full((64, 96, 3), 200, dtype=np.uint8)
            masker.create_mask(crop)  # no assertion needed; a download would raise above

    def test_checkpoint_metadata_is_read_correctly(self) -> None:
        masker = ManetBubbleMasker(self.checkpoint_path)
        self.assertEqual(masker.threshold, 0.45)
        self.assertEqual(masker.metadata.version, "Roboflow manga-segment_v2 v5")
        self.assertIsNone(masker.metadata.source_url)
        self.assertIsNone(masker.metadata.license_spdx)
        self.assertIsNotNone(masker.metadata.sha256)

    def test_threshold_override_takes_precedence_over_checkpoint(self) -> None:
        masker = ManetBubbleMasker(self.checkpoint_path, threshold=0.9)
        self.assertEqual(masker.threshold, 0.9)

    def test_result_follows_the_masker_contract(self) -> None:
        masker = ManetBubbleMasker(self.checkpoint_path)
        self.assertIsInstance(masker, BubbleMasker)
        crop = np.full((70, 130, 3), 210, dtype=np.uint8)
        mask = masker.create_mask(crop)
        validate_gray_mask(mask, crop.shape[:2])
        self.assertTrue(set(np.unique(mask).tolist()).issubset({0, 255}))

    def test_sigmoid_threshold_and_padding_removal(self) -> None:
        import torch

        masker = ManetBubbleMasker(self.checkpoint_path, threshold=0.5)
        crop = np.zeros((50, 100, 3), dtype=np.uint8)  # non-square -> real padding
        _, (x0, y0, content_width, content_height) = _preprocess(crop)
        self.assertGreater(y0, 0)  # sanity: this crop really does get padded

        # Padding scores strongly foreground, content scores strongly
        # background: if padding removal or resizing were wrong, some of that
        # positive padding signal would leak into the restored mask.
        logits = torch.full((1, 1, 512, 512), -10.0)
        logits[:, :, :y0, :] = 10.0
        logits[:, :, y0 + content_height :, :] = 10.0

        with patch.object(masker, "_model", return_value=logits):
            mask = masker.create_mask(crop)

        self.assertEqual(mask.shape, crop.shape[:2])
        self.assertTrue(np.all(mask == 0))

        logits.fill_(10.0)
        with patch.object(masker, "_model", return_value=logits):
            mask = masker.create_mask(crop)
        self.assertTrue(np.all(mask == 255))


@unittest.skipUnless(
    os.environ.get("TOONTRA_MANET_CHECKPOINT"),
    "set TOONTRA_MANET_CHECKPOINT to run the local MA-Net integration test",
)
class ManetBubbleMaskerLocalIntegrationTests(unittest.TestCase):
    def test_real_checkpoint_obeys_the_mask_contract(self) -> None:
        checkpoint = os.environ["TOONTRA_MANET_CHECKPOINT"]
        device = os.environ.get("TOONTRA_MANET_DEVICE", "cpu")
        masker = ManetBubbleMasker(checkpoint, device=device)

        crop = np.full((260, 420, 3), 255, dtype=np.uint8)
        crop[100:160, 120:300] = 0
        before = crop.copy()

        mask = masker.create_mask(crop)

        np.testing.assert_array_equal(crop, before)
        self.assertEqual(mask.shape, crop.shape[:2])
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(set(np.unique(mask).tolist()).issubset({0, 255}))


if __name__ == "__main__":
    unittest.main()
