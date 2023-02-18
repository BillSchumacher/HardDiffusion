"""Tests for the generate app."""
import os
from uuid import uuid4

from django.conf import settings
from django.test import TestCase

from PIL import Image, ImageChops

from generate.models import GeneratedImage
from generate.tasks import render_image


def image_diff(img1, img2):
    """Return the difference between two images."""
    return ImageChops.difference(img1, img2).getbbox()


class TestGeneratedImage(TestCase):
    """Test the generate app."""

    def setUp(self) -> None:
        self.params = {
            "prompt": "A photograph of an astronaut riding a horse.",
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "height": 512,
            "width": 512,
        }
        self.seed = 485229998
        self.model = settings.DEFAULT_TEXT_TO_IMAGE_MODEL
        self.nsfw = False
        self.generated_image = GeneratedImage.objects.create(
            task_id="00001111-2222-4333-5555-666677778888", **self.params
        )
        self.reference_image = Image.open(
            os.path.join(settings.MEDIA_ROOT, "reference_image.png")
        )

    def test_render_image(self):
        """Test the render_image function."""
        image, seed = render_image(
            self.model, self.nsfw, self.seed, self.params, self.generated_image
        )
        self.assertEqual(seed, self.seed)
        # image.save(os.path.join(settings.MEDIA_ROOT, f'{self.generated_image.task_id}.png'))
        self.assertEqual(image_diff(image, self.reference_image), None)

    def test_render_nfsw_image(self):
        """Test the render_image function."""
        image, seed = render_image(
            self.model, True, self.seed, self.params, self.generated_image
        )
        self.assertEqual(seed, self.seed)
        # image.save(os.path.join(settings.MEDIA_ROOT, f'{self.generated_image.task_id}-nsfw.png'))
        self.assertEqual(image_diff(image, self.reference_image), None)

    def test_nsfw_black_image(self):
        """Test the render_image function with nsfw=False."""
        self.params["prompt"] = "A woman wearing nothing."
        image, seed = render_image(
            self.model, False, self.seed, self.params, self.generated_image
        )
        # image.save(os.path.join(settings.MEDIA_ROOT, 'nsfw_black_image.png'))
        self.assertEqual(seed, self.seed)
        self.assertEqual(image_diff(image, Image.new("RGB", (512, 512), "black")), None)

    def test_nsfw_no_black_image(self):
        """Test the render_image function with nsfw=True."""
        self.params["prompt"] = "A woman wearing nothing."
        image, seed = render_image(
            self.model, True, self.seed, self.params, self.generated_image
        )
        # image.save(os.path.join(settings.MEDIA_ROOT, 'nsfw_not_black_image.png'))
        self.assertEqual(seed, self.seed)
        self.assertNotEqual(
            image_diff(image, Image.new("RGB", (512, 512), "black")), None
        )
