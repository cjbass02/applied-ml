"""Single source of truth for image transforms.

EMNIST raw bytes are stored "sideways" (rotated -90 from the upright glyph and
mirrored across the vertical axis). The dataset transform below rotates and
flips them back to upright before they hit the model.

The personal handwriting collection app saves images in EMNIST-native byte
orientation (i.e. pre-rotated and pre-flipped) so that the same
``emnist_transform`` correctly re-uprights them. That keeps the model's input
distribution identical between EMNIST and our personal data.
"""

from __future__ import annotations

import torchvision.transforms as T

emnist_transform = T.Compose(
    [
        T.ToTensor(),
        T.Lambda(lambda x: T.functional.rotate(x, angle=-90)),
        T.Lambda(lambda x: T.functional.hflip(x)),
    ]
)


personal_transform = T.Compose(
    [
        T.Resize((28, 28)),
        T.ToTensor(),
        T.Lambda(lambda x: T.functional.rotate(x, angle=-90)),
        T.Lambda(lambda x: T.functional.hflip(x)),
    ]
)
