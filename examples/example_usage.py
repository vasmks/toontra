from pathlib import Path

from toontra import Toontra

sample = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "toontra"
    / "assets"
    / "sample_webtoon.png"
)

image_paths = [sample]

toontra = Toontra()
results = toontra.process(image_paths)

for page in results:
    cleaned_image = page.cleaned

    for bubble in page.bubbles:
        box = bubble.detection.box.as_tuple()
        text = bubble.recognition.text if bubble.recognition else None
        translation = bubble.translation

print(f"Processed {len(results)} page(s)")
print(f"Detected {sum(len(page.bubbles) for page in results)} bubble(s)")
