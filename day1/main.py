from PIL import ImageDraw
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
from os import path

detect_raw = './cats.jpg'
image = Image.open(detect_raw)

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
# processor = DetrImageProcessor.from_pretrained("./hub")
# model = DetrForObjectDetection.from_pretrained("./hub")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
print(f"target_sizes: {target_sizes}")
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.9)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )


def draw_detect_result(img, results):
    draw = ImageDraw.Draw(img)
    offset = 20
    color = 'blue'
    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
        box = [round(i, 2) for i in box.tolist()]
        label = model.config.id2label[label.item()]
        score = round(score.item(), 3)
        draw.rectangle(box, outline='red', width=3)
        draw.text((box[0], box[1]), '{}:{}'.format(label, score), fill=color)

    return img




def out_name(in_name):
    last_dot = in_name.rfind('.')
    v = in_name[:last_dot] + '_detect' + in_name[last_dot:]
    return v

img = draw_detect_result(
    image, results)
# save img
img.save(path.join("out",out_name(detect_raw)))
