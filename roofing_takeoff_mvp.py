"""
Roofing Takeoff MVP
Single-file prototype that trains (optional) and runs inference for detecting roof areas
on construction drawings, computes areas using a provided scale, and exports a CSV

Files / features included in this single-file prototype:
- Dataset scaffold for COCO-style annotations (for fine-tuning Mask R-CNN)
- Training loop (torch / torchvision) to fine-tune a pretrained Mask R-CNN
- Inference pipeline to run the model on images/PDFs and extract roof masks
- Simple area calculation (pixels -> m^2) using user-supplied scale (px_per_meter)
- Streamlit-based lightweight UI for uploading drawings, setting scale, visual review,
  and exporting an Excel/CSV takeoff per drawing.

Limitations & assumptions (please read):
1. This is an MVP scaffold. Accurate roofing detection requires annotated training
   data (images + polygon masks labeled as 'roof'). You said you have past plans —
   those can be annotated (COCO format) and used to fine-tune this model.
2. Automatic scale detection from title blocks is hard and fragile. The app will ask
   you to provide a scale (pixels per meter / pixels per mm), or a known dimension
   on the plan to calibrate automatically.
3. Running training/inference requires a GPU for reasonable speed. CPU will work but
   be slow.

How to use (quick):
1. Create a virtualenv and install requirements: pip install -r requirements.txt
   (see `requirements()` in this file for the set)
2. Optional: annotate a small dataset (10-200 images) in COCO format with category
   'roof' and use the `--train` mode to fine-tune.
3. Run the Streamlit app: streamlit run roofing_takeoff_mvp.py -- --app
4. Upload drawings (images or multipage PDF), set scale, run inference, review,
   and export CSV/XLSX.

Notes about "original code":
This file is your original codebase to iterate on. It's intentionally explicit and
commented so you can extend it (add symbol recognition, title-block scale parsing, or
trade-specific measurement rules).

"""

# ---- Imports -----------------------------------------------------------------
import os
import io
import json
import math
import argparse
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
import pandas as pd

# PyTorch / Torchvision for model
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T

# For PDF -> image conversion
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

# For OCR (optional scale parsing)
try:
    import pytesseract
except Exception:
    pytesseract = None

# Streamlit UI
try:
    import streamlit as st
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st = None

# ---- Helper functions --------------------------------------------------------

def requirements() -> List[str]:
    """Return the pip packages required for this script."""
    return [
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "pillow",
        "numpy",
        "pandas",
        "pdf2image",
        "pytesseract",
        "streamlit",
        "streamlit-drawable-canvas",
        "openpyxl",
    ]


# ----------------- Dataset scaffold (COCO-like) -------------------------------
class RoofingDataset(Dataset):
    """
    A minimal COCO-style dataset wrapper.
    Expects a folder with images and an annotations JSON file in COCO format.
    The category name for roofs should be 'roof' (or id 1).
    """

    def __init__(self, images_dir: str, annotations_json: str, transforms=None):
        with open(annotations_json, 'r') as f:
            self.coco = json.load(f)

        # map image id -> file
        self.images = {img['id']: img for img in self.coco['images']}
        # annotations grouped by image id
        self.anns = {}
        for ann in self.coco['annotations']:
            self.anns.setdefault(ann['image_id'], []).append(ann)

        self.images_dir = images_dir
        self.ids = list(self.images.keys())
        self.transforms = transforms

        # category mapping (name -> id)
        self.cat_map = {c['name']: c['id'] for c in self.coco['categories']}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_meta = self.images[self.ids[idx]]
        img_path = os.path.join(self.images_dir, img_meta['file_name'])
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        h, w = img.shape[:2]

        anns = self.anns.get(img_meta['id'], [])
        masks = []
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            # COCO polygons -> mask
            poly = ann.get('segmentation', [])
            if not poly:
                continue
            # here we handle single polygon
            rle_mask = poly_to_mask(poly, h, w)
            masks.append(rle_mask)
            # bounding box
            bbox = ann.get('bbox', [0, 0, 0, 0])
            boxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
            labels.append(ann.get('category_id', 1))
            areas.append(ann.get('area', float(np.sum(rle_mask))))
            iscrowd.append(ann.get('iscrowd', 0))

        target = {}
        if masks:
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
            target['masks'] = torch.as_tensor(np.stack(masks, 0), dtype=torch.uint8)
            target['image_id'] = torch.tensor([img_meta['id']])
            target['area'] = torch.as_tensor(areas)
            target['iscrowd'] = torch.as_tensor(iscrowd)
        else:
            # empty target
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['masks'] = torch.zeros((0, h, w), dtype=torch.uint8)
            target['image_id'] = torch.tensor([img_meta['id']])
            target['area'] = torch.zeros((0,))
            target['iscrowd'] = torch.zeros((0,))

        img = Image.fromarray(img)
        if self.transforms:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)

        return img, target


def poly_to_mask(polys: List[List[float]], h: int, w: int) -> np.ndarray:
    """Convert COCO polygon segmentation (list of floats) to binary mask."""
    import shapely.geometry as geom
    import shapely.ops as ops
    from shapely.geometry import Polygon
    from shapely.affinity import affine_transform

    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        if not poly:
            continue
        # poly is [x1,y1,x2,y2,...]
        coords = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
        try:
            p = Polygon(coords)
            if not p.is_valid:
                p = p.buffer(0)
            if p.area == 0:
                continue
            # rasterize using PIL ImageDraw
            from PIL import ImageDraw
            img = Image.new('L', (w, h), 0)
            ImageDraw.Draw(img).polygon(coords, outline=1, fill=1)
            mask = np.maximum(mask, np.array(img, dtype=np.uint8))
        except Exception:
            continue
    return mask


# ----------------- Model helpers ------------------------------------------------

def get_instance_segmentation_model(num_classes: int):
    """Load a Mask R-CNN pre-trained on COCO and adjust heads for `num_classes`."""
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


# ----------------- Training loop (simple) -------------------------------------

def train_model(train_images_dir: str, train_annotations: str, out_dir: str,
                num_epochs: int = 10, batch_size: int = 2, device: str = 'cuda'):
    os.makedirs(out_dir, exist_ok=True)
    dataset = RoofingDataset(train_images_dir, train_annotations,
                              transforms=T.Compose([T.ToTensor()]))

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    num_classes = max([c['id'] for c in dataset.coco['categories']]) + 1
    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        i = 0
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch} iter {i} loss {losses.item():.4f}")
            i += 1

        lr_scheduler.step()
        torch.save(model.state_dict(), os.path.join(out_dir, f'model_epoch_{epoch}.pth'))
    print('Training complete. Models saved to', out_dir)


def collate_fn(batch):
    return tuple(zip(*batch))


# ----------------- Inference ---------------------------------------------------

def load_model_for_inference(model_path: str = None, num_classes: int = 2, device: str = 'cuda'):
    model = get_instance_segmentation_model(num_classes)
    if model_path and os.path.exists(model_path):
        sd = torch.load(model_path, map_location='cpu')
        model.load_state_dict(sd)
    model.eval()
    model.to(device)
    return model


def image_from_pdf(pdf_path: str) -> List[Image.Image]:
    if convert_from_path is None:
        raise RuntimeError('pdf2image not installed')
    pages = convert_from_path(pdf_path, dpi=300)
    return pages


def infer_on_image(model, pil_image: Image.Image, device: str = 'cuda',
                   score_threshold: float = 0.5) -> List[Dict]:
    """Run inference and return a list of detections with masks and boxes.

    Returns: list of {mask: np.ndarray (H,W), box: [x1,y1,x2,y2], score: float}
    """
    img = pil_image.convert('RGB')
    img_t = T.ToTensor()(img).to(device)
    with torch.no_grad():
        outputs = model([img_t])
    out = outputs[0]
    results = []
    for i in range(len(out['scores'])):
        score = float(out['scores'][i])
        if score < score_threshold:
            continue
        mask = out['masks'][i, 0].mul(255).byte().cpu().numpy()
        box = out['boxes'][i].cpu().numpy().tolist()
        label = int(out['labels'][i].cpu().numpy())
        results.append({'mask': mask, 'box': box, 'score': score, 'label': label})
    return results


def mask_area_pixels(mask: np.ndarray) -> int:
    return int(np.sum(mask > 127))


def compute_area_m2(mask_px: np.ndarray, px_per_meter: float) -> float:
    """Convert mask pixels area to square meters using pixels-per-meter calibration."""
    px_area = mask_area_pixels(mask_px)
    if px_per_meter <= 0:
        return 0.0
    meters2 = px_area / (px_per_meter ** 2)
    return float(meters2)


# ----------------- Utilities: visualization & export --------------------------

def visualize_detections(pil_image: Image.Image, detections: List[Dict]) -> Image.Image:
    """Overlay detections on image (masks + boxes + scores)."""
    import matplotlib.pyplot as plt
    from matplotlib import patches
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(pil_image)
    for d in detections:
        mask = d['mask']
        box = d['box']
        score = d['score']
        # contour
        from skimage import measure
        contours = measure.find_contours(mask > 127, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        # box
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1]-5, f"{score:.2f}", color='yellow', fontsize=12, backgroundcolor='black')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def export_takeoff_csv(results: List[Dict], out_path: str):
    rows = []
    for r in results:
        rows.append({
            'image': r.get('image_name', ''),
            'detection_id': r.get('id', ''),
            'label': r.get('label', 1),
            'score': r.get('score', 0),
            'area_m2': r.get('area_m2', 0.0),
            'area_px': r.get('area_px', 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print('Exported takeoff to', out_path)


# ----------------- Streamlit app ------------------------------------------------

def run_streamlit_app(model_path: str = None, device: str = 'cuda'):
    if st is None:
        raise RuntimeError('Streamlit or streamlit-drawable-canvas not installed')

    st.title('Roofing Takeoff MVP')
    st.markdown('Upload drawings (images or PDF). Provide pixels-per-meter scale or click a known dim.')

    uploaded_files = st.file_uploader('Upload images or PDFs', accept_multiple_files=True, type=['pdf', 'png', 'jpg', 'jpeg', 'tiff'])

    px_per_meter = st.number_input('Pixels per meter (px/m). If you know a known dimension on the plan, enter it and click Calibrate.', min_value=1.0, value=100.0)

    score_threshold = st.slider('Score threshold', 0.0, 1.0, 0.5)

    if st.button('Load model'):
        with st.spinner('Loading model...'):
            model = load_model_for_inference(model_path, num_classes=2, device=device)
        st.session_state['model'] = model
        st.success('Model loaded')

    if 'model' not in st.session_state:
        st.info('Load the model first (you can use a pre-trained checkpoint or your fine-tuned model).')

    if uploaded_files and 'model' in st.session_state:
        all_results = []
        for f in uploaded_files:
            name = f.name
            if name.lower().endswith('.pdf'):
                if convert_from_path is None:
                    st.error('pdf2image not installed - cannot read PDF')
                    continue
                # save temp
                tmp_path = os.path.join('tmp', name)
                os.makedirs('tmp', exist_ok=True)
                with open(tmp_path, 'wb') as out:
                    out.write(f.getbuffer())
                pages = image_from_pdf(tmp_path)
                images = pages
            else:
                images = [Image.open(f).convert('RGB')]

            for i, img in enumerate(images):
                st.write(f'### {name} - page {i+1}')
                model = st.session_state['model']
                with st.spinner('Running inference...'):
                    dets = infer_on_image(model, img, device=device, score_threshold=score_threshold)
                # compute areas
                page_results = []
                for idx, d in enumerate(dets):
                    area_px = mask_area_pixels(d['mask'])
                    area_m2 = compute_area_m2(d['mask'], px_per_meter)
                    page_results.append({
                        'image_name': f'{name}_p{i+1}',
                        'id': f'{name}_p{i+1}_{idx}',
                        'label': d['label'],
                        'score': d['score'],
                        'area_px': area_px,
                        'area_m2': area_m2,
                    })
                st.write(pd.DataFrame(page_results))
                vis = visualize_detections(img, dets)
                st.image(vis)
                all_results.extend(page_results)

        if all_results:
            out_csv = st.text_input('Output CSV filename', value='roofing_takeoff.csv')
            if st.button('Export CSV'):
                export_takeoff_csv(all_results, out_csv)
                st.success(f'Exported to {out_csv}')


# ----------------- CLI --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Roofing takeoff MVP utility')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--train_images', type=str, help='Train images dir')
    parser.add_argument('--train_ann', type=str, help='Train annotations JSON (COCO)')
    parser.add_argument('--out_dir', type=str, default='models', help='Output models dir')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model_path', type=str, help='Model checkpoint for inference')
    parser.add_argument('--infer_image', type=str, help='Image file (or PDF) to run inference on')
    parser.add_argument('--px_per_meter', type=float, default=100.0, help='Pixels per meter calibration')
    parser.add_argument('--app', action='store_true', help='Run Streamlit app')
    parser.add_argument('--device', type=str, default='cuda', help='Device for torch')

    args = parser.parse_args()

    if args.train:
        if not args.train_images or not args.train_ann:
            print('Provide --train_images and --train_ann')
            return
        train_model(args.train_images, args.train_ann, args.out_dir, num_epochs=args.epochs, device=args.device)
        return

    if args.app:
        run_streamlit_app(model_path=args.model_path, device=args.device)
        return

    if args.infer_image:
        model = load_model_for_inference(args.model_path, num_classes=2, device=args.device)
        # support pdf
        if args.infer_image.lower().endswith('.pdf'):
            pages = image_from_pdf(args.infer_image)
            images = pages
        else:
            images = [Image.open(args.infer_image).convert('RGB')]

        results = []
        for i, img in enumerate(images):
            dets = infer_on_image(model, img, device=args.device, score_threshold=0.5)
            for idx, d in enumerate(dets):
                area_px = mask_area_pixels(d['mask'])
                area_m2 = compute_area_m2(d['mask'], args.px_per_meter)
                results.append({
                    'image_name': f'{os.path.basename(args.infer_image)}_p{i+1}',
                    'id': f'{i}_{idx}',
                    'label': d['label'],
                    'score': d['score'],
                    'area_px': area_px,
                    'area_m2': area_m2,
                })
        out = os.path.splitext(os.path.basename(args.infer_image))[0] + '_takeoff.csv'
        export_takeoff_csv(results, out)
        return

    parser.print_help()


if __name__ == '__main__':
    main()
