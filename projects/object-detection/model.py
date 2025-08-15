import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # COCO class names
        self.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
        ]
    
    def _load_model(self, model_path):
        """Load pre-trained or custom model"""
        if model_path:
            model = torch.load(model_path, map_location=self.device)
        else:
            # Use YOLOv5 for demonstration
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        model.to(self.device)
        model.eval()
        return model
    
    def detect(self, image_path, confidence_threshold=0.5):
        """
        Detect objects in image
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detections with bounding boxes and labels
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        with torch.no_grad():
            results = self.model(image_rgb)
        
        # Process results
        detections = []
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf > confidence_threshold:
                x1, y1, x2, y2 = map(int, box)
                label = self.classes[int(cls)] if int(cls) < len(self.classes) else f"class_{int(cls)}"
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'label': label,
                    'class_id': int(cls)
                })
        
        return detections
    
    def draw_detections(self, image_path, detections, output_path=None):
        """Draw bounding boxes on image"""
        image = cv2.imread(image_path)
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            label = f"{detection['label']}: {detection['confidence']:.2f}"
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image

# Example usage
if __name__ == "__main__":
    detector = ObjectDetector()
    
    # Detect objects
    detections = detector.detect("sample_image.jpg", confidence_threshold=0.5)
    
    # Draw results
    result_image = detector.draw_detections("sample_image.jpg", detections, "output.jpg")
    
    print(f"Found {len(detections)} objects:")
    for det in detections:
        print(f"- {det['label']}: {det['confidence']:.2f}")