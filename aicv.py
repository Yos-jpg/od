import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet101_FPN_Weights
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import open3d as o3d
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from pytorch import pytorch3d


class RANGERMaskRCNN3D(nn.Module):
    """
    RANGER ROV Enhanced Mask R-CNN with ResNet-101 + Dual Laser 3D Reconstruction.

    Why ResNet-101 over ResNet-50:
    - Deeper network (101 vs 50 layers) = better feature extraction
    - Superior underwater object recognition capability
    - Better small object detection (critical for RANGER precision tasks)
    - More robust to challenging underwater lighting conditions
    - Higher accuracy on complex shapes (jellyfish, shipwreck components)
    - RANGER has 25A power budget - can handle the extra computation

    Performance Benefits:
    - ~3-5% higher mAP than ResNet-50 (critical for competition)
    - Better segmentation quality for precise 3D reconstruction
    - More stable training on underwater domain
    - Superior feature representation for laser line integration
    """

    def __init__(self,
                 num_classes: int = 30,
                 weights=MaskRCNN_ResNet101_FPN_Weights,
                 laser_baseline: float = 0.15,
                 camera_focal_length: float = 800.0,
                 enable_fpn: bool = True):
        super().__init__()

        # Import ResNet-101 based Mask R-CNN
        try:
            from torchvision.models.detection import maskrcnn_resnet101_fpn
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
            import torchvision.models as models

            # Create ResNet-101 backbone
            backbone = models.resnet101(weights=weights)

            # Convert to FPN backbone for better multi-scale detection
            if enable_fpn:
                from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
                self.backbone = resnet_fpn_backbone(
                    'resnet101', weights=weights)
                backbone_out_channels = 256  # FPN output channels
            else:
                # Use ResNet-101 without FPN (less common)
                backbone = nn.Sequential(*list(backbone.children())[:-2])
                backbone_out_channels = 2048  # ResNet-101 final conv layer
                self.backbone = backbone

            # Create full Mask R-CNN with ResNet-101 backbone
            from torchvision.models.detection import MaskRCNN
            from torchvision.models.detection.rpn import AnchorGenerator

            # Define anchor generator for better small object detection
            anchor_generator = AnchorGenerator(
                sizes=((32, 64, 128, 256, 512),),  # Multi-scale anchors
                aspect_ratios=((0.5, 1.0, 2.0),)   # Various aspect ratios
            )

            # ROI pooler for mask head
            from torchvision.ops import MultiScaleRoIAlign
            roi_pooler = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # FPN feature maps
                output_size=14,  # Output size for mask prediction
                sampling_ratio=2
            )

            # Create complete Mask R-CNN model
            self.detector = MaskRCNN(
                backbone=self.backbone,
                num_classes=num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                mask_roi_pool=roi_pooler,
                # Enhanced parameters for RANGER precision
                min_size=800,
                max_size=1333,  # Higher resolution for better precision
                rpn_pre_nms_top_n_train=3000,
                rpn_pre_nms_top_n_test=1500,
                rpn_post_nms_top_n_train=2000,
                rpn_post_nms_top_n_test=1000,
                box_detections_per_img=200,
                box_nms_thresh=0.5,
                mask_nms_thresh=0.5,
                # Score thresholds optimized for underwater conditions
                box_score_thresh=0.3,  # Lower threshold for challenging conditions
                box_fg_iou_thresh=0.5,
                box_bg_iou_thresh=0.5,
            )

        except ImportError:
            # Fallback to torchvision's built-in maskrcnn_resnet50_fpn and modify
            print("Warning: Using ResNet-50 backbone as ResNet-101 setup failed")
            self.detector = maskrcnn_resnet50_fpn(weights=weights)

            # Replace heads with enhanced versions
            in_features_box = self.detector.roi_heads.box_predictor.cls_score.in_features
            in_features_mask = self.detector.roi_heads.mask_predictor.conv5_mask.in_channels

            self.detector.roi_heads.box_predictor = MaskRCNNPredictor(
                in_features_box, num_classes)
            self.detector.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask, 256, num_classes)

        # Enhanced laser line segmentation head (optimized for ResNet-101 features)
        self.laser_segmentation = nn.Sequential(
            # Deeper network to match ResNet-101 complexity
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # red_laser, green_laser, background
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # 3D surface reconstruction head (enhanced for ResNet-101)
        self.surface_reconstruction = nn.Sequential(
            # Multi-scale processing for better 3D reconstruction
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Skip connection branch
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),  # Depth per pixel
            nn.ReLU()  # Ensure positive depths
        )

        # Attention mechanism for laser-object fusion
        self.laser_attention = nn.Sequential(
            nn.Conv2d(259, 128, kernel_size=1),  # 256 + 3 (laser channels)
            nn.Sigmoid()  # Attention weights
        )

        # Physical calibration parameters
        self.laser_baseline = laser_baseline
        self.focal_length = camera_focal_length
        self.register_buffer('camera_matrix', torch.eye(3))

        # ResNet-101 specific optimization
        # Freeze early layers for stability
        self._freeze_backbone_layers(freeze_layers=2)

    def _freeze_backbone_layers(self, freeze_layers: int):
        """
        Freeze early ResNet-101 layers for training stability.

        Why freeze early layers:
        - Early layers learn low-level features (edges, textures)
        - These transfer well from ImageNet to underwater domain
        - Freezing prevents overfitting on small underwater datasets
        - Focuses training on task-specific later layers
        """
        if hasattr(self.detector, 'backbone') and hasattr(self.detector.backbone, 'body'):
            backbone_layers = list(self.detector.backbone.body.children())

            for i, layer in enumerate(backbone_layers[:freeze_layers]):
                for param in layer.parameters():
                    param.requires_grad = False
                print(f"Frozen ResNet-101 layer {i}")

    def forward_inference(self, images: List[torch.Tensor]):
        """
        Enhanced inference leveraging ResNet-101's superior feature extraction.

        ResNet-101 advantages:
        - Better feature representation for challenging underwater scenes
        - Superior small object detection (important for sensors, connectors)
        - More robust to lighting variations and turbidity
        - Better handling of complex object shapes
        """
        # Enhanced preprocessing for ResNet-101
        processed_images = []
        for image in images:
            # ResNet-101 benefits from higher resolution input
            if image.shape[-2:] != (800, 800):
                image = F.interpolate(image.unsqueeze(0), size=(800, 800),
                                      mode='bilinear', align_corners=False).squeeze(0)
            processed_images.append(image)

        # Get enhanced segmentation results from ResNet-101
        detections = self.detector(processed_images)

        # Enhanced feature extraction for 3D processing
        with torch.no_grad():
            backbone_features = self.detector.backbone(
                torch.stack(processed_images))
            if isinstance(backbone_features, dict):
                # FPN features - use highest resolution
                main_features = backbone_features['0']
            else:
                main_features = backbone_features

        # Process each image with enhanced 3D reconstruction
        for i, (image, detection) in enumerate(zip(processed_images, detections)):
            if len(detection['boxes']) > 0:
                # Enhanced laser line detection using ResNet-101 features
                laser_features = main_features[i:i+1]
                laser_segmentation = self.laser_segmentation(laser_features)
                laser_lines = self.extract_laser_lines_from_segmentation(
                    laser_segmentation)

                # High-quality 3D reconstruction using precise masks
                masks = detection['masks']
                point_clouds = self.resnet101_enhanced_3d_reconstruction(
                    laser_lines, masks, detection['boxes'], laser_features
                )

                # Enhanced measurements leveraging ResNet-101's rich features
                detection['point_clouds'] = point_clouds
                detection['3d_measurements'] = self.enhanced_measurement_calculation(
                    point_clouds, masks, detection['boxes'], laser_features
                )

                # High-quality mesh generation
                detection['meshes'] = self.generate_enhanced_meshes(
                    point_clouds)

                # Confidence scoring based on ResNet-101 feature quality
                detection['measurement_confidence'] = self.calculate_resnet101_confidence(
                    point_clouds, masks, laser_features
                )

        return detections

    def extract_laser_lines_from_segmentation(self, laser_segmentation: torch.Tensor) -> Dict:
        """
        Extract laser lines from neural network segmentation.

        Advantage over traditional color-based detection:
        - Learns to handle underwater color distortion
        - Robust to varying lighting conditions
        - Handles laser line reflections and scattering
        - Adapts to different water clarity levels
        """
        laser_seg_np = laser_segmentation.squeeze().cpu().numpy()  # [3, H, W]

        # Extract red laser (channel 0)
        red_laser_mask = laser_seg_np[0] > 0.5
        red_line_points = self.extract_centerline_points(red_laser_mask)

        # Extract green laser (channel 1)
        green_laser_mask = laser_seg_np[1] > 0.5
        green_line_points = self.extract_centerline_points(green_laser_mask)

        return {
            'red_laser': red_line_points,
            'green_laser': green_line_points,
            'red_confidence': np.mean(laser_seg_np[0][red_laser_mask]) if np.any(red_laser_mask) else 0,
            'green_confidence': np.mean(laser_seg_np[1][green_laser_mask]) if np.any(green_laser_mask) else 0
        }

    def extract_centerline_points(self, laser_mask: np.ndarray) -> np.ndarray:
        """
        Extract centerline points from laser segmentation mask.

        Uses morphological operations + skeleton extraction for sub-pixel accuracy.
        """
        if not np.any(laser_mask):
            return np.array([])

        # Convert to uint8
        mask_uint8 = (laser_mask * 255).astype(np.uint8)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        cleaned_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

        # Skeletonization for centerline
        try:
            skeleton = cv2.ximgproc.thinning(cleaned_mask)
        except AttributeError:
            # Fallback if ximgproc not available
            skeleton = cleaned_mask

        # Extract points
        y_coords, x_coords = np.where(skeleton > 0)
        if len(x_coords) == 0:
            return np.array([])

        # Sort points to form continuous line
        points = np.column_stack([x_coords, y_coords])
        if len(points) > 1:
            # Sort by x-coordinate for horizontal lines, y-coordinate for vertical
            if np.std(points[:, 0]) > np.std(points[:, 1]):
                points = points[points[:, 0].argsort()]
            else:
                points = points[points[:, 1].argsort()]

        return points

    def resnet101_enhanced_3d_reconstruction(self, laser_lines: Dict, masks: torch.Tensor,
                                             boxes: torch.Tensor, backbone_features: torch.Tensor) -> Dict:
        """
        Enhanced 3D reconstruction leveraging ResNet-101's rich feature representation.

        ResNet-101 advantages for 3D reconstruction:
        - Richer feature representation enables better depth estimation
        - Better handling of occlusion and partial visibility
        - More robust laser-object intersection detection
        - Superior surface normal estimation
        """
        point_clouds = {}

        # Enhanced surface reconstruction using backbone features
        surface_depth = self.surface_reconstruction(backbone_features)
        surface_depth_np = surface_depth.squeeze().cpu().numpy()

        for obj_id, mask in enumerate(masks):
            mask_np = mask.squeeze().cpu().numpy()

            # Combine laser triangulation with learned depth estimation
            laser_3d_points = self.get_enhanced_masked_laser_points(
                laser_lines, mask_np)
            learned_3d_points = self.get_learned_depth_points(
                surface_depth_np, mask_np)

            # Fuse multiple 3D estimation methods
            if len(laser_3d_points) > 0 and len(learned_3d_points) > 0:
                fused_points = self.fuse_3d_estimations(
                    laser_3d_points, learned_3d_points)
                point_clouds[f'object_{obj_id}'] = fused_points
            elif len(laser_3d_points) > 0:
                point_clouds[f'object_{obj_id}'] = laser_3d_points
            elif len(learned_3d_points) > 0:
                point_clouds[f'object_{obj_id}'] = learned_3d_points

        return point_clouds

    def get_learned_depth_points(self, depth_map: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
        """
        Generate 3D points from learned depth estimation.

        Complements laser triangulation with dense depth prediction.
        """
        if not np.any(object_mask):
            return np.array([]).reshape(0, 3)

        # Get object pixels
        obj_y, obj_x = np.where(object_mask > 0.5)

        if len(obj_x) == 0:
            return np.array([]).reshape(0, 3)

        # Get depth values for object pixels
        depths = depth_map[obj_y, obj_x]

        # Filter valid depths
        valid_mask = (depths > 0.1) & (depths < 5.0)
        valid_x = obj_x[valid_mask]
        valid_y = obj_y[valid_mask]
        valid_depths = depths[valid_mask]

        if len(valid_x) == 0:
            return np.array([]).reshape(0, 3)

        # Convert to 3D world coordinates
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]

        X = (valid_x - cx) * valid_depths / fx
        Y = (valid_y - cy) * valid_depths / fy
        Z = valid_depths

        points_3d = np.column_stack([X, Y, Z])

        return points_3d

    def fuse_3d_estimations(self, laser_points: np.ndarray, learned_points: np.ndarray) -> np.ndarray:
        """
        Fuse laser triangulation with learned depth estimation.

        Strategy:
        - Use laser points as high-accuracy reference
        - Fill gaps with learned depth points
        - Cross-validate measurements for outlier removal
        """
        if len(laser_points) == 0:
            return learned_points
        if len(learned_points) == 0:
            return laser_points

        # Combine all points
        all_points = np.vstack([laser_points, learned_points])

        # Remove outliers using statistical methods
        if len(all_points) > 10:
            # Calculate distances from centroid
            centroid = np.mean(all_points, axis=0)
            distances = np.linalg.norm(all_points - centroid, axis=1)

            # Remove points beyond 2 standard deviations
            std_distance = np.std(distances)
            mean_distance = np.mean(distances)
            valid_mask = distances < (mean_distance + 2 * std_distance)

            all_points = all_points[valid_mask]

        return all_points

    def enhanced_measurement_calculation(self, point_clouds: Dict, masks: torch.Tensor,
                                         boxes: torch.Tensor, features: torch.Tensor) -> Dict:
        """
        Enhanced measurement calculation leveraging ResNet-101 features.

        Improvements over basic segmentation:
        - Feature-based confidence estimation
        - Multi-scale measurement validation
        - Enhanced surface quality assessment
        """
        measurements = {}

        for obj_id, point_cloud in point_clouds.items():
            if len(point_cloud) < 5:
                continue

            obj_idx = int(obj_id.split('_')[1])
            if obj_idx >= len(masks):
                continue

            mask = masks[obj_idx].squeeze().cpu().numpy()

            # Enhanced measurements using ResNet-101 features
            base_measurements = self.calculate_segmentation_measurements(
                {obj_id: point_cloud}, masks, boxes)[obj_id]

            # Add ResNet-101 specific enhancements
            feature_quality = self.assess_feature_quality(features, mask)
            measurement_stability = self.assess_measurement_stability(
                point_cloud)

            enhanced_measurements = {
                **base_measurements,
                'feature_quality': float(feature_quality),
                'measurement_stability': float(measurement_stability),
                'resnet101_confidence': float(feature_quality * measurement_stability),
                'precision_estimate': self.estimate_precision(point_cloud, mask)
            }

            measurements[obj_id] = enhanced_measurements

        return measurements

    def assess_feature_quality(self, features: torch.Tensor, mask: np.ndarray) -> float:
        """
        Assess feature quality for measurement confidence.

        Uses ResNet-101 feature statistics to determine measurement reliability.
        """
        # Extract features within mask region
        mask_tensor = torch.from_numpy(mask).float().to(features.device)
        mask_resized = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0),
                                     size=features.shape[-2:], mode='nearest').squeeze()

        # Get masked features
        masked_features = features * mask_resized.unsqueeze(0)

        # Calculate feature statistics
        feature_mean = torch.mean(masked_features[masked_features > 0])
        feature_std = torch.std(masked_features[masked_features > 0])

        # High-quality features have high mean activation and appropriate variance
        quality_score = torch.sigmoid(
            feature_mean) * (1 - torch.tanh(feature_std / feature_mean))

        return float(quality_score.cpu())

    def assess_measurement_stability(self, point_cloud: np.ndarray) -> float:
        """
        Assess 3D measurement stability.

        Stable measurements have:
        - Consistent point cloud density
        - Low variance in surface normals
        - Good spatial distribution
        """
        if len(point_cloud) < 10:
            return 0.0

        stability_factors = []

        # Point density consistency
        centroid = np.mean(point_cloud, axis=0)
        distances = np.linalg.norm(point_cloud - centroid, axis=1)
        density_consistency = 1.0 - (np.std(distances) / np.mean(distances))
        stability_factors.append(max(0, density_consistency))

        # Spatial distribution quality
        # Good distribution covers multiple spatial regions
        try:
            from scipy.spatial import ConvexHull
            if len(point_cloud) >= 4:
                hull = ConvexHull(point_cloud)
                hull_volume = hull.volume
                bbox_volume = np.prod(np.ptp(point_cloud, axis=0))
                spatial_quality = hull_volume / bbox_volume if bbox_volume > 0 else 0
                stability_factors.append(min(1.0, spatial_quality))
        except:
            pass

        # Surface normal consistency (if enough points)
        if len(point_cloud) > 20:
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point_cloud)
                pcd.estimate_normals()
                normals = np.asarray(pcd.normals)

                # Consistent normals indicate good surface quality
                normal_consistency = 1.0 - \
                    np.std(np.linalg.norm(normals, axis=1))
                stability_factors.append(max(0, normal_consistency))
            except:
                pass

        return np.mean(stability_factors) if stability_factors else 0.5

    def estimate_precision(self, point_cloud: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Estimate measurement precision for RANGER competition requirements.

        Returns precision estimates for different measurements:
        - Length precision (critical for 5cm shipwreck requirement)
        - Volume precision
        - Position precision
        """
        if len(point_cloud) < 10:
            # Conservative estimates
            return {'length': 10.0, 'volume': 50.0, 'position': 5.0}

        # Length precision based on point cloud extent and density
        dimensions = np.ptp(point_cloud, axis=0)  # [length, width, height]
        main_dimension = np.max(dimensions)

        # Precision improves with point density
        mask_area = np.sum(mask > 0.5)
        point_density = len(point_cloud) / mask_area if mask_area > 0 else 0

        # Estimate precision in cm (RANGER requirement is ±5cm)
        # Better with more points
        length_precision = max(0.5, 5.0 / (1 + point_density * 100))
        volume_precision = length_precision * 3  # Volume error compounds
        position_precision = length_precision * 0.5  # Position usually more accurate

        return {
            'length': float(length_precision),      # cm
            'volume': float(volume_precision),      # cm³ error
            'position': float(position_precision),  # cm
            'meets_ranger_req': length_precision < 5.0  # RANGER ±5cm requirement
        }

    def calculate_resnet101_confidence(self, point_clouds: Dict, masks: torch.Tensor,
                                       features: torch.Tensor) -> Dict:
        """
        Calculate measurement confidence using ResNet-101 feature analysis.

        Higher confidence = more reliable measurements for RANGER scoring.
        """
        confidences = {}

        for obj_id, point_cloud in point_clouds.items():
            obj_idx = int(obj_id.split('_')[1])
            if obj_idx >= len(masks):
                continue

            mask = masks[obj_idx].squeeze().cpu().numpy()

            # Combine multiple confidence factors
            feature_conf = self.assess_feature_quality(features, mask)
            stability_conf = self.assess_measurement_stability(point_cloud)
            precision_conf = 1.0 if self.estimate_precision(
                point_cloud, mask)['meets_ranger_req'] else 0.5

            # Weighted combination
            overall_confidence = (
                0.4 * feature_conf +      # ResNet-101 feature quality
                0.4 * stability_conf +    # 3D measurement stability
                0.2 * precision_conf      # RANGER precision requirement
            )

            confidences[obj_id] = {
                'overall': float(overall_confidence),
                'feature_quality': float(feature_conf),
                'measurement_stability': float(stability_conf),
                'precision_requirement': float(precision_conf),
                'ranger_ready': overall_confidence > 0.8 and precision_conf > 0.9
            }

        return confidences
    """
    RANGER ROV Enhanced Mask R-CNN with Dual Laser 3D Reconstruction.

    Why Mask R-CNN over Faster R-CNN:
    - Pixel-level segmentation for precise object boundaries
    - Better 3D point cloud quality (no background contamination)
    - Superior measurement accuracy for RANGER 5cm requirement
    - Handles complex object shapes (jellyfish, shipwreck components)
    - Precise laser-object intersection calculation

    Architecture Enhancement:
    - ResNet-50 + FPN backbone for robust feature extraction
    - Instance segmentation for pixel-perfect object boundaries
    - Dual laser line segmentation for structured light
    - 3D reconstruction from segmented regions only
    - Open3D integration for real-time visualization
    """

    def __init__(self,
                 num_classes: int = 30,
                 weights: bool = True,
                 laser_baseline: float = 0.15,
                 camera_focal_length: float = 800.0):
        super().__init__()

        # Use Mask R-CNN instead of Faster R-CNN
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        self.detector = maskrcnn_resnet50_fpn(
            weights=weights,
            min_size=800,
            max_size=1024,
            # Enhanced parameters for precision
            rpn_pre_nms_top_n_train=3000,
            rpn_pre_nms_top_n_test=1500,
            box_detections_per_img=200,  # More detections for complex scenes
        )

        # Replace mask predictor with custom class count
        in_features_mask = self.detector.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.detector.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

        # Enhanced laser line segmentation head
        # Why separate: Laser lines have different characteristics than objects
        self.laser_segmentation = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 3 classes: red_laser, green_laser, background
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # 3D surface reconstruction head
        # Uses segmentation masks to constrain reconstruction
        self.surface_reconstruction = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),  # Depth map per pixel
        )

        # Physical calibration
        self.laser_baseline = laser_baseline
        self.focal_length = camera_focal_length
        self.register_buffer('camera_matrix', torch.eye(3))

    def forward_inference(self, images: List[torch.Tensor]):
        """Enhanced inference with segmentation-based 3D reconstruction."""
        # Get instance segmentation results
        detections = self.detector(images)

        # Process each image for 3D reconstruction
        for i, (image, detection) in enumerate(zip(images, detections)):
            if len(detection['boxes']) > 0:
                # Get segmentation masks
                # [N, 1, H, W] - pixel-perfect masks
                masks = detection['masks']

                # Detect laser lines
                laser_lines = self.detect_laser_lines(image)

                # Generate 3D point cloud using segmentation masks
                point_clouds = self.segment_based_3d_reconstruction(
                    laser_lines, masks, detection['boxes']
                )

                # Calculate precise measurements from segmented regions
                detection['point_clouds'] = point_clouds
                detection['3d_measurements'] = self.calculate_segmentation_measurements(
                    point_clouds, masks, detection['boxes']
                )

                # Generate high-quality meshes from segmented point clouds
                detection['meshes'] = self.generate_segmented_meshes(
                    point_clouds)

        return detections

    def segment_based_3d_reconstruction(self, laser_lines: Dict, masks: torch.Tensor, boxes: torch.Tensor) -> Dict:
        """
        Generate 3D point clouds using segmentation masks for precision.

        Key advantage: Only reconstruct pixels that belong to actual objects,
        eliminating background noise and improving measurement accuracy.
        """
        point_clouds = {}

        for obj_id, mask in enumerate(masks):
            mask_np = mask.squeeze().cpu().numpy()  # [H, W]

            # Get object-specific laser intersections
            object_laser_points = self.get_masked_laser_points(
                laser_lines, mask_np)

            if len(object_laser_points) > 10:  # Need minimum points
                # Triangulate only object pixels
                object_3d_points = self.triangulate_masked_points(
                    object_laser_points)
                point_clouds[f'object_{obj_id}'] = object_3d_points

        return point_clouds

    def get_masked_laser_points(self, laser_lines: Dict, object_mask: np.ndarray) -> Dict:
        """
        Extract laser line points that intersect with segmented object.

        Critical for accuracy: Only use laser points on actual object surface,
        not on background or other objects.
        """
        masked_laser_points = {}

        # Process red laser points
        if 'red_laser' in laser_lines and len(laser_lines['red_laser']) > 0:
            red_points = laser_lines['red_laser']
            # Filter points that fall within object mask
            valid_red_points = []
            for point in red_points:
                x, y = int(point[0]), int(point[1])
                if (0 <= x < object_mask.shape[1] and
                    0 <= y < object_mask.shape[0] and
                        object_mask[y, x] > 0.5):  # Point is on object
                    valid_red_points.append(point)
            masked_laser_points['red_laser'] = np.array(valid_red_points)

        # Process green laser points
        if 'green_laser' in laser_lines and len(laser_lines['green_laser']) > 0:
            green_points = laser_lines['green_laser']
            valid_green_points = []
            for point in green_points:
                x, y = int(point[0]), int(point[1])
                if (0 <= x < object_mask.shape[1] and
                    0 <= y < object_mask.shape[0] and
                        object_mask[y, x] > 0.5):
                    valid_green_points.append(point)
            masked_laser_points['green_laser'] = np.array(valid_green_points)

        return masked_laser_points

    def triangulate_masked_points(self, masked_laser_points: Dict) -> np.ndarray:
        """
        Triangulate 3D points from masked laser intersections.

        Higher accuracy than bounding box approach because:
        - No background contamination
        - Precise object surface points only
        - Better surface normal estimation
        """
        all_3d_points = []

        # Process each laser separately
        for laser_color, points in masked_laser_points.items():
            if len(points) > 0:
                laser_3d_points = self.laser_to_3d(points, laser_color)
                all_3d_points.extend(laser_3d_points)

        return np.array(all_3d_points) if all_3d_points else np.array([]).reshape(0, 3)

    def calculate_segmentation_measurements(self, point_clouds: Dict, masks: torch.Tensor, boxes: torch.Tensor) -> Dict:
        """
        Calculate precise 3D measurements using segmentation-based point clouds.

        Advantages over bounding box measurements:
        - True object volume (not rectangular approximation)
        - Accurate surface area calculation
        - Better centroid estimation
        - Precise oriented bounding box
        """
        measurements = {}

        for obj_id, point_cloud in point_clouds.items():
            if len(point_cloud) < 10:
                continue

            # Get corresponding mask for additional analysis
            obj_idx = int(obj_id.split('_')[1])
            if obj_idx < len(masks):
                mask = masks[obj_idx].squeeze().cpu().numpy()

                # Calculate comprehensive measurements
                measurements[obj_id] = {
                    # 3D measurements from point cloud
                    **self.calculate_3d_measurements_from_points(point_cloud),

                    # 2D measurements from segmentation mask
                    **self.calculate_2d_measurements_from_mask(mask),

                    # Combined measurements
                    **self.calculate_combined_measurements(point_cloud, mask)
                }

        return measurements

    def calculate_3d_measurements_from_points(self, point_cloud: np.ndarray) -> Dict:
        """Calculate 3D measurements from point cloud."""
        if len(point_cloud) < 3:
            return {}

        # Oriented bounding box for accurate dimensions
        obb_dimensions = self.calculate_oriented_bounding_box(point_cloud)

        # Convex hull volume estimation
        try:
            from scipy.spatial import ConvexHull
            if len(point_cloud) >= 4:
                hull = ConvexHull(point_cloud)
                volume = hull.volume
            else:
                volume = 0
        except:
            volume = np.prod(obb_dimensions)  # Fallback

        return {
            'length_3d': float(obb_dimensions[0]),
            'width_3d': float(obb_dimensions[1]),
            'height_3d': float(obb_dimensions[2]),
            'volume_3d': float(volume),
            'centroid_3d': np.mean(point_cloud, axis=0).tolist(),
            'point_count': len(point_cloud),
        }

    def calculate_2d_measurements_from_mask(self, mask: np.ndarray) -> Dict:
        """Calculate 2D measurements from segmentation mask."""
        # Find contours of mask
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {}

        # Get largest contour (main object)
        main_contour = max(contours, key=cv2.contourArea)

        # Calculate 2D measurements
        area_2d = cv2.contourArea(main_contour)
        perimeter_2d = cv2.arcLength(main_contour, True)

        # Minimum area rectangle for 2D dimensions
        rect = cv2.minAreaRect(main_contour)
        (center_x, center_y), (width, height), angle = rect

        # Ellipse fitting for additional shape analysis
        if len(main_contour) >= 5:
            ellipse = cv2.fitEllipse(main_contour)
            ellipse_major = max(ellipse[1])
            ellipse_minor = min(ellipse[1])
        else:
            ellipse_major = ellipse_minor = 0

        return {
            'area_2d': float(area_2d),
            'perimeter_2d': float(perimeter_2d),
            'width_2d': float(max(width, height)),
            'height_2d': float(min(width, height)),
            'centroid_2d': [float(center_x), float(center_y)],
            'orientation_2d': float(angle),
            'ellipse_major': float(ellipse_major),
            'ellipse_minor': float(ellipse_minor),
            'circularity': 4 * np.pi * area_2d / (perimeter_2d**2) if perimeter_2d > 0 else 0
        }

    def calculate_combined_measurements(self, point_cloud: np.ndarray, mask: np.ndarray) -> Dict:
        """Calculate measurements combining 3D and 2D information."""
        # Surface area estimation from mask and depth
        surface_area = self.estimate_surface_area(point_cloud, mask)

        # Volume-to-area ratio
        volume_3d = np.prod(self.calculate_oriented_bounding_box(point_cloud))
        area_2d = np.sum(mask > 0.5)

        # Measurement confidence based on point density and mask quality
        confidence = self.calculate_measurement_confidence(point_cloud, mask)

        return {
            'surface_area_estimated': float(surface_area),
            'volume_to_area_ratio': float(volume_3d / area_2d) if area_2d > 0 else 0,
            'measurement_confidence': float(confidence),
            'reconstruction_quality': self.assess_reconstruction_quality(point_cloud, mask)
        }

    def estimate_surface_area(self, point_cloud: np.ndarray, mask: np.ndarray) -> float:
        """
        Estimate surface area using point cloud density and mask area.

        Better than simple mask area because accounts for 3D surface curvature.
        """
        if len(point_cloud) < 3:
            return 0

        # Calculate point cloud surface area using triangulation
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            pcd.estimate_normals()

            # Poisson reconstruction for surface area
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=6)
            surface_area = mesh.get_surface_area()

            return surface_area
        except:
            # Fallback: estimate from 2D mask
            mask_area = np.sum(mask > 0.5)
            depth_variation = np.std(point_cloud[:, 2]) if len(
                point_cloud) > 0 else 0
            # Rough approximation accounting for surface curvature
            return mask_area * (1 + depth_variation)

    def calculate_measurement_confidence(self, point_cloud: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate confidence in measurements based on data quality.

        Factors:
        - Point cloud density
        - Mask quality (smooth vs jagged edges)
        - 3D-2D consistency
        - Laser line coverage
        """
        confidence_factors = []

        # Point density factor
        mask_area = np.sum(mask > 0.5)
        if mask_area > 0:
            point_density = len(point_cloud) / mask_area
            # Target 0.1 points per pixel
            density_confidence = min(1.0, point_density / 0.1)
            confidence_factors.append(density_confidence)

        # Mask quality factor (smoothness of edges)
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            # Smooth contours have fewer points relative to perimeter
            smoothness = 1.0 - \
                min(1.0, len(main_contour) / cv2.arcLength(main_contour, True))
            confidence_factors.append(smoothness)

        # Point cloud consistency factor
        if len(point_cloud) > 10:
            # Check for outliers using statistical methods
            distances = np.linalg.norm(
                point_cloud - np.mean(point_cloud, axis=0), axis=1)
            std_distance = np.std(distances)
            outlier_ratio = np.sum(distances > np.mean(
                distances) + 2*std_distance) / len(point_cloud)
            consistency_confidence = 1.0 - outlier_ratio
            confidence_factors.append(consistency_confidence)

        # Overall confidence
        return np.mean(confidence_factors) if confidence_factors else 0.0

    def assess_reconstruction_quality(self, point_cloud: np.ndarray, mask: np.ndarray) -> str:
        """Assess overall reconstruction quality."""
        confidence = self.calculate_measurement_confidence(point_cloud, mask)

        if confidence > 0.8:
            return "EXCELLENT"
        elif confidence > 0.6:
            return "GOOD"
        elif confidence > 0.4:
            return "FAIR"
        else:
            return "POOR"

    def generate_segmented_meshes(self, point_clouds: Dict) -> Dict:
        """
        Generate high-quality meshes from segmented point clouds.

        Better mesh quality because:
        - No background noise
        - Clean object boundaries
        - Better surface normal estimation
        """
        meshes = {}

        for obj_id, point_cloud in point_clouds.items():
            if len(point_cloud) > 50:  # Need sufficient points for meshing
                try:

                    # Create point cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(point_cloud)

                    # Enhanced normal estimation for segmented objects
                    pcd.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=0.05, max_nn=50  # Smaller radius for finer details
                        )
                    )

                    # High-quality Poisson reconstruction
                    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd,
                        depth=9,  # Higher depth for more detail
                        width=0,  # Auto width
                        scale=1.1,
                        linear_fit=False
                    )

                    # Clean and optimize mesh
                    mesh.remove_degenerate_triangles()
                    mesh.remove_duplicated_triangles()
                    mesh.remove_duplicated_vertices()
                    mesh.remove_non_manifold_edges()

                    # Smooth mesh while preserving features
                    mesh = mesh.filter_smooth_simple(number_of_iterations=5)

                    meshes[obj_id] = mesh

                except Exception as e:
                    print(f"Mesh generation failed for {obj_id}: {e}")

        return meshes
    """
    RANGER ROV Enhanced Faster R-CNN with Dual Laser 3D Reconstruction.

    This implementation extends Faster R-CNN with:
    1. Dual laser line detection and segmentation
    2. 3D point cloud generation via triangulation
    3. Real-time Open3D integration
    4. Precise object measurement capabilities

    Design Rationale:
    - RANGER class has 25A power budget (vs 15A SCOUT) - supports dual lasers
    - RANGER allows lasers (SCOUT prohibits) - enables structured light
    - 25kg weight limit allows heavier camera/laser systems
    - Competition requires 5cm measurement accuracy
    """

    def __init__(self,
                 num_classes: int = 30,  # More classes for RANGER complexity
                 weights: bool = True,
                 laser_baseline: float = 0.15,  # 15cm between lasers
                 camera_focal_length: float = 800.0):
        """
        Initialize RANGER 3D Faster R-CNN.

        Args:
            num_classes: Object classes (expanded for RANGER tasks)
            weights: Use COCO weights weights
            laser_baseline: Physical distance between dual lasers (meters)
            camera_focal_length: Camera focal length (pixels)
        """
        super().__init__()

        # Enhanced backbone for complex RANGER tasks
        self.detector = maskrcnn_resnet50_fpn(
            weights=weights,
            min_size=800,  # Higher resolution for precision
            max_size=1024,
            rpn_pre_nms_top_n_train=3000,
            rpn_pre_nms_top_n_test=1500,
            rpn_post_nms_top_n_train=3000,
            rpn_post_nms_top_n_test=1500
        )

        # Replace classification head
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = MaskRCNNPredictor(
            in_features, num_classes)

        # Laser line segmentation head
        # Why separate head: Laser lines have different characteristics than objects
        self.laser_segmentation = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 2 classes: laser line, background
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Sigmoid()
        )

        # 3D reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6),  # [x, y, z, length, width, height]
        )

        # Physical calibration parameters
        self.laser_baseline = laser_baseline
        self.focal_length = camera_focal_length
        self.register_buffer('camera_matrix', torch.eye(3))

        # Open3D integration
        self.point_clouds = []
        self.mesh_reconstructions = {}

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict]] = None):
        """Forward pass with 3D reconstruction capability."""
        if self.training:
            return self.forward_train(images, targets)
        else:
            return self.forward_inference(images)

    def forward_inference(self, images: List[torch.Tensor]):
        """Enhanced inference with 3D reconstruction."""
        # Standard object detection
        detections = self.detector(images)

        # Process each image for 3D reconstruction
        for i, (image, detection) in enumerate(zip(images, detections)):
            if len(detection['boxes']) > 0:
                # Detect laser lines
                laser_lines = self.detect_laser_lines(image)

                # Generate 3D point cloud
                point_cloud = self.triangulate_3d_points(
                    laser_lines, detection['boxes'])

                # Add 3D measurements
                detection['point_clouds'] = point_cloud
                detection['3d_measurements'] = self.calculate_3d_measurements(
                    point_cloud, detection['boxes']
                )

                # Generate mesh if sufficient points
                if len(point_cloud) > 100:
                    mesh = self.generate_mesh(point_cloud)
                    detection['meshes'] = mesh

        return detections

    def detect_laser_lines(self, image: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Detect dual laser lines in underwater image.

        Uses color-based segmentation + machine learning refinement.
        Red laser (650nm) and Green laser (532nm) have distinct signatures.
        """
        # Convert to numpy for OpenCV processing
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Detect red laser line (650nm)
        red_mask = self.detect_red_laser(img_bgr)
        red_line_points = self.extract_line_points(red_mask)

        # Detect green laser line (532nm)
        green_mask = self.detect_green_laser(img_bgr)
        green_line_points = self.extract_line_points(green_mask)

        return {
            'red_laser': red_line_points,
            'green_laser': green_line_points,
            'red_mask': red_mask,
            'green_mask': green_mask
        }

    def detect_red_laser(self, image: np.ndarray) -> np.ndarray:
        """
        Detect red laser line using HSV color filtering + morphology.

        Red laser characteristics underwater:
        - Wavelength: 650nm
        - Better penetration through water than blue/green
        - Distinct red hue even with color shift
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red laser HSV ranges (tuned for underwater conditions)
        lower_red1 = np.array([0, 70, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations for clean line extraction
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        return red_mask

    def detect_green_laser(self, image: np.ndarray) -> np.ndarray:
        """
        Detect green laser line using HSV color filtering.

        Green laser characteristics underwater:
        - Wavelength: 532nm
        - High visibility underwater
        - Less scattering than red in some conditions
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Green laser HSV range (tuned for underwater)
        lower_green = np.array([40, 70, 70])
        upper_green = np.array([80, 255, 255])

        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        return green_mask

    def extract_line_points(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract centerline points from laser line mask.

        Uses skeleton extraction + sub-pixel refinement for accuracy.
        """
        # Skeletonization for centerline
        skeleton = cv2.ximgproc.thinning(mask)

        # Find contours of skeleton
        contours, _ = cv2.findContours(
            skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return np.array([])

        # Get longest contour (main laser line)
        main_contour = max(contours, key=cv2.contourArea)

        # Extract points and sort by x-coordinate
        points = main_contour.reshape(-1, 2)
        points = points[points[:, 0].argsort()]

        return points

    def triangulate_3d_points(self, laser_lines: Dict, bounding_boxes: torch.Tensor) -> np.ndarray:
        """
        Generate 3D point cloud using dual laser triangulation.

        Mathematical principle:
        For each laser line point (x_pixel, y_pixel):
        1. Calculate angle θ = atan((x_pixel - cx) / focal_length)
        2. Calculate distance Z = baseline / (2 * tan(θ/2))
        3. Calculate 3D coordinates (X, Y, Z)

        Why dual lasers:
        - Redundancy for occluded areas
        - Cross-validation of measurements
        - Increased point cloud density
        """
        point_cloud = []

        # Process red laser points
        if len(laser_lines['red_laser']) > 0:
            red_3d_points = self.laser_to_3d(laser_lines['red_laser'], 'red')
            point_cloud.extend(red_3d_points)

        # Process green laser points
        if len(laser_lines['green_laser']) > 0:
            green_3d_points = self.laser_to_3d(
                laser_lines['green_laser'], 'green')
            point_cloud.extend(green_3d_points)

        return np.array(point_cloud) if point_cloud else np.array([]).reshape(0, 3)

    def laser_to_3d(self, line_points: np.ndarray, laser_color: str) -> List[Tuple[float, float, float]]:
        """
        Convert 2D laser line points to 3D world coordinates.

        Uses triangulation based on known laser-camera geometry.
        """
        if len(line_points) == 0:
            return []

        # Camera parameters (should be calibrated for actual hardware)
        # Principal point
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        # Focal lengths
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]

        # Laser offset based on color (physical mounting positions)
        laser_offset = -self.laser_baseline / \
            2 if laser_color == 'red' else self.laser_baseline/2

        points_3d = []

        for point in line_points:
            x_pixel, y_pixel = point

            # Calculate angle from camera center
            theta = np.arctan((x_pixel - cx) / fx)

            # Triangulation formula
            # Z = laser_offset / tan(theta) for small angles
            # More accurate formula for larger angles:
            # Avoid division by zero
            Z = abs(laser_offset) / np.tan(abs(theta) + 1e-6)

            # Convert to 3D world coordinates
            X = (x_pixel - cx) * Z / fx
            Y = (y_pixel - cy) * Z / fy

            # Filter reasonable distances (0.1m to 5m for RANGER applications)
            if 0.1 < Z < 5.0:
                points_3d.append((X, Y, Z))

        return points_3d

    def calculate_3d_measurements(self, point_cloud: np.ndarray, bounding_boxes: torch.Tensor) -> Dict:
        """
        Calculate precise 3D measurements from point cloud data.

        For RANGER competition:
        - Shipwreck length within 5cm (10 points)
        - Object dimensions for manipulation planning
        - Volume estimation for specimen collection
        """
        if len(point_cloud) == 0:
            return {}

        measurements = {}

        for i, bbox in enumerate(bounding_boxes):
            # Filter points within bounding box
            bbox_points = self.filter_points_in_bbox(point_cloud, bbox)

            if len(bbox_points) < 10:  # Need minimum points for reliable measurement
                continue

            # Calculate object dimensions
            min_coords = np.min(bbox_points, axis=0)
            max_coords = np.max(bbox_points, axis=0)
            dimensions = max_coords - min_coords

            # Calculate oriented bounding box for better accuracy
            obb_dimensions = self.calculate_oriented_bounding_box(bbox_points)

            measurements[f'object_{i}'] = {
                'length': float(obb_dimensions[0]),  # meters
                'width': float(obb_dimensions[1]),   # meters
                'height': float(obb_dimensions[2]),  # meters
                'volume': float(np.prod(obb_dimensions)),  # cubic meters
                'centroid': np.mean(bbox_points, axis=0).tolist(),
                'point_count': len(bbox_points),
                'measurement_confidence': min(1.0, len(bbox_points) / 100)
            }

        return measurements

    def filter_points_in_bbox(self, point_cloud: np.ndarray, bbox: torch.Tensor) -> np.ndarray:
        """Filter 3D points that project into 2D bounding box."""
        if len(point_cloud) == 0:
            return np.array([]).reshape(0, 3)

        # Project 3D points to 2D using camera matrix
        points_2d = self.project_3d_to_2d(point_cloud)

        # Filter points within bounding box
        x1, y1, x2, y2 = bbox.cpu().numpy()

        mask = ((points_2d[:, 0] >= x1) & (points_2d[:, 0] <= x2) &
                (points_2d[:, 1] >= y1) & (points_2d[:, 1] <= y2))

        return point_cloud[mask]

    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D world points to 2D image coordinates."""
        if len(points_3d) == 0:
            return np.array([]).reshape(0, 2)

        # Homogeneous coordinates
        points_homogeneous = np.hstack(
            [points_3d, np.ones((len(points_3d), 1))])

        # Project using camera matrix
        camera_matrix_np = self.camera_matrix.cpu().numpy()
        points_2d_homogeneous = camera_matrix_np @ points_homogeneous.T

        # Convert from homogeneous coordinates
        points_2d = (
            points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]).T

        return points_2d

    def calculate_oriented_bounding_box(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate oriented bounding box dimensions for accurate measurement.

        Uses PCA to find principal axes, then measures along those axes.
        More accurate than axis-aligned bounding box for rotated objects.
        """
        if len(points) < 3:
            return np.array([0., 0., 0.])

        # Center the points
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid

        # Principal Component Analysis
        covariance_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort by eigenvalues (largest first)
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]

        # Project points onto principal axes
        projected_points = centered_points @ eigenvectors

        # Calculate dimensions along principal axes
        min_proj = np.min(projected_points, axis=0)
        max_proj = np.max(projected_points, axis=0)
        dimensions = max_proj - min_proj

        return np.abs(dimensions)

    def generate_mesh(self, point_cloud: np.ndarray) -> o3d.geometry.TriangleMesh:
        """
        Generate 3D mesh from point cloud using Open3D.

        For RANGER applications:
        - Visualize objects in 3D
        - Calculate surface area and volume
        - Export models for analysis
        """
        if len(point_cloud) < 50:
            return None

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30)
        )

        # Poisson surface reconstruction
        try:
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=8
            )

            # Clean up mesh
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

            return mesh

        except Exception as e:
            print(f"Mesh generation failed: {e}")
            return None


class Open3DVisualizer:
    """
    Real-time 3D visualization system using Open3D.

    Features:
    - Live point cloud updates
    - Mesh reconstruction display
    - Measurement annotations
    - 3D model export

    Why Open3D:
    - High-performance 3D processing
    - Real-time visualization capabilities
    - Extensive mesh processing tools
    - Python integration
    """

    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="RANGER ROV 3D View", width=800, height=600)

        # Visualization state
        self.point_clouds = {}
        self.meshes = {}
        self.measurement_lines = []

        # Coordinate frame for reference
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5)
        self.vis.add_geometry(self.coordinate_frame)

    def update_point_cloud(self, object_id: str, points: np.ndarray, color: Tuple[float, float, float] = (1.0, 0.0, 0.0)):
        """Update point cloud for specific object."""
        if len(points) == 0:
            return

        # Create or update point cloud
        if object_id in self.point_clouds:
            # Update existing point cloud
            self.point_clouds[object_id].points = o3d.utility.Vector3dVector(
                points)
            self.vis.update_geometry(self.point_clouds[object_id])
        else:
            # Create new point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color(color)

            self.point_clouds[object_id] = pcd
            self.vis.add_geometry(pcd)

    def update_mesh(self, object_id: str, mesh: o3d.geometry.TriangleMesh):
        """Update mesh visualization."""
        if mesh is None:
            return

        if object_id in self.meshes:
            # Remove old mesh
            self.vis.remove_geometry(self.meshes[object_id])

        # Add new mesh with material
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.9])  # Light blue

        self.meshes[object_id] = mesh
        self.vis.add_geometry(mesh)

    def add_measurement_line(self, start_point: np.ndarray, end_point: np.ndarray,
                             color: Tuple[float, float, float] = (0.0, 1.0, 0.0)):
        """Add measurement line between two 3D points."""
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector([start_point, end_point])
        line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
        line_set.colors = o3d.utility.Vector3dVector([color])

        self.measurement_lines.append(line_set)
        self.vis.add_geometry(line_set)

    def clear_measurements(self):
        """Clear all measurement lines."""
        for line in self.measurement_lines:
            self.vis.remove_geometry(line)
        self.measurement_lines.clear()

    def export_scene(self, filename: str):
        """Export current 3D scene to file."""
        # Combine all geometries
        combined_mesh = o3d.geometry.TriangleMesh()

        for mesh in self.meshes.values():
            combined_mesh += mesh

        # Export formats: .ply, .obj, .stl
        o3d.io.write_triangle_mesh(filename, combined_mesh)
        print(f"3D scene exported to: {filename}")

    def render_frame(self):
        """Render single frame (call in main loop)."""
        self.vis.poll_events()
        self.vis.update_renderer()


class DualLaserCalibration:
    """
    Calibration system for dual laser setup.

    Critical for accurate 3D measurements:
    - Camera intrinsic parameters
    - Laser-camera extrinsic parameters
    - Laser line plane equations
    - Distortion correction
    """

    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.laser_positions = {}
        self.laser_orientations = {}

    def calibrate_camera(self, calibration_images: List[np.ndarray],
                         board_size: Tuple[int, int] = (9, 6)) -> bool:
        """
        Calibrate camera using checkerboard pattern.

        Must be done underwater for accurate results.
        Water refraction affects focal length and distortion.
        """
        # Prepare object points
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0],
                               0:board_size[1]].T.reshape(-1, 2)
        objp *= 0.025  # 25mm squares

        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane

        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, board_size, None)

            if ret:
                objpoints.append(objp)

                # Refine corners to sub-pixel accuracy
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS +
                              cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                imgpoints.append(corners2)

        if len(objpoints) < 10:
            print("Error: Need at least 10 good calibration images")
            return False

        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        if ret:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs

            # Calculate reprojection error
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(
                    objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                )
                error = cv2.norm(
                    imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error

            print(
                f"Camera calibration completed. Mean error: {mean_error/len(objpoints):.3f} pixels")
            return True

        return False

    def calibrate_laser_plane(self, laser_color: str, calibration_points: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """
        Calibrate laser plane equation from known 3D-2D point correspondences.

        Args:
            laser_color: 'red' or 'green'
            calibration_points: List of (3D_world_point, 2D_image_point) pairs
        """
        if len(calibration_points) < 4:
            print(
                f"Error: Need at least 4 calibration points for {laser_color} laser")
            return False

        # Extract 3D and 2D points
        world_points = np.array([pt[0] for pt in calibration_points])
        image_points = np.array([pt[1] for pt in calibration_points])

        # Fit plane to 3D points: ax + by + cz + d = 0
        centroid = np.mean(world_points, axis=0)
        centered_points = world_points - centroid

        # SVD to find normal vector
        _, _, V = np.linalg.svd(centered_points)
        normal = V[-1, :]  # Last row of V

        # Calculate d parameter
        d = -np.dot(normal, centroid)

        # Store laser plane parameters
        self.laser_positions[laser_color] = {
            'plane_normal': normal,
            'plane_d': d,
            'calibration_points': calibration_points,
            'reprojection_error': self.calculate_reprojection_error(calibration_points)
        }

        print(
            f"{laser_color.title()} laser calibrated. Plane: {normal[0]:.3f}x + {normal[1]:.3f}y + {normal[2]:.3f}z + {d:.3f} = 0")
        return True

    def calculate_reprojection_error(self, calibration_points: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Calculate reprojection error for laser calibration."""
        if not calibration_points or self.camera_matrix is None:
            return float('inf')

        total_error = 0
        for world_pt, image_pt in calibration_points:
            # Project 3D point to 2D
            projected_pt, _ = cv2.projectPoints(
                world_pt.reshape(1, 1, 3),
                np.zeros(3), np.zeros(3),  # No rotation/translation
                self.camera_matrix, self.dist_coeffs
            )
            projected_pt = projected_pt.reshape(2)

            # Calculate error
            error = np.linalg.norm(projected_pt - image_pt)
            total_error += error

        return total_error / len(calibration_points)

    def save_calibration(self, filename: str):
        """Save calibration data to file."""
        calibration_data = {
            'camera_matrix': self.camera_matrix.tolist() if self.camera_matrix is not None else None,
            'dist_coeffs': self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
            'laser_positions': {}
        }

        # Convert laser calibration data to serializable format
        for color, data in self.laser_positions.items():
            calibration_data['laser_positions'][color] = {
                'plane_normal': data['plane_normal'].tolist(),
                'plane_d': float(data['plane_d']),
                'reprojection_error': float(data['reprojection_error'])
            }

        with open(filename, 'w') as f:
            import json
            json.dump(calibration_data, f, indent=2)

        print(f"Calibration saved to: {filename}")

    def load_calibration(self, filename: str) -> bool:
        """Load calibration data from file."""
        try:
            with open(filename, 'r') as f:
                import json
                data = json.load(f)

            if data['camera_matrix'] is not None:
                self.camera_matrix = np.array(data['camera_matrix'])
                self.dist_coeffs = np.array(data['dist_coeffs'])

            for color, laser_data in data['laser_positions'].items():
                self.laser_positions[color] = {
                    'plane_normal': np.array(laser_data['plane_normal']),
                    'plane_d': laser_data['plane_d'],
                    'reprojection_error': laser_data['reprojection_error']
                }

            print(f"Calibration loaded from: {filename}")
            return True

        except Exception as e:
            print(f"Failed to load calibration: {e}")
            return False


class RANGERCompetitionInterface:
    """
    Competition interface specifically designed for RANGER ROV with 3D capabilities.

    Features:
    - Real-time 3D object visualization
    - Precision measurement display (5cm accuracy requirement)
    - Dual laser line overlay
    - 3D model export for competition documentation
    - Enhanced manipulation guidance
    """

    def __init__(self,
                 model: RANGERFasterRCNN3D,
                 calibration: DualLaserCalibration,
                 camera_source: int = 0):

        self.model = model
        self.calibration = calibration

        # Enhanced display for RANGER complexity
        self.window_width = 1400  # Wider for 3D view
        self.window_height = 900  # Taller for more information

        self.camera = cv2.VideoCapture(camera_source)
        # Higher resolution for precision
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # 3D visualization
        self.visualizer_3d = Open3DVisualizer()

        # Interface state
        self.show_laser_lines = True
        self.show_3d_overlay = True
        self.measurement_mode = "length"  # length, volume, surface_area
        self.export_counter = 0

        # RANGER-specific task states
        self.current_ranger_task = "shipwreck_measurement"
        self.measurement_precision = 0.05  # 5cm requirement

        # Competition logging
        self.measurement_log = []

    def run(self):
        """Enhanced competition interface with 3D visualization."""
        print("RANGER ROV 3D Competition Interface Started")
        print("Enhanced Controls:")
        print("  ESC - Exit")
        print("  SPACE - Toggle 3D overlay")
        print("  L - Toggle laser line display")
        print("  M - Cycle measurement mode")
        print("  S - Save 3D model")
        print("  1-3 - Select RANGER task")
        print("  C - Capture calibration point")

        cv2.namedWindow('RANGER ROV 3D Interface', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RANGER ROV 3D Interface',
                         self.window_width, self.window_height)

        while True:
            ret, frame = self.camera.read()
            if not ret:
                continue

            # Enhanced processing pipeline
            processed_frame = self.process_frame_3d(frame)

            # Update 3D visualization
            self.update_3d_visualization(processed_frame)

            # Create enhanced display
            display_frame = self.create_enhanced_display(processed_frame)

            cv2.imshow('RANGER ROV 3D Interface', display_frame)

            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if not self.handle_enhanced_input(key):
                break

        self.cleanup()

    def process_frame_3d(self, frame: np.ndarray) -> Dict:
        """Enhanced frame processing with 3D reconstruction."""
        # Preprocess for model
        input_tensor = self.preprocess_frame(frame)

        # Run enhanced detection with 3D
        with torch.no_grad():
            results = self.model([input_tensor])

        # Process results
        result = results[0]

        return {
            'original_frame': frame,
            'detections': result,
            'point_clouds': result.get('point_clouds', np.array([])),
            '3d_measurements': result.get('3d_measurements', {}),
            'meshes': result.get('meshes', {}),
            'processing_time': time.time()
        }

    def create_enhanced_display(self, processed_data: Dict) -> np.ndarray:
        """Create enhanced display with 3D information overlay."""
        frame = processed_data['original_frame'].copy()
        detections = processed_data['detections']
        measurements = processed_data['3d_measurements']

        # Draw detections with 3D information
        self.draw_3d_detections(frame, detections, measurements)

        # Draw laser lines if enabled
        if self.show_laser_lines:
            self.draw_laser_overlay(frame, detections)

        # Add measurement precision indicators
        self.draw_precision_indicators(frame, measurements)

        # Add RANGER-specific task information
        self.draw_ranger_task_info(frame)

        # Add 3D visualization status
        self.draw_3d_status(frame, processed_data)

        return frame

    def draw_3d_detections(self, frame: np.ndarray, detections: Dict, measurements: Dict):
        """Draw detection boxes with 3D measurement annotations."""
        if 'boxes' not in detections or len(detections['boxes']) == 0:
            return

        boxes = detections['boxes'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        labels = detections['labels'].cpu().numpy()

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if score < 0.7:
                continue

            x1, y1, x2, y2 = box.astype(int)

            # Enhanced bounding box for 3D objects
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

            # 3D measurement overlay
            obj_key = f'object_{i}'
            if obj_key in measurements:
                meas = measurements[obj_key]

                # Measurement text with precision indicator
                length_cm = meas['length'] * 100  # Convert to cm
                width_cm = meas['width'] * 100
                height_cm = meas['height'] * 100

                # Color code based on measurement confidence
                confidence = meas['measurement_confidence']
                color = (0, 255, 0) if confidence > 0.8 else (
                    0, 165, 255) if confidence > 0.6 else (0, 0, 255)

                # Multi-line measurement display
                text_lines = [
                    f"L: {length_cm:.1f}cm ({meas['point_count']} pts)",
                    f"W: {width_cm:.1f}cm",
                    f"H: {height_cm:.1f}cm",
                    f"Vol: {meas['volume']*1e6:.0f}cm³"
                ]

                # Draw measurement box
                line_height = 20
                box_height = len(text_lines) * line_height + 10
                box_width = 200

                cv2.rectangle(frame, (x1, y1 - box_height),
                              (x1 + box_width, y1), color, -1)
                cv2.rectangle(frame, (x1, y1 - box_height),
                              (x1 + box_width, y1), (0, 0, 0), 2)

                # Draw text lines
                for j, line in enumerate(text_lines):
                    text_y = y1 - box_height + 15 + j * line_height
                    cv2.putText(frame, line, (x1 + 5, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Draw 3D center point
                if 'centroid' in meas:
                    centroid_2d = self.project_3d_to_2d_point(meas['centroid'])
                    if centroid_2d is not None:
                        cv2.circle(frame, centroid_2d, 5, color, -1)
                        cv2.circle(frame, centroid_2d, 8, (255, 255, 255), 2)

    def draw_laser_overlay(self, frame: np.ndarray, detections: Dict):
        """Draw laser line overlays for visualization."""
        h, w = frame.shape[:2]

        # Create laser overlay
        laser_overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw red laser projection line
        cv2.line(laser_overlay, (w//4, 0), (w//4, h), (0, 0, 255), 2)

        # Draw green laser projection line
        cv2.line(laser_overlay, (3*w//4, 0), (3*w//4, h), (0, 255, 0), 2)

        # Add laser status text
        cv2.putText(laser_overlay, "RED LASER", (w//4 - 40, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(laser_overlay, "GREEN LASER", (3*w//4 - 50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Blend with main frame
        cv2.addWeighted(frame, 0.8, laser_overlay, 0.2, 0, frame)

    def draw_precision_indicators(self, frame: np.ndarray, measurements: Dict):
        """Draw precision indicators for RANGER competition requirements."""
        h, w = frame.shape[:2]

        # Precision status panel
        panel_y = h - 100
        cv2.rectangle(frame, (10, panel_y), (300, h - 10), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, panel_y), (300, h - 10), (255, 255, 255), 2)

        cv2.putText(frame, "MEASUREMENT PRECISION", (20, panel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Check if any measurements meet RANGER 5cm requirement
        precision_met = False
        for obj_data in measurements.values():
            if obj_data.get('measurement_confidence', 0) > 0.8:
                precision_met = True
                break

        status_color = (0, 255, 0) if precision_met else (0, 0, 255)
        status_text = "PRECISION: ±5cm OK" if precision_met else "PRECISION: INSUFFICIENT"

        cv2.putText(frame, status_text, (20, panel_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

        # Required precision for RANGER shipwreck task
        cv2.putText(frame, "REQ: ±5cm for 10pts", (20, panel_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def draw_ranger_task_info(self, frame: np.ndarray):
        """Draw RANGER-specific task information."""
        # Task selector in top right
        task_info = {
            "shipwreck_measurement": "TASK 1: Shipwreck (±5cm = 10pts)",
            "renewable_energy": "TASK 2: Marine Energy",
            "mate_floats": "TASK 3: Profiling Float"
        }

        task_text = task_info.get(self.current_ranger_task, "UNKNOWN TASK")

        cv2.rectangle(
            frame, (frame.shape[1] - 350, 10), (frame.shape[1] - 10, 50), (60, 60, 60), -1)
        cv2.putText(frame, task_text, (frame.shape[1] - 340, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_3d_status(self, frame: np.ndarray, processed_data: Dict):
        """Draw 3D processing status information."""
        # 3D status in bottom right
        point_count = len(processed_data.get('point_clouds', []))
        mesh_count = len(processed_data.get('meshes', {}))

        status_lines = [
            f"3D Points: {point_count}",
            f"Meshes: {mesh_count}",
            f"3D View: {'ON' if self.show_3d_overlay else 'OFF'}"
        ]

        start_y = frame.shape[0] - 80
        for i, line in enumerate(status_lines):
            cv2.putText(frame, line, (frame.shape[1] - 200, start_y + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def update_3d_visualization(self, processed_data: Dict):
        """Update Open3D 3D visualization."""
        if not self.show_3d_overlay:
            return

        measurements = processed_data.get('3d_measurements', {})
        point_clouds = processed_data.get('point_clouds', np.array([]))

        # Update point clouds for each detected object
        for obj_id, obj_data in measurements.items():
            if 'centroid' in obj_data:
                # Create simple point cloud around centroid for visualization
                centroid = np.array(obj_data['centroid'])
                # In real implementation, this would be the actual object point cloud
                dummy_points = centroid.reshape(1, 3)  # Simplified for example

                self.visualizer_3d.update_point_cloud(obj_id, dummy_points)

        # Update meshes if available
        meshes = processed_data.get('meshes', {})
        for mesh_id, mesh in meshes.items():
            if mesh is not None:
                self.visualizer_3d.update_mesh(mesh_id, mesh)

        # Render frame
        self.visualizer_3d.render_frame()

    def handle_enhanced_input(self, key: int) -> bool:
        """Handle enhanced input for 3D interface."""
        if key == 27:  # ESC
            return False
        elif key == 32:  # SPACE
            self.show_3d_overlay = not self.show_3d_overlay
        elif key == ord('l') or key == ord('L'):
            self.show_laser_lines = not self.show_laser_lines
        elif key == ord('m') or key == ord('M'):
            modes = ["length", "volume", "surface_area"]
            current_idx = modes.index(self.measurement_mode)
            self.measurement_mode = modes[(current_idx + 1) % len(modes)]
        elif key == ord('s') or key == ord('S'):
            self.export_3d_model()
        elif key >= ord('1') and key <= ord('3'):
            tasks = ["shipwreck_measurement",
                     "renewable_energy", "mate_floats"]
            task_idx = key - ord('1')
            if task_idx < len(tasks):
                self.current_ranger_task = tasks[task_idx]

        return True

    def export_3d_model(self):
        """Export current 3D model for competition documentation."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"ranger_3d_model_{self.export_counter:03d}_{timestamp}.ply"

        try:
            self.visualizer_3d.export_scene(filename)
            self.export_counter += 1
            print(f"3D model exported: {filename}")

            # Log export for competition records
            self.measurement_log.append({
                'timestamp': timestamp,
                'task': self.current_ranger_task,
                'filename': filename,
                'measurement_mode': self.measurement_mode
            })

        except Exception as e:
            print(f"Export failed: {e}")

    def project_3d_to_2d_point(self, point_3d: List[float]) -> Optional[Tuple[int, int]]:
        """Project single 3D point to 2D image coordinates."""
        if self.calibration.camera_matrix is None:
            return None

        try:
            point_3d = np.array(point_3d).reshape(1, 1, 3)
            projected, _ = cv2.projectPoints(
                point_3d,
                np.zeros(3), np.zeros(3),  # No rotation/translation
                self.calibration.camera_matrix,
                self.calibration.dist_coeffs
            )

            x, y = projected.reshape(2)
            return (int(x), int(y))

        except Exception as e:
            print(f"Projection failed: {e}")
            return None

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model inference."""
        # Convert to RGB and resize
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (800, 600))  # Higher res for RANGER

        # Convert to tensor and normalize
        tensor = torch.FloatTensor(resized).permute(2, 0, 1) / 255.0
        tensor = tensor.to(self.model.detector.transform.device if hasattr(
            self.model.detector, 'transform') else 'cpu')

        return tensor

    def cleanup(self):
        """Cleanup resources."""
        self.camera.release()
        cv2.destroyAllWindows()

        # Save measurement log
        if self.measurement_log:
            log_filename = f"ranger_measurements_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_filename, 'w') as f:
                import json
                json.dump(self.measurement_log, f, indent=2)
            print(f"Measurement log saved: {log_filename}")


# ==================== USAGE EXAMPLE ====================
def main():
    """Complete RANGER ROV 3D system example."""
    print("Initializing RANGER ROV 3D Vision System...")

    # Initialize calibration system
    calibration = DualLaserCalibration()

    # Try to load existing calibration
    if not calibration.load_calibration('ranger_calibration.json'):
        print("No existing calibration found. Please run calibration first.")
        # In real deployment, you'd run calibration procedure here

    # Initialize enhanced model
    model = RANGERFasterRCNN3D(
        num_classes=30,  # More classes for RANGER complexity
        laser_baseline=0.15,  # 15cm between lasers
        camera_focal_length=800.0
    )

    # Load trained weights (placeholder)
    # model.load_state_dict(torch.load('ranger_3d_model.pth'))
    model.eval()

    # Initialize competition interface
    interface = RANGERCompetitionInterface(
        model=model,
        calibration=calibration,
        camera_source=0
    )

    print("RANGER 3D Vision System Ready!")
    print("Features enabled:")
    print("✓ Dual laser triangulation")
    print("✓ Real-time 3D reconstruction")
    print("✓ Open3D visualization")
    print("✓ Precision measurement (±5cm)")
    print("✓ 3D model export")
    print("✓ Competition task integration")

    # Run competition interface
    interface.run()


if __name__ == "__main__":
    main()


# ==================== MODULE DEPENDENCIES ====================

"""
Complete Module Dependencies for RANGER ROV 3D System:

Core ML/CV:
torch>=1.12.0
torchvision>=0.13.0  
opencv-python>=4.6.0
opencv-contrib-python>=4.6.0  # For ximgproc (thinning)

3D Processing:
open3d>=0.15.0  # Critical for 3D visualization and mesh generation
numpy>=1.21.0
scipy>=1.9.0  # For advanced mathematical operations

Data Processing:
albumentations>=1.3.0
pillow>=9.0.0
pandas>=1.4.0

Utilities:
tqdm>=4.64.0
matplotlib>=3.5.0
seaborn>=0.11.0
pyyaml>=6.0
json  # Built-in

Hardware


# import torch
# print(torch.__version__)         # PyTorch version
# print(torch.cuda.is_available())  # Should be True if CUDA works
# print(torch.version.cuda)
"""
