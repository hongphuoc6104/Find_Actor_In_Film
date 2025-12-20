"""
Dataset Validator for warehouse/labeled_faces

Validates structure and generates statistics for ground truth dataset.
"""

import os
import json
from pathlib import Path
from typing import Dict, List
from collections import Counter
from PIL import Image


def validate_labeled_faces_dataset(dataset_path: str = "warehouse/labeled_faces") -> Dict:
    """
    Validate warehouse/labeled_faces structure and gather statistics.
    
    Expected structure:
        warehouse/labeled_faces/
            Actor1/
                image1.jpg
                image2.jpg
            Actor2/
                image3.jpg
            ...
    
    Args:
        dataset_path: Path to labeled_faces directory
        
    Returns:
        Dict with validation results and statistics
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        return {
            'valid': False,
            'error': f"Dataset path does not exist: {dataset_path}",
            'actors': [],
            'total_images': 0
        }
    
    if not dataset_path.is_dir():
        return {
            'valid': False,
            'error': f"Dataset path is not a directory: {dataset_path}",
            'actors': [],
            'total_images': 0
        }
    
    # Scan for actor folders
    actors = []
    issues = []
    total_images = 0
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    for actor_dir in sorted(dataset_path.iterdir()):
        if not actor_dir.is_dir():
            issues.append(f"Non-directory item found: {actor_dir.name}")
            continue
        
        actor_name = actor_dir.name
        
        # Count valid images
        image_files = []
        for img_path in actor_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                # Try to open to verify it's valid
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    image_files.append(img_path.name)
                except Exception as e:
                    issues.append(f"Invalid image {actor_name}/{img_path.name}: {str(e)}")
        
        if len(image_files) == 0:
            issues.append(f"Actor folder '{actor_name}' has no valid images")
        
        actors.append({
            'name': actor_name,
            'image_count': len(image_files),
            'images': image_files[:5]  # First 5 for preview
        })
        
        total_images += len(image_files)
    
    # Check if dataset is empty
    if len(actors) == 0:
        return {
            'valid': False,
            'error': "No actor folders found in dataset",
            'actors': [],
            'total_images': 0,
            'issues': issues
        }
    
    # Statistics
    image_counts = [a['image_count'] for a in actors]
    
    result = {
        'valid': True,
        'dataset_path': str(dataset_path),
        'num_actors': len(actors),
        'total_images': total_images,
        'actors': actors,
        'statistics': {
            'avg_images_per_actor': total_images / len(actors) if actors else 0,
            'min_images': min(image_counts) if image_counts else 0,
            'max_images': max(image_counts) if image_counts else 0,
            'distribution': Counter(image_counts)
        },
        'issues': issues
    }
    
    return result


def print_validation_report(validation_result: Dict):
    """
    Print human-readable validation report.
    
    Args:
        validation_result: Dict from validate_labeled_faces_dataset()
    """
    print("\n" + "=" * 60)
    print("📁 LABELED FACES DATASET VALIDATION")
    print("=" * 60)
    
    if not validation_result['valid']:
        print(f"\n❌ VALIDATION FAILED")
        print(f"Error: {validation_result.get('error', 'Unknown error')}")
        if 'issues' in validation_result and validation_result['issues']:
            print(f"\nIssues found:")
            for issue in validation_result['issues']:
                print(f"  - {issue}")
        print("=" * 60 + "\n")
        return
    
    # Valid dataset
    print(f"\n✅ VALIDATION PASSED\n")
    print(f"📂 Dataset Path: {validation_result['dataset_path']}")
    print(f"👥 Number of Actors: {validation_result['num_actors']}")
    print(f"🖼️  Total Images: {validation_result['total_images']}")
    
    # Statistics
    stats = validation_result['statistics']
    print(f"\n📊 Statistics:")
    print(f"   • Average images per actor: {stats['avg_images_per_actor']:.1f}")
    print(f"   • Min images: {stats['min_images']}")
    print(f"   • Max images: {stats['max_images']}")
    
    # Actor breakdown
    print(f"\n👤 Actors:")
    for actor in validation_result['actors']:
        print(f"   • {actor['name']:30s}: {actor['image_count']:3d} images")
    
    # Issues (warnings)
    if validation_result['issues']:
        print(f"\n⚠️  Warnings ({len(validation_result['issues'])}):")
        for issue in validation_result['issues'][:10]:  # Show first 10
            print(f"   - {issue}")
        if len(validation_result['issues']) > 10:
            print(f"   ... and {len(validation_result['issues']) - 10} more")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if stats['min_images'] < 10:
        print(f"   ⚠️  Some actors have fewer than 10 images. Consider adding more for better evaluation.")
    if validation_result['num_actors'] < 5:
        print(f"   ℹ️  Only {validation_result['num_actors']} actors. Consider adding more for comprehensive evaluation.")
    if stats['min_images'] >= 10 and validation_result['num_actors'] >= 5:
        print(f"   ✅ Dataset looks good for evaluation!")
    
    print("\n" + "=" * 60 + "\n")


def save_dataset_metadata(validation_result: Dict, output_path: str = "warehouse/labeled_faces/metadata.json"):
    """
    Save dataset metadata to JSON.
    
    Args:
        validation_result: Dict from validate_labeled_faces_dataset()
        output_path: Path to save metadata
    """
    if not validation_result['valid']:
        print(f"[Warning] Cannot save metadata for invalid dataset")
        return
    
    metadata = {
        'dataset_name': 'Labeled Faces Ground Truth',
        'num_actors': validation_result['num_actors'],
        'total_images': validation_result['total_images'],
        'actors': [a['name'] for a in validation_result['actors']],
        'statistics': validation_result['statistics']
    }
    
    # Convert Counter to dict for JSON serialization
    if 'distribution' in metadata['statistics']:
        metadata['statistics']['distribution'] = dict(metadata['statistics']['distribution'])
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"[Info] Dataset metadata saved to: {output_path}")


if __name__ == "__main__":
    """Run validation as standalone script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate labeled_faces dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="warehouse/labeled_faces",
        help="Path to labeled_faces directory"
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save metadata.json file"
    )
    
    args = parser.parse_args()
    
    result = validate_labeled_faces_dataset(args.dataset)
    print_validation_report(result)
    
    if args.save_metadata and result['valid']:
        save_dataset_metadata(result)
