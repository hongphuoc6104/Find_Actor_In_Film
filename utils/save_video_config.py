#!/usr/bin/env python
"""
Utility to save video-specific configuration parameters.

Usage:
    python utils/save_video_config.py --movie VIDEO_NAME --params params.yaml
    python utils/save_video_config.py --movie VIDEO_NAME --from-current
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def save_video_config(
    movie_name: str,
    params_to_save: Dict[str, Any],
    description: Optional[str] = None,
    output_dir: Path = Path("configs/videos")
):
    """
    Save tuned parameters for a specific video.
    
    Args:
        movie_name: Video name
        params_to_save: Dict of params to override (e.g., clustering, merge, post_merge)
        description: Optional comment about why these params work
        output_dir: Directory to save config files
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare output path
    video_cfg_path = output_dir / f"{movie_name}.yaml"
    
    # Prepare config data with metadata
    config_data = {
        "# Video-specific config for": movie_name,
        "# Created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    if description:
        config_data["# Description"] = description
    
    # Add the actual parameters
    config_data.update(params_to_save)
    
    # Write to file
    with open(video_cfg_path, 'w', encoding='utf-8') as f:
        # Write comments separately to ensure proper formatting
        f.write(f"# Video-specific config for: {movie_name}\n")
        f.write(f"# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if description:
            f.write(f"# Description: {description}\n")
        f.write("\n")
        
        # Write actual YAML data (without the comment keys)
        actual_params = {k: v for k, v in params_to_save.items() if not k.startswith("#")}
        yaml.dump(actual_params, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    
    print(f"✅ Saved config for '{movie_name}' to: {video_cfg_path}")
    print(f"   Next time you run: python flows/pipeline.py --movie {movie_name}")
    print(f"   It will automatically use these parameters!")
    
    return video_cfg_path


def load_current_config() -> Dict[str, Any]:
    """Load current base config."""
    from utils.config_loader import load_config
    return load_config()


def main():
    parser = argparse.ArgumentParser(description="Save video-specific configuration")
    parser.add_argument("--movie", required=True, help="Movie/video name")
    parser.add_argument("--params", help="Path to YAML file with parameters to save")
    parser.add_argument("--from-current", action="store_true", 
                       help="Save current loaded config")
    parser.add_argument("--description", help="Description of why these params work")
    
    # Individual parameter overrides
    parser.add_argument("--distance-threshold", type=float, 
                       help="Clustering distance threshold")
    parser.add_argument("--merge-threshold", type=float,
                       help="Merge within movie threshold")
    parser.add_argument("--post-merge-threshold", type=float,
                       help="Post-merge distance threshold")
    parser.add_argument("--min-size", type=int,
                       help="Minimum cluster size")
    
    args = parser.parse_args()
    
    # Determine params to save
    if args.params:
        # Load from provided YAML file
        with open(args.params, 'r', encoding='utf-8') as f:
            params_to_save = yaml.safe_load(f)
    elif args.from_current:
        # Use current config
        current_cfg = load_current_config()
        # Extract key parameters (you can customize this)
        params_to_save = {
            "clustering": current_cfg.get("clustering", {}),
            "merge": current_cfg.get("merge", {}),
            "post_merge": current_cfg.get("post_merge", {}),
            "filter_clusters": current_cfg.get("filter_clusters", {}),
            "quality_filters": current_cfg.get("quality_filters", {})
        }
    else:
        # Build from individual args
        params_to_save = {}
        
        if args.distance_threshold is not None:
            params_to_save.setdefault("clustering", {})["distance_threshold"] = {
                "default": args.distance_threshold
            }
        
        if args.merge_threshold is not None:
            params_to_save["merge"] = {"within_movie_threshold": args.merge_threshold}
        
        if args.post_merge_threshold is not None:
            params_to_save["post_merge"] = {"distance_threshold": args.post_merge_threshold}
        
        if args.min_size is not None:
            params_to_save["filter_clusters"] = {"min_size": args.min_size}
        
        if not params_to_save:
            print("❌ Error: No parameters specified!")
            print("   Use --params, --from-current, or individual parameter flags")
            parser.print_help()
            return
    
    # Save the config
    save_video_config(
        movie_name=args.movie,
        params_to_save=params_to_save,
        description=args.description
    )


if __name__ == "__main__":
    main()
