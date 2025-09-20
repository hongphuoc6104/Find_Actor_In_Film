import argparse
import os
from typing import Any, Dict, List

import cv2

from services.recognition import recognize
from utils.config_loader import load_config


def _display(title: str, paths: List[str]) -> None:
    """Show a list of images sequentially."""
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        cv2.imshow(title, img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize search results")
    parser.add_argument("--image", required=True, help="Query image path")
    parser.add_argument("--top-k", type=int, default=None, help="Number of candidates")
    args = parser.parse_args()

    cfg = load_config()
    frames_root = cfg["storage"].get("frames_root", "")

    result = recognize(args.image, top_k=args.top_k)
    candidates: Dict[str, List[Dict[str, Any]]] = result.get("candidates", {})

    if result.get("is_unknown", True):
        print("Unknown face. Showing nearest candidates...")
        for movie_id, movie_candidates in candidates.items():
            for cand in movie_candidates:
                movie = cand.get("movie") or movie_id
                print(f"Candidate {cand['character_id']} - Movie: {movie}")
                images: List[str] = []
                rep = cand.get("rep_image")
                if rep:
                    rep_movie = rep.get("movie")
                    frame = rep.get("frame")
                    if rep_movie and frame:
                        images.append(os.path.join(frames_root, rep_movie, frame))
                images.extend(cand.get("preview_paths", []))
                _display(f"{cand['character_id']}:{movie}", images)
    else:
        print("Recognized face. Showing frames by movie...")
        for movie_id, movie_candidates in candidates.items():
            movie_label = movie_candidates[0].get("movie") if movie_candidates else movie_id
            print(f"Movie: {movie_label}")
            for cand in movie_candidates:
                print(f"  Character {cand['character_id']}")
                images: List[str] = []
                rep = cand.get("rep_image")
                if rep:
                    rep_movie = rep.get("movie")
                    frame = rep.get("frame")
                    if rep_movie and frame:
                        images.append(os.path.join(frames_root, rep_movie, frame))
                images.extend(cand.get("preview_paths", []))
                _display(f"{cand['character_id']}:{movie_label}", images)


if __name__ == "__main__":
    main()
